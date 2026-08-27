import torch

from src.data import utils


class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)       # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, 10)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                    batched_eigenvalues)))
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)

            return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                                     E=extra_edge_attr,
                                     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)

        if self.mode == 'eigenvalues':
            eigvals, _ = eigh_real_nodes(L, mask, compute_eigenvectors=False)
            eigvals = eigvals.type_as(A)                                 # bs, n
            eigvals = eigvals / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = eigh_real_nodes(L, mask, compute_eigenvectors=True)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors.type_as(A)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=noisy_data['node_mask'],
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def eigh_real_nodes(laplacian, node_mask, compute_eigenvectors, bucket_size=16):
    """Diagonalize valid graph blocks with a batched fast path and safe fallbacks.

    ``to_dense_batch`` right-pads graphs to the largest graph in a batch.  Sending
    that padded tensor directly to a symmetric eigensolver creates a large,
    repeated artificial spectrum and was the source of intermittent CUDA
    convergence failures. Graphs with similar valid-node counts are grouped
    into small size buckets. Padding inside a bucket receives distinct high
    diagonal values, so it cannot mix with the molecular spectrum or recreate
    the repeated-eigenvalue failure.

    The normal path uses the input floating-point dtype (float32 in training),
    which is substantially faster than unconditional per-graph float64.  If a
    grouped solve fails, only the affected group is retried graph-by-graph in
    float64, first on the current device and then on CPU with a final tiny
    diagonal jitter.  Results are returned in the input dtype so mixed-precision
    training does not retain large float64 tensors in the autograd graph.
    """
    if laplacian.ndim != 3 or laplacian.shape[-1] != laplacian.shape[-2]:
        raise ValueError(f"Expected a batched square Laplacian, got {laplacian.shape}")
    if node_mask.shape != laplacian.shape[:2]:
        raise ValueError(
            f"node_mask shape {node_mask.shape} does not match Laplacian {laplacian.shape}"
        )
    if bucket_size < 1:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    batch_size, max_nodes, _ = laplacian.shape
    output_dtype = laplacian.dtype
    if output_dtype not in (torch.float32, torch.float64):
        # CUDA symmetric eigensolvers do not support fp16/bf16. ExtraFeatures
        # currently constructs float32 adjacency, but keep this helper robust.
        output_dtype = torch.float32
    eigenvalues = torch.full(
        (batch_size, max_nodes),
        2 * max_nodes,
        dtype=output_dtype,
        device=laplacian.device,
    )
    eigenvectors = (
        torch.zeros(
            (batch_size, max_nodes, max_nodes),
            dtype=output_dtype,
            device=laplacian.device,
        )
        if compute_eigenvectors
        else None
    )

    node_counts = node_mask.sum(dim=1, dtype=torch.long)
    if (node_counts == 0).any():
        empty = torch.nonzero(node_counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Cannot compute spectral features for empty graphs {empty}")

    # PyG's to_dense_batch guarantees right-padded masks. Fail loudly if this
    # invariant changes, because slicing [:n, :n] would then select wrong nodes.
    packed_mask = torch.arange(max_nodes, device=node_mask.device).unsqueeze(0) < node_counts.unsqueeze(1)
    if not torch.equal(node_mask.bool(), packed_mask):
        raise ValueError("Spectral batching requires right-padded contiguous node masks")

    bucket_sizes = torch.clamp(
        ((node_counts + bucket_size - 1) // bucket_size) * bucket_size,
        max=max_nodes,
    )
    for bucket_nodes_tensor in torch.unique(bucket_sizes, sorted=True):
        bucket_nodes = int(bucket_nodes_tensor.item())
        graph_indices = torch.nonzero(
            bucket_sizes == bucket_nodes_tensor, as_tuple=False
        ).flatten()
        group_node_counts = node_counts.index_select(0, graph_indices)
        graph_laplacians = laplacian.index_select(0, graph_indices)[
            :, :bucket_nodes, :bucket_nodes
        ]
        graph_laplacians = graph_laplacians.to(output_dtype)
        graph_laplacians = (
            graph_laplacians + graph_laplacians.transpose(-1, -2)
        ) / 2
        valid_in_bucket = (
            torch.arange(bucket_nodes, device=node_mask.device).unsqueeze(0)
            < group_node_counts.unsqueeze(1)
        )
        graph_laplacians = _stabilize_spectral_bucket(
            graph_laplacians, valid_in_bucket
        )
        if not torch.isfinite(graph_laplacians).all():
            bad = graph_indices[
                ~torch.isfinite(graph_laplacians).flatten(1).all(dim=1)
            ].tolist()
            raise ValueError(f"Non-finite Laplacian entries in graphs {bad}")

        try:
            values, vectors = _eigh(graph_laplacians, compute_eigenvectors)
        except RuntimeError:
            values, vectors = _eigh_group_with_fallback(
                graph_laplacians,
                graph_indices,
                compute_eigenvectors,
                destination_device=laplacian.device,
                output_dtype=output_dtype,
            )

        eigenvalues[graph_indices, :bucket_nodes] = values.to(output_dtype)
        if eigenvectors is not None:
            vectors = vectors.to(output_dtype) * valid_in_bucket.unsqueeze(-1)
            eigenvectors[graph_indices, :bucket_nodes, :bucket_nodes] = vectors

    return eigenvalues, eigenvectors


def _stabilize_spectral_bucket(graph_laplacians, valid_node_mask):
    """Make right padding and isolated-node degeneracy safe for batched eigh."""
    bucket_nodes = graph_laplacians.shape[-1]
    node_index = torch.arange(
        bucket_nodes,
        device=graph_laplacians.device,
        dtype=graph_laplacians.dtype,
    )
    diagonal = graph_laplacians.diagonal(dim1=-2, dim2=-1)

    # A combinatorial Laplacian has lambda_max <= 2 * max_degree. Values above
    # 4 * bucket_nodes are safely outside the molecular spectrum. Distinct
    # padding values avoid an artificial degenerate eigenspace.
    padding_diagonal = 4 * bucket_nodes + node_index + 1
    diagonal.copy_(torch.where(valid_node_mask, diagonal, padding_diagonal))

    # Explicit RetroBridge dummy nodes can create a large exact zero eigenspace.
    # This deterministic tie-break stays below the 1e-5 connected-component
    # threshold while choosing a stable basis for that degenerate subspace.
    isolated = valid_node_mask & (diagonal == 0)
    tie_break = (node_index + 1) * (5e-6 / (bucket_nodes + 1))
    diagonal.add_(isolated * tie_break)
    return graph_laplacians


def _eigh_group_with_fallback(
    graph_laplacians,
    graph_indices,
    compute_eigenvectors,
    destination_device,
    output_dtype,
):
    """Retry a failed batched solve while isolating pathological graphs."""
    value_rows = []
    vector_rows = []
    for local_index, batch_index_tensor in enumerate(graph_indices):
        batch_index = int(batch_index_tensor.item())
        graph_laplacian = graph_laplacians[local_index].to(torch.float64)
        try:
            values, vectors = _eigh(graph_laplacian, compute_eigenvectors)
        except RuntimeError:
            cpu_laplacian = graph_laplacian.detach().cpu()
            try:
                values, vectors = _eigh(cpu_laplacian, compute_eigenvectors)
            except RuntimeError:
                jitter = torch.linspace(
                    0,
                    1e-10,
                    graph_laplacian.shape[0],
                    dtype=torch.float64,
                    device="cpu",
                )
                try:
                    values, vectors = _eigh(
                        cpu_laplacian + torch.diag(jitter),
                        compute_eigenvectors,
                    )
                except RuntimeError as cpu_error:
                    raise RuntimeError(
                        f"Eigen decomposition failed for graph {batch_index} "
                        f"with {graph_laplacian.shape[0]} valid nodes"
                    ) from cpu_error

        value_rows.append(values.to(device=destination_device, dtype=output_dtype))
        if compute_eigenvectors:
            vector_rows.append(vectors.to(device=destination_device, dtype=output_dtype))

    values = torch.stack(value_rows)
    vectors = torch.stack(vector_rows) if compute_eigenvectors else None
    return values, vectors


def _eigh(matrix, compute_eigenvectors):
    if compute_eigenvectors:
        return torch.linalg.eigh(matrix)
    return torch.linalg.eigvalsh(matrix), None


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


def bmv(m, v):
    """
    Batched matrix-vector product
    :param m: (b, n, m)
    :param v: (b, m)
    :return: (b, n)
    """
    return m.bmm(v.unsqueeze(-1)).squeeze(-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        # Triangle count matrix (indicates for each node i how many triangles it shares with node j)
        T = self.k1_matrix * self.k2_matrix
        c5 = diag_a5 - 2 * bmv(T, self.d) - 2 * self.d * triangles - bmv(self.k1_matrix, triangles) + 5 * triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
