"""Permutation-equivariant graph policy for executable inverse electron flow.

The policy never emits atom-map numbers or tool syntax.  Atom maps are private
executor handles used only to connect a scored graph node/container to the
existing MechET runtime.  The learned action is factorized as

``action family -> source container -> sink container -> continue/commit``.

Endpoint-context molecules are selected with a graph-to-graph retrieval head.
Electron-participating imports instead use a typed, map-free molecular-graph
decoder and remain subject to the same executor gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from rdkit import Chem

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by optional install
    raise ImportError("graph_electron_policy requires the 'train' extra") from exc

from .forward_expert import ElectronContainer, ElectronMove
from .graph_fragment_actions import (
    REACTIVE_ROLES,
    FragmentAtom,
    FragmentBond,
    ReactiveFragmentProgram,
    replay_reactive_fragment,
)
from .transactional_event_space import MoveInventory


ACTION_FAMILIES = ("FLOW", "BE_DELTA", "IMPORT_ENV", "IMPORT_REACTIVE", "FINISH")
CONTAINER_KINDS = {"LP": 0, "ATOM": 1, "BOND": 2, "RADICAL_PAIR": 3}
FRAGMENT_BOND_TYPES = (1, 2, 3, 12, 17)


@dataclass(frozen=True)
class GraphTensor:
    """One disconnected molecular graph and private map-to-row correspondence."""

    atoms: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    maps: tuple[int, ...]

    def to(self, device: torch.device | str) -> "GraphTensor":
        return GraphTensor(
            atoms=self.atoms.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            maps=self.maps,
        )


def _charge_bucket(charge: int) -> int:
    return max(-5, min(5, int(charge))) + 5


@lru_cache(maxsize=16384)
def _smiles_to_graph_cached(
    smiles: str,
    target_maps: tuple[int, ...],
    require_maps: bool,
) -> GraphTensor:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid graph SMILES: {smiles!r}")
    maps = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    if require_maps and (any(value <= 0 for value in maps) or len(set(maps)) != len(maps)):
        raise ValueError("current and target states require unique positive private maps")
    target_set = set(target_maps)
    atom_rows: list[list[int]] = []
    for atom in mol.GetAtoms():
        atom_rows.append(
            [
                min(118, atom.GetAtomicNum()),
                _charge_bucket(atom.GetFormalCharge()),
                min(8, atom.GetDegree()),
                min(8, atom.GetTotalNumHs()),
                int(atom.GetIsAromatic()),
                min(4, atom.GetNumRadicalElectrons()),
                int(not target_set or atom.GetAtomMapNum() in target_set),
            ]
        )
    edges: list[tuple[int, int]] = []
    edge_rows: list[list[int]] = []
    for bond in mol.GetBonds():
        order = min(3, max(1, int(round(bond.GetBondTypeAsDouble()))))
        feature = [order, int(bond.GetIsConjugated()), int(bond.IsInRing())]
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend(((left, right), (right, left)))
        edge_rows.extend((feature, feature))
    return GraphTensor(
        atoms=torch.tensor(atom_rows, dtype=torch.long),
        edge_index=(
            torch.tensor(edges, dtype=torch.long).T
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        ),
        edge_attr=(
            torch.tensor(edge_rows, dtype=torch.long)
            if edge_rows
            else torch.empty((0, 3), dtype=torch.long)
        ),
        maps=maps,
    )


def smiles_to_graph(
    smiles: str,
    *,
    target_maps: Sequence[int] | None = None,
    require_maps: bool = True,
) -> GraphTensor:
    """Convert mapped SMILES to cached tensors without exposing map IDs."""

    return _smiles_to_graph_cached(
        str(smiles), tuple(int(value) for value in (target_maps or ())), require_maps
    )


def strip_atom_maps(smiles: str) -> str:
    """Canonical fragment-bank key; atom maps never define fragment identity."""

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid fragment SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        aggregate = torch.zeros_like(nodes)
        degree = nodes.new_zeros((nodes.shape[0], 1))
        if edge_index.numel():
            source, destination = edge_index
            messages = self.message(torch.cat((nodes[source], edges), dim=-1))
            # Autocast may produce BF16 messages while the numerically stable
            # accumulation buffer remains FP32. ``index_add_`` requires an
            # exact dtype match, so promote only at the reduction boundary.
            aggregate.index_add_(0, destination, messages.to(aggregate.dtype))
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=nodes.dtype)[:, None])
        aggregate = aggregate / degree.clamp_min(1.0).sqrt()
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class MolecularGraphEncoder(nn.Module):
    """Small dependency-free GNN; no PyG/DGL requirement."""

    def __init__(self, hidden_dim: int = 192, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.atom_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_dim) for size in (119, 11, 9, 9, 2, 5, 2)
        )
        self.edge_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_dim) for size in (4, 2, 2)
        )
        self.layers = nn.ModuleList(
            EdgeMessageLayer(hidden_dim, dropout) for _ in range(num_layers)
        )

    def forward(self, graph: GraphTensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = sum(
            embedding(graph.atoms[:, column])
            for column, embedding in enumerate(self.atom_embeddings)
        )
        edges = (
            sum(
                embedding(graph.edge_attr[:, column])
                for column, embedding in enumerate(self.edge_embeddings)
            )
            if graph.edge_attr.numel()
            else nodes.new_zeros((0, nodes.shape[-1]))
        )
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index, edges)
        return nodes, nodes.mean(dim=0)

    def forward_batch(
        self, graphs: Sequence[GraphTensor]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
        """Encode disconnected graphs in one packed GNN invocation."""

        if not graphs:
            raise ValueError("graph batch cannot be empty")
        device = self.atom_embeddings[0].weight.device
        atom_parts: list[torch.Tensor] = []
        edge_index_parts: list[torch.Tensor] = []
        edge_attr_parts: list[torch.Tensor] = []
        graph_parts: list[torch.Tensor] = []
        slices: list[tuple[int, int]] = []
        offset = 0
        for graph_index, graph in enumerate(graphs):
            count = int(graph.atoms.shape[0])
            if count == 0:
                raise ValueError("molecular graph cannot be empty")
            atom_parts.append(graph.atoms.to(device, non_blocking=True))
            graph_parts.append(
                torch.full((count,), graph_index, dtype=torch.long, device=device)
            )
            if graph.edge_index.numel():
                edge_index_parts.append(
                    graph.edge_index.to(device, non_blocking=True) + offset
                )
                edge_attr_parts.append(graph.edge_attr.to(device, non_blocking=True))
            slices.append((offset, offset + count))
            offset += count
        atoms = torch.cat(atom_parts)
        edge_index = (
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )
        edge_attr = (
            torch.cat(edge_attr_parts)
            if edge_attr_parts
            else torch.empty((0, 3), dtype=torch.long, device=device)
        )
        nodes = sum(
            embedding(atoms[:, column])
            for column, embedding in enumerate(self.atom_embeddings)
        )
        edges = (
            sum(
                embedding(edge_attr[:, column])
                for column, embedding in enumerate(self.edge_embeddings)
            )
            if edge_attr.numel()
            else nodes.new_zeros((0, nodes.shape[-1]))
        )
        for layer in self.layers:
            nodes = layer(nodes, edge_index, edges)
        graph_index = torch.cat(graph_parts)
        pools = nodes.new_zeros((len(graphs), nodes.shape[-1]))
        pools.index_add_(0, graph_index, nodes)
        counts = torch.bincount(graph_index, minlength=len(graphs)).to(nodes.dtype)
        pools = pools / counts[:, None].clamp_min(1.0)
        return nodes, pools, tuple(slices)


@dataclass
class PolicyContext:
    current_nodes: torch.Tensor
    current_maps: tuple[int, ...]
    vector: torch.Tensor


class GraphElectronPolicy(nn.Module):
    """Hierarchical graph actor with value and action-value heads."""

    def __init__(self, hidden_dim: int = 192, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = MolecularGraphEncoder(hidden_dim, num_layers, dropout)
        self.context = nn.Sequential(
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.family_head = nn.Linear(hidden_dim, len(ACTION_FAMILIES))
        self.kind_embedding = nn.Embedding(len(CONTAINER_KINDS), hidden_dim)
        self.container = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.source_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.sink_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.event_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.commit_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.be_operation_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.be_pair_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.be_atom_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.be_delta_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.import_query = nn.Linear(hidden_dim, hidden_dim)
        self.import_key = nn.Linear(hidden_dim, hidden_dim)
        self.reactive_role_head = nn.Linear(hidden_dim, len(REACTIVE_ROLES))
        self.reactive_role_embedding = nn.Embedding(len(REACTIVE_ROLES), hidden_dim)
        self.empty_fragment = nn.Parameter(torch.zeros(hidden_dim))
        self.fragment_state = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        # ADD_ATOM, ADD_BOND, COMMIT_FRAGMENT
        self.fragment_operation_head = nn.Linear(hidden_dim, 3)
        self.fragment_element_head = nn.Linear(hidden_dim, 119)
        self.fragment_charge_head = nn.Linear(hidden_dim, 11)
        self.fragment_h_head = nn.Linear(hidden_dim, 9)
        self.fragment_no_implicit_head = nn.Linear(hidden_dim, 2)
        self.fragment_radical_head = nn.Linear(hidden_dim, 5)
        self.fragment_chiral_head = nn.Linear(hidden_dim, 8)
        self.fragment_parent_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.fragment_pair_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.fragment_bond_type_head = nn.Linear(hidden_dim, len(FRAGMENT_BOND_TYPES))
        self.fragment_active_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.finish_embedding = nn.Parameter(torch.zeros(hidden_dim))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_context(self, current_smiles: str, target_smiles: str) -> PolicyContext:
        target = smiles_to_graph(target_smiles).to(self.device)
        target_maps = target.maps
        current = smiles_to_graph(current_smiles, target_maps=target_maps).to(self.device)
        current_nodes, current_pool = self.encoder(current)
        _, target_pool = self.encoder(target)
        vector = self.context(
            torch.cat(
                (current_pool, target_pool, torch.abs(current_pool - target_pool), current_pool * target_pool),
                dim=-1,
            )
        )
        return PolicyContext(current_nodes=current_nodes, current_maps=current.maps, vector=vector)

    def encode_context_batch(
        self,
        current_graphs: Sequence[GraphTensor],
        target_graphs: Sequence[GraphTensor],
    ) -> list[PolicyContext]:
        """Encode a real mini-batch while retaining private map/node bindings."""

        if len(current_graphs) != len(target_graphs) or not current_graphs:
            raise ValueError("current/target graph batches must be equally non-empty")
        current_nodes, current_pools, current_slices = self.encoder.forward_batch(
            current_graphs
        )
        _, target_pools, _ = self.encoder.forward_batch(target_graphs)
        vectors = self.context(
            torch.cat(
                (
                    current_pools,
                    target_pools,
                    torch.abs(current_pools - target_pools),
                    current_pools * target_pools,
                ),
                dim=-1,
            )
        )
        return [
            PolicyContext(
                current_nodes=current_nodes[start:end],
                current_maps=current_graphs[index].maps,
                vector=vectors[index],
            )
            for index, (start, end) in enumerate(current_slices)
        ]

    def encode_fragment_batch(self, graphs: Sequence[GraphTensor]) -> torch.Tensor:
        """Return pooled embeddings for a packed fragment candidate batch."""

        return self.encoder.forward_batch(graphs)[1]

    def family_logits(self, context: PolicyContext) -> torch.Tensor:
        return self.family_head(context.vector)

    def value(self, context: PolicyContext) -> torch.Tensor:
        return self.value_head(context.vector).squeeze(-1)

    def encode_containers(
        self, context: PolicyContext, containers: Sequence[ElectronContainer]
    ) -> torch.Tensor:
        if not containers:
            return context.vector.new_empty((0, self.hidden_dim))
        positions = {atom_map: index for index, atom_map in enumerate(context.current_maps)}
        rows: list[torch.Tensor] = []
        for item in containers:
            try:
                left = context.current_nodes[positions[item.atoms[0]]]
                right = (
                    context.current_nodes[positions[item.atoms[1]]]
                    if len(item.atoms) == 2
                    else torch.zeros_like(left)
                )
            except KeyError as exc:
                raise ValueError(f"container references missing private map: {item}") from exc
            kind = self.kind_embedding.weight[CONTAINER_KINDS[item.kind]]
            rows.append(self.container(torch.cat((left + right, torch.abs(left - right), kind, left))))
        return torch.stack(rows)

    def source_logits(
        self,
        context: PolicyContext,
        sources: Sequence[ElectronContainer],
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_containers(context, sources)
        history = torch.zeros_like(context.vector) if history is None else history
        repeated = torch.cat(
            (
                encoded,
                context.vector.expand(encoded.shape[0], -1),
                history.expand(encoded.shape[0], -1),
            ),
            dim=-1,
        )
        return self.source_head(repeated).squeeze(-1), encoded

    def sink_logits(
        self,
        context: PolicyContext,
        source_embedding: torch.Tensor,
        sinks: Sequence[ElectronContainer],
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_containers(context, sinks)
        history = torch.zeros_like(context.vector) if history is None else history
        repeated = torch.cat(
            (
                encoded,
                source_embedding.expand(encoded.shape[0], -1),
                context.vector.expand(encoded.shape[0], -1),
                history.expand(encoded.shape[0], -1),
            ),
            dim=-1,
        )
        return self.sink_head(repeated).squeeze(-1), encoded

    def flow_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        moves: Sequence[ElectronMove | Mapping[str, Any]],
        *,
        context: PolicyContext | None = None,
        inventory: MoveInventory | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced NLL for one atomic coupled electron event."""

        if not moves:
            raise ValueError("FLOW event requires at least one move")
        context = context or self.encode_context(current_smiles, target_smiles)
        inventory = inventory or MoveInventory.from_state(current_smiles)
        parsed = [item if isinstance(item, ElectronMove) else ElectronMove.parse(item) for item in moves]
        family_target = torch.tensor(ACTION_FAMILIES.index("FLOW"), device=self.device)
        family_loss = F.cross_entropy(self.family_logits(context)[None, :], family_target[None])
        history = torch.zeros_like(context.vector)
        source_loss = context.vector.new_zeros(())
        sink_loss = context.vector.new_zeros(())
        commit_loss = context.vector.new_zeros(())
        action_embedding = context.vector.new_zeros((self.hidden_dim,))
        for index, move in enumerate(parsed):
            if move.source not in inventory.sources:
                raise ValueError(f"gold source outside state-derived inventory: {move.source}")
            sources = inventory.sources
            source_scores, source_embeddings = self.source_logits(context, sources, history)
            source_index = sources.index(move.source)
            source_loss = source_loss + F.cross_entropy(
                source_scores[None, :], torch.tensor([source_index], device=self.device)
            )
            sinks = inventory.compatible_sinks(move.source)
            if move.sink not in sinks:
                raise ValueError(f"gold sink outside source-conditioned inventory: {move.sink}")
            selected_source = source_embeddings[source_index]
            sink_scores, sink_embeddings = self.sink_logits(context, selected_source, sinks, history)
            sink_index = sinks.index(move.sink)
            sink_loss = sink_loss + F.cross_entropy(
                sink_scores[None, :], torch.tensor([sink_index], device=self.device)
            )
            action_embedding = torch.tanh(selected_source + sink_embeddings[sink_index])
            history = self.event_cell(action_embedding[None, :], history[None, :]).squeeze(0)
            # class 0 = continue adding arrows, class 1 = atomically commit event
            commit_target = int(index == len(parsed) - 1)
            commit_scores = self.commit_head(torch.cat((context.vector, history), dim=-1))
            commit_loss = commit_loss + F.cross_entropy(
                commit_scores[None, :], torch.tensor([commit_target], device=self.device)
            )
        total = family_loss + source_loss + sink_loss + commit_loss
        return total, {
            "family": float(family_loss.detach()),
            "source": float(source_loss.detach()),
            "sink": float(sink_loss.detach()),
            "commit": float(commit_loss.detach()),
        }

    def import_logits(
        self,
        context: PolicyContext,
        fragments: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not fragments:
            raise ValueError("fragment candidate set is empty")
        pools = []
        for fragment in fragments:
            graph = smiles_to_graph(strip_atom_maps(fragment), require_maps=False).to(self.device)
            _, pool = self.encoder(graph)
            pools.append(pool)
        keys = self.import_key(torch.stack(pools))
        query = F.normalize(self.import_query(context.vector), dim=-1)
        keys = F.normalize(keys, dim=-1)
        return keys @ query, keys

    def be_delta_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        payload: Mapping[str, Any],
        *,
        context: PolicyContext | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """NLL for a sparse BE-matrix edit sequence followed by atomic commit.

        Candidate bond positions are all unordered atom pairs, so this head can
        represent both bond deletion/order reduction and new-bond formation in
        ``O(n^2)`` choices per sparse edit.  Charge changes use ``O(n)`` atom
        choices.  The executor still validates the coupled edit atomically.
        """

        if payload.get("mode") != "BE_DELTA":
            raise ValueError("BE head requires a BE_DELTA payload")
        context = context or self.encode_context(current_smiles, target_smiles)
        family_target = torch.tensor(ACTION_FAMILIES.index("BE_DELTA"), device=self.device)
        family_loss = F.cross_entropy(self.family_logits(context)[None, :], family_target[None])
        history = torch.zeros_like(context.vector)
        operation_loss = context.vector.new_zeros(())
        position_loss = context.vector.new_zeros(())
        delta_loss = context.vector.new_zeros(())
        positions = {atom_map: index for index, atom_map in enumerate(context.current_maps)}

        operations: list[tuple[int, tuple[int, ...], int]] = []
        for item in payload.get("bond_deltas") or ():
            atoms = tuple(sorted(int(value) for value in item.get("atoms") or ()))
            delta = int(item.get("delta") or 0)
            if len(atoms) != 2 or delta not in {-1, 1}:
                raise ValueError(f"unsupported sparse bond delta: {item}")
            operations.append((0, atoms, delta))
        for item in payload.get("charge_actions") or ():
            atom_map = int(item.get("atom_map") or 0)
            delta = int(item.get("q1") or 0) - int(item.get("q0") or 0)
            if atom_map <= 0 or delta not in {-1, 1}:
                raise ValueError(f"unsupported sparse charge delta: {item}")
            operations.append((1, (atom_map,), delta))
        if not operations:
            raise ValueError("BE_DELTA payload contains no sparse edits")

        pair_maps = tuple(
            (context.current_maps[left], context.current_maps[right])
            for left in range(len(context.current_maps))
            for right in range(left + 1, len(context.current_maps))
        )
        pair_containers = tuple(ElectronContainer("BOND", pair) for pair in pair_maps)
        pair_embeddings = self.encode_containers(context, pair_containers)
        atom_maps = tuple(context.current_maps)
        atom_embeddings = context.current_nodes

        for operation, location, delta in operations:
            operation_scores = self.be_operation_head(torch.cat((context.vector, history), dim=-1))
            operation_loss = operation_loss + F.cross_entropy(
                operation_scores[None, :], torch.tensor([operation], device=self.device)
            )
            if operation == 0:
                normalized = tuple(sorted(location))
                normalized_pairs = tuple(tuple(sorted(pair)) for pair in pair_maps)
                if normalized not in normalized_pairs:
                    raise ValueError(f"BE pair references missing private maps: {location}")
                selected = normalized_pairs.index(normalized)
                repeated = torch.cat(
                    (
                        pair_embeddings,
                        context.vector.expand(len(pair_maps), -1),
                        history.expand(len(pair_maps), -1),
                        pair_embeddings * context.vector.expand(len(pair_maps), -1),
                    ),
                    dim=-1,
                )
                scores = self.be_pair_head(repeated).squeeze(-1)
                action_embedding = pair_embeddings[selected]
            else:
                if location[0] not in positions:
                    raise ValueError(f"BE charge references missing private map: {location[0]}")
                selected = atom_maps.index(location[0])
                repeated = torch.cat(
                    (
                        atom_embeddings,
                        context.vector.expand(len(atom_maps), -1),
                        history.expand(len(atom_maps), -1),
                    ),
                    dim=-1,
                )
                scores = self.be_atom_head(repeated).squeeze(-1)
                action_embedding = atom_embeddings[selected]
            position_loss = position_loss + F.cross_entropy(
                scores[None, :], torch.tensor([selected], device=self.device)
            )
            delta_scores = self.be_delta_head(torch.cat((context.vector, action_embedding), dim=-1))
            delta_target = 0 if delta == -1 else 1
            delta_loss = delta_loss + F.cross_entropy(
                delta_scores[None, :], torch.tensor([delta_target], device=self.device)
            )
            history = self.event_cell(action_embedding[None, :], history[None, :]).squeeze(0)

        commit_scores = self.be_operation_head(torch.cat((context.vector, history), dim=-1))
        operation_loss = operation_loss + F.cross_entropy(
            commit_scores[None, :], torch.tensor([2], device=self.device)
        )
        total = family_loss + operation_loss + position_loss + delta_loss
        return total, {
            "family": float(family_loss.detach()),
            "operation": float(operation_loss.detach()),
            "position": float(position_loss.detach()),
            "delta": float(delta_loss.detach()),
        }

    def import_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        fragments: Sequence[str],
        gold_index: int,
        *,
        context: PolicyContext | None = None,
        candidate_pools: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = context or self.encode_context(current_smiles, target_smiles)
        family_target = torch.tensor(ACTION_FAMILIES.index("IMPORT_ENV"), device=self.device)
        family_loss = F.cross_entropy(self.family_logits(context)[None, :], family_target[None])
        if candidate_pools is None:
            scores, _ = self.import_logits(context, fragments)
        else:
            if candidate_pools.shape[0] != len(fragments):
                raise ValueError("candidate pool count does not match fragment count")
            keys = F.normalize(self.import_key(candidate_pools), dim=-1)
            query = F.normalize(self.import_query(context.vector), dim=-1)
            scores = keys @ query
        fragment_loss = F.cross_entropy(
            scores[None, :], torch.tensor([int(gold_index)], device=self.device)
        )
        return family_loss + fragment_loss

    def _fragment_prefix_graph(
        self,
        program: ReactiveFragmentProgram,
        atom_count: int,
        extra_bond_count: int,
    ) -> GraphTensor:
        if atom_count < 1:
            raise ValueError("empty fragment uses the learned empty embedding")
        if atom_count > len(program.atoms) or extra_bond_count > len(program.extra_bonds):
            raise ValueError("fragment prefix exceeds reference program")
        bonds: list[tuple[int, int, int]] = []
        for index, atom in enumerate(program.atoms[:atom_count]):
            if atom.parent is not None:
                if atom.parent >= atom_count:
                    raise ValueError("fragment parent is outside prefix")
                bonds.append((index, int(atom.parent), int(atom.parent_bond_type)))
        for bond in program.extra_bonds[:extra_bond_count]:
            if max(bond.atoms) >= atom_count:
                raise ValueError("extra bond is outside atom prefix")
            bonds.append((int(bond.atoms[0]), int(bond.atoms[1]), int(bond.bond_type)))
        degree = [0 for _ in range(atom_count)]
        edge_index: list[tuple[int, int]] = []
        edge_attr: list[list[int]] = []
        for left, right, bond_type in bonds:
            degree[left] += 1
            degree[right] += 1
            order = bond_type if bond_type in {1, 2, 3} else 1
            edge_index.extend(((left, right), (right, left)))
            edge_attr.extend(([order, 0, 0], [order, 0, 0]))
        atom_rows = []
        for index, atom in enumerate(program.atoms[:atom_count]):
            atom_rows.append(
                [
                    min(118, int(atom.atomic_num)),
                    _charge_bucket(int(atom.formal_charge)),
                    min(8, degree[index]),
                    min(8, int(atom.explicit_h)),
                    0,
                    min(4, int(atom.radical_electrons)),
                    0,
                ]
            )
        return GraphTensor(
            atoms=torch.tensor(atom_rows, dtype=torch.long, device=self.device),
            edge_index=(
                torch.tensor(edge_index, dtype=torch.long, device=self.device).T
                if edge_index
                else torch.empty((2, 0), dtype=torch.long, device=self.device)
            ),
            edge_attr=(
                torch.tensor(edge_attr, dtype=torch.long, device=self.device)
                if edge_attr
                else torch.empty((0, 3), dtype=torch.long, device=self.device)
            ),
            maps=tuple(range(1, atom_count + 1)),
        )

    def _reactive_fragment_state(
        self,
        context: PolicyContext,
        program: ReactiveFragmentProgram,
        atom_count: int,
        extra_bond_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        role_index = REACTIVE_ROLES.index(program.role)
        role = self.reactive_role_embedding.weight[role_index]
        if atom_count:
            nodes, pool = self.encoder(
                self._fragment_prefix_graph(program, atom_count, extra_bond_count)
            )
        else:
            nodes = context.vector.new_empty((0, self.hidden_dim))
            pool = self.empty_fragment
        state = self.fragment_state(torch.cat((context.vector, role, pool), dim=-1))
        return nodes, state

    def reactive_fragment_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        program: ReactiveFragmentProgram,
        *,
        context: PolicyContext | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced graph-program loss for an open-vocabulary import."""

        context = context or self.encode_context(current_smiles, target_smiles)
        family_target = torch.tensor(
            ACTION_FAMILIES.index("IMPORT_REACTIVE"), device=self.device
        )
        family_loss = F.cross_entropy(
            self.family_logits(context)[None, :], family_target[None]
        )
        role_index = REACTIVE_ROLES.index(program.role)
        role_loss = F.cross_entropy(
            self.reactive_role_head(context.vector)[None, :],
            torch.tensor([role_index], device=self.device),
        )
        operation_loss = context.vector.new_zeros(())
        atom_loss = context.vector.new_zeros(())
        position_loss = context.vector.new_zeros(())
        bond_loss = context.vector.new_zeros(())

        for atom_index, atom in enumerate(program.atoms):
            nodes, state = self._reactive_fragment_state(
                context, program, atom_index, 0
            )
            operation_loss = operation_loss + F.cross_entropy(
                self.fragment_operation_head(state)[None, :],
                torch.tensor([0], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_element_head(state)[None, :],
                torch.tensor([int(atom.atomic_num)], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_charge_head(state)[None, :],
                torch.tensor([_charge_bucket(int(atom.formal_charge))], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_h_head(state)[None, :],
                torch.tensor([min(8, int(atom.explicit_h))], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_no_implicit_head(state)[None, :],
                torch.tensor([int(atom.no_implicit)], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_radical_head(state)[None, :],
                torch.tensor([min(4, int(atom.radical_electrons))], device=self.device),
            )
            atom_loss = atom_loss + F.cross_entropy(
                self.fragment_chiral_head(state)[None, :],
                torch.tensor([min(7, int(atom.chiral_tag))], device=self.device),
            )
            if atom.parent is not None:
                parent_scores = self.fragment_parent_head(
                    torch.cat((nodes, state.expand(nodes.shape[0], -1)), dim=-1)
                ).squeeze(-1)
                position_loss = position_loss + F.cross_entropy(
                    parent_scores[None, :],
                    torch.tensor([int(atom.parent)], device=self.device),
                )
                if int(atom.parent_bond_type) not in FRAGMENT_BOND_TYPES:
                    raise ValueError(f"unsupported fragment bond type: {atom.parent_bond_type}")
                bond_loss = bond_loss + F.cross_entropy(
                    self.fragment_bond_type_head(state)[None, :],
                    torch.tensor(
                        [FRAGMENT_BOND_TYPES.index(int(atom.parent_bond_type))],
                        device=self.device,
                    ),
                )

        existing = {
            tuple(sorted((index, int(atom.parent))))
            for index, atom in enumerate(program.atoms)
            if atom.parent is not None
        }
        for extra_index, bond in enumerate(program.extra_bonds):
            nodes, state = self._reactive_fragment_state(
                context, program, len(program.atoms), extra_index
            )
            operation_loss = operation_loss + F.cross_entropy(
                self.fragment_operation_head(state)[None, :],
                torch.tensor([1], device=self.device),
            )
            candidates = tuple(
                (left, right)
                for left in range(nodes.shape[0])
                for right in range(left + 1, nodes.shape[0])
                if (left, right) not in existing
            )
            target_pair = tuple(sorted(map(int, bond.atoms)))
            if target_pair not in candidates:
                raise ValueError(f"extra bond outside legal pair inventory: {target_pair}")
            pair_embeddings = torch.stack(
                [
                    torch.cat(
                        (
                            nodes[left] + nodes[right],
                            torch.abs(nodes[left] - nodes[right]),
                            state,
                        ),
                        dim=-1,
                    )
                    for left, right in candidates
                ]
            )
            pair_scores = self.fragment_pair_head(pair_embeddings).squeeze(-1)
            position_loss = position_loss + F.cross_entropy(
                pair_scores[None, :],
                torch.tensor([candidates.index(target_pair)], device=self.device),
            )
            if int(bond.bond_type) not in FRAGMENT_BOND_TYPES:
                raise ValueError(f"unsupported fragment bond type: {bond.bond_type}")
            bond_loss = bond_loss + F.cross_entropy(
                self.fragment_bond_type_head(state)[None, :],
                torch.tensor(
                    [FRAGMENT_BOND_TYPES.index(int(bond.bond_type))], device=self.device
                ),
            )
            existing.add(target_pair)

        nodes, state = self._reactive_fragment_state(
            context, program, len(program.atoms), len(program.extra_bonds)
        )
        operation_loss = operation_loss + F.cross_entropy(
            self.fragment_operation_head(state)[None, :],
            torch.tensor([2], device=self.device),
        )
        active_logits = self.fragment_active_head(
            torch.cat((nodes, state.expand(nodes.shape[0], -1)), dim=-1)
        ).squeeze(-1)
        active_targets = torch.zeros_like(active_logits)
        active_targets[list(program.active_atoms)] = 1.0
        active_loss = F.binary_cross_entropy_with_logits(active_logits, active_targets)

        # One reaction-level decision must not receive a gradient proportional
        # to reagent size.  Average each factor over its supervised choices;
        # otherwise a large reactive fragment overwhelms FLOW/FINISH examples.
        operation_loss = operation_loss / (
            len(program.atoms) + len(program.extra_bonds) + 1
        )
        atom_loss = atom_loss / (6 * len(program.atoms))
        located_bonds = max(0, len(program.atoms) - 1) + len(program.extra_bonds)
        if located_bonds:
            position_loss = position_loss / located_bonds
            bond_loss = bond_loss / located_bonds
        total = (
            family_loss
            + role_loss
            + operation_loss
            + atom_loss
            + position_loss
            + bond_loss
            + active_loss
        )
        return total, {
            "family": float(family_loss.detach()),
            "role": float(role_loss.detach()),
            "operation": float(operation_loss.detach()),
            "atom": float(atom_loss.detach()),
            "position": float(position_loss.detach()),
            "bond": float(bond_loss.detach()),
            "active": float(active_loss.detach()),
        }

    @staticmethod
    def _sample_index(
        logits: torch.Tensor,
        *,
        greedy: bool,
        temperature: float,
        allowed: torch.Tensor | None = None,
    ) -> tuple[int, float]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scores = logits / temperature
        if allowed is not None:
            if allowed.shape != scores.shape or not bool(allowed.any()):
                raise ValueError("action mask has no legal choice")
            scores = scores.masked_fill(~allowed, -torch.inf)
        distribution = torch.distributions.Categorical(logits=scores)
        index = int(scores.argmax()) if greedy else int(distribution.sample())
        return index, float(distribution.log_prob(torch.tensor(index, device=scores.device)))

    @torch.no_grad()
    def rollout_reactive_fragment(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        role: str | None = None,
        max_atoms: int = 32,
        max_extra_bonds: int = 8,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Generate one reactive fragment option for executor/RL interaction.

        The rollout is map-free.  On successful commit it returns canonical
        unmapped SMILES and one active atom *position*; the environment creates
        private maps only when the fragment is imported.  Invalid valence or
        sanitization is an explicit rejected transition, suitable for replay.
        """

        if max_atoms < 1 or max_extra_bonds < 0:
            raise ValueError("invalid fragment rollout limits")
        context = self.encode_context(current_smiles, target_smiles)
        total_logprob = 0.0
        if role is None:
            role_index, value = self._sample_index(
                self.reactive_role_head(context.vector),
                greedy=greedy,
                temperature=temperature,
            )
            role = REACTIVE_ROLES[role_index]
            total_logprob += value
        elif role not in REACTIVE_ROLES:
            raise ValueError(f"unknown reactive role: {role}")

        atoms: list[FragmentAtom] = []
        extra_bonds: list[FragmentBond] = []
        operations: list[dict[str, Any]] = []
        max_operations = max_atoms + max_extra_bonds + 1
        for _ in range(max_operations):
            # A one-carbon dummy makes the typed program valid while the empty
            # prefix path still uses the learned empty-fragment embedding.
            holder_atoms = tuple(atoms) or (FragmentAtom(6),)
            holder = ReactiveFragmentProgram(
                role=role,
                atoms=holder_atoms,
                extra_bonds=tuple(extra_bonds),
                active_atoms=(0,),
                source_unmapped_smiles="",
            )
            nodes, state = self._reactive_fragment_state(
                context, holder, len(atoms), len(extra_bonds)
            )
            existing = {
                tuple(sorted((index, int(atom.parent))))
                for index, atom in enumerate(atoms)
                if atom.parent is not None
            }
            existing.update(tuple(sorted(bond.atoms)) for bond in extra_bonds)
            pair_candidates = tuple(
                (left, right)
                for left in range(len(atoms))
                for right in range(left + 1, len(atoms))
                if (left, right) not in existing
            )
            allowed = torch.tensor(
                [
                    len(atoms) < max_atoms,
                    bool(pair_candidates) and len(extra_bonds) < max_extra_bonds,
                    bool(atoms),
                ],
                dtype=torch.bool,
                device=self.device,
            )
            operation, value = self._sample_index(
                self.fragment_operation_head(state),
                greedy=greedy,
                temperature=temperature,
                allowed=allowed,
            )
            total_logprob += value
            if operation == 2:
                active_logits = self.fragment_active_head(
                    torch.cat((nodes, state.expand(nodes.shape[0], -1)), dim=-1)
                ).squeeze(-1)
                active, value = self._sample_index(
                    active_logits, greedy=greedy, temperature=temperature
                )
                total_logprob += value
                program = ReactiveFragmentProgram(
                    role=role,
                    atoms=tuple(atoms),
                    extra_bonds=tuple(extra_bonds),
                    active_atoms=(active,),
                    source_unmapped_smiles="",
                )
                try:
                    fragment = replay_reactive_fragment(program)
                except Exception as exc:
                    return {
                        "ok": False,
                        "code": "INVALID_REACTIVE_FRAGMENT",
                        "message": str(exc),
                        "role": role,
                        "program": program,
                        "operations": operations,
                        "logprob": total_logprob,
                    }
                return {
                    "ok": True,
                    "fragment": fragment,
                    "role": role,
                    "active_atom": active,
                    "program": program,
                    "operations": operations,
                    "logprob": total_logprob,
                }
            if operation == 0:
                element_allowed = torch.ones(119, dtype=torch.bool, device=self.device)
                element_allowed[0] = False
                atomic_num, value = self._sample_index(
                    self.fragment_element_head(state),
                    greedy=greedy,
                    temperature=temperature,
                    allowed=element_allowed,
                )
                total_logprob += value
                charge, value = self._sample_index(
                    self.fragment_charge_head(state), greedy=greedy, temperature=temperature
                )
                total_logprob += value
                explicit_h, value = self._sample_index(
                    self.fragment_h_head(state), greedy=greedy, temperature=temperature
                )
                total_logprob += value
                no_implicit, value = self._sample_index(
                    self.fragment_no_implicit_head(state), greedy=greedy, temperature=temperature
                )
                total_logprob += value
                radical, value = self._sample_index(
                    self.fragment_radical_head(state), greedy=greedy, temperature=temperature
                )
                total_logprob += value
                chiral, value = self._sample_index(
                    self.fragment_chiral_head(state), greedy=greedy, temperature=temperature
                )
                total_logprob += value
                parent = parent_bond_type = None
                if atoms:
                    parent_logits = self.fragment_parent_head(
                        torch.cat((nodes, state.expand(nodes.shape[0], -1)), dim=-1)
                    ).squeeze(-1)
                    parent, value = self._sample_index(
                        parent_logits, greedy=greedy, temperature=temperature
                    )
                    total_logprob += value
                    bond_index, value = self._sample_index(
                        self.fragment_bond_type_head(state),
                        greedy=greedy,
                        temperature=temperature,
                    )
                    total_logprob += value
                    parent_bond_type = FRAGMENT_BOND_TYPES[bond_index]
                atom = FragmentAtom(
                    atomic_num=atomic_num,
                    formal_charge=charge - 5,
                    explicit_h=explicit_h,
                    no_implicit=bool(no_implicit),
                    radical_electrons=radical,
                    chiral_tag=chiral,
                    parent=parent,
                    parent_bond_type=parent_bond_type,
                )
                atoms.append(atom)
                operations.append({"op": "ADD_ATOM", "atom": asdict(atom)})
                continue

            pair_embeddings = torch.stack(
                [
                    torch.cat(
                        (
                            nodes[left] + nodes[right],
                            torch.abs(nodes[left] - nodes[right]),
                            state,
                        ),
                        dim=-1,
                    )
                    for left, right in pair_candidates
                ]
            )
            pair_scores = self.fragment_pair_head(pair_embeddings).squeeze(-1)
            pair_index, value = self._sample_index(
                pair_scores, greedy=greedy, temperature=temperature
            )
            total_logprob += value
            bond_index, value = self._sample_index(
                self.fragment_bond_type_head(state),
                greedy=greedy,
                temperature=temperature,
            )
            total_logprob += value
            bond = FragmentBond(
                atoms=pair_candidates[pair_index],
                bond_type=FRAGMENT_BOND_TYPES[bond_index],
            )
            extra_bonds.append(bond)
            operations.append({"op": "ADD_BOND", "bond": asdict(bond)})
        return {
            "ok": False,
            "code": "REACTIVE_FRAGMENT_BUDGET",
            "message": "fragment rollout reached its operation budget",
            "role": role,
            "operations": operations,
            "logprob": total_logprob,
        }

    def finish_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        context: PolicyContext | None = None,
    ) -> torch.Tensor:
        context = context or self.encode_context(current_smiles, target_smiles)
        target = torch.tensor(ACTION_FAMILIES.index("FINISH"), device=self.device)
        return F.cross_entropy(self.family_logits(context)[None, :], target[None])

    def q_value(self, context: PolicyContext, action_embedding: torch.Tensor) -> torch.Tensor:
        return self.q_head(torch.cat((context.vector, action_embedding), dim=-1)).squeeze(-1)


def expectile_loss(residual: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """IQL value loss for ``residual = Q.detach() - V``."""

    if not 0.5 <= expectile < 1.0:
        raise ValueError("expectile must be in [0.5, 1.0)")
    weight = torch.where(residual >= 0, expectile, 1.0 - expectile)
    return (weight * residual.square()).mean()


def graph_iql_losses(
    *,
    q: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    log_prob: torch.Tensor,
    discount: float = 0.99,
    expectile: float = 0.7,
    temperature: float = 3.0,
    max_weight: float = 100.0,
) -> dict[str, torch.Tensor]:
    """Stable offline-IQL/AWR objectives for executor-labelled transitions."""

    target_q = reward + discount * (1.0 - done.float()) * next_value.detach()
    critic = F.mse_loss(q, target_q)
    residual = q.detach() - value
    value_loss = expectile_loss(residual, expectile)
    weights = torch.exp(temperature * residual).clamp(max=max_weight).detach()
    actor = -(weights * log_prob).mean()
    return {"critic": critic, "value": value_loss, "actor": actor}
