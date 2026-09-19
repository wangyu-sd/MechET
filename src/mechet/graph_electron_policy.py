"""Permutation-equivariant graph policy for executable inverse electron flow.

The policy never emits atom-map numbers or tool syntax.  Atom maps are private
executor handles used only to connect a scored graph node/container to the
existing MechET runtime.  The learned action is factorized as

``action family -> source container -> sink container -> continue/commit``.

Imported molecules are selected with a graph-to-graph retrieval head.  This is
intentionally a closed fragment bank for the first implementation: arbitrary
SMILES generation is a separate problem and must not weaken the executor gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rdkit import Chem

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by optional install
    raise ImportError("graph_electron_policy requires the 'train' extra") from exc

from .forward_expert import ElectronContainer, ElectronMove
from .transactional_event_space import MoveInventory


ACTION_FAMILIES = ("FLOW", "BE_DELTA", "IMPORT", "FINISH")
CONTAINER_KINDS = {"LP": 0, "ATOM": 1, "BOND": 2, "RADICAL_PAIR": 3}


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


def smiles_to_graph(
    smiles: str,
    *,
    target_maps: Sequence[int] | None = None,
    require_maps: bool = True,
) -> GraphTensor:
    """Convert mapped SMILES to graph tensors without exposing map IDs as features."""

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid graph SMILES: {smiles!r}")
    maps = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    if require_maps and (any(value <= 0 for value in maps) or len(set(maps)) != len(maps)):
        raise ValueError("current and target states require unique positive private maps")
    target_set = set(int(value) for value in (target_maps or ()))
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
            aggregate.index_add_(0, destination, messages)
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced NLL for one atomic coupled electron event."""

        if not moves:
            raise ValueError("FLOW event requires at least one move")
        context = self.encode_context(current_smiles, target_smiles)
        inventory = MoveInventory.from_state(current_smiles)
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """NLL for a sparse BE-matrix edit sequence followed by atomic commit.

        Candidate bond positions are all unordered atom pairs, so this head can
        represent both bond deletion/order reduction and new-bond formation in
        ``O(n^2)`` choices per sparse edit.  Charge changes use ``O(n)`` atom
        choices.  The executor still validates the coupled edit atomically.
        """

        if payload.get("mode") != "BE_DELTA":
            raise ValueError("BE head requires a BE_DELTA payload")
        context = self.encode_context(current_smiles, target_smiles)
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
    ) -> torch.Tensor:
        context = self.encode_context(current_smiles, target_smiles)
        family_target = torch.tensor(ACTION_FAMILIES.index("IMPORT"), device=self.device)
        family_loss = F.cross_entropy(self.family_logits(context)[None, :], family_target[None])
        scores, _ = self.import_logits(context, fragments)
        fragment_loss = F.cross_entropy(
            scores[None, :], torch.tensor([int(gold_index)], device=self.device)
        )
        return family_loss + fragment_loss

    def finish_nll(self, current_smiles: str, target_smiles: str) -> torch.Tensor:
        context = self.encode_context(current_smiles, target_smiles)
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
