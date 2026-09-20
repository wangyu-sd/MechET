"""Permutation-equivariant graph policy for executable inverse electron flow.

The policy never emits atom-map numbers or tool syntax.  Atom maps are private
executor handles used only to connect a scored graph node/container to the
existing MechET runtime.  The learned action is factorized as

``action family -> container kind -> conditional node pointers -> commit``.

Both environment and electron-participating imports use a typed, map-free
molecular-graph decoder.  Chemistry rules do not construct candidate action
lists; the executor validates the model-proposed event after generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
from typing import Any, Mapping, Sequence

from rdkit import Chem

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised by optional install
    raise ImportError("graph_electron_policy requires the 'train' extra") from exc

from .forward_expert import ElectronContainer, ElectronMove
from .electron_policy_protocol import CompressedTrajectory, IMPORT_ROLES
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
FLOW_SOURCE_KINDS = ("LP", "BOND", "RADICAL_PAIR")
FLOW_SINK_KINDS = ("ATOM", "BOND", "RADICAL_PAIR")
FRAGMENT_BOND_TYPES = (1, 2, 3, 12, 17)
# Frozen strict universe maxima are 69 atoms and 9 non-tree bonds.  Small
# headroom keeps inference well-defined without excluding any training row.
MAX_FRAGMENT_ATOMS = 72
MAX_FRAGMENT_EXTRA_BONDS = 12
MAX_BE_EDITS = 6


@dataclass(frozen=True)
class GraphTensor:
    """One disconnected molecular graph and private map-to-row correspondence."""

    atoms: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    alignment: torch.Tensor
    maps: tuple[int, ...]

    def to(self, device: torch.device | str) -> "GraphTensor":
        return GraphTensor(
            atoms=self.atoms.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            alignment=self.alignment.to(device),
            maps=self.maps,
        )


def _charge_bucket(charge: int) -> int:
    return max(-5, min(5, int(charge))) + 5


@lru_cache(maxsize=16384)
def _smiles_to_graph_cached(
    smiles: str,
    target_smiles: str,
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
    target_by_map: dict[int, Chem.Atom] = {}
    target_bonds: dict[int, dict[int, float]] = {}
    if target_smiles:
        target_mol = Chem.MolFromSmiles(target_smiles, params)
        if target_mol is None:
            raise ValueError(f"invalid target graph SMILES: {target_smiles!r}")
        target_by_map = {
            int(atom.GetAtomMapNum()): atom for atom in target_mol.GetAtoms()
        }
        target_set = set(target_by_map)
        for atom_map, atom in target_by_map.items():
            target_bonds[atom_map] = {
                int(neighbour.GetAtomMapNum()): float(
                    target_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbour.GetIdx()).GetBondTypeAsDouble()
                )
                for neighbour in atom.GetNeighbors()
            }
    atom_rows: list[list[int]] = []
    alignment_rows: list[list[float]] = []
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
                min(255, int(atom.GetIsotope())),
                min(7, int(atom.GetChiralTag())),
                int(atom.GetNoImplicit()),
                min(15, int(atom.GetHybridization())),
            ]
        )
        target_atom = target_by_map.get(int(atom.GetAtomMapNum()))
        if target_atom is None:
            alignment_rows.append([0.0] * 8)
        else:
            current_bonds = {
                int(neighbour.GetAtomMapNum()): float(
                    mol.GetBondBetweenAtoms(atom.GetIdx(), neighbour.GetIdx()).GetBondTypeAsDouble()
                )
                for neighbour in atom.GetNeighbors()
                if int(neighbour.GetAtomMapNum()) in target_set
            }
            reference_bonds = target_bonds.get(int(atom.GetAtomMapNum()), {})
            bond_keys = set(current_bonds) | set(reference_bonds)
            bond_delta = sum(
                abs(current_bonds.get(key, 0.0) - reference_bonds.get(key, 0.0))
                for key in bond_keys
            )
            alignment_rows.append(
                [
                    1.0,
                    float(atom.GetFormalCharge() - target_atom.GetFormalCharge()) / 5.0,
                    float(atom.GetDegree() - target_atom.GetDegree()) / 8.0,
                    float(atom.GetTotalNumHs() - target_atom.GetTotalNumHs()) / 8.0,
                    float(int(atom.GetIsAromatic()) - int(target_atom.GetIsAromatic())),
                    float(atom.GetNumRadicalElectrons() - target_atom.GetNumRadicalElectrons()) / 4.0,
                    min(bond_delta / 6.0, 1.0),
                    float(atom.GetAtomicNum() == target_atom.GetAtomicNum()),
                ]
            )
    edges: list[tuple[int, int]] = []
    edge_rows: list[list[int]] = []
    for bond in mol.GetBonds():
        bond_type = min(31, max(0, int(bond.GetBondType())))
        feature = [
            bond_type,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            min(7, int(bond.GetStereo())),
            min(7, int(bond.GetBondDir())),
        ]
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
            else torch.empty((0, 5), dtype=torch.long)
        ),
        alignment=torch.tensor(alignment_rows, dtype=torch.float32),
        maps=maps,
    )


def smiles_to_graph(
    smiles: str,
    *,
    target_smiles: str | None = None,
    target_maps: Sequence[int] | None = None,
    require_maps: bool = True,
) -> GraphTensor:
    """Convert mapped SMILES to cached tensors without exposing map IDs."""

    return _smiles_to_graph_cached(
        str(smiles),
        str(target_smiles or ""),
        tuple(int(value) for value in (target_maps or ())),
        require_maps,
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
            nn.Embedding(size, hidden_dim)
            for size in (119, 11, 9, 9, 2, 5, 2, 256, 8, 2, 16)
        )
        self.edge_embeddings = nn.ModuleList(
            nn.Embedding(size, hidden_dim) for size in (32, 2, 2, 8, 8)
        )
        self.alignment_projection = nn.Linear(8, hidden_dim, bias=False)
        self.layers = nn.ModuleList(
            EdgeMessageLayer(hidden_dim, dropout) for _ in range(num_layers)
        )

    def forward(self, graph: GraphTensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = sum(
            embedding(graph.atoms[:, column])
            for column, embedding in enumerate(self.atom_embeddings)
        )
        nodes = nodes + self.alignment_projection(graph.alignment)
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
        alignment_parts: list[torch.Tensor] = []
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
            alignment_parts.append(graph.alignment.to(device, non_blocking=True))
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
            else torch.empty((0, 5), dtype=torch.long, device=device)
        )
        nodes = sum(
            embedding(atoms[:, column])
            for column, embedding in enumerate(self.atom_embeddings)
        )
        nodes = nodes + self.alignment_projection(torch.cat(alignment_parts))
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


class CompressedHistoryEncoder(nn.Module):
    """Encode the map-free accepted-action ledger without state repetition."""

    _families = ("NONE",) + ACTION_FAMILIES
    _containers = ("NONE", "LP", "ATOM", "BOND", "RADICAL_PAIR")

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.family = nn.Embedding(len(self._families), hidden_dim)
        self.source = nn.Embedding(len(self._containers), hidden_dim)
        self.sink = nn.Embedding(len(self._containers), hidden_dim)
        self.role = nn.Embedding(len(IMPORT_ROLES), hidden_dim)
        self.site = nn.Embedding(65536, hidden_dim)
        self.outcome = nn.Embedding(2, hidden_dim)
        self.error = nn.Embedding(256, hidden_dim)
        self.counts = nn.Linear(6, hidden_dim)
        self.cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.empty = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self, histories: Sequence[CompressedTrajectory], *, device: torch.device
    ) -> torch.Tensor:
        if not histories:
            return self.empty.new_empty((0, self.hidden_dim))
        hidden = self.empty.to(device).expand(len(histories), -1)
        maximum = max((len(item.events) for item in histories), default=0)
        for step in range(maximum):
            rows: list[list[float]] = []
            families: list[int] = []
            sources: list[int] = []
            sinks: list[int] = []
            roles: list[int] = []
            site_rows: list[torch.Tensor] = []
            outcomes: list[int] = []
            errors: list[int] = []
            active: list[bool] = []
            for history in histories:
                if step >= len(history.events):
                    event = None
                    active.append(False)
                else:
                    event = history.events[step]
                    active.append(True)
                family = event.family if event is not None else "NONE"
                source = event.source_kinds[0] if event and event.source_kinds else "NONE"
                sink = event.sink_kinds[0] if event and event.sink_kinds else "NONE"
                role = event.import_role if event is not None else "NONE"
                families.append(self._families.index(family))
                sources.append(self._containers.index(source))
                sinks.append(self._containers.index(sink))
                roles.append(IMPORT_ROLES.index(role))
                codes = tuple(event.site_codes) if event is not None else ()
                site_rows.append(
                    self.site(torch.tensor(codes, dtype=torch.long, device=device)).mean(dim=0)
                    if codes
                    else self.empty.to(device)
                )
                outcomes.append(int(bool(event.accepted)) if event is not None else 1)
                error_text = event.error_code if event is not None else ""
                errors.append(
                    int.from_bytes(hashlib.sha256(error_text.encode()).digest()[:1], "big")
                    if error_text
                    else 0
                )
                rows.append(
                    [
                        min((event.electron_moves if event else 0) / 8.0, 1.0),
                        min((event.fragment_atoms if event else 0) / 32.0, 1.0),
                        min((event.active_atoms if event else 0) / 8.0, 1.0),
                        min((event.extra_bonds if event else 0) / 8.0, 1.0),
                        min((event.bond_edits if event else 0) / 8.0, 1.0),
                        min((event.charge_edits if event else 0) / 8.0, 1.0),
                    ]
                )
            event_vector = (
                self.family(torch.tensor(families, device=device))
                + self.source(torch.tensor(sources, device=device))
                + self.sink(torch.tensor(sinks, device=device))
                + self.role(torch.tensor(roles, device=device))
                + torch.stack(site_rows)
                + self.outcome(torch.tensor(outcomes, device=device))
                + self.error(torch.tensor(errors, device=device))
                + self.counts(torch.tensor(rows, dtype=torch.float32, device=device))
            )
            candidate = self.cell(event_vector, hidden)
            mask = torch.tensor(active, dtype=torch.bool, device=device)[:, None]
            hidden = torch.where(mask, candidate, hidden)
        return hidden


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
        self.history_encoder = CompressedHistoryEncoder(hidden_dim)
        self.history_residual = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Stage 1 is exactly state-only.  Stage 2 starts from that checkpoint
        # and learns to open this residual gate for useful trajectory context.
        self.history_gate = nn.Parameter(torch.zeros(()))
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
        # Direct electron-event decoder.  It predicts container kinds and
        # graph-node pointers without asking chemistry rules to construct a
        # candidate action list.  Two-atom containers use conditional pointers
        # p(i)p(j|i); a symmetric set likelihood removes arbitrary endpoint
        # ordering and avoids materializing O(n^2) pairs.
        self.direct_source_kind_head = nn.Linear(
            2 * hidden_dim, len(FLOW_SOURCE_KINDS)
        )
        self.direct_source_first_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.direct_source_second_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.direct_sink_kind_head = nn.Linear(3 * hidden_dim, len(FLOW_SINK_KINDS))
        self.direct_sink_first_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.direct_sink_second_head = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.direct_source_kind_embedding = nn.Embedding(
            len(FLOW_SOURCE_KINDS), hidden_dim
        )
        self.direct_sink_kind_embedding = nn.Embedding(
            len(FLOW_SINK_KINDS), hidden_dim
        )
        self.direct_action = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh()
        )
        self.be_operation_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        self.be_pair_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.be_pair_first_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.be_pair_second_head = nn.Sequential(
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
        self.fragment_pair_first_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.fragment_pair_second_head = nn.Sequential(
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

    def freeze_enumerated_action_heads(self) -> None:
        """Exclude legacy candidate-ranking heads from direct-policy training."""

        legacy = (
            self.kind_embedding,
            self.container,
            self.source_head,
            self.sink_head,
            self.import_query,
            self.import_key,
            self.be_pair_head,
            self.fragment_pair_head,
        )
        for module in legacy:
            module.requires_grad_(False)

    def _add_history(
        self,
        vectors: torch.Tensor,
        histories: Sequence[CompressedTrajectory] | None,
    ) -> torch.Tensor:
        if histories is None:
            return vectors
        if len(histories) != vectors.shape[0]:
            raise ValueError("history batch does not match graph batch")
        encoded = self.history_encoder(histories, device=vectors.device)
        update = self.history_residual(torch.cat((vectors, encoded), dim=-1))
        return vectors + torch.tanh(self.history_gate) * update

    def encode_context(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        history: CompressedTrajectory | None = None,
    ) -> PolicyContext:
        target = smiles_to_graph(target_smiles).to(self.device)
        target_maps = target.maps
        current = smiles_to_graph(
            current_smiles,
            target_smiles=target_smiles,
            target_maps=target_maps,
        ).to(self.device)
        current_nodes, current_pool = self.encoder(current)
        _, target_pool = self.encoder(target)
        vector = self.context(
            torch.cat(
                (current_pool, target_pool, torch.abs(current_pool - target_pool), current_pool * target_pool),
                dim=-1,
            )
        )
        vector = self._add_history(
            vector[None, :], None if history is None else (history,)
        ).squeeze(0)
        return PolicyContext(current_nodes=current_nodes, current_maps=current.maps, vector=vector)

    def encode_context_batch(
        self,
        current_graphs: Sequence[GraphTensor],
        target_graphs: Sequence[GraphTensor],
        histories: Sequence[CompressedTrajectory] | None = None,
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
        vectors = self._add_history(vectors, histories)
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

    def forward(
        self,
        decisions: Sequence[Mapping[str, Any]],
        current_graphs: Sequence[GraphTensor],
        target_graphs: Sequence[GraphTensor],
        histories: Sequence[CompressedTrajectory] | None = None,
    ) -> torch.Tensor:
        """Return per-decision BC losses for standard DDP training.

        Keeping every trainable operation inside ``forward`` lets PyTorch DDP
        discover used parameters and bucket gradients.  Diagnostic scalar
        conversion is disabled here so the asynchronous CUDA queue is not
        drained once per loss component.
        """

        if len(decisions) != len(current_graphs):
            raise ValueError("decision and graph batches differ")
        contexts = self.encode_context_batch(
            current_graphs, target_graphs, histories
        )
        losses: list[torch.Tensor] = []
        for value, context in zip(decisions, contexts):
            kind = str(value["kind"])
            if kind == "FLOW":
                loss = self.direct_flow_nll(
                    str(value["current"]),
                    str(value["target"]),
                    value["moves"],
                    context=context,
                    return_parts=False,
                )[0]
            elif kind == "BE_DELTA":
                loss = self.be_delta_nll(
                    str(value["current"]),
                    str(value["target"]),
                    value["moves"][0],
                    context=context,
                    return_parts=False,
                )[0]
            elif kind in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
                loss = self.reactive_fragment_nll(
                    str(value["current"]),
                    str(value["target"]),
                    value["program"],
                    context=context,
                    return_parts=False,
                )[0]
            elif kind == "FINISH":
                loss = self.finish_nll(
                    str(value["current"]), str(value["target"]), context=context
                )
            else:
                raise ValueError(f"unknown decision kind: {kind}")
            losses.append(loss)
        return torch.stack(losses)

    def family_logits(self, context: PolicyContext) -> torch.Tensor:
        return self.family_head(context.vector)

    @torch.no_grad()
    def sample_next_family(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        trajectory: CompressedTrajectory | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Sample one action family directly, without a candidate inventory."""

        context = self.encode_context(
            current_smiles, target_smiles, history=trajectory
        )
        index, logprob = self._sample_index(
            self.family_logits(context), greedy=greedy, temperature=temperature
        )
        return {
            "family": ACTION_FAMILIES[index],
            "logprob": logprob,
            "candidate_enumeration": False,
        }

    def canonical_action_nll(
        self,
        decision: Mapping[str, Any],
        *,
        history: CompressedTrajectory | None = None,
    ) -> torch.Tensor:
        """Differentiable log-probability bridge for stage-3 PPO/GRPO.

        Rollout stores the directly sampled canonical action.  RL recomputes
        its log-probability as ``-canonical_action_nll``; no candidate set is
        reconstructed during collection or optimization.
        """

        current = str(decision["current"])
        target = str(decision["target"])
        context = self.encode_context(current, target, history=history)
        kind = str(decision["kind"])
        if kind == "FLOW":
            return self.direct_flow_nll(
                current,
                target,
                decision["moves"],
                context=context,
                return_parts=False,
            )[0]
        if kind == "BE_DELTA":
            return self.be_delta_nll(
                current,
                target,
                decision["moves"][0],
                context=context,
                return_parts=False,
            )[0]
        if kind in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
            program = decision["program"]
            if isinstance(program, Mapping):
                program = ReactiveFragmentProgram.from_dict(program)
            return self.reactive_fragment_nll(
                current,
                target,
                program,
                context=context,
                return_parts=False,
                normalize=False,
            )[0]
        if kind == "FINISH":
            return self.finish_nll(current, target, context=context)
        raise ValueError(f"unknown canonical action: {kind}")

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

    @staticmethod
    def _pointer_logits(
        nodes: torch.Tensor,
        conditioning: Sequence[torch.Tensor],
        head: nn.Module,
    ) -> torch.Tensor:
        repeated = [value.expand(nodes.shape[0], -1) for value in conditioning]
        return head(torch.cat((nodes, *repeated), dim=-1)).squeeze(-1)

    def _direct_container_nll(
        self,
        *,
        nodes: torch.Tensor,
        maps: tuple[int, ...],
        kind: str,
        atoms: tuple[int, ...],
        kinds: tuple[str, ...],
        kind_logits: torch.Tensor,
        first_conditioning: Sequence[torch.Tensor],
        first_head: nn.Module,
        second_conditioning: Sequence[torch.Tensor],
        second_head: nn.Module,
        kind_embedding: nn.Embedding,
        allowed_containers: Sequence[ElectronContainer],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kind not in kinds:
            raise ValueError(f"unsupported direct container kind: {kind}")
        kind_index = kinds.index(kind)
        allowed_set = set(allowed_containers)
        if ElectronContainer(kind, atoms) not in allowed_set:
            raise ValueError(
                f"gold container is not legal in the current state: {kind} {atoms}"
            )
        kind_mask = torch.tensor(
            [any(item.kind == candidate for item in allowed_set) for candidate in kinds],
            dtype=torch.bool,
            device=self.device,
        )
        kind_logits = kind_logits.masked_fill(~kind_mask, -torch.inf)
        loss = F.cross_entropy(
            kind_logits[None, :], torch.tensor([kind_index], device=self.device)
        )
        positions = {atom_map: index for index, atom_map in enumerate(maps)}
        try:
            target = tuple(positions[int(atom_map)] for atom_map in atoms)
        except KeyError as exc:
            raise ValueError(f"direct pointer target outside current graph: {atoms}") from exc
        first_logits = self._pointer_logits(nodes, first_conditioning, first_head)
        legal_of_kind = tuple(item for item in allowed_set if item.kind == kind)
        first_allowed = torch.zeros_like(first_logits, dtype=torch.bool)
        for item in legal_of_kind:
            for atom_map in item.atoms:
                if atom_map in positions:
                    first_allowed[positions[atom_map]] = True
        if not bool(first_allowed.any()):
            raise ValueError(f"no legal pointer for container kind {kind}")
        first_logits = first_logits.masked_fill(~first_allowed, -torch.inf)
        if len(target) == 1:
            loss = loss + F.cross_entropy(
                first_logits[None, :], torch.tensor([target[0]], device=self.device)
            )
            representation = nodes[target[0]]
        elif len(target) == 2:
            left, right = target

            def ordered_logprob(first: int, second: int) -> torch.Tensor:
                first_logprob = F.log_softmax(first_logits, dim=0)[first]
                second_logits = self._pointer_logits(
                    nodes,
                    (*second_conditioning, nodes[first]),
                    second_head,
                ).clone()
                second_allowed = torch.zeros_like(second_logits, dtype=torch.bool)
                first_map = maps[first]
                for item in legal_of_kind:
                    if first_map not in item.atoms:
                        continue
                    for atom_map in item.atoms:
                        if atom_map != first_map and atom_map in positions:
                            second_allowed[positions[atom_map]] = True
                second_logits = second_logits.masked_fill(~second_allowed, -torch.inf)
                return first_logprob + F.log_softmax(second_logits, dim=0)[second]

            # Either endpoint order is correct.  This set likelihood prevents
            # private atom-map ordering from becoming a supervision shortcut.
            loss = loss - torch.logaddexp(
                ordered_logprob(left, right), ordered_logprob(right, left)
            )
            representation = nodes[left] + nodes[right]
        else:
            raise ValueError(f"invalid direct container arity: {kind} {atoms}")
        representation = torch.tanh(
            representation + kind_embedding.weight[kind_index]
        )
        return loss, representation

    def direct_flow_nll(
        self,
        current_smiles: str,
        target_smiles: str,
        moves: Sequence[ElectronMove | Mapping[str, Any]],
        *,
        context: PolicyContext | None = None,
        return_parts: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced FLOW loss with no state-derived action enumeration."""

        if not moves:
            raise ValueError("FLOW event requires at least one move")
        context = context or self.encode_context(current_smiles, target_smiles)
        inventory = MoveInventory.from_state(current_smiles)
        parsed = [
            item if isinstance(item, ElectronMove) else ElectronMove.parse(item)
            for item in moves
        ]
        family_target = torch.tensor(ACTION_FAMILIES.index("FLOW"), device=self.device)
        family_loss = F.cross_entropy(
            self.family_logits(context)[None, :], family_target[None]
        )
        history = torch.zeros_like(context.vector)
        source_loss = context.vector.new_zeros(())
        sink_loss = context.vector.new_zeros(())
        commit_loss = context.vector.new_zeros(())
        for index, move in enumerate(parsed):
            source_state = torch.cat((context.vector, history), dim=-1)
            source_kind_logits = self.direct_source_kind_head(source_state)
            value, source_embedding = self._direct_container_nll(
                nodes=context.current_nodes,
                maps=context.current_maps,
                kind=move.source.kind,
                atoms=move.source.atoms,
                kinds=FLOW_SOURCE_KINDS,
                kind_logits=source_kind_logits,
                first_conditioning=(source_state,),
                first_head=self.direct_source_first_head,
                second_conditioning=(source_state,),
                second_head=self.direct_source_second_head,
                kind_embedding=self.direct_source_kind_embedding,
                allowed_containers=inventory.sources,
            )
            source_loss = source_loss + value
            sink_state = torch.cat(
                (context.vector, history, source_embedding), dim=-1
            )
            sink_kind_logits = self.direct_sink_kind_head(sink_state)
            value, sink_embedding = self._direct_container_nll(
                nodes=context.current_nodes,
                maps=context.current_maps,
                kind=move.sink.kind,
                atoms=move.sink.atoms,
                kinds=FLOW_SINK_KINDS,
                kind_logits=sink_kind_logits,
                first_conditioning=(sink_state,),
                first_head=self.direct_sink_first_head,
                second_conditioning=(sink_state,),
                second_head=self.direct_sink_second_head,
                kind_embedding=self.direct_sink_kind_embedding,
                allowed_containers=inventory.compatible_sinks(move.source),
            )
            sink_loss = sink_loss + value
            action_embedding = self.direct_action(
                torch.cat((source_embedding, sink_embedding), dim=-1)
            )
            history = self.event_cell(
                action_embedding[None, :], history[None, :]
            ).squeeze(0)
            commit_target = int(index == len(parsed) - 1)
            commit_scores = self.commit_head(
                torch.cat((context.vector, history), dim=-1)
            )
            commit_loss = commit_loss + F.cross_entropy(
                commit_scores[None, :],
                torch.tensor([commit_target], device=self.device),
            )
        total = family_loss + source_loss + sink_loss + commit_loss
        parts = (
            {
                "family": float(family_loss.detach()),
                "source": float(source_loss.detach()),
                "sink": float(sink_loss.detach()),
                "commit": float(commit_loss.detach()),
            }
            if return_parts
            else {}
        )
        return total, parts

    def _sample_direct_container(
        self,
        *,
        context: PolicyContext,
        kinds: tuple[str, ...],
        kind_logits: torch.Tensor,
        first_conditioning: Sequence[torch.Tensor],
        first_head: nn.Module,
        second_conditioning: Sequence[torch.Tensor],
        second_head: nn.Module,
        kind_embedding: nn.Embedding,
        allowed_containers: Sequence[ElectronContainer],
        greedy: bool,
        temperature: float,
    ) -> tuple[ElectronContainer, torch.Tensor, float]:
        allowed_set = set(allowed_containers)
        kind_allowed = torch.tensor(
            [any(item.kind == candidate for item in allowed_set) for candidate in kinds],
            dtype=torch.bool,
            device=self.device,
        )
        kind_index, kind_logprob = self._sample_index(
            kind_logits,
            greedy=greedy,
            temperature=temperature,
            allowed=kind_allowed,
        )
        kind = kinds[kind_index]
        positions = {
            atom_map: index for index, atom_map in enumerate(context.current_maps)
        }
        legal_of_kind = tuple(item for item in allowed_set if item.kind == kind)
        first_logits = self._pointer_logits(
            context.current_nodes, first_conditioning, first_head
        )
        first_allowed = torch.zeros_like(first_logits, dtype=torch.bool)
        for item in legal_of_kind:
            for atom_map in item.atoms:
                if atom_map in positions:
                    first_allowed[positions[atom_map]] = True
        first, value = self._sample_index(
            first_logits,
            greedy=greedy,
            temperature=temperature,
            allowed=first_allowed,
        )
        indices = [first]
        if kind in {"BOND", "RADICAL_PAIR"}:
            second_logits = self._pointer_logits(
                context.current_nodes,
                (*second_conditioning, context.current_nodes[first]),
                second_head,
            )
            allowed = torch.zeros_like(second_logits, dtype=torch.bool)
            first_map = context.current_maps[first]
            for item in legal_of_kind:
                if first_map not in item.atoms:
                    continue
                for atom_map in item.atoms:
                    if atom_map != first_map and atom_map in positions:
                        allowed[positions[atom_map]] = True
            second, value = self._sample_index(
                second_logits,
                greedy=greedy,
                temperature=temperature,
                allowed=allowed,
            )
            indices.append(second)
            first_log_probs = F.log_softmax(
                first_logits.masked_fill(~first_allowed, -torch.inf) / temperature,
                dim=0,
            )

            def ordered_logprob(left: int, right: int) -> torch.Tensor:
                conditional = self._pointer_logits(
                    context.current_nodes,
                    (*second_conditioning, context.current_nodes[left]),
                    second_head,
                )
                conditional_allowed = torch.zeros_like(conditional, dtype=torch.bool)
                left_map = context.current_maps[left]
                for item in legal_of_kind:
                    if left_map not in item.atoms:
                        continue
                    for atom_map in item.atoms:
                        if atom_map != left_map and atom_map in positions:
                            conditional_allowed[positions[atom_map]] = True
                conditional = conditional.masked_fill(~conditional_allowed, -torch.inf)
                return first_log_probs[left] + F.log_softmax(
                    conditional / temperature, dim=0
                )[right]

            total_logprob = kind_logprob + float(
                torch.logaddexp(
                    ordered_logprob(first, second), ordered_logprob(second, first)
                )
            )
        else:
            total_logprob = kind_logprob + value
        atoms = tuple(context.current_maps[index] for index in indices)
        embedding = torch.tanh(
            context.current_nodes[indices].sum(dim=0)
            + kind_embedding.weight[kind_index]
        )
        return ElectronContainer(kind, atoms), embedding, total_logprob

    @torch.no_grad()
    def rollout_direct_flow(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        trajectory: CompressedTrajectory | None = None,
        max_arrows: int = 8,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Generate a complete electron event without a legal-action inventory."""

        context = self.encode_context(
            current_smiles, target_smiles, history=trajectory
        )
        history = torch.zeros_like(context.vector)
        inventory = MoveInventory.from_state(current_smiles)
        moves: list[ElectronMove] = []
        total_logprob = 0.0
        for _ in range(max_arrows):
            source_state = torch.cat((context.vector, history), dim=-1)
            try:
                source, source_embedding, value = self._sample_direct_container(
                    context=context,
                    kinds=FLOW_SOURCE_KINDS,
                    kind_logits=self.direct_source_kind_head(source_state),
                    first_conditioning=(source_state,),
                    first_head=self.direct_source_first_head,
                    second_conditioning=(source_state,),
                    second_head=self.direct_source_second_head,
                    kind_embedding=self.direct_source_kind_embedding,
                    allowed_containers=inventory.sources,
                    greedy=greedy,
                    temperature=temperature,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "code": "DIRECT_POINTER_INVALID",
                    "message": str(exc),
                    "moves": [move.to_dict() for move in moves],
                    "logprob": total_logprob,
                }
            total_logprob += value
            sink_state = torch.cat(
                (context.vector, history, source_embedding), dim=-1
            )
            try:
                sink, sink_embedding, value = self._sample_direct_container(
                    context=context,
                    kinds=FLOW_SINK_KINDS,
                    kind_logits=self.direct_sink_kind_head(sink_state),
                    first_conditioning=(sink_state,),
                    first_head=self.direct_sink_first_head,
                    second_conditioning=(sink_state,),
                    second_head=self.direct_sink_second_head,
                    kind_embedding=self.direct_sink_kind_embedding,
                    allowed_containers=inventory.compatible_sinks(source),
                    greedy=greedy,
                    temperature=temperature,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "code": "DIRECT_POINTER_INVALID",
                    "message": str(exc),
                    "moves": [move.to_dict() for move in moves],
                    "logprob": total_logprob,
                }
            total_logprob += value
            moves.append(ElectronMove(source, sink))
            action_embedding = self.direct_action(
                torch.cat((source_embedding, sink_embedding), dim=-1)
            )
            history = self.event_cell(
                action_embedding[None, :], history[None, :]
            ).squeeze(0)
            commit_logits = self.commit_head(
                torch.cat((context.vector, history), dim=-1)
            )
            commit, value = self._sample_index(
                commit_logits, greedy=greedy, temperature=temperature
            )
            total_logprob += value
            if commit == 1:
                return {
                    "ok": True,
                    "moves": [move.to_dict() for move in moves],
                    "logprob": total_logprob,
                }
        return {
            "ok": False,
            "code": "DIRECT_FLOW_BUDGET",
            "moves": [move.to_dict() for move in moves],
            "logprob": total_logprob,
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
        return_parts: bool = True,
        max_edits: int = MAX_BE_EDITS,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """NLL for a sparse BE-matrix edit sequence followed by atomic commit.

        Bond positions are emitted by two conditional node pointers rather
        than an enumerated pair list.  The executor validates the resulting
        sparse edit atomically after generation.
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

        if len(operations) > max_edits:
            raise ValueError(
                f"BE_DELTA has {len(operations)} edits, above the rollout limit {max_edits}"
            )
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(current_smiles, params)
        if mol is None:
            raise ValueError("invalid current state for BE_DELTA")
        atom_maps = tuple(context.current_maps)
        atom_embeddings = context.current_nodes
        by_map = {int(atom.GetAtomMapNum()): atom for atom in mol.GetAtoms()}
        seen_pairs: set[tuple[int, int]] = set()
        seen_atoms: set[int] = set()
        charge_phase = False

        for edit_index, (operation, location, delta) in enumerate(operations):
            if charge_phase and operation == 0:
                raise ValueError("BE_DELTA canonical order requires bonds before charges")
            charge_phase = charge_phase or operation == 1
            operation_scores = self.be_operation_head(torch.cat((context.vector, history), dim=-1))
            operation_allowed = torch.tensor(
                [not charge_phase or operation == 0, True, edit_index > 0],
                dtype=torch.bool,
                device=self.device,
            )
            # Once a charge edit is emitted the canonical program cannot go
            # back to bond edits.  The current charge itself is still legal.
            if charge_phase:
                operation_allowed[0] = False
            operation_loss = operation_loss - F.log_softmax(
                operation_scores.masked_fill(~operation_allowed, -torch.inf), dim=0
            )[operation]
            if operation == 0:
                try:
                    left, right = (positions[value] for value in location)
                except KeyError as exc:
                    raise ValueError(
                        f"BE pair references missing private maps: {location}"
                    ) from exc
                state = torch.cat((context.vector, history), dim=-1)
                first_logits = self._pointer_logits(
                    atom_embeddings, (state,), self.be_pair_first_head
                )

                pair_allowed = torch.zeros(
                    (len(atom_maps), len(atom_maps)), dtype=torch.bool, device=self.device
                )
                for first in range(len(atom_maps)):
                    for second in range(first + 1, len(atom_maps)):
                        pair = tuple(sorted((atom_maps[first], atom_maps[second])))
                        if pair in seen_pairs:
                            continue
                        left_atom, right_atom = by_map[pair[0]], by_map[pair[1]]
                        bond = mol.GetBondBetweenAtoms(left_atom.GetIdx(), right_atom.GetIdx())
                        order = 0.0 if bond is None else float(bond.GetBondTypeAsDouble())
                        if order > 0.0 or order < 3.0:
                            pair_allowed[first, second] = True
                            pair_allowed[second, first] = True
                first_allowed = pair_allowed.any(dim=1)

                def ordered_pair_logprob(first: int, second: int) -> torch.Tensor:
                    second_logits = self._pointer_logits(
                        atom_embeddings,
                        (state, atom_embeddings[first]),
                        self.be_pair_second_head,
                    ).clone()
                    return F.log_softmax(
                        first_logits.masked_fill(~first_allowed, -torch.inf), dim=0
                    )[first] + F.log_softmax(
                        second_logits.masked_fill(~pair_allowed[first], -torch.inf), dim=0
                    )[second]

                position_loss = position_loss - torch.logaddexp(
                    ordered_pair_logprob(left, right),
                    ordered_pair_logprob(right, left),
                )
                action_embedding = torch.tanh(
                    atom_embeddings[left] + atom_embeddings[right]
                )
                pair = tuple(sorted(location))
                seen_pairs.add(pair)
                left_atom, right_atom = by_map[pair[0]], by_map[pair[1]]
                bond = mol.GetBondBetweenAtoms(left_atom.GetIdx(), right_atom.GetIdx())
                order = 0.0 if bond is None else float(bond.GetBondTypeAsDouble())
                delta_allowed = torch.tensor(
                    [order > 0.0, order < 3.0], dtype=torch.bool, device=self.device
                )
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
                atom_allowed = torch.tensor(
                    [atom_map not in seen_atoms for atom_map in atom_maps],
                    dtype=torch.bool,
                    device=self.device,
                )
                position_loss = position_loss - F.log_softmax(
                    scores.masked_fill(~atom_allowed, -torch.inf), dim=0
                )[selected]
                seen_atoms.add(location[0])
                q0 = int(by_map[location[0]].GetFormalCharge())
                delta_allowed = torch.tensor(
                    [q0 > -5, q0 < 5], dtype=torch.bool, device=self.device
                )
            delta_scores = self.be_delta_head(torch.cat((context.vector, action_embedding), dim=-1))
            delta_target = 0 if delta == -1 else 1
            if not bool(delta_allowed[delta_target]):
                raise ValueError(f"illegal BE delta {delta} at {location}")
            delta_loss = delta_loss - F.log_softmax(
                delta_scores.masked_fill(~delta_allowed, -torch.inf), dim=0
            )[delta_target]
            history = self.event_cell(action_embedding[None, :], history[None, :]).squeeze(0)

        commit_scores = self.be_operation_head(torch.cat((context.vector, history), dim=-1))
        commit_allowed = torch.tensor(
            [not charge_phase and len(operations) < max_edits, len(operations) < max_edits, True],
            dtype=torch.bool,
            device=self.device,
        )
        operation_loss = operation_loss - F.log_softmax(
            commit_scores.masked_fill(~commit_allowed, -torch.inf), dim=0
        )[2]
        total = family_loss + operation_loss + position_loss + delta_loss
        parts = (
            {
                "family": float(family_loss.detach()),
                "operation": float(operation_loss.detach()),
                "position": float(position_loss.detach()),
                "delta": float(delta_loss.detach()),
            }
            if return_parts
            else {}
        )
        return total, parts

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
            order = min(31, max(0, int(bond_type)))
            edge_index.extend(((left, right), (right, left)))
            edge_attr.extend(([order, 0, 0, 0, 0], [order, 0, 0, 0, 0]))
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
                    min(255, int(atom.isotope)),
                    min(7, int(atom.chiral_tag)),
                    int(atom.no_implicit),
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
                else torch.empty((0, 5), dtype=torch.long, device=self.device)
            ),
            alignment=torch.zeros((atom_count, 8), dtype=torch.float32, device=self.device),
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
        return_parts: bool = True,
        normalize: bool = True,
        max_atoms: int = MAX_FRAGMENT_ATOMS,
        max_extra_bonds: int = MAX_FRAGMENT_EXTRA_BONDS,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Teacher-forced graph-program loss for an open-vocabulary import."""

        context = context or self.encode_context(current_smiles, target_smiles)
        family = "IMPORT_ENV" if program.role == "ENVIRONMENT" else "IMPORT_REACTIVE"
        family_target = torch.tensor(ACTION_FAMILIES.index(family), device=self.device)
        family_loss = F.cross_entropy(
            self.family_logits(context)[None, :], family_target[None]
        )
        role_index = REACTIVE_ROLES.index(program.role)
        role_allowed = torch.tensor(
            [
                candidate == "ENVIRONMENT"
                if family == "IMPORT_ENV"
                else candidate != "ENVIRONMENT"
                for candidate in REACTIVE_ROLES
            ],
            dtype=torch.bool,
            device=self.device,
        )
        role_logits = self.reactive_role_head(context.vector).masked_fill(
            ~role_allowed, -torch.inf
        )
        role_loss = F.cross_entropy(
            role_logits[None, :],
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
            operation_allowed = torch.tensor(
                [atom_index < max_atoms, atom_index >= 2, atom_index > 0],
                dtype=torch.bool,
                device=self.device,
            )
            operation_loss = operation_loss - F.log_softmax(
                self.fragment_operation_head(state).masked_fill(
                    ~operation_allowed, -torch.inf
                ),
                dim=0,
            )[0]
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

        for extra_index, bond in enumerate(program.extra_bonds):
            nodes, state = self._reactive_fragment_state(
                context, program, len(program.atoms), extra_index
            )
            operation_allowed = torch.tensor(
                [False, extra_index < max_extra_bonds, True],
                dtype=torch.bool,
                device=self.device,
            )
            operation_loss = operation_loss - F.log_softmax(
                self.fragment_operation_head(state).masked_fill(
                    ~operation_allowed, -torch.inf
                ),
                dim=0,
            )[1]
            target_pair = tuple(sorted(map(int, bond.atoms)))
            if min(target_pair) < 0 or max(target_pair) >= nodes.shape[0]:
                raise ValueError(f"extra bond outside fragment: {target_pair}")
            first_logits = self._pointer_logits(
                nodes, (state,), self.fragment_pair_first_head
            )

            occupied = {
                tuple(sorted((index, int(atom.parent))))
                for index, atom in enumerate(program.atoms)
                if atom.parent is not None
            }
            occupied.update(
                tuple(sorted(map(int, existing.atoms)))
                for existing in program.extra_bonds[:extra_index]
            )
            pair_allowed = torch.ones(
                (nodes.shape[0], nodes.shape[0]), dtype=torch.bool, device=self.device
            )
            pair_allowed.fill_diagonal_(False)
            for first, second in occupied:
                pair_allowed[first, second] = False
                pair_allowed[second, first] = False
            first_allowed = pair_allowed.any(dim=1)

            def ordered_pair_logprob(first: int, second: int) -> torch.Tensor:
                second_logits = self._pointer_logits(
                    nodes, (state, nodes[first]), self.fragment_pair_second_head
                ).clone()
                return F.log_softmax(
                    first_logits.masked_fill(~first_allowed, -torch.inf), dim=0
                )[first] + F.log_softmax(
                    second_logits.masked_fill(~pair_allowed[first], -torch.inf), dim=0
                )[second]

            left, right = target_pair
            position_loss = position_loss - torch.logaddexp(
                ordered_pair_logprob(left, right),
                ordered_pair_logprob(right, left),
            )
            if int(bond.bond_type) not in FRAGMENT_BOND_TYPES:
                raise ValueError(f"unsupported fragment bond type: {bond.bond_type}")
            bond_loss = bond_loss + F.cross_entropy(
                self.fragment_bond_type_head(state)[None, :],
                torch.tensor(
                    [FRAGMENT_BOND_TYPES.index(int(bond.bond_type))], device=self.device
                ),
            )

        nodes, state = self._reactive_fragment_state(
            context, program, len(program.atoms), len(program.extra_bonds)
        )
        terminal_allowed = torch.tensor(
            [
                not program.extra_bonds and len(program.atoms) < max_atoms,
                len(program.atoms) >= 2
                and len(program.extra_bonds) < max_extra_bonds,
                bool(program.atoms),
            ],
            dtype=torch.bool,
            device=self.device,
        )
        operation_loss = operation_loss - F.log_softmax(
            self.fragment_operation_head(state).masked_fill(
                ~terminal_allowed, -torch.inf
            ),
            dim=0,
        )[2]
        active_logits = self.fragment_active_head(
            torch.cat((nodes, state.expand(nodes.shape[0], -1)), dim=-1)
        ).squeeze(-1)
        active_targets = torch.zeros_like(active_logits)
        active_targets[list(program.active_atoms)] = 1.0
        active_loss = F.binary_cross_entropy_with_logits(
            active_logits, active_targets, reduction="sum"
        )
        if program.role != "ENVIRONMENT":
            # Active sites are sampled as a Bernoulli set conditioned on being
            # non-empty.  Use the same normalized distribution for SFT and RL.
            log_empty = F.logsigmoid(-active_logits).sum()
            log_nonempty = torch.log1p(-torch.exp(log_empty).clamp(max=1.0 - 1e-7))
            active_loss = active_loss + log_nonempty

        # One reaction-level decision must not receive a gradient proportional
        # to reagent size.  Average each factor over its supervised choices;
        # otherwise a large reactive fragment overwhelms FLOW/FINISH examples.
        if normalize:
            operation_loss = operation_loss / (
                len(program.atoms) + len(program.extra_bonds) + 1
            )
            atom_loss = atom_loss / (6 * len(program.atoms))
            located_bonds = max(0, len(program.atoms) - 1) + len(program.extra_bonds)
            if located_bonds:
                position_loss = position_loss / located_bonds
                bond_loss = bond_loss / located_bonds
            active_loss = active_loss / len(program.atoms)
        total = (
            family_loss
            + role_loss
            + operation_loss
            + atom_loss
            + position_loss
            + bond_loss
            + active_loss
        )
        parts = (
            {
                "family": float(family_loss.detach()),
                "role": float(role_loss.detach()),
                "operation": float(operation_loss.detach()),
                "atom": float(atom_loss.detach()),
                "position": float(position_loss.detach()),
                "bond": float(bond_loss.detach()),
                "active": float(active_loss.detach()),
            }
            if return_parts
            else {}
        )
        return total, parts

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

    @staticmethod
    def _sample_active_set(
        logits: torch.Tensor,
        *,
        greedy: bool,
        require_nonempty: bool,
    ) -> tuple[tuple[int, ...], float]:
        """Sample the same multi-label active-site distribution used by SFT."""

        probabilities = torch.sigmoid(logits)
        if greedy:
            selected = probabilities >= 0.5
            if require_nonempty and not bool(selected.any()):
                selected[int(probabilities.argmax())] = True
        else:
            selected = torch.bernoulli(probabilities).bool()
            retries = 0
            while require_nonempty and not bool(selected.any()) and retries < 128:
                selected = torch.bernoulli(probabilities).bool()
                retries += 1
            if require_nonempty and not bool(selected.any()):
                selected[int(probabilities.argmax())] = True
        raw_logprob = torch.where(
            selected, F.logsigmoid(logits), F.logsigmoid(-logits)
        ).sum()
        if require_nonempty:
            log_empty = F.logsigmoid(-logits).sum()
            raw_logprob = raw_logprob - torch.log1p(
                -torch.exp(log_empty).clamp(max=1.0 - 1e-7)
            )
        indices = tuple(int(index) for index in torch.nonzero(selected).flatten())
        return indices, float(raw_logprob)

    @torch.no_grad()
    def rollout_reactive_fragment(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        trajectory: CompressedTrajectory | None = None,
        family: str | None = None,
        role: str | None = None,
        max_atoms: int = MAX_FRAGMENT_ATOMS,
        max_extra_bonds: int = MAX_FRAGMENT_EXTRA_BONDS,
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
        context = self.encode_context(
            current_smiles, target_smiles, history=trajectory
        )
        total_logprob = 0.0
        if family is None:
            family = "IMPORT_ENV" if role == "ENVIRONMENT" else "IMPORT_REACTIVE"
        if family not in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
            raise ValueError(f"invalid import family: {family}")
        role_allowed = torch.tensor(
            [
                candidate == "ENVIRONMENT"
                if family == "IMPORT_ENV"
                else candidate != "ENVIRONMENT"
                for candidate in REACTIVE_ROLES
            ],
            dtype=torch.bool,
            device=self.device,
        )
        if role is None:
            role_index, value = self._sample_index(
                self.reactive_role_head(context.vector),
                greedy=greedy,
                temperature=temperature,
                allowed=role_allowed,
            )
            role = REACTIVE_ROLES[role_index]
            total_logprob += value
        elif role not in REACTIVE_ROLES:
            raise ValueError(f"unknown reactive role: {role}")
        elif not bool(role_allowed[REACTIVE_ROLES.index(role)]):
            raise ValueError(f"role {role} is incompatible with {family}")

        atoms: list[FragmentAtom] = []
        extra_bonds: list[FragmentBond] = []
        operations: list[dict[str, Any]] = []
        bond_phase = False
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
            allowed = torch.tensor(
                [
                    not bond_phase and len(atoms) < max_atoms,
                    len(atoms) >= 2 and len(extra_bonds) < max_extra_bonds,
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
                active_atoms, value = self._sample_active_set(
                    active_logits,
                    greedy=greedy,
                    require_nonempty=role != "ENVIRONMENT",
                )
                if role == "ENVIRONMENT":
                    active_atoms = ()
                    value = float(F.logsigmoid(-active_logits).sum())
                total_logprob += value
                program = ReactiveFragmentProgram(
                    role=role,
                    atoms=tuple(atoms),
                    extra_bonds=tuple(extra_bonds),
                    active_atoms=active_atoms,
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
                    "active_atoms": active_atoms,
                    "active_atom": active_atoms[0] if active_atoms else None,
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

            first_logits = self._pointer_logits(
                nodes, (state,), self.fragment_pair_first_head
            )
            bond_phase = True
            occupied = {
                tuple(sorted((index, int(atom.parent))))
                for index, atom in enumerate(atoms)
                if atom.parent is not None
            }
            occupied.update(tuple(sorted(map(int, bond.atoms))) for bond in extra_bonds)
            pair_allowed = torch.ones(
                (len(atoms), len(atoms)), dtype=torch.bool, device=self.device
            )
            pair_allowed.fill_diagonal_(False)
            for left, right in occupied:
                pair_allowed[left, right] = False
                pair_allowed[right, left] = False
            first_allowed = pair_allowed.any(dim=1)
            if not bool(first_allowed.any()):
                return {
                    "ok": False,
                    "code": "FRAGMENT_BOND_SPACE_EXHAUSTED",
                    "role": role,
                    "operations": operations,
                    "logprob": total_logprob,
                }
            first, value = self._sample_index(
                first_logits,
                greedy=greedy,
                temperature=temperature,
                allowed=first_allowed,
            )
            second_logits = self._pointer_logits(
                nodes, (state, nodes[first]), self.fragment_pair_second_head
            )
            second_allowed = pair_allowed[first]
            second, value = self._sample_index(
                second_logits,
                greedy=greedy,
                temperature=temperature,
                allowed=second_allowed,
            )
            # The action is an undirected bond.  Its probability is the sum
            # of the two ordered pointer paths, exactly as in teacher forcing.
            def ordered_pair_logprob(left: int, right: int) -> torch.Tensor:
                second_scores = self._pointer_logits(
                    nodes, (state, nodes[left]), self.fragment_pair_second_head
                )
                return F.log_softmax(
                    (first_logits / temperature).masked_fill(~first_allowed, -torch.inf), dim=0
                )[left] + F.log_softmax(
                    (second_scores / temperature).masked_fill(
                        ~pair_allowed[left], -torch.inf
                    ),
                    dim=0,
                )[right]

            total_logprob += float(
                torch.logaddexp(
                    ordered_pair_logprob(first, second),
                    ordered_pair_logprob(second, first),
                )
            )
            bond_index, value = self._sample_index(
                self.fragment_bond_type_head(state),
                greedy=greedy,
                temperature=temperature,
            )
            total_logprob += value
            bond = FragmentBond(
                atoms=tuple(sorted((first, second))),
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

    @torch.no_grad()
    def rollout_be_delta(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        trajectory: CompressedTrajectory | None = None,
        max_edits: int = MAX_BE_EDITS,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Sample a sparse BE edit program with structural legality masks."""

        context = self.encode_context(
            current_smiles, target_smiles, history=trajectory
        )
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(current_smiles, params)
        if mol is None:
            return {"ok": False, "code": "INVALID_CURRENT_STATE", "logprob": 0.0}
        positions = {value: index for index, value in enumerate(context.current_maps)}
        by_map = {
            int(atom.GetAtomMapNum()): atom for atom in mol.GetAtoms()
        }
        history = torch.zeros_like(context.vector)
        bond_deltas: list[dict[str, Any]] = []
        charge_actions: list[dict[str, Any]] = []
        seen_pairs: set[tuple[int, int]] = set()
        seen_atoms: set[int] = set()
        charge_phase = False
        total_logprob = 0.0
        for edit_index in range(max_edits + 1):
            operation_logits = self.be_operation_head(
                torch.cat((context.vector, history), dim=-1)
            )
            operation_allowed = torch.tensor(
                [
                    edit_index < max_edits and not charge_phase,
                    edit_index < max_edits,
                    edit_index > 0,
                ],
                dtype=torch.bool,
                device=self.device,
            )
            operation, value = self._sample_index(
                operation_logits,
                greedy=greedy,
                temperature=temperature,
                allowed=operation_allowed,
            )
            total_logprob += value
            if operation == 2:
                return {
                    "ok": True,
                    "moves": [
                        {
                            "mode": "BE_DELTA",
                            "bond_deltas": bond_deltas,
                            "charge_actions": charge_actions,
                        }
                    ],
                    "logprob": total_logprob,
                }
            state = torch.cat((context.vector, history), dim=-1)
            if operation == 0:
                first_logits = self._pointer_logits(
                    context.current_nodes, (state,), self.be_pair_first_head
                )
                pair_allowed = torch.zeros(
                    (len(context.current_maps), len(context.current_maps)),
                    dtype=torch.bool,
                    device=self.device,
                )
                for left_index in range(len(context.current_maps)):
                    for right_index in range(left_index + 1, len(context.current_maps)):
                        pair = tuple(
                            sorted(
                                (
                                    context.current_maps[left_index],
                                    context.current_maps[right_index],
                                )
                            )
                        )
                        if pair in seen_pairs:
                            continue
                        left_atom, right_atom = by_map[pair[0]], by_map[pair[1]]
                        bond = mol.GetBondBetweenAtoms(
                            left_atom.GetIdx(), right_atom.GetIdx()
                        )
                        order = 0.0 if bond is None else float(bond.GetBondTypeAsDouble())
                        if order > 0.0 or order < 3.0:
                            pair_allowed[left_index, right_index] = True
                            pair_allowed[right_index, left_index] = True
                first_allowed = pair_allowed.any(dim=1)
                if not bool(first_allowed.any()):
                    return {
                        "ok": False,
                        "code": "BE_PAIR_SPACE_EXHAUSTED",
                        "logprob": total_logprob,
                    }
                first, value = self._sample_index(
                    first_logits,
                    greedy=greedy,
                    temperature=temperature,
                    allowed=first_allowed,
                )
                second_logits = self._pointer_logits(
                    context.current_nodes,
                    (state, context.current_nodes[first]),
                    self.be_pair_second_head,
                )
                second_allowed = pair_allowed[first]
                second, value = self._sample_index(
                    second_logits,
                    greedy=greedy,
                    temperature=temperature,
                    allowed=second_allowed,
                )
                def ordered_pair_logprob(left_index: int, right_index: int) -> torch.Tensor:
                    right_logits = self._pointer_logits(
                        context.current_nodes,
                        (state, context.current_nodes[left_index]),
                        self.be_pair_second_head,
                    )
                    return F.log_softmax(
                        (first_logits / temperature).masked_fill(
                            ~first_allowed, -torch.inf
                        ),
                        dim=0,
                    )[left_index] + F.log_softmax(
                        (right_logits / temperature).masked_fill(
                            ~pair_allowed[left_index], -torch.inf
                        ),
                        dim=0,
                    )[right_index]

                total_logprob += float(
                    torch.logaddexp(
                        ordered_pair_logprob(first, second),
                        ordered_pair_logprob(second, first),
                    )
                )
                pair = tuple(sorted((context.current_maps[first], context.current_maps[second])))
                seen_pairs.add(pair)
                left = by_map[pair[0]]
                right = by_map[pair[1]]
                bond = mol.GetBondBetweenAtoms(left.GetIdx(), right.GetIdx())
                order = 0.0 if bond is None else float(bond.GetBondTypeAsDouble())
                delta_allowed = torch.tensor(
                    [order > 0.0, order < 3.0], dtype=torch.bool, device=self.device
                )
                action_embedding = torch.tanh(
                    context.current_nodes[first] + context.current_nodes[second]
                )
                delta_index, value = self._sample_index(
                    self.be_delta_head(torch.cat((context.vector, action_embedding), dim=-1)),
                    greedy=greedy,
                    temperature=temperature,
                    allowed=delta_allowed,
                )
                total_logprob += value
                bond_deltas.append({"atoms": list(pair), "delta": -1 if delta_index == 0 else 1})
            else:
                charge_phase = True
                atom_logits = self.be_atom_head(
                    torch.cat(
                        (
                            context.current_nodes,
                            context.vector.expand(len(context.current_maps), -1),
                            history.expand(len(context.current_maps), -1),
                        ),
                        dim=-1,
                    )
                ).squeeze(-1)
                allowed = torch.tensor(
                    [atom_map not in seen_atoms for atom_map in context.current_maps],
                    dtype=torch.bool,
                    device=self.device,
                )
                if not bool(allowed.any()):
                    return {"ok": False, "code": "BE_CHARGE_SPACE_EXHAUSTED", "logprob": total_logprob}
                selected, value = self._sample_index(
                    atom_logits,
                    greedy=greedy,
                    temperature=temperature,
                    allowed=allowed,
                )
                total_logprob += value
                atom_map = context.current_maps[selected]
                seen_atoms.add(atom_map)
                action_embedding = context.current_nodes[selected]
                q0 = int(by_map[atom_map].GetFormalCharge())
                delta_allowed = torch.tensor(
                    [q0 > -5, q0 < 5], dtype=torch.bool, device=self.device
                )
                delta_index, value = self._sample_index(
                    self.be_delta_head(torch.cat((context.vector, action_embedding), dim=-1)),
                    greedy=greedy,
                    temperature=temperature,
                    allowed=delta_allowed,
                )
                total_logprob += value
                delta = -1 if delta_index == 0 else 1
                charge_actions.append({"atom_map": atom_map, "q0": q0, "q1": q0 + delta})
            history = self.event_cell(
                action_embedding[None, :], history[None, :]
            ).squeeze(0)
        return {"ok": False, "code": "BE_DELTA_BUDGET", "logprob": total_logprob}

    @torch.no_grad()
    def sample_action(
        self,
        current_smiles: str,
        target_smiles: str,
        *,
        trajectory: CompressedTrajectory | None = None,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict[str, Any]:
        """Sample one complete canonical action from the unified graph policy."""

        context = self.encode_context(
            current_smiles, target_smiles, history=trajectory
        )
        family_index, family_logprob = self._sample_index(
            self.family_logits(context), greedy=greedy, temperature=temperature
        )
        family = ACTION_FAMILIES[family_index]
        if family == "FINISH":
            return {"ok": True, "action": {"kind": family}, "logprob": family_logprob}
        if family == "FLOW":
            sampled = self.rollout_direct_flow(
                current_smiles,
                target_smiles,
                trajectory=trajectory,
                greedy=greedy,
                temperature=temperature,
            )
        elif family == "BE_DELTA":
            sampled = self.rollout_be_delta(
                current_smiles,
                target_smiles,
                trajectory=trajectory,
                greedy=greedy,
                temperature=temperature,
            )
        else:
            sampled = self.rollout_reactive_fragment(
                current_smiles,
                target_smiles,
                trajectory=trajectory,
                family=family,
                greedy=greedy,
                temperature=temperature,
            )
        if not sampled.get("ok"):
            return {**sampled, "family": family, "logprob": family_logprob + float(sampled.get("logprob", 0.0))}
        action: dict[str, Any] = {"kind": family}
        if family in {"FLOW", "BE_DELTA"}:
            action["moves"] = sampled["moves"]
        else:
            action["program"] = sampled["program"].to_dict()
        return {
            "ok": True,
            "action": action,
            "logprob": family_logprob + float(sampled["logprob"]),
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
