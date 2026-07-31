"""Compact forward electron-flow expert and formal source-sink verifier.

The inverse Qwen actor remains unchanged. This module supplies an independent,
small graph model that scores forward electron moves, target-product recovery,
and target-versus-competitor selectivity. Closed-shell two-electron polar
chemistry is supported in v1; unsupported chemistry returns UNKNOWN/FAIL rather
than being guessed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from rdkit import Chem
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True, order=True)
class ElectronContainer:
    kind: str
    atoms: tuple[int, ...]

    def __post_init__(self):
        kind = self.kind.upper().replace("LONE_PAIR", "LP")
        atoms = (
            tuple(sorted(int(value) for value in self.atoms))
            if kind == "BOND"
            else tuple(int(value) for value in self.atoms)
        )
        expected = 2 if kind == "BOND" else 1
        if kind not in {"LP", "ATOM", "BOND"} or len(atoms) != expected:
            raise ValueError(f"invalid electron container: {kind} {atoms}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "atoms", atoms)

    @property
    def id(self) -> str:
        return f"{self.kind}:" + ",".join(map(str, self.atoms))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "atoms": list(self.atoms), "id": self.id}

    @classmethod
    def parse(cls, value: Any, role: str) -> "ElectronContainer":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            kind = value.get("kind") or value.get("type") or (
                "LP" if role == "source" else "ATOM"
            )
            atoms = value.get("atoms") or value.get("atom_maps") or value.get("map")
        else:
            atoms = value
            kind = (
                "BOND"
                if isinstance(value, (list, tuple)) and len(value) == 2
                else ("LP" if role == "source" else "ATOM")
            )
        if isinstance(atoms, int):
            atoms = [atoms]
        return cls(str(kind), tuple(atoms or ()))


@dataclass(frozen=True)
class ElectronMove:
    source: ElectronContainer
    sink: ElectronContainer
    electrons: int = 2

    def __post_init__(self):
        if self.electrons != 2:
            raise ValueError("v1 supports two-electron moves only")

    @property
    def id(self) -> str:
        return f"{self.source.id}->{self.sink.id}/2e"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "sink": self.sink.to_dict(),
            "electrons": 2,
            "id": self.id,
        }

    @classmethod
    def parse(cls, value: Any) -> "ElectronMove":
        if isinstance(value, cls):
            return value
        return cls(
            ElectronContainer.parse(value["source"], "source"),
            ElectronContainer.parse(value["sink"], "sink"),
        )


@dataclass(frozen=True)
class ForwardEvidence:
    formal_compatible: bool
    step_logprob: float | None = None
    target_score: float | None = None
    target_rank: int | None = None
    best_competitor_score: float | None = None
    selectivity_margin: float | None = None
    uncertainty: float | None = None
    verdict: str = "UNKNOWN"
    diagnostics: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ValueError("all atoms require unique positive atom maps")
    return mol


def _lp_electrons(atom: Chem.Atom) -> int:
    table = Chem.GetPeriodicTable()
    return int(
        table.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetFormalCharge()
        - sum(round(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())
        - atom.GetTotalNumHs()
    )


def enumerate_containers(
    smiles: str,
) -> tuple[tuple[ElectronContainer, ...], tuple[ElectronContainer, ...]]:
    mol = _mol(smiles)
    sources: list[ElectronContainer] = []
    sinks: list[ElectronContainer] = []
    atoms = list(mol.GetAtoms())
    for atom in atoms:
        atom_map = atom.GetAtomMapNum()
        sinks.append(ElectronContainer("ATOM", (atom_map,)))
        if _lp_electrons(atom) >= 2:
            sources.append(ElectronContainer("LP", (atom_map,)))
    for bond in mol.GetBonds():
        pair = (
            bond.GetBeginAtom().GetAtomMapNum(),
            bond.GetEndAtom().GetAtomMapNum(),
        )
        sources.append(ElectronContainer("BOND", pair))
        if round(bond.GetBondTypeAsDouble()) < 3 and not bond.GetIsAromatic():
            sinks.append(ElectronContainer("BOND", pair))
    # New bonds are candidates. The formal verifier rejects impossible ones.
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            if mol.GetBondBetweenAtoms(left.GetIdx(), right.GetIdx()) is None:
                sinks.append(
                    ElectronContainer(
                        "BOND",
                        (left.GetAtomMapNum(), right.GetAtomMapNum()),
                    )
                )
    return tuple(sorted(set(sources))), tuple(sorted(set(sinks)))


def verify_electron_step(
    smiles: str,
    moves: Sequence[ElectronMove | dict[str, Any]],
) -> dict[str, Any]:
    """Apply coupled arrows atomically and return the sanitized next state."""
    try:
        parsed = [ElectronMove.parse(value) for value in moves]
        mol = _mol(smiles)
        atom_index = {
            atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()
        }
        bond_delta: dict[tuple[int, int], int] = {}
        charge_delta: dict[int, int] = {}

        def add_charge(atom_map: int, delta: int) -> None:
            charge_delta[atom_map] = charge_delta.get(atom_map, 0) + delta

        for move in parsed:
            if set(move.source.atoms + move.sink.atoms) - set(atom_index):
                raise ValueError("ATOM_MAP_MISSING")
            if move.source.kind == "LP":
                donor = move.source.atoms[0]
                if _lp_electrons(mol.GetAtomWithIdx(atom_index[donor])) < 2:
                    raise ValueError("SOURCE_HAS_NO_ELECTRON_PAIR")
                pair = (
                    move.sink.atoms
                    if move.sink.kind == "BOND"
                    else tuple(sorted((donor, move.sink.atoms[0])))
                )
                if donor not in pair:
                    raise ValueError("SOURCE_NOT_IN_TARGET_BOND")
                acceptor = pair[0] if pair[1] == donor else pair[1]
                bond_delta[pair] = bond_delta.get(pair, 0) + 1
                add_charge(donor, 1)
                add_charge(acceptor, -1)
            elif move.source.kind == "BOND" and move.sink.kind in {"ATOM", "LP"}:
                pair = move.source.atoms
                target = move.sink.atoms[0]
                if target not in pair:
                    raise ValueError("CLEAVAGE_TARGET_NOT_IN_BOND")
                other = pair[0] if pair[1] == target else pair[1]
                bond_delta[pair] = bond_delta.get(pair, 0) - 1
                add_charge(target, -1)
                add_charge(other, 1)
            elif move.source.kind == move.sink.kind == "BOND":
                shared = set(move.source.atoms) & set(move.sink.atoms)
                if len(shared) != 1:
                    raise ValueError("NONLOCAL_BOND_SHIFT")
                centre = next(iter(shared))
                old = next(value for value in move.source.atoms if value != centre)
                new = next(value for value in move.sink.atoms if value != centre)
                bond_delta[move.source.atoms] = (
                    bond_delta.get(move.source.atoms, 0) - 1
                )
                bond_delta[move.sink.atoms] = (
                    bond_delta.get(move.sink.atoms, 0) + 1
                )
                add_charge(old, 1)
                add_charge(new, -1)
            else:
                raise ValueError("UNSUPPORTED_MOVE")

        rw = Chem.RWMol(mol)
        for pair, delta in bond_delta.items():
            left = atom_index[pair[0]]
            right = atom_index[pair[1]]
            old_bond = rw.GetBondBetweenAtoms(left, right)
            old_order = (
                int(round(old_bond.GetBondTypeAsDouble())) if old_bond else 0
            )
            new_order = old_order + delta
            if old_bond:
                rw.RemoveBond(left, right)
            if new_order not in {0, 1, 2, 3}:
                raise ValueError("INVALID_BOND_ORDER")
            if new_order:
                rw.AddBond(
                    left,
                    right,
                    {
                        1: Chem.BondType.SINGLE,
                        2: Chem.BondType.DOUBLE,
                        3: Chem.BondType.TRIPLE,
                    }[new_order],
                )
        for atom_map, delta in charge_delta.items():
            atom = rw.GetAtomWithIdx(atom_index[atom_map])
            atom.SetFormalCharge(atom.GetFormalCharge() + delta)
        output = rw.GetMol()
        Chem.SanitizeMol(output)
        return {
            "ok": True,
            "state_smiles": Chem.MolToSmiles(
                output,
                canonical=True,
                isomericSmiles=True,
            ),
            "code": "PASS",
        }
    except Exception as exc:
        return {
            "ok": False,
            "state_smiles": "",
            "code": "CHEMICAL_STATE_INVALID",
            "message": str(exc),
        }


def _graph(smiles: str) -> dict[str, Any]:
    mol = _mol(smiles)
    atoms = torch.tensor(
        [
            [
                min(atom.GetAtomicNum(), 118),
                max(0, min(8, atom.GetFormalCharge() + 4)),
                min(atom.GetDegree(), 6),
                min(atom.GetTotalNumHs(), 4),
                int(atom.GetIsAromatic()),
                int(atom.IsInRing()),
                min(int(atom.GetTotalValence()), 8),
            ]
            for atom in mol.GetAtoms()
        ],
        dtype=torch.long,
    )
    edges: list[tuple[int, int]] = []
    edge_attributes: list[list[int]] = []
    for bond in mol.GetBonds():
        pairs = (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
        for source, destination in pairs:
            edges.append((source, destination))
            edge_attributes.append(
                [
                    min(3, int(round(bond.GetBondTypeAsDouble()))),
                    int(bond.GetIsConjugated()),
                    int(bond.GetIsAromatic()),
                    int(bond.IsInRing()),
                ]
            )
    return {
        "atoms": atoms,
        "edge_index": (
            torch.tensor(edges, dtype=torch.long).T
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_attr": (
            torch.tensor(edge_attributes, dtype=torch.long)
            if edge_attributes
            else torch.empty((0, 4), dtype=torch.long)
        ),
        "maps": tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms()),
    }


def _condition_vector(value: Any, size: int = 64) -> torch.Tensor:
    """Deterministic hashed bag-of-tokens for sparse condition metadata."""
    if value in (None, "", {}, []):
        return torch.zeros(size, dtype=torch.float32)
    text = (
        json.dumps(value, sort_keys=True, ensure_ascii=False)
        if not isinstance(value, str)
        else value
    )
    vector = torch.zeros(size, dtype=torch.float32)
    for token in text.lower().replace(",", " ").replace(";", " ").split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = vector.norm()
    return vector / norm if norm > 0 else vector


class ForwardElectronExpert(nn.Module):
    """Small graph pointer model for moves and precursor-product compatibility."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        condition_dim: int = 64,
    ):
        super().__init__()
        hidden = hidden_dim
        self.config = {
            "hidden_dim": hidden,
            "num_layers": num_layers,
            "dropout": dropout,
            "condition_dim": condition_dim,
        }
        self.atom_emb = nn.ModuleList(
            [nn.Embedding(size, hidden) for size in (128, 9, 7, 5, 2, 2, 9)]
        )
        self.bond_emb = nn.ModuleList(
            [nn.Embedding(size, hidden) for size in (4, 2, 2, 2)]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(3 * hidden, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                )
                for _ in range(num_layers)
            ]
        )
        self.kind = nn.Embedding(3, hidden)
        self.container = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.condition_dim = condition_dim
        self.condition = nn.Sequential(
            nn.Linear(condition_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.source_head = nn.Linear(2 * hidden, 1)
        self.sink_head = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.reaction_head = nn.Sequential(
            nn.Linear(5 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _to_device(self, graph: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in graph.items()
        }

    def encode(self, graph: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self._to_device(graph)
        atoms = sum(
            embedding(graph["atoms"][:, index])
            for index, embedding in enumerate(self.atom_emb)
        )
        edge = (
            sum(
                embedding(graph["edge_attr"][:, index])
                for index, embedding in enumerate(self.bond_emb)
            )
            if graph["edge_attr"].numel()
            else atoms.new_zeros((0, atoms.shape[-1]))
        )
        for layer in self.layers:
            aggregate = torch.zeros_like(atoms)
            if graph["edge_index"].numel():
                source, destination = graph["edge_index"]
                aggregate.index_add_(0, destination, atoms[source] + edge)
            atoms = atoms + layer(
                torch.cat([atoms, aggregate, atoms * aggregate], dim=-1)
            )
        return atoms, atoms.mean(dim=0)

    def encode_containers(
        self,
        atom_embeddings: torch.Tensor,
        maps: Sequence[int],
        values: Sequence[ElectronContainer],
    ) -> torch.Tensor:
        positions = {atom_map: index for index, atom_map in enumerate(maps)}
        rows = []
        for container in values:
            left = atom_embeddings[positions[container.atoms[0]]]
            right = (
                atom_embeddings[positions[container.atoms[1]]]
                if len(container.atoms) == 2
                else torch.zeros_like(left)
            )
            kind = self.kind.weight[
                {"LP": 0, "ATOM": 1, "BOND": 2}[container.kind]
            ]
            rows.append(
                self.container(
                    torch.cat([left + right, torch.abs(left - right), kind, left])
                )
            )
        return torch.stack(rows)

    def move_logits(
        self,
        smiles: str,
        conditions: Any = None,
    ) -> tuple[
        tuple[ElectronContainer, ...],
        tuple[ElectronContainer, ...],
        torch.Tensor,
        torch.Tensor,
    ]:
        sources, sinks = enumerate_containers(smiles)
        graph = _graph(smiles)
        atoms, pooled = self.encode(graph)
        source_embeddings = self.encode_containers(atoms, graph["maps"], sources)
        sink_embeddings = self.encode_containers(atoms, graph["maps"], sinks)
        condition = self.condition(
            _condition_vector(conditions, self.condition_dim).to(self.device)
        )
        source_logits = self.source_head(
            torch.cat(
                [source_embeddings, condition.expand(len(sources), -1)],
                dim=-1,
            )
        ).squeeze(-1)
        sink_logits = []
        for index in range(len(sources)):
            selected_source = source_embeddings[index].expand(len(sinks), -1)
            graph_context = pooled.expand(len(sinks), -1)
            condition_context = condition.expand(len(sinks), -1)
            sink_logits.append(
                self.sink_head(
                    torch.cat(
                        [
                            sink_embeddings,
                            selected_source,
                            graph_context,
                            condition_context,
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)
            )
        return sources, sinks, source_logits, torch.stack(sink_logits)

    def rank_moves(
        self,
        smiles: str,
        top_k: int = 20,
        conditions: Any = None,
    ) -> list[dict[str, Any]]:
        sources, sinks, source_logits, sink_logits = self.move_logits(
            smiles,
            conditions=conditions,
        )
        source_logprobs = F.log_softmax(source_logits, dim=0)
        output = []
        for source_index, source in enumerate(sources):
            sink_logprobs = F.log_softmax(sink_logits[source_index], dim=0)
            for sink_index, sink in enumerate(sinks):
                output.append(
                    {
                        "source": source.to_dict(),
                        "sink": sink.to_dict(),
                        "logprob": float(
                            (source_logprobs[source_index] + sink_logprobs[sink_index])
                            .detach()
                            .cpu()
                        ),
                    }
                )
        return sorted(
            output,
            key=lambda item: item["logprob"],
            reverse=True,
        )[:top_k]

    def reaction_score(
        self,
        reactants: str,
        product: str,
        conditions: Any = None,
    ) -> torch.Tensor:
        _, reactant_embedding = self.encode(_graph(reactants))
        _, product_embedding = self.encode(_graph(product))
        condition = self.condition(
            _condition_vector(conditions, self.condition_dim).to(self.device)
        )
        return self.reaction_head(
            torch.cat(
                [
                    reactant_embedding,
                    product_embedding,
                    torch.abs(reactant_embedding - product_embedding),
                    reactant_embedding * product_embedding,
                    condition,
                ]
            )
        ).squeeze()

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "model.pt")
        (output / "config.json").write_text(
            json.dumps(self.config, indent=2),
            encoding="utf-8",
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata or {}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "ForwardElectronExpert":
        checkpoint = Path(path)
        model = cls(**json.loads((checkpoint / "config.json").read_text()))
        model.load_state_dict(
            torch.load(
                checkpoint / "model.pt",
                map_location=device,
                weights_only=True,
            )
        )
        return model.to(device).eval()


def score_reaction(
    model: ForwardElectronExpert,
    reactants: str,
    target: str,
    competitors: Iterable[str] = (),
    conditions: Any = None,
) -> ForwardEvidence:
    products = [target, *competitors]
    scores = [
        float(
            torch.sigmoid(
                model.reaction_score(
                    reactants,
                    product,
                    conditions=conditions,
                )
            )
            .detach()
            .cpu()
        )
        for product in products
    ]
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    rank = order.index(0) + 1
    best = max(scores[1:]) if len(scores) > 1 else None
    margin = scores[0] - best if best is not None else None
    verdict = (
        "FORWARD_UNRECOVERABLE"
        if rank != 1
        else (
            "SELECTIVITY_AMBIGUOUS"
            if margin is not None and margin < 0.1
            else "FORWARD_SUPPORTED"
        )
    )
    return ForwardEvidence(
        formal_compatible=True,
        target_score=scores[0],
        target_rank=rank,
        best_competitor_score=best,
        selectivity_margin=margin,
        uncertainty=1 - abs(scores[0] - 0.5) * 2,
        verdict=verdict,
    )


def forward_edge_cost(
    actor_logprob: float,
    evidence: ForwardEvidence,
    *,
    forward_weight: float = 1.0,
    selectivity_weight: float = 0.5,
    uncertainty_weight: float = 0.25,
) -> float:
    """Convert forward evidence into a soft route-search edge cost."""
    if not evidence.formal_compatible:
        return float("inf")
    cost = -float(actor_logprob)
    if evidence.target_score is not None:
        cost -= forward_weight * float(evidence.target_score)
    if evidence.selectivity_margin is not None:
        cost -= selectivity_weight * float(evidence.selectivity_margin)
    if evidence.uncertainty is not None:
        cost += uncertainty_weight * float(evidence.uncertainty)
    return cost
