"""Provenance-aware mechanistic primitive retrieval and soft evidence.

The library binds generic textbook-derived roles to atom-mapped molecular states
and instantiates candidate source-to-sink electron moves. It never overrides the
deterministic executor and never treats an unmatched move as impossible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

_TOKEN_RE = re.compile(r"[a-z0-9_+-]+")
VERIFIED = ("executor_verified", "chemist_reviewed", "released")
RETRIEVABLE = ("text_supported", *VERIFIED)


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    locator: str = ""
    evidence_kind: str = "reference"
    license: str = ""
    revision: str = ""

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "SourceRef":
        return cls(
            str(row.get("source_id") or ""),
            str(row.get("locator") or row.get("term_id") or ""),
            str(row.get("evidence_kind") or "reference"),
            str(row.get("license") or ""),
            str(row.get("revision") or ""),
        )


@dataclass(frozen=True)
class Pattern:
    smarts: str
    roles: dict[str, int]
    optional: bool = False

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "Pattern":
        return cls(
            str(row.get("smarts") or ""),
            {str(k): int(v) for k, v in dict(row.get("roles") or {}).items()},
            bool(row.get("optional", False)),
        )


@dataclass(frozen=True)
class MoveTemplate:
    source_kind: str
    source_roles: tuple[str, ...]
    sink_kind: str
    sink_roles: tuple[str, ...]
    electrons: int = 2

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "MoveTemplate":
        source, sink = dict(row.get("source") or {}), dict(row.get("sink") or {})
        return cls(
            str(source.get("kind") or "").upper(),
            tuple(map(str, source.get("roles") or ())),
            str(sink.get("kind") or "").upper(),
            tuple(map(str, sink.get("roles") or ())),
            int(row.get("electrons", 2)),
        )


@dataclass(frozen=True)
class Primitive:
    primitive_id: str
    version: str
    name: str
    level: str
    description: str
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    patterns: tuple[Pattern, ...] = ()
    moves: tuple[MoveTemplate, ...] = ()
    preconditions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    competing_primitives: tuple[str, ...] = ()
    followups: tuple[str, ...] = ()
    retrieval_only: bool = False
    prior: float = 1.0
    sources: tuple[SourceRef, ...] = ()
    status: str = "draft"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "Primitive":
        return cls(
            primitive_id=str(row.get("primitive_id") or ""),
            version=str(row.get("version") or "1.0"),
            name=str(row.get("name") or row.get("primitive_id") or ""),
            level=str(row.get("level") or "motif"),
            description=str(row.get("description") or ""),
            aliases=tuple(map(str, row.get("aliases") or ())),
            tags=tuple(map(str, row.get("tags") or ())),
            patterns=tuple(Pattern.parse(x) for x in row.get("patterns") or ()),
            moves=tuple(MoveTemplate.parse(x) for x in row.get("moves") or ()),
            preconditions=tuple(map(str, row.get("preconditions") or ())),
            warnings=tuple(map(str, row.get("warnings") or ())),
            competing_primitives=tuple(map(str, row.get("competing_primitives") or ())),
            followups=tuple(map(str, row.get("followups") or ())),
            retrieval_only=bool(row.get("retrieval_only", False)),
            prior=float(row.get("prior", 1.0)),
            sources=tuple(SourceRef.parse(x) for x in row.get("sources") or ()),
            status=str(row.get("status") or "draft"),
            metadata=dict(row.get("metadata") or {}),
        )

    @property
    def search_text(self) -> str:
        return " ".join((self.primitive_id, self.name, self.description, *self.aliases, *self.tags, *self.preconditions, *self.warnings)).lower()


@dataclass(frozen=True)
class PrimitiveMatch:
    primitive_id: str
    primitive_name: str
    score: float
    role_bindings: dict[str, int]
    moves: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    preconditions: tuple[str, ...]
    competing_primitives: tuple[str, ...]
    followups: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    status: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["moves"] = list(self.moves)
        row["sources"] = [asdict(x) for x in self.sources]
        return row


def _load(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("install mechet[knowledge]")
    return yaml.safe_load(text)


def _mapped_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles or ""), sanitize=True)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if any(x <= 0 for x in maps) or len(maps) != len(set(maps)):
        raise ValueError("unique positive atom maps are required")
    return mol


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).lower()))


def _role_indexes(query: Chem.Mol, roles: Mapping[str, int]) -> dict[str, int]:
    lookup = {a.GetAtomMapNum(): a.GetIdx() for a in query.GetAtoms() if a.GetAtomMapNum()}
    missing = set(roles.values()) - set(lookup)
    if missing:
        raise ValueError(f"SMARTS roles are missing mapped atoms: {sorted(missing)}")
    return {role: lookup[number] for role, number in roles.items()}


def _bindings(mol: Chem.Mol, pattern: Pattern) -> list[dict[str, int]]:
    query = Chem.MolFromSmarts(pattern.smarts)
    if query is None:
        raise ValueError(f"invalid SMARTS: {pattern.smarts}")
    indexes = _role_indexes(query, pattern.roles)
    return [
        {role: mol.GetAtomWithIdx(match[index]).GetAtomMapNum() for role, index in indexes.items()}
        for match in mol.GetSubstructMatches(query, uniquify=True)
    ]


def _merge(parts: Sequence[Mapping[str, int]]) -> dict[str, int] | None:
    output: dict[str, int] = {}
    for part in parts:
        for role, atom_map in part.items():
            if role in output and output[role] != atom_map:
                return None
            output[role] = atom_map
    return output


def _moves(primitive: Primitive, binding: Mapping[str, int]) -> tuple[dict[str, Any], ...]:
    output = []
    for move in primitive.moves:
        if set(move.source_roles + move.sink_roles) - set(binding):
            return ()
        output.append({
            "source": {"kind": move.source_kind, "atoms": [binding[x] for x in move.source_roles]},
            "sink": {"kind": move.sink_kind, "atoms": [binding[x] for x in move.sink_roles]},
            "electrons": move.electrons,
        })
    return tuple(output)


def _move_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    source, sink = dict(row.get("source") or {}), dict(row.get("sink") or {})
    sk, tk = str(source.get("kind") or "").upper(), str(sink.get("kind") or "").upper()
    sa, ta = tuple(map(int, source.get("atoms") or ())), tuple(map(int, sink.get("atoms") or ()))
    if sk == "BOND": sa = tuple(sorted(sa))
    if tk == "BOND": ta = tuple(sorted(ta))
    return sk, sa, tk, ta, int(row.get("electrons", 2))


def _expected_deltas(moves: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for row in moves:
        source, sink = dict(row.get("source") or {}), dict(row.get("sink") or {})
        for side, sign in ((source, -1.0), (sink, 1.0)):
            if str(side.get("kind") or "").upper() == "BOND" and len(side.get("atoms") or ()) == 2:
                pair = tuple(sorted(map(int, side["atoms"])))
                out[pair] = out.get(pair, 0.0) + sign
    return {k: v for k, v in out.items() if abs(v) > 1e-8}


def _bond_inventory(smiles: str) -> dict[tuple[int, int], float]:
    mol = _mapped_mol(smiles)
    return {
        tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))): float(b.GetBondTypeAsDouble())
        for b in mol.GetBonds()
    }


class PrimitiveLibrary:
    def __init__(self, primitives: Iterable[Primitive], source_registry: Mapping[str, Any] | None = None, metadata: Mapping[str, Any] | None = None):
        self.primitives = tuple(primitives)
        self.by_id = {x.primitive_id: x for x in self.primitives}
        self.source_registry = dict(source_registry or {})
        self.metadata = dict(metadata or {})
        self.validate()

    @classmethod
    def load(cls, path: str | Path, source_registry: str | Path | None = None) -> "PrimitiveLibrary":
        root = Path(path)
        files = sorted(root.rglob("*.y*ml")) + sorted(root.rglob("*.json")) if root.is_dir() else [root]
        primitives, metadata = [], {}
        for file in files:
            payload = _load(file) or {}
            rows = payload if isinstance(payload, list) else payload.get("primitives") or []
            if isinstance(payload, Mapping): metadata.update(payload.get("metadata") or {})
            primitives.extend(Primitive.parse(x) for x in rows)
        registry = _load(Path(source_registry)) if source_registry else {}
        return cls(primitives, registry or {}, metadata)

    def validate(self) -> None:
        if not self.primitives or len(self.by_id) != len(self.primitives):
            raise ValueError("primitive library is empty or has duplicate IDs")
        allowed_status = {"draft", "text_supported", "executor_verified", "chemist_reviewed", "released", "deprecated"}
        registered = self.source_registry.get("sources", self.source_registry)
        registered = set(registered) if isinstance(registered, Mapping) else set()
        for primitive in self.primitives:
            if not primitive.primitive_id or not primitive.description or primitive.status not in allowed_status:
                raise ValueError(f"invalid primitive record: {primitive.primitive_id}")
            if not primitive.patterns and not primitive.retrieval_only:
                raise ValueError(f"executable primitive has no patterns: {primitive.primitive_id}")
            for pattern in primitive.patterns:
                query = Chem.MolFromSmarts(pattern.smarts)
                if query is None: raise ValueError(f"invalid SMARTS: {pattern.smarts}")
                _role_indexes(query, pattern.roles)
            for source in primitive.sources:
                if registered and source.source_id not in registered:
                    raise ValueError(f"unregistered source {source.source_id}")
            for move in primitive.moves:
                if move.electrons != 2 or move.source_kind not in {"LP", "ATOM", "BOND"} or move.sink_kind not in {"LP", "ATOM", "BOND"}:
                    raise ValueError(f"unsupported move in {primitive.primitive_id}")

    def retrieve(self, state_smiles: str, query: str = "", top_k: int = 8, statuses: Sequence[str] = RETRIEVABLE, max_bindings_per_primitive: int = 8) -> list[PrimitiveMatch]:
        mol, qtokens, output = _mapped_mol(state_smiles), _tokens(query), []
        for primitive in self.primitives:
            if primitive.status not in statuses or primitive.status == "deprecated": continue
            if primitive.retrieval_only:
                if not qtokens or not (qtokens & _tokens(primitive.search_text)): continue
                output.append(PrimitiveMatch(primitive.primitive_id, primitive.name, primitive.prior + 0.25 * len(qtokens & _tokens(primitive.search_text)), {}, (), primitive.warnings, primitive.preconditions, primitive.competing_primitives, primitive.followups, primitive.sources, primitive.status))
                continue
            groups, optional_hits, failed = [], 0, False
            for pattern in primitive.patterns:
                found = _bindings(mol, pattern)
                if pattern.optional: optional_hits += int(bool(found))
                elif not found: failed = True; break
                else: groups.append(found[:max_bindings_per_primitive])
            if failed: continue
            seen = set()
            for parts in product(*groups) if groups else [()]:
                binding = _merge(parts)
                if binding is None or tuple(sorted(binding.items())) in seen: continue
                seen.add(tuple(sorted(binding.items())))
                moves = _moves(primitive, binding)
                if primitive.moves and not moves: continue
                score = primitive.prior + 0.2 * len(binding) + 0.1 * optional_hits + 0.25 * len(qtokens & _tokens(primitive.search_text))
                output.append(PrimitiveMatch(primitive.primitive_id, primitive.name, float(score), binding, moves, primitive.warnings, primitive.preconditions, primitive.competing_primitives, primitive.followups, primitive.sources, primitive.status))
                if len(seen) >= max_bindings_per_primitive: break
        return sorted(output, key=lambda x: (x.score, x.primitive_id), reverse=True)[:max(int(top_k), 0)]

    def support_moves(self, state_smiles: str, moves: Sequence[Mapping[str, Any]], top_k: int = 64) -> dict[str, Any]:
        target = sorted(_move_key(x) for x in moves)
        matches = self.retrieve(state_smiles, top_k=top_k, statuses=VERIFIED)
        supported = [x for x in matches if x.moves and sorted(_move_key(y) for y in x.moves) == target]
        raw = max((x.score for x in supported), default=0.0)
        return {"supported": bool(supported), "support_score": raw / (1 + raw) if raw else 0.0, "primitive_ids": [x.primitive_id for x in supported], "matches": [x.to_dict() for x in supported[:8]], "soft_evidence_only": True}

    def reaction_evidence(self, reactants: str, products: str, top_k: int = 64) -> list[dict[str, Any]]:
        before, after = _bond_inventory(reactants), _bond_inventory(products)
        actual = {p: after.get(p, 0.0) - before.get(p, 0.0) for p in set(before) | set(after)}
        output = []
        for match in self.retrieve(reactants, top_k=top_k, statuses=VERIFIED):
            expected = _expected_deltas(match.moves)
            if not expected: continue
            matched = {p: actual[p] for p, v in expected.items() if p in actual and math.copysign(1, actual[p]) == math.copysign(1, v) and abs(actual[p]) + 1e-8 >= abs(v)}
            output.append({"primitive_id": match.primitive_id, "role_bindings": match.role_bindings, "expected_bond_deltas": {f"{p[0]}-{p[1]}": v for p, v in expected.items()}, "matched_bond_deltas": {f"{p[0]}-{p[1]}": v for p, v in matched.items()}, "support": len(matched) / len(expected), "complete": len(matched) == len(expected)})
        return sorted(output, key=lambda x: (x["support"], x["primitive_id"]), reverse=True)

    def annotate_reaction(self, reactants: str, products: str, top_k: int = 16) -> dict[str, Any]:
        rows = self.reaction_evidence(reactants, products, max(top_k, 32))[:top_k]
        return {"primitive_ids": [x["primitive_id"] for x in rows if x["support"] > 0], "complete_primitive_ids": [x["primitive_id"] for x in rows if x["complete"]], "primitive_evidence": rows, "best_support": max((x["support"] for x in rows), default=0.0), "soft_evidence_only": True}

    def render_context(self, state_smiles: str, query: str = "", top_k: int = 6) -> str:
        matches = self.retrieve(state_smiles, query=query, top_k=top_k)
        if not matches: return "No reviewed mechanistic primitive matched the current state."
        lines = ["Reviewed mechanistic primitive candidates (soft guidance):"]
        for match in matches:
            roles = ", ".join(f"{k}=map:{v}" for k, v in sorted(match.role_bindings.items())) or "no atom binding"
            lines.append(f"- {match.primitive_id}: {match.primitive_name}; {roles}")
            if match.moves: lines.append("  proposed E_MOVE set: " + json.dumps(list(match.moves)))
            if match.warnings: lines.append("  warnings: " + "; ".join(match.warnings))
        lines.append("These records do not override the executor and do not prove selectivity or feasibility.")
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {"n_primitives": len(self.primitives), "primitive_ids": sorted(self.by_id), "statuses": {s: sum(x.status == s for x in self.primitives) for s in sorted({x.status for x in self.primitives})}, "metadata": self.metadata}
