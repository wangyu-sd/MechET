"""Partial-order equivalence and compositional signatures for MECH_PROOF v1.

The model may linearize independent electron-flow events in different orders. This
module canonicalizes a proof into a state-id-, edge-order-, and atom-map-label-
invariant signature. Dependencies are retained only when two events lie on the
same proof path and touch a common mapped atom; disjoint events are commuting.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

from rdkit import Chem

from mechet.proof_program import (
    ProofEdge,
    ProofProgram,
    ProofProgramError,
    execute_proof,
    parse_proof_program,
)


@dataclass(frozen=True)
class ProofEquivalenceSignature:
    """Canonical proof signature independent of serialization choices."""

    target: str
    root_imports: tuple[str, ...]
    event_counts: tuple[tuple[str, int], ...]
    dependency_counts: tuple[tuple[str, str, int], ...]
    endpoint: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        raise ProofProgramError(
            f"unparseable species in proof signature: {smiles[:120]}"
        )
    return mol


def _canonical_species(smiles: str, *, ignore_maps: bool) -> str:
    mol = _mol(smiles)
    if ignore_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _canonical_multiset(
    smiles_items: Iterable[str],
    *,
    ignore_maps: bool,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            _canonical_species(item, ignore_maps=ignore_maps)
            for item in smiles_items
            if item
        )
    )


def _fragment_maps(smiles: str) -> set[int]:
    return {
        atom.GetAtomMapNum()
        for atom in _mol(smiles).GetAtoms()
        if atom.GetAtomMapNum() > 0
    }


def edge_touched_maps(edge: ProofEdge) -> frozenset[int]:
    maps: set[int] = set()
    for i, j, _ in edge.bonds:
        maps.update((int(i), int(j)))
    maps.update(int(atom_map) for atom_map, _ in edge.lone_pairs)
    maps.update(int(action.atom_map) for action in edge.charges)
    for fragment in edge.imports:
        maps.update(_fragment_maps(fragment))
    return frozenset(maps)


def _source_atom_features(
    source_smiles: str,
    edge: ProofEdge,
) -> dict[int, tuple[int, int, int, int]]:
    combined = ".".join(
        part for part in [source_smiles, *edge.imports] if part
    )
    imported_maps = {
        atom_map
        for fragment in edge.imports
        for atom_map in _fragment_maps(fragment)
    }
    features: dict[int, tuple[int, int, int, int]] = {}
    for atom in _mol(combined).GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map <= 0:
            continue
        features[atom_map] = (
            int(atom.GetAtomicNum()),
            int(atom.GetFormalCharge()),
            int(atom.GetIsAromatic()),
            int(atom_map in imported_maps),
        )
    return features


def _canonical_local_maps(
    edge: ProofEdge,
    source_smiles: str,
) -> dict[int, int]:
    """Canonicalize local action roles without using original map labels."""
    touched = sorted(edge_touched_maps(edge))
    features = _source_atom_features(source_smiles, edge)
    lp = dict(edge.lone_pairs)
    charge = {
        action.atom_map: (action.q0, action.q1)
        for action in edge.charges
    }
    adjacency: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    source_mol = _mol(
        ".".join(part for part in [source_smiles, *edge.imports] if part)
    )
    map_to_atom = {
        atom.GetAtomMapNum(): atom
        for atom in source_mol.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }
    for i, j, delta in edge.bonds:
        current = 0
        if i in map_to_atom and j in map_to_atom:
            bond = source_mol.GetBondBetweenAtoms(
                map_to_atom[i].GetIdx(),
                map_to_atom[j].GetIdx(),
            )
            current = (
                int(round(bond.GetBondTypeAsDouble()))
                if bond is not None
                else 0
            )
        adjacency[i].append((j, current, int(delta)))
        adjacency[j].append((i, current, int(delta)))

    labels: dict[int, str] = {}
    for atom_map in touched:
        descriptor = {
            "atom": features.get(atom_map, (0, 0, 0, 0)),
            "lp": int(lp.get(atom_map, 0)),
            "charge": charge.get(atom_map, (0, 0)),
            "degree": len(adjacency.get(atom_map, [])),
        }
        labels[atom_map] = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        )
    for _ in range(max(len(touched), 1)):
        refined: dict[int, str] = {}
        for atom_map in touched:
            neighborhood = sorted(
                (labels[neighbor], current, delta)
                for neighbor, current, delta in adjacency.get(atom_map, [])
            )
            payload = json.dumps(
                {"self": labels[atom_map], "neighbors": neighborhood},
                sort_keys=True,
                separators=(",", ":"),
            )
            refined[atom_map] = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()
        labels = refined

    ordered = sorted(
        touched,
        key=lambda atom_map: (
            labels[atom_map],
            sorted(
                (labels[neighbor], current, delta)
                for neighbor, current, delta in adjacency.get(atom_map, [])
            ),
            features.get(atom_map, (0, 0, 0, 0)),
            int(lp.get(atom_map, 0)),
            charge.get(atom_map, (0, 0)),
            atom_map,
        ),
    )
    return {atom_map: index for index, atom_map in enumerate(ordered)}


def edge_primitive_signature(
    edge: ProofEdge,
    *,
    source_smiles: str = "",
    abstract_maps: bool = True,
) -> str:
    """Return a deterministic chemistry-role label for one proof event."""
    map_ids = (
        _canonical_local_maps(edge, source_smiles)
        if abstract_maps and source_smiles
        else {
            atom_map: atom_map
            for atom_map in edge_touched_maps(edge)
        }
    )
    payload = {
        "imports": _canonical_multiset(edge.imports, ignore_maps=True),
        "bonds": sorted(
            (map_ids[i], map_ids[j], int(delta))
            for i, j, delta in edge.bonds
        ),
        "lone_pairs": sorted(
            (map_ids[atom_map], int(delta))
            for atom_map, delta in edge.lone_pairs
        ),
        "charges": sorted(
            (map_ids[action.atom_map], int(action.q0), int(action.q1))
            for action in edge.charges
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _event_reachability(program: ProofProgram) -> set[tuple[int, int]]:
    outgoing_by_state: dict[str, list[int]] = defaultdict(list)
    for index, edge in enumerate(program.edges):
        outgoing_by_state[edge.src].append(index)
    precedence: set[tuple[int, int]] = set()
    for source_index, source_edge in enumerate(program.edges):
        queue = deque([source_edge.dst])
        seen_states = {source_edge.dst}
        while queue:
            state = queue.popleft()
            for target_index in outgoing_by_state.get(state, []):
                if target_index != source_index:
                    precedence.add((source_index, target_index))
                next_state = program.edges[target_index].dst
                if next_state not in seen_states:
                    seen_states.add(next_state)
                    queue.append(next_state)
    return precedence


def canonical_partial_order_signature(
    program_or_text: ProofProgram | str,
    *,
    require_executable: bool = True,
) -> ProofEquivalenceSignature:
    """Canonicalize a proof modulo state ids, map labels, and commuting order."""
    program = (
        parse_proof_program(program_or_text)
        if isinstance(program_or_text, str)
        else program_or_text
    )
    execution = execute_proof(program)
    if require_executable and not execution.ok:
        message = (
            execution.diagnostics[0].get(
                "message",
                "proof execution failed",
            )
            if execution.diagnostics
            else "proof execution failed"
        )
        raise ProofProgramError(message)

    labels = [
        edge_primitive_signature(
            edge,
            source_smiles=execution.states.get(edge.src, ""),
        )
        for edge in program.edges
    ]
    touched = [edge_touched_maps(edge) for edge in program.edges]
    event_counts = Counter(labels)
    dependency_counts: Counter[tuple[str, str]] = Counter()
    for left, right in _event_reachability(program):
        if touched[left] & touched[right]:
            dependency_counts[(labels[left], labels[right])] += 1

    root_imports = tuple(
        sorted(
            item
            for imports in program.roots.values()
            for item in _canonical_multiset(imports, ignore_maps=True)
        )
    )
    endpoint = ""
    if execution.ok:
        endpoint = ".".join(
            sorted(
                _canonical_species(fragment, ignore_maps=True)
                for fragment in execution.precursor_smiles.split(".")
                if fragment
            )
        )
    return ProofEquivalenceSignature(
        target=_canonical_species(program.target_smiles, ignore_maps=True),
        root_imports=root_imports,
        event_counts=tuple(sorted(event_counts.items())),
        dependency_counts=tuple(
            sorted(
                (left, right, count)
                for (left, right), count in dependency_counts.items()
            )
        ),
        endpoint=endpoint,
    )


def primitive_signatures(
    program_or_text: ProofProgram | str,
) -> tuple[str, ...]:
    program = (
        parse_proof_program(program_or_text)
        if isinstance(program_or_text, str)
        else program_or_text
    )
    execution = execute_proof(program)
    if not execution.ok:
        message = (
            execution.diagnostics[0].get(
                "message",
                "proof execution failed",
            )
            if execution.diagnostics
            else "proof execution failed"
        )
        raise ProofProgramError(message)
    return tuple(
        sorted(
            {
                edge_primitive_signature(
                    edge,
                    source_smiles=execution.states.get(edge.src, ""),
                )
                for edge in program.edges
            }
        )
    )


def composition_signature(program_or_text: ProofProgram | str) -> str:
    """Digest action composition while excluding target and endpoint molecules."""
    signature = canonical_partial_order_signature(program_or_text)
    payload = {
        "event_counts": signature.event_counts,
        "dependency_counts": signature.dependency_counts,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def proofs_equivalent(
    left: ProofProgram | str,
    right: ProofProgram | str,
    *,
    require_same_endpoint: bool = True,
) -> bool:
    left_signature = canonical_partial_order_signature(left)
    right_signature = canonical_partial_order_signature(right)
    if not require_same_endpoint:
        left_signature = ProofEquivalenceSignature(
            left_signature.target,
            left_signature.root_imports,
            left_signature.event_counts,
            left_signature.dependency_counts,
            "",
        )
        right_signature = ProofEquivalenceSignature(
            right_signature.target,
            right_signature.root_imports,
            right_signature.event_counts,
            right_signature.dependency_counts,
            "",
        )
    return left_signature == right_signature
