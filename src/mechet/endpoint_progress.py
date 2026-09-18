"""Deterministic reward-side distance to a frozen precursor endpoint.

The model never receives values produced by this module.  Atom maps are used
only as executor-private identities.  Callers may provide a current-to-reference
map alignment when an imported fragment received fresh private map numbers.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from rdkit import Chem


@dataclass(frozen=True)
class EndpointDistanceWeights:
    bond: float = 1.0
    charge: float = 0.5
    connectivity: float = 0.5
    fragment: float = 0.5


@dataclass(frozen=True)
class EndpointDistance:
    bond: float
    charge: float
    connectivity: float
    fragment: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bond": self.bond,
            "charge": self.charge,
            "connectivity": self.connectivity,
            "fragment": self.fragment,
            "total": self.total,
        }


@dataclass(frozen=True)
class _MappedGraph:
    atoms: Mapping[int, tuple[int, int, int, int]]
    bonds: Mapping[tuple[int, int], float]
    components: tuple[frozenset[int], ...]


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("endpoint distance requires parseable mapped SMILES")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
        raise ValueError("endpoint distance requires unique positive atom maps")
    return mol


def _graph(
    smiles: str,
    *,
    map_alignment: Mapping[int, int] | None = None,
) -> _MappedGraph:
    mol = _mapped_mol(smiles)
    alignment = {int(key): int(value) for key, value in (map_alignment or {}).items()}

    def identity(atom: Chem.Atom) -> int:
        source = int(atom.GetAtomMapNum())
        return alignment.get(source, source)

    mapped = [identity(atom) for atom in mol.GetAtoms()]
    if len(mapped) != len(set(mapped)):
        raise ValueError("map alignment is not injective over the current state")
    atoms = {
        identity(atom): (
            int(atom.GetAtomicNum()),
            int(atom.GetIsotope()),
            int(atom.GetFormalCharge()),
            int(atom.GetNumRadicalElectrons()),
        )
        for atom in mol.GetAtoms()
    }
    bonds: dict[tuple[int, int], float] = {}
    for bond in mol.GetBonds():
        left = identity(bond.GetBeginAtom())
        right = identity(bond.GetEndAtom())
        bonds[tuple(sorted((left, right)))] = float(bond.GetBondTypeAsDouble())
    components = tuple(
        frozenset(identity(mol.GetAtomWithIdx(index)) for index in indices)
        for indices in Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    )
    return _MappedGraph(atoms=atoms, bonds=bonds, components=components)


def _same_component(graph: _MappedGraph, left: int, right: int) -> bool:
    return any(left in component and right in component for component in graph.components)


def endpoint_distance(
    current: str,
    reference: str,
    *,
    target_atom_maps: Sequence[int],
    contributing_atom_maps: Sequence[int] = (),
    current_to_reference_maps: Mapping[int, int] | None = None,
    weights: EndpointDistanceWeights | None = None,
    extra_fragment_cap: int = 2,
) -> EndpointDistance:
    """Measure mapped bond/electron progress without rewarding spectators.

    ``target_atom_maps`` and ``contributing_atom_maps`` are frozen reference
    identities.  Only those identities participate in bond, charge and
    connectivity terms.  Unaligned current components count as extra fragments,
    but their total contribution is capped so free solvents cannot dominate.
    """

    if extra_fragment_cap < 0:
        raise ValueError("extra_fragment_cap must be non-negative")
    active_weights = weights or EndpointDistanceWeights()
    current_graph = _graph(current, map_alignment=current_to_reference_maps)
    reference_graph = _graph(reference)
    relevant = {
        *(int(value) for value in target_atom_maps),
        *(int(value) for value in contributing_atom_maps),
    }
    if not relevant:
        raise ValueError("endpoint distance requires at least one relevant atom")

    current_atoms = set(current_graph.atoms) & relevant
    reference_atoms = set(reference_graph.atoms) & relevant
    atom_union = current_atoms | reference_atoms

    bond_keys = {
        key
        for key in set(current_graph.bonds) | set(reference_graph.bonds)
        if key[0] in relevant and key[1] in relevant
    }
    bond_term = sum(
        abs(current_graph.bonds.get(key, 0.0) - reference_graph.bonds.get(key, 0.0))
        for key in bond_keys
    )

    charge_term = 0.0
    atom_identity_mismatches = 0
    for atom_map in atom_union:
        current_atom = current_graph.atoms.get(atom_map)
        reference_atom = reference_graph.atoms.get(atom_map)
        if current_atom is None or reference_atom is None:
            continue
        if current_atom[:2] != reference_atom[:2]:
            atom_identity_mismatches += 1
        charge_term += abs(current_atom[2] - reference_atom[2])
        charge_term += abs(current_atom[3] - reference_atom[3])

    connectivity_term = sum(
        _same_component(current_graph, left, right)
        != _same_component(reference_graph, left, right)
        for left, right in combinations(sorted(atom_union), 2)
        if left in current_atoms
        and right in current_atoms
        and left in reference_atoms
        and right in reference_atoms
    )

    def relevant_components(graph: _MappedGraph) -> Counter[frozenset[int]]:
        return Counter(
            frozenset(component & relevant)
            for component in graph.components
            if component & relevant
        )

    current_components = relevant_components(current_graph)
    reference_components = relevant_components(reference_graph)
    component_difference = sum((current_components - reference_components).values())
    component_difference += sum((reference_components - current_components).values())
    missing_atoms = len(reference_atoms - current_atoms)
    unexpected_aligned_atoms = len(current_atoms - reference_atoms)
    reference_spectator_maps = set(reference_graph.atoms) - relevant
    unaligned_components = sum(
        1
        for component in current_graph.components
        if not component & relevant
        and not component <= reference_spectator_maps
    )
    fragment_term = float(
        component_difference
        + missing_atoms
        + unexpected_aligned_atoms
        + atom_identity_mismatches
        + min(unaligned_components, extra_fragment_cap)
    )

    total = (
        active_weights.bond * bond_term
        + active_weights.charge * charge_term
        + active_weights.connectivity * float(connectivity_term)
        + active_weights.fragment * fragment_term
    )
    return EndpointDistance(
        bond=float(bond_term),
        charge=float(charge_term),
        connectivity=float(connectivity_term),
        fragment=fragment_term,
        total=float(total),
    )


def potential_progress_reward(
    before: EndpointDistance | float,
    after: EndpointDistance | float,
    start: EndpointDistance | float,
    *,
    event_cap: float = 0.25,
    epsilon: float = 1e-8,
) -> float:
    """Return clipped potential improvement for one committed event."""

    if event_cap <= 0:
        raise ValueError("event_cap must be positive")
    before_value = before.total if isinstance(before, EndpointDistance) else float(before)
    after_value = after.total if isinstance(after, EndpointDistance) else float(after)
    start_value = start.total if isinstance(start, EndpointDistance) else float(start)
    raw = (before_value - after_value) / max(start_value, epsilon)
    return max(-event_cap, min(event_cap, raw))


def capped_progress_increment(
    accumulated: float,
    proposed: float,
    *,
    total_cap: float = 1.0,
) -> tuple[float, float]:
    """Apply one shaping increment while respecting a rollout-wide cap."""

    if total_cap <= 0:
        raise ValueError("total_cap must be positive")
    updated = max(-total_cap, min(total_cap, accumulated + proposed))
    return updated - accumulated, updated
