"""Endpoint normalization for full executable and structural precursor states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

from .proof_program import ProofProgramError, _canonical_mapped, sides_equal


ROUTE_ENDPOINT_CONTRACT = 'target_heavy_atom_contributors_v1'


@dataclass(frozen=True)
class PrecursorEndpoints:
    """Two endpoint views used by execution and scientific evaluation.

    ``full`` is the complete executor-derived molecular state. ``structural``
    retains fragments containing at least one atom map from the target product;
    the remaining fragments are reported as ``auxiliary`` rather than silently
    mixed into structural precursor accuracy.
    """

    full: str
    structural: str
    auxiliary: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "full_precursor_state": self.full,
            "structural_precursor": self.structural,
            "auxiliary_fragments": list(self.auxiliary),
        }


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ProofProgramError(f"unparseable mapped SMILES: {str(smiles)[:120]}")
    return mol


def _fragment_smiles(smiles: str) -> list[str]:
    mol = _mapped_mol(smiles)
    return [
        _canonical_mapped(fragment)
        for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    ]


def atom_maps(smiles: str) -> set[int]:
    return {
        int(atom.GetAtomMapNum())
        for atom in _mapped_mol(smiles).GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }


def _split_by_target_maps(full_precursor: str, target_maps: set[int]) -> PrecursorEndpoints:
    structural: list[str] = []
    auxiliary: list[str] = []
    fragments = _fragment_smiles(full_precursor)
    for fragment in fragments:
        maps = atom_maps(fragment)
        (structural if maps & target_maps else auxiliary).append(fragment)
    return PrecursorEndpoints(
        full=".".join(sorted(fragments)),
        structural=".".join(sorted(structural)),
        auxiliary=tuple(sorted(auxiliary)),
    )


def split_precursor_endpoints(full_precursor: str, target_smiles: str) -> PrecursorEndpoints:
    """Historical single-reaction view: contributors include target hydrogens."""
    return _split_by_target_maps(full_precursor, atom_maps(target_smiles))


def split_route_precursor_endpoints(full_precursor: str, target_smiles: str) -> PrecursorEndpoints:
    """Route obligations require target heavy atoms; keep auxiliaries as evidence.

    Retain every contributing fragment, including small heteroatom donors and
    complete leaving groups attached to a structural precursor. A transferred
    target hydrogen alone does not make a reagent a route obligation.
    """
    heavy_maps = [int(atom.GetAtomMapNum()) for atom in _mapped_mol(target_smiles).GetAtoms()
                  if atom.GetAtomicNum() > 1]
    if not heavy_maps or any(v <= 0 for v in heavy_maps) or len(set(heavy_maps)) != len(heavy_maps):
        raise ProofProgramError('ROUTE_TARGET_HEAVY_ATOM_MAPS_REQUIRED')
    return _split_by_target_maps(full_precursor, set(heavy_maps))


def reference_structural_precursor(row: dict[str, object]) -> str:
    """Resolve the frozen structural precursor reference from a dataset row."""

    metadata = dict(row.get("metadata") or {})
    explicit = str(
        row.get("structural_precursor")
        or metadata.get("structural_precursor")
        or metadata.get("core_precursor")
        or ""
    ).strip()
    if explicit:
        return explicit
    full = str(
        row.get("expected_precursor")
        or row.get("full_precursor_state")
        or metadata.get("full_precursor_state")
        or metadata.get("derived_precursor")
        or ""
    ).strip()
    target = str(row.get("target_smiles") or metadata.get("target_smiles") or "").strip()
    if full and target:
        return split_precursor_endpoints(full, target).structural
    return ""


def structural_exact(predicted: str, expected: str) -> bool:
    """Compare structural precursor multisets independently of atom-map labels."""

    return sides_equal(predicted, expected, ignore_maps=True)


def mapped_exact(predicted: str, expected: str) -> bool:
    """Compare structural precursor multisets while preserving atom-map labels."""

    return sides_equal(predicted, expected, ignore_maps=False)
