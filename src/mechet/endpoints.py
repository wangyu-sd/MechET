"""Endpoint normalization for full executable and structural precursor states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rdkit import Chem

from .proof_program import ProofProgramError, _canonical_mapped, sides_equal


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


def split_precursor_endpoints(full_precursor: str, target_smiles: str) -> PrecursorEndpoints:
    """Split a full precursor state into structural and auxiliary fragments."""

    target_maps = atom_maps(target_smiles)
    structural: list[str] = []
    auxiliary: list[str] = []
    for fragment in _fragment_smiles(full_precursor):
        maps = atom_maps(fragment)
        (structural if maps & target_maps else auxiliary).append(fragment)
    return PrecursorEndpoints(
        full=".".join(sorted(_fragment_smiles(full_precursor))),
        structural=".".join(sorted(structural)),
        auxiliary=tuple(sorted(auxiliary)),
    )


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
