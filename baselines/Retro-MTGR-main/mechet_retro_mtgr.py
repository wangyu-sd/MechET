"""MechET adapter primitives for Retro-MTGR-style center/LG prediction.

The upstream snapshot is kept untouched.  This module implements the minimum
strict adapter needed to audit its native one-bond/two-leaving-group target:
unsupported reactions are retained with an explicit reason instead of being
silently removed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem
import torch


ATOM_SYMBOLS = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
    "Fe", "Al", "I", "B", "K", "Se", "Zn", "H", "Cu", "Mn", "*", "unknown",
)
BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(row: dict[str, Any]) -> str:
    value = row.get("stable_id") or row.get("id")
    if not value:
        raise ValueError("row is missing stable_id/id")
    return str(value)


def canonical_unmapped(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def structural_key(smiles: str) -> tuple[str, ...] | None:
    if not smiles:
        return None
    parts: list[str] = []
    for fragment in smiles.split("."):
        value = canonical_unmapped(fragment.strip())
        if not value:
            return None
        parts.append(value)
    return tuple(sorted(parts))


def _bond_type_name(bond: Chem.Bond) -> str:
    return str(bond.GetBondType())


def _bond_type(value: str) -> Chem.BondType:
    lookup = {str(item): item for item in BOND_TYPES}
    if value not in lookup:
        raise ValueError(f"unsupported bond type: {value}")
    return lookup[value]


def _map_atoms(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    result: dict[int, Chem.Atom] = {}
    for atom in mol.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        if atom_map:
            if atom_map in result:
                raise ValueError(f"duplicate atom map {atom_map}")
            result[atom_map] = atom
    return result


def _mapped_bonds(mol: Chem.Mol) -> dict[tuple[int, int], str]:
    result: dict[tuple[int, int], str] = {}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom().GetAtomMapNum()
        end = bond.GetEndAtom().GetAtomMapNum()
        if begin and end:
            result[tuple(sorted((begin, end)))] = _bond_type_name(bond)
    return result


def _atom_patch(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "formal_charge": atom.GetFormalCharge(),
        "explicit_hs": atom.GetNumExplicitHs(),
        "no_implicit": atom.GetNoImplicit(),
        "isotope": atom.GetIsotope(),
        "chiral_tag": int(atom.GetChiralTag()),
    }


def _copy_atom(atom: Chem.Atom, *, clear_map: bool = True) -> Chem.Atom:
    copied = Chem.Atom(atom)
    if clear_map:
        copied.SetAtomMapNum(0)
        if copied.HasProp("molAtomMapNumber"):
            copied.ClearProp("molAtomMapNumber")
    return copied


def _label_spec(
    precursor: Chem.Mol,
    product_maps: set[int],
    anchor_map: int,
) -> dict[str, Any]:
    precursor_atoms = _map_atoms(precursor)
    if anchor_map not in precursor_atoms:
        raise ValueError(f"anchor map {anchor_map} absent from precursor")
    anchor = precursor_atoms[anchor_map]
    fragment_ids = Chem.GetMolFrags(precursor, asMols=False, sanitizeFrags=False)
    component = next(ids for ids in fragment_ids if anchor.GetIdx() in ids)
    extra_ids = {
        index
        for index in component
        if precursor.GetAtomWithIdx(index).GetAtomMapNum() not in product_maps
    }
    attached_core_ids: set[int] = set()
    for index in extra_ids:
        for neighbor in precursor.GetAtomWithIdx(index).GetNeighbors():
            if neighbor.GetIdx() not in extra_ids:
                attached_core_ids.add(neighbor.GetIdx())
    if any(precursor.GetAtomWithIdx(index).GetAtomMapNum() != anchor_map for index in attached_core_ids):
        raise ValueError("leaving group attaches outside the reaction-center endpoint")

    rw = Chem.RWMol()
    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(1)
    dummy_index = rw.AddAtom(dummy)
    old_to_new: dict[int, int] = {}
    for old_index in sorted(extra_ids):
        old_to_new[old_index] = rw.AddAtom(_copy_atom(precursor.GetAtomWithIdx(old_index)))
    for bond in precursor.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in extra_ids and end in extra_ids:
            rw.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
        elif begin == anchor.GetIdx() and end in extra_ids:
            rw.AddBond(dummy_index, old_to_new[end], bond.GetBondType())
        elif end == anchor.GetIdx() and begin in extra_ids:
            rw.AddBond(dummy_index, old_to_new[begin], bond.GetBondType())
    attachment = Chem.MolToSmiles(rw.GetMol(), canonical=True, isomericSmiles=True)
    spec = {"anchor": _atom_patch(anchor), "attachment": attachment}
    spec["key"] = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return spec


def _apply_atom_patch(atom: Chem.Atom, patch: dict[str, Any]) -> None:
    atom.SetFormalCharge(int(patch["formal_charge"]))
    atom.SetNumExplicitHs(int(patch["explicit_hs"]))
    atom.SetNoImplicit(bool(patch["no_implicit"]))
    atom.SetIsotope(int(patch["isotope"]))
    atom.SetChiralTag(Chem.ChiralType(int(patch["chiral_tag"])))


def _attach_label(rw: Chem.RWMol, anchor_index: int, spec: dict[str, Any]) -> None:
    _apply_atom_patch(rw.GetAtomWithIdx(anchor_index), spec["anchor"])
    fragment = Chem.MolFromSmiles(spec["attachment"], sanitize=False)
    if fragment is None:
        raise ValueError(f"cannot parse attachment label: {spec['attachment']}")
    dummy_indices = [
        atom.GetIdx()
        for atom in fragment.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == 1
    ]
    if len(dummy_indices) != 1:
        raise ValueError("attachment label must contain exactly one mapped dummy")
    dummy_index = dummy_indices[0]
    fragment_to_output: dict[int, int] = {}
    for atom in fragment.GetAtoms():
        if atom.GetIdx() != dummy_index:
            fragment_to_output[atom.GetIdx()] = rw.AddAtom(_copy_atom(atom))
    for bond in fragment.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if dummy_index in (begin, end):
            neighbor = end if begin == dummy_index else begin
            rw.AddBond(anchor_index, fragment_to_output[neighbor], bond.GetBondType())
        else:
            rw.AddBond(fragment_to_output[begin], fragment_to_output[end], bond.GetBondType())


def apply_retro_edit(
    product_smiles: str,
    bond_indices: tuple[int, int] | list[int],
    left_spec: dict[str, Any],
    right_spec: dict[str, Any],
) -> str:
    product = Chem.MolFromSmiles(product_smiles)
    if product is None:
        return ""
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(0)
    left, right = (int(value) for value in bond_indices)
    if product.GetBondBetweenAtoms(left, right) is None:
        return ""
    try:
        rw = Chem.RWMol(product)
        rw.RemoveBond(left, right)
        _attach_label(rw, left, left_spec)
        _attach_label(rw, right, right_spec)
        result = rw.GetMol()
        Chem.SanitizeMol(result)
        return Chem.MolToSmiles(result, canonical=True, isomericSmiles=True)
    except (ValueError, RuntimeError):
        return ""


@dataclass
class Analysis:
    target: dict[str, Any]
    label_specs: list[dict[str, Any]]


def analyze_row(row: dict[str, Any]) -> Analysis:
    identifier = stable_id(row)
    target: dict[str, Any] = {
        "stable_id": identifier,
        "product_mapped": row.get("product_mapped", ""),
        "product_unmapped": row.get("product_unmapped", ""),
        "reference_precursors": row.get("precursor_unmapped", ""),
        "status": "unsupported",
        "reason": "",
    }
    product = Chem.MolFromSmiles(target["product_mapped"])
    precursor = Chem.MolFromSmiles(row.get("precursor_mapped", ""))
    if product is None or precursor is None:
        target["reason"] = "parse_failure"
        return Analysis(target, [])
    try:
        product_atoms = _map_atoms(product)
        precursor_atoms = _map_atoms(precursor)
    except ValueError as error:
        target["reason"] = "duplicate_atom_map"
        target["detail"] = str(error)
        return Analysis(target, [])
    missing_maps = sorted(set(product_atoms) - set(precursor_atoms))
    if missing_maps:
        target["reason"] = "product_maps_missing_from_precursor"
        target["detail"] = missing_maps
        return Analysis(target, [])
    fragments = Chem.GetMolFrags(precursor, asMols=False, sanitizeFrags=False)
    if len(fragments) != 2:
        target["reason"] = "precursor_fragment_count_not_two"
        target["detail"] = len(fragments)
        return Analysis(target, [])

    product_bonds = _mapped_bonds(product)
    precursor_bonds = _mapped_bonds(precursor)
    removed = sorted(key for key in product_bonds if key not in precursor_bonds)
    changed = sorted(
        key
        for key in product_bonds.keys() & precursor_bonds.keys()
        if product_bonds[key] != precursor_bonds[key]
    )
    if len(removed) != 1 or changed:
        target["reason"] = "reaction_center_not_single_removed_bond"
        target["detail"] = {"removed_bonds": len(removed), "changed_bonds": len(changed)}
        return Analysis(target, [])
    left_map, right_map = removed[0]
    left_index = product_atoms[left_map].GetIdx()
    right_index = product_atoms[right_map].GetIdx()
    if left_index > right_index:
        left_index, right_index = right_index, left_index
        left_map, right_map = right_map, left_map
    try:
        left_spec = _label_spec(precursor, set(product_atoms), left_map)
        right_spec = _label_spec(precursor, set(product_atoms), right_map)
    except ValueError as error:
        target["reason"] = "nonlocal_leaving_group"
        target["detail"] = str(error)
        return Analysis(target, [])

    reconstructed = apply_retro_edit(
        target["product_mapped"], (left_index, right_index), left_spec, right_spec
    )
    if structural_key(reconstructed) != structural_key(target["reference_precursors"]):
        target["reason"] = "native_edit_does_not_reconstruct_reference"
        target["reconstructed"] = reconstructed
        return Analysis(target, [left_spec, right_spec])
    target.update({
        "status": "supported",
        "reason": "",
        "center_atom_indices": [left_index, right_index],
        "center_atom_maps": [left_map, right_map],
        "center_bond_type": product_bonds[tuple(sorted((left_map, right_map)))],
        "left_label_key": left_spec["key"],
        "right_label_key": right_spec["key"],
        "reconstructed": reconstructed,
    })
    return Analysis(target, [left_spec, right_spec])


def atom_features(mol: Chem.Mol) -> torch.Tensor:
    symbol_to_index = {symbol: index for index, symbol in enumerate(ATOM_SYMBOLS)}
    rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol() if atom.GetSymbol() in symbol_to_index else "unknown"
        one_hot = [0.0] * len(ATOM_SYMBOLS)
        one_hot[symbol_to_index[symbol]] = 1.0
        rows.append(one_hot + [
            atom.GetTotalNumHs() / 4.0,
            atom.GetDegree() / 6.0,
            float(atom.GetIsAromatic()),
            atom.GetFormalCharge() / 4.0,
            atom.GetMass() / 200.0,
        ])
    return torch.tensor(rows, dtype=torch.float32)


def graph_tensors(smiles: str, device: torch.device | str = "cpu") -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid molecule: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    features = atom_features(mol).to(device)
    adjacency = torch.eye(mol.GetNumAtoms(), dtype=torch.float32, device=device)
    bonds: list[tuple[int, int]] = []
    bond_features: list[list[float]] = []
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        adjacency[left, right] = 1.0
        adjacency[right, left] = 1.0
        if left > right:
            left, right = right, left
        bonds.append((left, right))
        bond_features.append([
            float(bond.GetBondType() == bond_type) for bond_type in BOND_TYPES
        ] + [float(bond.IsInRing()), float(bond.GetIsConjugated())])
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    scale = degree.rsqrt()
    adjacency = scale[:, None] * adjacency * scale[None, :]
    return {
        "mol": mol,
        "x": features,
        "adj": adjacency,
        "bonds": bonds,
        "bond_features": torch.tensor(bond_features, dtype=torch.float32, device=device),
    }


def precursor_graph_smiles(smiles: str) -> str:
    return canonical_unmapped(smiles)

