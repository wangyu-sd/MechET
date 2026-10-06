"""Lossless endpoint-to-typed-decision compiler for JevRetro.

The compiler operates on an atom-mapped product and its atom-contributing
structural precursor. It factorizes the inverse transformation into:

1. atom-state changes on product-origin atoms;
2. bond-state changes between product-origin atoms;
3. deletion of product atoms absent from the precursor (kept explicit);
4. precursor-only residual fragments represented as attachment templates with
   dummy attachment slots.

The representation is intended for full single-step endpoint supervision. It
does not require a mechanistic trace and therefore can be built on complete
reaction-level training splits.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from rdkit import Chem


_BOND_TO_NAME = {
    Chem.BondType.SINGLE: "SINGLE",
    Chem.BondType.DOUBLE: "DOUBLE",
    Chem.BondType.TRIPLE: "TRIPLE",
    Chem.BondType.AROMATIC: "AROMATIC",
}
_NAME_TO_BOND = {value: key for key, value in _BOND_TO_NAME.items()}


def canonical_unmapped(smiles: str, *, isomeric: bool = True) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def _mapped_mol(smiles: str, *, role: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"{role} SMILES does not parse")
    seen: set[int] = set()
    for atom in mol.GetAtoms():
        atom_map = int(atom.GetAtomMapNum())
        if atom_map <= 0:
            raise ValueError(f"{role} contains an unmapped atom")
        if atom_map in seen:
            raise ValueError(f"{role} contains duplicate atom map {atom_map}")
        seen.add(atom_map)
    return mol


def _atom_by_map(mol: Chem.Mol) -> dict[int, Chem.Atom]:
    return {int(atom.GetAtomMapNum()): atom for atom in mol.GetAtoms()}


def _index_by_map(mol: Chem.Mol) -> dict[int, int]:
    return {int(atom.GetAtomMapNum()): atom.GetIdx() for atom in mol.GetAtoms()}


def _bond_type_name(bond: Chem.Bond) -> str:
    try:
        return _BOND_TO_NAME[bond.GetBondType()]
    except KeyError as exc:
        raise ValueError(f"unsupported bond type: {bond.GetBondType()}") from exc


def _bond_states(mol: Chem.Mol) -> dict[tuple[int, int], dict[str, Any]]:
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for bond in mol.GetBonds():
        left = int(bond.GetBeginAtom().GetAtomMapNum())
        right = int(bond.GetEndAtom().GetAtomMapNum())
        if left <= 0 or right <= 0:
            raise ValueError("bond endpoint is missing an atom map")
        stereo_atoms = []
        for atom_index in bond.GetStereoAtoms():
            atom_map = int(mol.GetAtomWithIdx(int(atom_index)).GetAtomMapNum())
            if atom_map > 0:
                stereo_atoms.append(atom_map)
        output[tuple(sorted((left, right)))] = {
            "type": _bond_type_name(bond),
            "stereo": int(bond.GetStereo()),
            "stereo_atoms": stereo_atoms,
        }
    return output


def _atom_state(atom: Chem.Atom) -> dict[str, Any]:
    return {
        "atomic_num": int(atom.GetAtomicNum()),
        "formal_charge": int(atom.GetFormalCharge()),
        "isotope": int(atom.GetIsotope()),
        "chiral_tag": int(atom.GetChiralTag()),
        "explicit_h": int(atom.GetNumExplicitHs()),
        "no_implicit": bool(atom.GetNoImplicit()),
        "aromatic": bool(atom.GetIsAromatic()),
        "radical_electrons": int(atom.GetNumRadicalElectrons()),
    }


def _copy_atom(atom: Chem.Atom, *, keep_map: bool) -> Chem.Atom:
    new = Chem.Atom(int(atom.GetAtomicNum()))
    new.SetFormalCharge(int(atom.GetFormalCharge()))
    new.SetIsotope(int(atom.GetIsotope()))
    new.SetChiralTag(atom.GetChiralTag())
    new.SetNumExplicitHs(int(atom.GetNumExplicitHs()))
    new.SetNoImplicit(bool(atom.GetNoImplicit()))
    new.SetIsAromatic(bool(atom.GetIsAromatic()))
    new.SetNumRadicalElectrons(int(atom.GetNumRadicalElectrons()))
    new.SetAtomMapNum(int(atom.GetAtomMapNum()) if keep_map else 0)
    return new


def _residual_components(
    precursor: Chem.Mol,
    *,
    product_maps: set[int],
) -> list[set[int]]:
    residual = {
        atom.GetIdx()
        for atom in precursor.GetAtoms()
        if int(atom.GetAtomMapNum()) not in product_maps
    }
    components: list[set[int]] = []
    unseen = set(residual)
    while unseen:
        root = unseen.pop()
        component = {root}
        queue: deque[int] = deque([root])
        while queue:
            current = queue.popleft()
            atom = precursor.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                idx = neighbor.GetIdx()
                if idx in unseen:
                    unseen.remove(idx)
                    component.add(idx)
                    queue.append(idx)
        components.append(component)
    return components


def _attachment_template(
    precursor: Chem.Mol,
    component: set[int],
    *,
    product_maps: set[int],
) -> dict[str, Any]:
    rw = Chem.RWMol()
    local: dict[int, int] = {}
    for old_index in sorted(component):
        local[old_index] = rw.AddAtom(
            _copy_atom(precursor.GetAtomWithIdx(old_index), keep_map=False)
        )

    for bond in precursor.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in component and right in component:
            rw.AddBond(local[left], local[right], bond.GetBondType())
            new_bond = rw.GetBondBetweenAtoms(local[left], local[right])
            if new_bond is not None:
                new_bond.SetIsAromatic(bool(bond.GetIsAromatic()))

    cuts: list[tuple[int, int, Chem.Bond]] = []
    for residual_index in sorted(component):
        atom = precursor.GetAtomWithIdx(residual_index)
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(residual_index)
            if other in component:
                continue
            anchor_map = int(precursor.GetAtomWithIdx(other).GetAtomMapNum())
            if anchor_map not in product_maps:
                raise ValueError("residual component touches an unknown non-product atom")
            cuts.append((anchor_map, residual_index, bond))
    cuts.sort(
        key=lambda item: (
            item[0],
            int(precursor.GetAtomWithIdx(item[1]).GetAtomMapNum()),
            _bond_type_name(item[2]),
        )
    )

    slots = []
    for slot, (anchor_map, residual_index, bond) in enumerate(cuts, start=1):
        dummy = Chem.Atom(0)
        dummy.SetIsotope(slot)
        dummy_index = rw.AddAtom(dummy)
        rw.AddBond(local[residual_index], dummy_index, bond.GetBondType())
        slots.append(
            {
                "slot": slot,
                "anchor_map": anchor_map,
                "bond": _bond_type_name(bond),
            }
        )

    template = rw.GetMol()
    try:
        Chem.SanitizeMol(template)
    except Exception:
        # Dummy-bearing residual fragments can be syntactically serializable
        # even when standalone valence perception is incomplete. The final
        # reconstructed molecule is sanitized again after attachment.
        template.UpdatePropertyCache(strict=False)
    smiles = Chem.MolToSmiles(template, canonical=True, isomericSmiles=True)
    return {"template": smiles, "slots": slots}


def derive_endpoint_program(product_smiles: str, precursor_smiles: str) -> dict[str, Any]:
    product = _mapped_mol(product_smiles, role="product")
    precursor = _mapped_mol(precursor_smiles, role="precursor")
    p_atoms = _atom_by_map(product)
    r_atoms = _atom_by_map(precursor)
    product_maps = set(p_atoms)
    precursor_maps = set(r_atoms)
    shared = product_maps & precursor_maps

    for atom_map in sorted(shared):
        if p_atoms[atom_map].GetAtomicNum() != r_atoms[atom_map].GetAtomicNum():
            raise ValueError(f"atom identity changes at map {atom_map}")

    atom_edits = []
    for atom_map in sorted(shared):
        before = _atom_state(p_atoms[atom_map])
        after = _atom_state(r_atoms[atom_map])
        if before != after:
            atom_edits.append({"map": atom_map, "from": before, "to": after})

    p_bonds = _bond_states(product)
    r_bonds = _bond_states(precursor)
    bond_edits = []
    pairs = {
        pair
        for pair in set(p_bonds) | set(r_bonds)
        if pair[0] in shared and pair[1] in shared
    }
    for pair in sorted(pairs):
        before = p_bonds.get(pair)
        after = r_bonds.get(pair)
        if before != after:
            bond_edits.append(
                {
                    "maps": list(pair),
                    "from": before,
                    "to": after,
                }
            )

    attachments = [
        _attachment_template(
            precursor,
            component,
            product_maps=product_maps,
        )
        for component in _residual_components(
            precursor,
            product_maps=product_maps,
        )
    ]
    attachments.sort(
        key=lambda item: (
            item["template"],
            tuple(
                (slot["anchor_map"], slot["slot"], slot["bond"])
                for slot in item["slots"]
            ),
        )
    )

    program = {
        "schema": "jevretro_endpoint_program_v1",
        "delete_product_maps": sorted(product_maps - precursor_maps),
        "atom_edits": atom_edits,
        "bond_edits": bond_edits,
        "attachments": attachments,
    }
    reconstructed = apply_endpoint_program(product_smiles, program)
    expected = canonical_unmapped(precursor_smiles, isomeric=True)
    if reconstructed != expected:
        raise ValueError(
            "endpoint program is not lossless: "
            f"reconstructed={reconstructed} expected={expected}"
        )
    return program


def _set_atom_state(atom: Chem.Atom, state: dict[str, Any]) -> None:
    if int(atom.GetAtomicNum()) != int(state["atomic_num"]):
        raise ValueError("endpoint program attempts to change atomic number")
    atom.SetFormalCharge(int(state["formal_charge"]))
    atom.SetIsotope(int(state["isotope"]))
    atom.SetChiralTag(Chem.ChiralType(int(state["chiral_tag"])))
    atom.SetNumExplicitHs(int(state["explicit_h"]))
    atom.SetNoImplicit(bool(state["no_implicit"]))
    atom.SetIsAromatic(bool(state["aromatic"]))
    atom.SetNumRadicalElectrons(int(state["radical_electrons"]))


def _set_bond_state(
    rw: Chem.RWMol,
    left: int,
    right: int,
    state: dict[str, Any] | None,
) -> None:
    bond = rw.GetBondBetweenAtoms(left, right)
    if state is None:
        if bond is not None:
            rw.RemoveBond(left, right)
        return
    bond_type = _NAME_TO_BOND[str(state["type"])]
    if bond is None:
        rw.AddBond(left, right, bond_type)
        bond = rw.GetBondBetweenAtoms(left, right)
    else:
        bond.SetBondType(bond_type)
    if bond is None:
        raise RuntimeError("failed to create edited bond")
    bond.SetIsAromatic(bond_type == Chem.BondType.AROMATIC)
    bond.SetStereo(Chem.BondStereo(int(state.get("stereo", 0))))


def _attach_template(
    rw: Chem.RWMol,
    attachment: dict[str, Any],
) -> None:
    template = Chem.MolFromSmiles(str(attachment["template"]), sanitize=False)
    if template is None:
        raise ValueError("attachment template does not parse")
    current = rw.GetMol()
    anchor_indices = _index_by_map(current)
    slots = {int(item["slot"]): item for item in attachment.get("slots") or []}

    real_old = [atom.GetIdx() for atom in template.GetAtoms() if atom.GetAtomicNum() != 0]
    new_index: dict[int, int] = {}
    for old in real_old:
        new_index[old] = rw.AddAtom(_copy_atom(template.GetAtomWithIdx(old), keep_map=False))

    for bond in template.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        left_dummy = template.GetAtomWithIdx(left).GetAtomicNum() == 0
        right_dummy = template.GetAtomWithIdx(right).GetAtomicNum() == 0
        if not left_dummy and not right_dummy:
            rw.AddBond(new_index[left], new_index[right], bond.GetBondType())
            added = rw.GetBondBetweenAtoms(new_index[left], new_index[right])
            if added is not None:
                added.SetIsAromatic(bool(bond.GetIsAromatic()))
            continue
        if left_dummy and right_dummy:
            raise ValueError("attachment template contains dummy-dummy bond")
        dummy_index = left if left_dummy else right
        real_index = right if left_dummy else left
        slot_id = int(template.GetAtomWithIdx(dummy_index).GetIsotope())
        slot = slots.get(slot_id)
        if slot is None:
            raise ValueError(f"attachment template missing slot metadata {slot_id}")
        anchor_map = int(slot["anchor_map"])
        if anchor_map not in anchor_indices:
            raise ValueError(f"attachment anchor map {anchor_map} absent from product")
        rw.AddBond(
            new_index[real_index],
            anchor_indices[anchor_map],
            _NAME_TO_BOND[str(slot["bond"])],
        )


def apply_endpoint_program(product_smiles: str, program: dict[str, Any]) -> str:
    if program.get("schema") != "jevretro_endpoint_program_v1":
        raise ValueError("unsupported endpoint program schema")
    product = _mapped_mol(product_smiles, role="product")
    rw = Chem.RWMol(product)

    indices = _index_by_map(rw.GetMol())
    for edit in program.get("atom_edits") or []:
        atom_map = int(edit["map"])
        if atom_map not in indices:
            raise ValueError(f"atom edit map {atom_map} absent from product")
        _set_atom_state(rw.GetAtomWithIdx(indices[atom_map]), dict(edit["to"]))

    for edit in program.get("bond_edits") or []:
        left_map, right_map = (int(x) for x in edit["maps"])
        indices = _index_by_map(rw.GetMol())
        if left_map not in indices or right_map not in indices:
            raise ValueError("bond edit references absent product atom")
        _set_bond_state(
            rw,
            indices[left_map],
            indices[right_map],
            edit.get("to"),
        )

    indices = _index_by_map(rw.GetMol())
    delete_indices = []
    for atom_map in program.get("delete_product_maps") or []:
        if int(atom_map) not in indices:
            raise ValueError(f"delete map {atom_map} absent from product")
        delete_indices.append(indices[int(atom_map)])
    for atom_index in sorted(delete_indices, reverse=True):
        rw.RemoveAtom(atom_index)

    for attachment in program.get("attachments") or []:
        _attach_template(rw, dict(attachment))

    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def attachment_templates(programs: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(attachment["template"])
        for program in programs
        for attachment in program.get("attachments") or []
    }


def program_summary(program: dict[str, Any]) -> dict[str, int]:
    slots = sum(
        len(attachment.get("slots") or [])
        for attachment in program.get("attachments") or []
    )
    return {
        "atom_edits": len(program.get("atom_edits") or []),
        "bond_edits": len(program.get("bond_edits") or []),
        "delete_product_atoms": len(program.get("delete_product_maps") or []),
        "attachments": len(program.get("attachments") or []),
        "attachment_slots": slots,
    }
