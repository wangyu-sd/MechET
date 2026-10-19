"""Conservative complete-reaction identity; local edit keys are retrieval only.

No desalting, tautomer choice, neutralization or stereo relaxation. A component
multiset, not a set, is used. Atom maps are addresses, never chemical identity.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json

from rdkit import Chem


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=200000)
def canonical_components(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('empty or non-string SMILES')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid SMILES: ' + smiles[:160])
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return tuple(sorted(Chem.MolToSmiles(part, canonical=True, isomericSmiles=True)
                        for part in Chem.GetMolFrags(mol, asMols=True)))


def _components(values):
    if isinstance(values, str):
        values = [values]
    return sorted(component for value in values for component in canonical_components(value))


def _mapped_graph(text):
    # Retain explicit hydrogens for correspondence; normalized identity above
    # uses RDKit's usual explicit/implicit-H equivalence, not H deletion edits.
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.MolFromSmiles(text, parser)
    if mol is None:
        return None
    atoms = {}
    for atom in mol.GetAtoms():
        key = atom.GetAtomMapNum()
        if not key or key in atoms:
            return None
        atoms[key] = atom.GetIdx()
    bonds = {tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))):
             (float(bond.GetBondTypeAsDouble()), str(bond.GetStereo())) for bond in mol.GetBonds()}
    return mol, atoms, bonds


def _environment(graph, key, radius):
    if key not in graph[1]:
        return None
    mol = Chem.Mol(graph[0])
    index = graph[1][key]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    selected = {index}
    for _ in range(radius):
        selected |= {neighbor.GetIdx() for i in list(selected) for neighbor in mol.GetAtomWithIdx(i).GetNeighbors()}
    # Root annotation avoids confusing inequivalent atoms in the same fragment.
    mol.GetAtomWithIdx(index).SetAtomMapNum(1)
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(selected), canonical=True, isomericSmiles=True)


def _mapped_center(precursors, product, *, radius=1):
    before, after = _mapped_graph(precursors), _mapped_graph(product)
    unknown = {'mapping_status':'mapping_unknown', 'reaction_family_key':None,
               'changed_bond_count':None, 'center_radius':radius}
    if before is None or after is None or not set(after[1]) <= set(before[1]):
        return unknown
    common = before[1].keys() & after[1].keys()
    for key in common:
        a = before[0].GetAtomWithIdx(before[1][key])
        b = after[0].GetAtomWithIdx(after[1][key])
        if (a.GetAtomicNum(), a.GetIsotope()) != (b.GetAtomicNum(), b.GetIsotope()):
            return unknown
    edits = []
    for pair in sorted(before[2].keys() | after[2].keys()):
        old, new = before[2].get(pair), after[2].get(pair)
        if old != new:
            ends = sorted([(_environment(before, k, radius), _environment(after, k, radius)) for k in pair], key=str)
            edits.append({'ends':ends, 'before':old, 'after':new})
    charges = []
    for key in common:
        old = before[0].GetAtomWithIdx(before[1][key]).GetFormalCharge()
        new = after[0].GetAtomWithIdx(after[1][key]).GetFormalCharge()
        if old != new:
            charges.append([_environment(before,key,radius), _environment(after,key,radius),old,new])
    return {'mapping_status':'mapped_center_available', 'reaction_family_key':
            stable_hash({'radius':radius, 'bonds':sorted(edits,key=str), 'charges':sorted(charges,key=str)}),
            'changed_bond_count':len(edits), 'center_radius':radius,
            'family_key_scope':'local_edit_retrieval_candidate_not_chemical_equivalence'}


def mapped_center(precursors, product, *, radius=1):
    try:
        return _mapped_center(precursors,product,radius=radius)
    except RuntimeError as error:
        # Full exact identity has already succeeded. A boundary stereobond can
        # be unserializable in RDKit's *partial* neighbourhood. Do not remove
        # stereo, invent a family or discard the intact complete reaction.
        return {'mapping_status':'mapping_unknown','reaction_family_key':None,
                'changed_bond_count':None,'center_radius':radius,
                'family_error':'local_stereochemical_fragment_serialization_failed',
                'family_error_details':str(error)[:500]}


def make_reaction(product, precursors, *, auxiliary=(), mapped_product=None, mapped_precursors=None, radius=1):
    if isinstance(precursors, str):
        precursors = [precursors]
    if not precursors:
        raise ValueError('complete reaction requires structural precursor SMILES')
    canonical_product = _components([product])
    canonical_precursors = _components(precursors)
    result = {'schema':'lgrv_reaction_v1', 'product_raw':product, 'precursors_raw':list(precursors),
              'product_components':canonical_product, 'structural_components':canonical_precursors,
              'auxiliary_raw':list(auxiliary), 'auxiliary_components':_components(auxiliary),
              'reaction_smiles':'.'.join(canonical_precursors)+'>>'+'.'.join(canonical_product),
              'reaction_key':stable_hash([canonical_precursors,canonical_product]),
              'normalization':'rdkit_map_free_isomeric_component_multiset_no_standardizer_v1'}
    result.update(mapped_center(mapped_precursors or '.'.join(precursors), mapped_product or product, radius=radius))
    return result
