"""Explicit material availability projection; never mutate executor chemistry.

The optional protonation convention is a workup/starting-material assumption,
not an electron-flow event, stability oracle, or general ion neutralization.
Only terminal O-/S- conjugate bases with an exact protonated stock entry qualify.
"""
from functools import lru_cache

from rdkit import Chem

from .route_training_contract import chemical_identity

EXACT_MATERIAL = 'exact_material_stock_v1'
PROTIC_MATERIAL = 'terminal_heteroanion_protonation_stock_v1'


def material_contract(cfg):
    value = cfg.get('stock_material_contract', EXACT_MATERIAL)
    if value not in (EXACT_MATERIAL, PROTIC_MATERIAL):
        raise ValueError('UNRECOGNIZED_STOCK_MATERIAL_CONTRACT')
    return value


@lru_cache(maxsize=16384)
def _protonated_material(identity):
    mol = Chem.MolFromSmiles(identity)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge()]
    if (len(charged) != 1 or charged[0].GetFormalCharge() != -1 or
            any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
        return None
    atom = charged[0]
    if (atom.GetAtomicNum() not in (8,16) or atom.GetIsAromatic() or
            atom.GetDegree() != 1 or atom.GetTotalNumHs() != 0):
        return None
    bond = atom.GetBonds()[0]
    if (bond.GetBondType() != Chem.BondType.SINGLE or
            atom.GetNeighbors()[0].GetAtomicNum() != 6):
        return None
    atom.SetFormalCharge(0)
    atom.SetNumExplicitHs(1)
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return chemical_identity(Chem.MolToSmiles(mol, isomericSmiles=True)), 'terminal_'+atom.GetSymbol()+'_minus_add_H'


def project_material_endpoints(public_structural, stock, contract=EXACT_MATERIAL):
    """Preserve whole-stock and occurrence semantics; aliases require stock.

    The raw mapped state is kept elsewhere. Returned public precursor strings
    retain graph H atoms unless this explicitly declared material alias applies.
    Counterions are never silently discarded here.
    """
    material_contract(dict(stock_material_contract=contract))
    structural = chemical_identity(public_structural)
    fragments = ([public_structural] if structural in stock else public_structural.split('.'))
    pairs = []
    for public in fragments:
        mechanism = chemical_identity(public)
        material, rule = mechanism, 'exact'
        if contract == PROTIC_MATERIAL and mechanism not in stock:
            candidate = _protonated_material(mechanism)
            if candidate is not None and candidate[0] in stock:
                material, rule = candidate
                public = material
        pairs.append((material, mechanism, public, rule))
    pairs.sort()
    return dict(precursors=[p[0] for p in pairs], public_precursors=[p[2] for p in pairs],
                materials=[dict(mechanistic=m,material=v,rule=r) for v,m,_,r in pairs])


def validate_material_edge(edge, stock, contract):
    """Independently derive the projection from full mapped endpoint evidence."""
    material_contract(dict(stock_material_contract=contract))
    if contract == EXACT_MATERIAL:
        return
    from .endpoints import split_route_precursor_endpoints, _mapped_mol
    try:
        initial, final = edge['initial_state'], edge['final_state']
        before, after = _mapped_mol(initial), _mapped_mol(final)
        initial_maps = [a.GetAtomMapNum() for a in before.GetAtoms()]
        final_maps = [a.GetAtomMapNum() for a in after.GetAtoms()]
        atom_identities = {a.GetAtomMapNum():(a.GetAtomicNum(),a.GetIsotope()) for a in after.GetAtoms()}
        if (any(m <= 0 for m in initial_maps+final_maps) or
                len(final_maps) != len(set(final_maps)) or
                len(initial_maps) != len(set(initial_maps)) or
                not set(initial_maps) <= set(final_maps) or
                any(atom_identities.get(a.GetAtomMapNum()) != (a.GetAtomicNum(),a.GetIsotope())
                    for a in before.GetAtoms()) or
                chemical_identity(initial) != chemical_identity(edge['product'])):
            raise ValueError('MATERIAL_STATE_MAP_OR_PRODUCT_MISMATCH')
        structural = split_route_precursor_endpoints(final, initial).structural
        expected = project_material_endpoints(structural, stock, contract)
        if (edge['precursors'] != expected['precursors'] or
                edge['endpoint_materials'] != expected['materials']):
            raise ValueError('MATERIAL_ENDPOINT_EVIDENCE_MISMATCH')
        if ('public_precursors' in edge and
                [chemical_identity(v) for v in edge['public_precursors']] != expected['precursors']):
            raise ValueError('MATERIAL_PUBLIC_ENDPOINT_MISMATCH')
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('INVALID_MATERIAL_EDGE:'+str(exc)) from None
