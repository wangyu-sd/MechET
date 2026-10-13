"""Versioned route assets and reference-only chemical complexity.

No plain mapped reaction is relabeled as a recorded electronic mechanism.
Canonicalization preserves charge, stereochemistry and isotopes; stock salt
entries remain whole entries rather than being flattened into purchasable ions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from rdkit import Chem
from .proof_program import _canonical_unmapped


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(16<<20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify_source(path: Path, sha256: str) -> None:
    if file_sha256(path) != sha256:
        raise ValueError(f'SHA256 mismatch: {path}')


def normalize_stock_smiles(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('empty stock structure')
    return '.'.join(sorted(_canonical_unmapped(x) for x in smiles.split('.')))


def classify_composition(reactions: list[dict], training_programs: set[str],
                         primitive_counts: dict[str, int]) -> dict:
    if not reactions or any(r.get('annotation_status') != 'verified_mechanism'
                            or not r.get('program_signature') or not r.get('primitive_signatures')
                            for r in reactions):
        return {'stratum':'Unknown','reason':'missing_verified_electronic_program'}
    if all(r['program_signature'] in training_programs for r in reactions):
        return {'stratum':'C0','reason':'all_complete_programs_in_training_exposure'}
    support = [primitive_counts.get(p,0) for r in reactions for p in r['primitive_signatures']]
    if any(n == 0 for n in support):
        return {'stratum':'C2','reason':'verified_primitive_zero_training_support'}
    if all(n >= 5 for n in support):
        return {'stratum':'C1','reason':'unseen_composition_all_primitives_support_at_least5'}
    return {'stratum':'Unknown','reason':'primitive_support_below_C1_threshold_but_nonzero'}


def mapped_complexity(reaction: str | None) -> dict:
    result = {'changed_bonds_shared_mapped_atoms':None,'mapped_reacting_atom_count_shared':None,
              'ring_count_change':None,'mapping_valid':False,'mapping_complete':False,
              'stereo_atom_count':None,'explicit_hydrogen_atom_count':None}
    if not reaction:
        return result
    try:
        left, _, right = reaction.split('>')
        params = Chem.SmilesParserParams()
        params.removeHs = False
        reactants = Chem.MolFromSmiles(left,params)
        product = Chem.MolFromSmiles(right,params)
        if reactants is None or product is None:
            return result
        maps = [[a.GetAtomMapNum() for a in mol.GetAtoms()] for mol in (reactants, product)]
        positive = [[x for x in side if x>0] for side in maps]
        if not all(positive) or any(len(side)!=len(set(side)) for side in positive):
            return result
        shared = set(positive[0]) & set(positive[1])
        def bonds(mol):
            found = {}
            for bond in mol.GetBonds():
                a,b = bond.GetBeginAtom().GetAtomMapNum(),bond.GetEndAtom().GetAtomMapNum()
                if a in shared and b in shared:
                    found[tuple(sorted((a,b)))] = bond.GetBondTypeAsDouble()
            return found
        before,after = bonds(reactants),bonds(product)
        changed = [key for key in set(before)|set(after) if before.get(key)!=after.get(key)]
        result.update(changed_bonds_shared_mapped_atoms=len(changed),
            mapped_reacting_atom_count_shared=len({atom for pair in changed for atom in pair}),
            ring_count_change=product.GetRingInfo().NumRings()-reactants.GetRingInfo().NumRings(),
            mapping_valid=True,mapping_complete=all(all(x>0 for x in side) for side in maps),
            stereo_atom_count=sum(a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in product.GetAtoms()),
            explicit_hydrogen_atom_count=sum(a.GetAtomicNum()==1 for a in product.GetAtoms()))
    except (ValueError, RuntimeError):
        pass
    return result


def extract_reference_route(route: dict, target_id: str, route_index: int) -> tuple[dict,list[dict]]:
    reference_id = route.get('content_hash') or f'{target_id}:reference:{route_index}'
    edges,leaves = [],[]
    def walk(node, path):
        step = node.get('synthesis_step')
        if not step:
            leaves.append(normalize_stock_smiles(node['smiles']))
            return 0
        children = step['reactants']
        mapped = step.get('mapped_smiles')
        complexity = mapped_complexity(mapped)
        edges.append({'target_id':target_id,'reference_route_id':reference_id,
            'reference_edge_id':f'{reference_id}:{path}','source_product_smiles':node['smiles'],
            'source_precursor_smiles':[child['smiles'] for child in children],
            'source_mapping_smiles':mapped,'source_mapping_id':step.get('metadata',{}).get('ID'),
            'annotation_status':'mapping_only' if complexity['mapping_valid'] else 'no_verified_match',
            'mechanism_match_status':'PENDING_EXACT_FROZEN_CORPUS_MATCH',
            'program_signature':None,'primitive_signatures':None,'action_vocabulary_covered':None,
            'structural_precursor_multiplicity':len(children),
            'source_is_convergent':step.get('is_convergent'),**complexity})
        return 1 + max((walk(child,f'{path}.{i}') for i,child in enumerate(children)),default=0)
    depth = walk(route['target'],'0')
    if route.get('length') is not None and depth != route['length']:
        raise ValueError(f'reference depth disagrees with source: {target_id}')
    info = {'target_id':target_id,'reference_route_id':reference_id,'reference_rank':route_index,
            'source_patent_id':route.get('metadata',{}).get('patent_id'),
            'source_content_hash':route.get('content_hash'),'depth':depth,
            'reaction_edge_count':len(edges),'leaf_smiles':sorted(leaves),
            'convergent_branch_count':sum(e['source_is_convergent'] is True for e in edges),
            'maximum_precursor_multiplicity':max((e['structural_precursor_multiplicity'] for e in edges),default=0),
            'reference_event_length':None,'coupled_electronic_moves':None,'imported_participants':None,
            'composition_stratum':'Unknown','composition_reason':'verified_program_annotation_pending',
            'reaction_ids':[e['reference_edge_id'] for e in edges]}
    return info,edges
