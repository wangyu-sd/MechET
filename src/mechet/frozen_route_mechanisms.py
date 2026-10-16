"""Exact frozen-record correspondence, never electronic-program inference.

Callers must authenticate the complete source artifact and audit manifest
before supplying records. Matching removes maps/component order only, retaining
charge, isotope, stereo and precursor multiplicity. Source programs are
executor-verified corpus witnesses, not independent experimental certificates.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import re
from rdkit import Chem
from .frozen_route_assets import normalize_stock_smiles, classify_composition
from .in_place_grounded_flow import atom_token_spans
from .natural_language_electron_flow import NaturalLanguageInventory, _resolve_source, _resolve_destination
from .forward_expert import ElectronMove
from .proof_to_trace import execution_primitive_signature


def _sft_call(row):
    assistants=[m for m in row['messages'] if m['role']=='assistant']
    tools=[m for m in row['messages'] if m['role']=='tool']
    if len(assistants)!=1 or len(tools)!=1 or len(assistants[0].get('tool_calls',[]))!=1:
        raise ValueError('actual SFT requires one paired tool exchange')
    call=assistants[0]['tool_calls'][0]['function']
    result=json.loads(tools[0]['content'])
    if (tools[0]['name']!=call['name'] or result.get('ok') is not True
            or row.get('metadata',{}).get('executor_replayed') is not True):
        raise ValueError('unverified actual SFT exchange')
    return call['name'],call['arguments'],result


def sft_event_features(row: dict) -> dict:
    """Resolve the SAVED SFT inventory/labels, never the old compiler moves.

    Temporary aliases are bound to their explicit atom-token positions, not
    regenerated after canonicalization. Private addresses reconstructed here
    are per-event only; they must not be stitched across trajectory steps.
    """
    name,args,result=_sft_call(row)
    if name!='apply_electron_flow' or args.get('direction')!='retrosynthetic':
        raise ValueError('not an actual retrosynthetic SFT electron event')
    users=[m['content'] for m in row['messages'] if m['role']=='user']
    if len(users)!=1:raise ValueError('one actual SFT user observation required')
    lines=[line.removeprefix('ANNOTATED CURRENT STATE: ') for line in users[0].splitlines()
           if line.startswith('ANNOTATED CURRENT STATE: ')]
    if len(lines)!=1:raise ValueError('missing unique annotated state')
    annotated=lines[0]
    clean=[];aliases=[];positions=[];cursor=0;length=0
    for match in re.finditer(r'<(A\d+)>',annotated):
        piece=annotated[cursor:match.start()];clean.append(piece);length+=len(piece)
        aliases.append(match[1]);positions.append(length);cursor=match.end()
    clean.append(annotated[cursor:]);visible=''.join(clean)
    if positions!=[start for start,_ in atom_token_spans(visible)] or len(set(aliases))!=len(aliases):
        raise ValueError('incomplete or duplicated annotated atoms')
    params=Chem.SmilesParserParams();params.removeHs=False
    mol=Chem.MolFromSmiles(visible,params)
    if mol is None or mol.GetNumAtoms()!=len(aliases):raise ValueError('invalid annotated state')
    mapping={alias:int(alias[1:]) for alias in aliases}
    if any(value<=0 for value in mapping.values()):raise ValueError('invalid annotated atom alias')
    for atom,alias in zip(mol.GetAtoms(),aliases,strict=True):atom.SetAtomMapNum(mapping[alias])
    state=Chem.MolToSmiles(mol,canonical=False,isomericSmiles=True)
    pairs={tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum()))) for b in mol.GetBonds()}
    inventory=NaturalLanguageInventory(visible,mapping,{v:k for k,v in mapping.items()},
        {},{pair:'' for pair in pairs},annotated)
    arrows=args.get('electron_flow') or []
    changes=bool(args.get('bond_order_changes') or args.get('charge_changes'))
    if arrows and changes:raise ValueError('mixed ordinary and BE_DELTA SFT labels')
    if not arrows and not changes:raise ValueError('empty actual SFT electron event')
    moves=[ElectronMove(_resolve_source(arrow['source'],inventory),
                       _resolve_destination(arrow['destination'],inventory)).to_dict() for arrow in arrows]
    return {'ordinary_source_sink_event':not changes,
            'primitive_sequence':[execution_primitive_signature(move,state) for move in moves],
            'recorded_tool_call':{'name':name,'arguments':deepcopy(args)},
            'annotated_state':annotated,'state_after':result.get('current_state'),
            'source_decision_id':row['id']}


def summarize_sft_reaction(rows: list[dict], split: str, *, structural_precursor: str | None) -> dict:
    if not rows:raise ValueError('empty actual SFT reaction')
    source=rows[0]['source_id'];steps=[];imported=0;sequence=[];ordinary=True;primitives=set()
    for index,row in enumerate(rows):
        if row['source_id']!=source or row.get('metadata',{}).get('decision_index')!=index:
            raise ValueError('actual SFT source/index discontinuity')
        name,args,_=_sft_call(row)
        if name=='apply_electron_flow':
            event=sft_event_features(row);steps.append(event)
            ordinary &= event['ordinary_source_sink_event']
            sequence.append(event['primitive_sequence']);primitives.update(event['primitive_sequence'])
        elif name=='import_fragments':
            imported+=sum(item['count'] for item in args['fragments'] if item['purpose']=='electron_participant')
        elif name=='finish_trace':
            if index!=len(rows)-1:raise ValueError('SFT finish is not final')
        else:raise ValueError('unexpected actual SFT tool')
    if _sft_call(rows[-1])[0]!='finish_trace':raise ValueError('incomplete actual SFT reaction: missing finish')
    supported=ordinary and bool(steps)
    signature=hashlib.sha256(json.dumps(sequence,sort_keys=True,separators=(',',':')).encode()).hexdigest() if supported else None
    return {'source_id':source,'source_split':split,'target_smiles':rows[0]['target_smiles'],
        'full_precursor_state':rows[0]['expected_precursor'],'structural_precursor':structural_precursor,
        'executor_replayed':True,'ordinary_source_sink_program':supported,
        'program_signature':signature,'primitive_signatures':sorted(primitives) if supported else None,
        'observed_ordinary_primitive_signatures':sorted(primitives),
        'signature_version':'source_to_sink_execution_moves_v1',
        'reference_event_length':len(steps),
        'coupled_electronic_moves':max((len(s) for s in sequence),default=0) if supported else None,
        'imported_participants':imported,'electronic_program':{
            'format':'recorded_natural_language_SFT_calls_with_per_step_aliases','steps':steps,
            'addresses_are_per_step_not_persistent_maps':True}}


def reaction_chemistry_key(product: str, precursors: list[str]) -> tuple:
    # Outer precursor entries can themselves be salt mixtures. Preserve their
    # component multiplicities for correspondence; this is NOT a stock lookup.
    components=[part for precursor in precursors for part in precursor.split('.')]
    return normalize_stock_smiles(product),tuple(sorted(normalize_stock_smiles(x) for x in components))


def resolve_reference_mechanism(reference: dict, records: list[dict]) -> dict:
    result=deepcopy(reference)
    result.update(program_signature=None,primitive_signatures=None,
                  electronic_programs=[],matched_source_ids=[],matched_source_splits=[],
                  signature_version=None,action_vocabulary_covered=None)
    if reference.get('mapping_valid') is not True:
        result.update(annotation_status='no_verified_match',
                      mechanism_match_status='REFERENCE_MAPPING_UNAVAILABLE')
        return result
    key=reaction_chemistry_key(reference['source_product_smiles'],reference['source_precursor_smiles'])
    matches=[row for row in records if reaction_chemistry_key(row['target_smiles'],
             [row['structural_precursor']])==key]
    if not matches:
        result.update(annotation_status='no_verified_match',mechanism_match_status='NO_EXACT_CORPUS_CHEMISTRY')
        return result
    matches.sort(key=lambda row:(row['source_split'],row['source_id']))
    result.update(matched_source_ids=sorted({row['source_id'] for row in matches}),
                  matched_source_splits=sorted({row['source_split'] for row in matches}))
    supported=[row for row in matches if row.get('executor_replayed') is True
               and row.get('ordinary_source_sink_program') is True
               and row.get('program_signature') and row.get('primitive_signatures')
               and row.get('electronic_program',{}).get('steps')
               and row.get('signature_version')=='source_to_sink_execution_moves_v1']
    # Do not choose a convenient supported witness while hiding conflicting or
    # unsupported source records for precisely the same reference chemistry.
    if len(supported)!=len(matches):
        result.update(annotation_status='unsupported_mechanism',
                      mechanism_match_status='UNVERIFIED_OR_NON_SOURCE_SINK_CORPUS_RECORD')
        return result
    identities={(row['program_signature'],tuple(sorted(row['primitive_signatures']))) for row in supported}
    if len(identities)!=1:
        result.update(annotation_status='unsupported_mechanism',mechanism_match_status='AMBIGUOUS_RECORDED_PROGRAMS')
        return result
    first=supported[0]
    result.update(annotation_status='verified_mechanism',mechanism_match_status='EXACT_FROZEN_CORPUS_CORRESPONDENCE',
                  program_signature=first['program_signature'],primitive_signatures=list(first['primitive_signatures']),
                  signature_version=first['signature_version'],action_vocabulary_covered=True,
                  electronic_programs=[deepcopy(row['electronic_program']) for row in supported],
                  mechanism_evidence='executor_verified_frozen_corpus_witness_not_experimental_certificate')
    for key in ['reference_event_length','coupled_electronic_moves','imported_participants']:
        values={row.get(key) for row in supported}
        result[key]=next(iter(values)) if len(values)==1 else None
    return result


def annotate_reference_targets(targets: list[dict], records_by_chemistry: dict,
                               training_programs: set[str], primitive_counts: dict[str,int]) -> list[dict]:
    """Annotate every fixed route; primary route choice never uses model output.

    Event length is the number of recorded FLOW decisions; coupling is the
    maximum arrows in one event; imports count electron-participant molecules
    with multiplicity, excluding explicitly endpoint-only context molecules.
    """
    result=deepcopy(targets)
    for target in result:
        annotated=[]
        for edge in target.get('reference_reactions',[]):
            key=reaction_chemistry_key(edge['source_product_smiles'],edge['source_precursor_smiles'])
            annotated.append(resolve_reference_mechanism(edge,records_by_chemistry.get(key,[])))
        target['reference_reactions']=annotated
        for route in target.get('reference_routes',[]):
            relevant=[edge for edge in annotated if edge['reference_route_id']==route['reference_route_id']]
            label=classify_composition(relevant,training_programs,primitive_counts)
            route.update(composition_stratum=label['stratum'],composition_reason=label['reason'])
            for key in ['reference_event_length','coupled_electronic_moves','imported_participants']:
                values=[edge.get(key) for edge in relevant]
                available=bool(values) and all(value is not None for value in values)
                route[key]=(max(values) if key=='coupled_electronic_moves' else sum(values)) if available else None
        primary=(target.get('reference_routes') or [{}])[0]
        target['composition_stratum']=primary.get('composition_stratum','Unknown')
    return result


def merge_reference_annotations(targets: list[dict], annotations: list[dict]) -> list[dict]:
    """Attach offline labels without changing frozen targets, stock or routes."""
    indexed={row['id']:row for row in annotations}
    if len(indexed)!=len(annotations) or len(targets)!=len(annotations) or set(indexed)!={row['id'] for row in targets}:
        raise ValueError('annotation full denominator/ID mismatch')
    edge_fields={'annotation_status','mechanism_match_status','program_signature',
        'primitive_signatures','action_vocabulary_covered'}
    route_fields={'composition_stratum','composition_reason','reference_event_length',
                  'coupled_electronic_moves','imported_participants'}
    result=deepcopy(targets)
    for target in result:
        other=indexed[target['id']]
        for key,value in target.items():
            if key not in {'reference_routes','reference_reactions','composition_stratum','already_in_stock'} and other.get(key)!=value:
                raise ValueError(f'annotation changes frozen target: {key}')
        edges={edge['reference_edge_id']:edge for edge in other.get('reference_reactions',[])}
        original=target.get('reference_reactions',[])
        if len(edges)!=len(other.get('reference_reactions',[])) or set(edges)!={e['reference_edge_id'] for e in original}:
            raise ValueError('annotation changes reference edge population')
        merged=[]
        for edge in original:
            new=edges[edge['reference_edge_id']]
            for key,value in edge.items():
                if key not in edge_fields and new.get(key)!=value:
                    raise ValueError(f'annotation changes frozen reference chemistry: {key}')
            merged.append(deepcopy(new))
        target['reference_reactions']=merged
        routes={route['reference_route_id']:route for route in other.get('reference_routes',[])}
        original_routes=target.get('reference_routes',[])
        if len(routes)!=len(other.get('reference_routes',[])) or set(routes)!={r['reference_route_id'] for r in original_routes}:
            raise ValueError('annotation changes reference route population')
        for route in original_routes:
            new=routes[route['reference_route_id']]
            for key,value in route.items():
                if key not in route_fields|{'all_reference_leaves_in_stock'} and new.get(key)!=value:
                    raise ValueError(f'annotation changes frozen reference route: {key}')
            for key in route_fields:
                if key in new:route[key]=deepcopy(new[key])
        target['composition_stratum']=other.get('composition_stratum','Unknown')
    return result
