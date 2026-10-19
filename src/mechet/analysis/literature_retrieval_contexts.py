"""Full-pool stereo-aware contexts from saved model programs, never GT."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import json
from pathlib import Path

from .literature_stereo_family import ALIAS_LIMIT,POLICY,family_context
from .literature_reaction_keys import make_reaction
from .literature_sources import file_sha256


def _calculate_context(pair):
    return family_context(*pair)


def prepare_contexts(queries,programs,*,actor_sha256,workers=1):
    if type(workers) is not int or not 1<=workers<=32:
        raise ValueError('invalid CPU context worker count')
    queries=list(queries);keys={q['reaction_key'] for q in queries}
    if len(keys)!=len(queries):raise ValueError('duplicate frozen query')
    grouped=defaultdict(list);unique={};seen=set()
    for p in programs:
        if (p['program_record_id'] in seen or p['reaction_key'] not in keys or
            p.get('actor_sha256')!=actor_sha256 or p.get('program_state_binding',{}).get('valid') is not True):
            raise ValueError('unbound/duplicate source program for retrieval context')
        seen.add(p['program_record_id'])
        pair=(p['mapped_structural_precursors'],p['mapped_product'])
        unique.setdefault(pair,None);grouped[p['reaction_key']].append((p['program_record_id'],pair))
    jobs=list(unique)
    if workers==1:computed=map(_calculate_context,jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            computed=list(pool.map(_calculate_context,jobs,chunksize=32))
    lookup=dict(zip(jobs,computed))
    result={}
    for q in queries:
        key=q['reaction_key'];sources=sorted(grouped[key])
        contexts=[lookup[pair] for _,pair in sources]
        if any(c['reaction_key']!=key for c in contexts):
            raise ValueError('native mapped chemistry differs from frozen retrieval reaction')
        base={'reaction_key':key,'policy':POLICY,'family_key_v2':None,
            'legacy_family_keys':(),'legacy_alias_space_complete':False,
            'legacy_alias_limit':ALIAS_LIMIT,'program_record_ids':[pid for pid,_ in sources]}
        if not sources:row={**base,'status':'native_mapping_unavailable'}
        else:
            families={c['family_key_v2'] for c in contexts}
            statuses={c['status'] for c in contexts}
            if len(families)>1:
                row={**base,'status':'conflicting_native_families'}
            elif statuses=={'available'}:
                aliases=tuple(sorted({a for c in contexts for a in c['legacy_family_keys']}))
                if len(aliases)>ALIAS_LIMIT:row={**base,'status':'alias_space_unresolved'}
                else:
                    row={**base,'status':'available','family_key_v2':contexts[0]['family_key_v2'],
                        'legacy_family_keys':aliases,'legacy_alias_space_complete':True}
            else:
                status=next(iter(statuses)) if len(statuses)==1 else 'native_family_unresolved'
                row={**base,'status':status}
            pid,pair=sources[0]
            row.update(representative_program_record_id=pid,mapped_precursors=pair[0],mapped_product=pair[1],
                representative_selection='lexicographically first saved program ID; no outcomes or GT')
        result[key]=row
    return result


def check_policy_controls(root,spec_path,*,freeze_sha256,source_catalogue,queries):
    """Re-execute the frozen controls, not just a self-certified pass flag."""
    from .literature_policy_controls import evaluate_controls
    root,spec_path,source_catalogue=map(Path,(root,spec_path,source_catalogue))
    path=root/'policy_control_manifest.json';manifest=json.loads(path.read_bytes())
    files=manifest.get('output_files',{})
    if not {'policy_control_results.json','control_sources.sqlite'}<=files.keys():
        raise ValueError('policy control artifacts missing')
    for name,value in files.items():
        if Path(name).name!=name or file_sha256(root/name)!=value['sha256']:
            raise ValueError('policy control artifact checksum differs')
    if (manifest.get('complete') is not True or manifest.get('all_controls_passed') is not True or
        manifest.get('family_policy')!=POLICY or manifest.get('control_spec_sha256')!=file_sha256(spec_path) or
        manifest.get('prediction_freeze_manifest_sha256')!=freeze_sha256 or
        manifest.get('source_catalogue_sha256')!=file_sha256(source_catalogue) or
        manifest.get('family_implementation_sha256')!=file_sha256(Path(__file__).with_name('literature_stereo_family.py')) or
        manifest.get('producer_sha256')!=file_sha256(Path(__file__).with_name('literature_policy_controls.py'))):
        raise ValueError('policy controls do not bind the frozen data/current implementation')
    spec=json.loads(spec_path.read_bytes());catalogue=json.loads(source_catalogue.read_bytes())
    selected=[r for r in catalogue['files'] if r['path']==spec['ord_relative_path']]
    relative=Path(spec['ord_relative_path'])
    if (relative.is_absolute() or '..' in relative.parts or len(selected)!=1 or
        file_sha256(source_catalogue.parent/relative)!=selected[0]['sha256'] or
        manifest.get('source_file_sha256')!=selected[0]['sha256']):
        raise ValueError('policy controls do not bind original source file')
    recorded=json.loads((root/'policy_control_results.json').read_bytes())
    expected=evaluate_controls(spec,':memory:',family_policy=POLICY)
    fields=('controls','total','passed','failed','all_passed','family_policy')
    if not expected['all_passed'] or any(recorded.get(k)!=expected[k] for k in fields):
        raise ValueError('policy control report differs from actual re-execution')
    source_keys={make_reaction(s['mapped_product'],[s['mapped_precursors']])['reaction_key'] for s in spec['sources']}
    if (source_keys&{q['reaction_key'] for q in queries} or
        recorded.get('source_independence_passed') is not True or
        recorded.get('source_key_overlap_with_frozen_mechet_pool')!=[]):
        raise ValueError('policy controls are not independent of the frozen generated query pool')
    return {'manifest_sha256':file_sha256(path),'spec_sha256':file_sha256(spec_path),
        'family_policy':POLICY,'control_count':expected['total'],'chemical_accuracy_established':False}
