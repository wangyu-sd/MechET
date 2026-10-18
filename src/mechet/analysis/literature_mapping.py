"""Enrich frozen retrieval keys from authenticated, saved native atom maps.

No model call, GT mapping, new reaction, source grade or evidence promotion.
Conflicting native correspondences remain unknown, not outcome-selected.
The original prediction freeze and evidence analyses are never rewritten.
"""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .literature_reaction_keys import make_reaction
from .literature_sources import file_sha256


def _rows(path):
    with gzip.open(path,'rt') as handle:
        yield from (json.loads(line) for line in handle)


def _authenticated_manifest(root, name, required_table):
    path=root/name
    manifest=json.loads(path.read_bytes())
    if manifest.get('complete') is not True:
        raise ValueError('complete frozen input required: '+name)
    if required_table not in manifest.get('output_files',{}):
        raise ValueError('required table lacks frozen checksum: '+required_table)
    for filename, description in manifest['output_files'].items():
        if Path(filename).name!=filename or file_sha256(root/filename)!=description['sha256']:
            raise ValueError('frozen input table identity differs: '+filename)
    return manifest,file_sha256(path)


def enrich_queries(freeze_root, program_root, output_root):
    """Write a separate full-pool mapping view after all input checks pass."""
    freeze_root,program_root,output_root=map(Path,(freeze_root,program_root,output_root))
    freeze,freeze_sha=_authenticated_manifest(freeze_root,'prediction_freeze_manifest.json','unique_reaction_queries.jsonl.gz')
    programs,program_sha=_authenticated_manifest(program_root,'program_join_manifest.json','program_reaction_links.jsonl.gz')
    if (programs['actor_sha256']!=freeze['actor_sha256'] or
        programs['prediction_freeze_manifest_sha256']!=freeze_sha or
        programs['unmatched_occurrence_count']!=0):
        raise ValueError('native programs do not bind complete frozen actor/population')
    queries={}
    for row in _rows(freeze_root/'unique_reaction_queries.jsonl.gz'):
        expected=make_reaction(row['product_raw'],row['precursors_raw'],auxiliary=row['auxiliary_raw'])
        key=row['reaction_key']
        if key!=expected['reaction_key'] or key in queries or row['center_radius']!=1:
            raise ValueError('duplicate, changed or unsupported frozen reaction query')
        # In the archived single-reaction serializer, auxiliary_components is
        # the full-mixture remainder, whereas auxiliary_raw may be empty. Its
        # frozen bytes/role basis must be preserved, not replaced from raw.
        for field in ('product_components','structural_components'):
            if row[field]!=expected[field]:
                raise ValueError('frozen chemical component identity differs')
        queries[key]=row
    linked=defaultdict(list);seen=set()
    for row in _rows(program_root/'program_reaction_links.jsonl.gz'):
        key=row['reaction_key'];pid=row['program_record_id']
        if (pid in seen or key not in queries or row['actor_sha256']!=freeze['actor_sha256'] or
            row['actor_exposure_manifest_sha256']!=programs['actor_exposure_manifest_sha256'] or
            row.get('program_state_binding',{}).get('valid') is not True):
            raise ValueError('unbound, duplicated or invalid native program')
        seen.add(pid)
        native=make_reaction(row['mapped_product'],[row['mapped_structural_precursors']],radius=1)
        if (native['reaction_key']!=key or
            native['mapping_status']!=row['mapping_status'] or
            native['reaction_family_key']!=row['reaction_family_key']):
            raise ValueError('native mapped chemistry or recomputed family differs')
        linked[key].append((pid,native))
    counts=Counter(query_count=len(queries),program_records=len(seen),newly_resolved=0,
                   conflicting=0,unavailable=0,mapped_after=0)
    enriched=[]
    mapping_fields=('mapping_status','reaction_family_key','changed_bond_count','center_radius',
                    'family_key_scope','family_error','family_error_details')
    for key,q in queries.items():
        sources=linked[key]
        families={r['reaction_family_key'] for _,r in sources if r.get('reaction_family_key')}
        row=dict(q)
        provenance={'program_record_ids':sorted(pid for pid,_ in sources),
            'family_keys':sorted(families),'original_mapping_status':q['mapping_status'],
            'original_reaction_family_key':q['reaction_family_key'],
            'basis':'saved native atom correspondence; no GT, new mapping model or evidence grade',
            'policy':'all authenticated native programs must agree on the radius-1 family'}
        if len(families)==1 and all(r.get('reaction_family_key') for _,r in sources):
            for field in mapping_fields:row.pop(field,None)
            row.update({field:sources[0][1][field] for field in mapping_fields if field in sources[0][1]})
            provenance['status']='unique_native_family'
            counts['newly_resolved']+=q['mapping_status']=='mapping_unknown'
        elif families:
            for field in mapping_fields:row.pop(field,None)
            row.update(mapping_status='mapping_unknown',reaction_family_key=None,
                       changed_bond_count=None,center_radius=1)
            provenance['status']='conflicting_native_families'
            counts['conflicting']+=1
        else:
            provenance['status']='native_mapping_unavailable'
            counts['unavailable']+=1
        counts['mapped_after']+=row['mapping_status']=='mapped_center_available'
        row['native_mapping']=provenance;enriched.append(row)
    # Only after complete validation; never overwrite a freeze or old analysis.
    output_root.mkdir(parents=True,exist_ok=False)
    name='unique_reaction_queries_enriched.jsonl.gz'
    with gzip.open(output_root/name,'wt') as handle:
        for row in enriched:handle.write(json.dumps(row)+'\n')
    manifest={'schema':'lgrv_native_mapping_enrichment_v1','complete':True,
        'actor_sha256':freeze['actor_sha256'],'prediction_freeze_manifest_sha256':freeze_sha,
        'program_join_manifest_sha256':program_sha,'counts':dict(counts),
        'original_prediction_freeze_root':str(freeze_root.resolve()),
        'program_join_root':str(program_root.resolve()),'center_radius':1,
        'policy':'all authenticated native programs must agree on the radius-1 family',
        'source_grades_unchanged':True,'new_model_inference':False,
        'analogue_calibration_complete':False,'retrieval_results_recomputed':False,
        'producer_files':{p.name:file_sha256(p) for p in
            (Path(__file__),Path(__file__).with_name('literature_reaction_keys.py'))},
        'output_files':{name:{'sha256':file_sha256(output_root/name)}}}
    (output_root/'mapping_enrichment_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prediction-freeze-root','program-join-root','output-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    result=enrich_queries(args.prediction_freeze_root,args.program_join_root,args.output_root)
    print(json.dumps({'phase':'native_mapping_enrichment_complete',**result['counts']}),flush=True)
