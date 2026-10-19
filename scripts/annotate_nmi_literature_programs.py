#!/usr/bin/env python3
"""Join every frozen saved reaction to actual terminal programs and actor exposure.

CPU-only analysis. It neither generates model programs nor changes the frozen
retrieval pool. Primary evidence is a later join, not supplied by signatures.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from mechet.analysis.literature_predictions import resolve_tree_artifact
from mechet.analysis.literature_programs import authenticate_exposure, bind_program_states, composition_metrics, join_report_programs
from mechet.analysis.literature_reaction_keys import stable_hash
from mechet.analysis.literature_sources import file_sha256
from mechet.analysis.tree_costs import observed_work_metrics


def rows(path):
    with gzip.open(path,'rt') as handle:
        yield from (json.loads(line) for line in handle)


def save(path,value):
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def annotate(freeze_root, exposure_path, output_root, *, workers=4, compatible_program_cache=None,
             cache_manifest_sha256=None,cache_links_sha256=None):
    freeze_root,exposure_path,output_root=map(Path,(freeze_root,exposure_path,output_root))
    if workers<1:
        raise ValueError('workers must be positive')
    freeze_path=freeze_root/'prediction_freeze_manifest.json'
    freeze=json.loads(freeze_path.read_bytes())
    if freeze.get('complete') is not True:
        raise ValueError('complete frozen prediction pool required')
    for name,description in freeze['output_files'].items():
        if file_sha256(freeze_root/name)!=description['sha256']:
            raise ValueError('frozen table checksum differs: '+name)
    exposure=json.loads(exposure_path.read_bytes())
    authenticate_exposure(exposure,freeze['actor_sha256'])
    @lru_cache(maxsize=4)
    def document(path):
        payload=Path(path).read_bytes()
        return json.loads(payload),hashlib.sha256(payload).hexdigest()
    def report_for(source):
        data,sha=document(source['report_path'])
        if sha!=source['report_sha256']:
            raise ValueError('frozen report checksum differs')
        report=data['rows'][source['report_row_index']] if 'report_row_index' in source else data
        if report['id']!=source['target_id']:
            raise ValueError('source report target differs')
        if source.get('manifest_path'):
            contract,sha=document(source['manifest_path'])
            if sha!=source['manifest_sha256'] or contract.get('adapter_sha')!=freeze['actor_sha256']:
                raise ValueError('frozen actor contract checksum or identity differs')
        return report
    # Check every frozen report/contract before creating the output population.
    for source in freeze['source_reports']:
        report_for(source)
    occurrences=defaultdict(list);populations=defaultdict(list);routes=defaultdict(list)
    for row in rows(freeze_root/'generated_reaction_steps.jsonl.gz'):
        occurrences[row['run_id'],row['target_id']].append(row)
    for row in rows(freeze_root/'target_population.jsonl.gz'):
        populations[row['run_id']].append(row['target_id'])
    for row in rows(freeze_root/'frozen_routes.jsonl.gz'):
        routes[row['run_id']].append(row)
    cached=defaultdict(list);adoption=None
    if compatible_program_cache:
        cache_root=Path(compatible_program_cache)
        old_path=cache_root/'program_join_manifest.json'
        old=json.loads(old_path.read_bytes())
        if cache_manifest_sha256 and file_sha256(old_path)!=cache_manifest_sha256:
            raise ValueError('cache manifest checksum differs')
        old_links=cache_root/'program_reaction_links.jsonl.gz'
        if old.get('complete'):
            for name,description in old['output_files'].items():
                if file_sha256(cache_root/name)!=description['sha256']:
                    raise ValueError('cache artifact checksum differs: '+name)
        elif not cache_manifest_sha256 or not cache_links_sha256:
            raise ValueError('partial cache requires external checksum pins for manifest and program links')
        if cache_links_sha256 and file_sha256(old_links)!=cache_links_sha256:
            raise ValueError('cache artifact checksum differs: program links')
        # Adapter changes are explicitly opted into; primitive/compiler and
        # endpoint semantics may not silently change under that opt-in.
        for name in ('tree_chemistry.py','natural_language_electron_flow.py','proof_to_trace.py','endpoints.py'):
            relative=('src/mechet/analysis/' if name=='tree_chemistry.py' else 'src/mechet/')+name
            if old['producer_files'].get(relative)!=file_sha256(ROOT/relative):
                raise ValueError('cached program compiler/endpoint producer is not compatible')
        if (old.get('schema')!='lgrv_saved_program_join_v1' or
            old.get('actor_sha256')!=freeze['actor_sha256'] or
            old.get('prediction_freeze_manifest_sha256')!=file_sha256(freeze_path) or
            old.get('actor_exposure_manifest_sha256')!=file_sha256(exposure_path)):
            raise ValueError('compatible program cache belongs to different frozen inputs')
        # An incomplete run's last atomic progress boundary is a prefix of
        # whole reports. Do not adopt potentially half-written later reports.
        boundary=old['counts'] if old.get('complete') else json.loads((cache_root/'progress.json').read_bytes())['counts']
        completed=freeze['source_reports'][:boundary['reports']]
        allowed={(r['run_id'],r['target_id'],r['report_sha256']) for r in completed}
        import itertools
        for row in itertools.islice(rows(old_links),boundary['program_records']):
            source=row['source'];key=(row['run_id'],row['target_id'],source['report_sha256'])
            if key not in allowed:
                raise ValueError('cached program is outside completed report prefix')
            if row['actor_exposure_manifest_sha256']!=old['actor_exposure_manifest_sha256']:
                raise ValueError('cached program actor exposure differs')
            cached[key].append(row)
        if sum(map(len,cached.values()))!=boundary['program_records']:
            raise ValueError('cached complete-report prefix count differs from atomic progress boundary')
        adoption={'cache_root':str(cache_root.resolve()),'manifest_sha256':file_sha256(old_path),
                  'program_links_sha256':file_sha256(old_links),'completed_report_prefix':boundary['reports'],
                  'source_producer_files':old['producer_files'],
                  'basis':'explicit compatible serializer extension; old nested-tree annotations and v1 primitive semantics unchanged; source bytes rechecked'}
    output_root.mkdir(parents=True,exist_ok=False)
    producer_paths=[Path(__file__),ROOT/'src/mechet/analysis/literature_programs.py',
        ROOT/'src/mechet/analysis/tree_chemistry.py',ROOT/'src/mechet/natural_language_electron_flow.py',
        ROOT/'src/mechet/proof_to_trace.py',ROOT/'src/mechet/endpoints.py']
    manifest={'schema':'lgrv_saved_program_join_v1','complete':False,
        'prediction_freeze_manifest_sha256':file_sha256(freeze_path),
        'actor_exposure_manifest_path':str(exposure_path.resolve()),
        'actor_exposure_manifest_sha256':file_sha256(exposure_path),
        'actor_sha256':freeze['actor_sha256'],
        'producer_files':{str(p.relative_to(ROOT)):file_sha256(p) for p in producer_paths},
        'exposure_scope':'actual actor training corpus, not historical strict H2',
        'new_model_inference':False,'source_reactions_frozen_before_join':True,
        'primary_evidence_supplied':False,'reference_C0_C1_C2':'Unknown for all160',
        'compatible_program_cache_adoption':adoption,'counts':{},'output_files':{}}
    save(output_root/'program_join_manifest.json',manifest)
    def prepare(source):
        report=report_for(source)
        selected=occurrences[source['run_id'],source['target_id']]
        work={'target_id':source['target_id'],'report_path':source['report_path'],
              'report_index':source.get('report_row_index'),'report_sha256':source['report_sha256'],
              'report':{'work':report.get('work',{}),'wall_seconds':report.get('wall_seconds')}}
        if not selected:
            return source,[],work,None
        tree_path=resolve_tree_artifact(report,source)
        payload=tree_path.read_bytes();tree_sha=hashlib.sha256(payload).hexdigest()
        if source.get('target_tree_sha256') and tree_sha!=source['target_tree_sha256']:
            raise ValueError('frozen native tree checksum differs')
        reused=cached.get((source['run_id'],source['target_id'],source['report_sha256']))
        if reused:
            if any(row['source']['tree_sha256']!=tree_sha for row in reused):
                raise ValueError('cached native tree checksum differs')
            native=json.loads(payload)
            for row in reused:
                components=row['program_source_pointer'].strip('/').split('/')
                value=native
                for part in components[:-1]:
                    part=part.replace('~1','/').replace('~0','~')
                    value=value[int(part)] if isinstance(value,list) else value[part]
                if value['id']!=row['native_context_id'] or stable_hash(value['actions'])!=row['program_payload_sha256']:
                    raise ValueError('cached program action payload differs from native source')
                inner=native.get('reaction_trees',{}).get(row['reaction_product'],native)
                root=next(c['state'] for c in inner['contexts'] if c['id']==inner['root_context'])
                binding=bind_program_states(value['actions'],root,value['state'])
                row['program_state_binding']=binding
                if not binding['valid']:
                    row.update(exposure_class='unavailable',complete_signature=None,
                               complete_signature_training_count=None,program_complete=False,
                               whole_program_unseen=None,support_error=binding['reason'])
            origin={**source,'tree_path':str(tree_path),'tree_sha256':tree_sha,'compatible_program_cache_reused':True}
            return source,reused,work,origin
        tree=json.loads(payload)
        if not report.get('target'):
            report={**report,'target':tree['target']}
        records=join_report_programs(report,tree,selected,exposure,actor_sha=freeze['actor_sha256'])
        origin={**source,'tree_path':str(tree_path),'tree_sha256':tree_sha}
        for row in records:
            row['source']=origin
            row['actor_exposure_manifest_sha256']=manifest['actor_exposure_manifest_sha256']
        return source,records,work,origin
    counts=Counter();programs=defaultdict(list);work_entries=defaultdict(list);matched=set()
    with ThreadPoolExecutor(max_workers=workers) as pool, gzip.open(output_root/'program_reaction_links.jsonl.gz','wt') as handle, gzip.open(output_root/'native_tree_sources.jsonl.gz','wt') as sources:
        for i,(source,records,work,origin) in enumerate(pool.map(prepare,freeze['source_reports']),1):
            counts['reports']+=1;work_entries[source['run_id']].append(work)
            if origin:
                sources.write(json.dumps(origin)+'\n');counts['native_trees_read']+=1
                if origin.get('compatible_program_cache_reused'):
                    counts['reports_reused']+=1
            else:
                counts['reports_without_eligible_reactions']+=1
            for row in records:
                handle.write(json.dumps(row,separators=(',',':'))+'\n')
                programs[source['run_id']].append(row);matched.update(row['occurrence_ids'])
                counts['program_records']+=1;counts[row['exposure_class']]+=1
            if i%100==0:
                progress={'phase':'saved_program_join','reports_done':i,
                          'reports_total':len(freeze['source_reports']),'counts':dict(counts)}
                handle.flush();sources.flush();save(output_root/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    missing=[r for batch in occurrences.values() for r in batch if r['occurrence_id'] not in matched]
    with gzip.open(output_root/'program_join_unavailable.jsonl.gz','wt') as handle:
        for row in missing:
            handle.write(json.dumps({'occurrence_id':row['occurrence_id'],'run_id':row['run_id'],
                'target_id':row['target_id'],'reaction_key':row['reaction_key'],
                'reason':'no matching saved terminal program in native tree','source':row['source']})+'\n')
    census={run:composition_metrics(programs[run],{},routes[run],universe=universe)
            for run,universe in populations.items()}
    # No experimental evidence has been attached at this stage. Do not publish
    # unassessed chemistry as zero supported reactions.
    for run in census.values():
        for label in ('seen_ordinary','unseen_ordinary','nonordinary','unavailable','unseen_combination_supported_primitives'):
            for key in ('source_supported_unique_reactions','source_supported_targets','source_checked_coverage_fraction','source_supported_program_fraction'):
                run[label][key]=None
        for key in ('UnseenProgramToSupportedReaction','NovelProgramPrecedentCoveredRoute','NovelProgramSupportedDisconnection'):
            run[key]=None
        run['primary_evidence_supplied']=False
    save(output_root/'composition_census.json',census)
    save(output_root/'observed_work.json',{run:observed_work_metrics(entries) for run,entries in work_entries.items()})
    manifest.update(complete=True,counts=dict(counts),unmatched_occurrence_count=len(missing),
                    all_eligible_reactions_linked=not missing)
    for path in sorted(output_root.iterdir()):
        if path.name not in {'program_join_manifest.json','progress.json'}:
            manifest['output_files'][path.name]={'sha256':file_sha256(path),'bytes':path.stat().st_size}
    save(output_root/'program_join_manifest.json',manifest)
    save(output_root/'progress.json',{'phase':'saved_program_join_complete','counts':dict(counts),
                                    'unmatched_occurrences':len(missing)})
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-freeze-root',type=Path,required=True)
    parser.add_argument('--actor-exposure-manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--compatible-program-cache',type=Path,
                        help='Explicit adoption of same-input, source-rehashed v1 nested-tree annotations')
    parser.add_argument('--cache-manifest-sha256')
    parser.add_argument('--cache-links-sha256')
    args=parser.parse_args()
    result=annotate(args.prediction_freeze_root,args.actor_exposure_manifest,args.output_root,
                    workers=args.workers,compatible_program_cache=args.compatible_program_cache,
                    cache_manifest_sha256=args.cache_manifest_sha256,cache_links_sha256=args.cache_links_sha256)
    print(json.dumps({'phase':'program_join_complete',**result['counts'],
                      'unmatched_occurrences':result['unmatched_occurrence_count']}),flush=True)


if __name__=='__main__':
    main()
