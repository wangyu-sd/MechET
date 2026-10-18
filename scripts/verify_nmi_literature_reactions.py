#!/usr/bin/env python3
"""Run frozen LGRV structure retrieval and checked-source route aggregation."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mechet.analysis.literature_evidence import assess,load_primary_registries
from mechet.analysis.literature_frozen_index import FrozenReactionIndex
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_route_metrics import aggregate_routes,disconnection_counts,paired_observed_targets,source_supported
from mechet.analysis.literature_sources import file_sha256
from mechet.analysis.mechanism_consistency import assess_mechanism
from mechet.analysis.literature_programs import composition_metrics
from mechet.analysis.literature_mapping import load_query_view
from mechet.analysis.literature_comparison import compare_disconnections, discovery_efficiency, reaction_target_outcomes


def rows(path):
    with gzip.open(path,'rt') as handle:
        yield from (json.loads(line) for line in handle)


def save(path,value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['tree-analysis-root','progress-snapshot','source-reaction-index','route-assets-manifest',
                 'output-root','retrieval-config','prediction-freeze-root']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--primary-source-registry',type=Path,action='append',default=[])
    parser.add_argument('--program-join-root',type=Path)
    parser.add_argument('--native-mapping-view',type=Path)
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.retrieval_config.read_text())
    if (config['literature']['protocol']!='LGRV-v1' or
        config['main_outcome']!='literature_evidence_for_complete_synthetic_reaction_steps_and_stock_routes'):
        raise ValueError('LGRV complete-reaction evidence protocol required, not reference ranking')
    freeze = json.loads((args.prediction_freeze_root/'prediction_freeze_manifest.json').read_bytes())
    if freeze.get('complete') is not True:
        raise ValueError('prediction pool is not frozen')
    checks = [(args.progress_snapshot,freeze['progress_snapshot_sha256']),
              (args.route_assets_manifest,freeze['route_assets_manifest_sha256']),
              (args.tree_analysis_root/'source_manifest.json',freeze['prior_analysis_source_manifest_sha256'])]
    checks += [(args.prediction_freeze_root/name,value['sha256']) for name,value in freeze['output_files'].items()]
    for path,expected in checks:
        if file_sha256(path)!=expected:
            raise ValueError('frozen input checksum differs: '+str(path))
    if freeze['actor_sha256']!=config['actor']['weight_sha256']:
        raise ValueError('actor identity differs')
    programs=defaultdict(list);program_manifest=None
    if args.program_join_root:
        program_manifest=json.loads((args.program_join_root/'program_join_manifest.json').read_bytes())
        if (program_manifest.get('complete') is not True or
            program_manifest.get('actor_sha256')!=freeze['actor_sha256'] or
            program_manifest.get('prediction_freeze_manifest_sha256')!=file_sha256(args.prediction_freeze_root/'prediction_freeze_manifest.json')):
            raise ValueError('program join does not belong to completed frozen actor/population')
        for name,description in program_manifest['output_files'].items():
            if file_sha256(args.program_join_root/name)!=description['sha256']:
                raise ValueError('program join artifact checksum differs: '+name)
        for row in rows(args.program_join_root/'program_reaction_links.jsonl.gz'):
            programs[row['run_id']].append(row)
    native_mapping=None
    query_rows=rows(args.prediction_freeze_root/'unique_reaction_queries.jsonl.gz')
    if args.native_mapping_view:
        if args.program_join_root is None:
            raise ValueError('native mapping view requires the authenticated complete program join')
        query_rows,native_mapping=load_query_view(args.prediction_freeze_root,args.program_join_root,
                                                args.native_mapping_view)
    corpus = json.loads((args.source_reaction_index.parent/'source_corpus_manifest.json').read_bytes())
    if corpus.get('complete') is not True:
        raise ValueError('source index incomplete or checksum differs')
    source_index=FrozenReactionIndex(args.source_reaction_index,index_sha256=corpus['index_sha256'])
    registry = load_primary_registries(args.primary_source_registry)
    primary = defaultdict(list)
    for record in registry['records']:
        key = make_reaction(record['source_product'],record['source_precursors'])['reaction_key']
        primary[key].append(record)
        if record.get('analogue_query_reaction_key'):
            primary[record['analogue_query_reaction_key']].append(record)
        if record.get('contradicted_reaction_key') and record['contradicted_reaction_key']!=key:
            primary[record['contradicted_reaction_key']].append(record)
    output = args.output_root;output.mkdir(parents=True,exist_ok=False)
    save(output/'source_corpus_manifest.json',corpus);save(output/'prediction_freeze_manifest.json',freeze)
    if native_mapping:
        for name in ('mapping_enrichment_manifest.json','unique_reaction_queries_enriched.jsonl.gz'):
            shutil.copyfile(args.native_mapping_view/name,output/name)
    evidence = {};tiers=Counter();retrieval=Counter()
    with gzip.open(output/'literature_hits.jsonl.gz','wt') as hits_file, gzip.open(output/'reaction_evidence.jsonl.gz','wt') as evidence_file:
        for number,reaction in enumerate(query_rows,1):
            key = reaction['reaction_key']
            hits = source_index.query(reaction,cap=int(config['literature']['primary_search_cap_per_unique_reaction']))
            for hit in hits:
                hits_file.write(json.dumps({'query_reaction_key':key,**hit})+'\n')
            evidence[key] = assess(reaction,hits,primary.get(key,[]));tiers[evidence[key]['tier']] += 1
            retrieval['unique_reactions_with_index_hit'] += bool(hits)
            retrieval['source_records_retrieved'] += len(hits)
            evidence_file.write(json.dumps(evidence[key])+'\n')
            if number%500==0:
                progress = {'phase':'literature_retrieval','unique_reactions_done':number,'tiers':dict(tiers),**retrieval}
                save(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    source_index.close()
    populations=defaultdict(list);observed=defaultdict(set);occurrences=defaultdict(list);route_rows=defaultdict(list)
    for row in rows(args.prediction_freeze_root/'target_population.jsonl.gz'):
        populations[row['run_id']].append(row['target_id'])
        if row['report_status']=='observed':
            observed[row['run_id']].add(row['target_id'])
    shutil.copyfile(args.prediction_freeze_root/'generated_reaction_steps.jsonl.gz',output/'generated_reaction_steps.jsonl.gz')
    with gzip.open(output/'mechanism_consistency.jsonl.gz','wt') as handle:
        for row in rows(output/'generated_reaction_steps.jsonl.gz'):
            occurrences[row['run_id']].append(row)
            handle.write(json.dumps(assess_mechanism(row,evidence[row['reaction_key']]))+'\n')
    for route in rows(args.prediction_freeze_root/'frozen_routes.jsonl.gz'):
        route_rows[route['run_id']].append(route)
    metrics = {'schema':'lgrv_results_v1','protocol':'LGRV-v1','literature_tiers':dict(tiers),
               'retrieval':dict(retrieval),'runs':{},'paired':{},'ranking_is_primary':False,'no_new_inference':True,
               'source_overlap':'unknown unless separately documented','unsupported_is_not_invalid':True,
               'r2_reference_C0_C1_C2':'Unknown for all160; not inferred from generated signature novelty',
               'r2_complete_program_join':('complete' if program_manifest['unmatched_occurrence_count']==0 else 'partial') if program_manifest else 'pending',
               'r2_program_join_manifest_sha256':file_sha256(args.program_join_root/'program_join_manifest.json') if program_manifest else None,
               'composition':{},'paired_disconnections':{},'efficiency':{},'matched_compute_claim':False}
    observed_work={}
    if program_manifest:
        shutil.copyfile(args.program_join_root/'program_reaction_links.jsonl.gz',output/'program_reaction_links.jsonl.gz')
        shutil.copyfile(args.program_join_root/'observed_work.json',output/'observed_work.json')
        observed_work=json.loads((args.program_join_root/'observed_work.json').read_bytes())
    with gzip.open(output/'route_evidence.jsonl.gz','wt') as handle:
        for run,universe in populations.items():
            result = {'denominator':len(universe),'observed_reports':len(observed[run]),
                'pending_reports':len(set(universe)-observed[run]),
                'supported_disconnections':disconnection_counts(occurrences[run],evidence,universe=universe)}
            result['reaction_source_outcomes']=reaction_target_outcomes(occurrences[run],evidence,universe=universe)
            if freeze['runs'][run]['mode']=='route':
                route_metrics = aggregate_routes(route_rows[run],evidence,universe=universe,observed_ids=observed[run])
                for route in route_metrics.pop('route_evidence'):
                    handle.write(json.dumps(route)+'\n')
                result.update(route_metrics)
            else:
                exact = {r['target_id'] for r in occurrences[run] if source_supported(evidence[r['reaction_key']]) and evidence[r['reaction_key']]['tier']=='A'}
                result.update({'AnySourceSupportedReaction':result['supported_disconnections']['any_supported_targets'],
                               'ExactSourceSupported':len(exact),'attempt_budget':10 if 'mcts' in run else 'one_greedy_episode',
                               'source_protocol':'frozen rxn_prod_min proxy DIAGNOSTIC; not stock-route planning'})
            metrics['runs'][run] = result
            if program_manifest:
                metrics['composition'][run]=composition_metrics(programs[run],evidence,route_rows[run],universe=universe)
            metrics['efficiency'][run]=discovery_efficiency(observed_work.get(run,{}),
                supported_disconnections=result['supported_disconnections']['unique_supported_reactions'],
                covered_targets=result.get('counts',{}).get('PrecedentCoveredSolved@B'))
    with (output/'baseline_comparison.csv').open('w') as handle:
        fields=['baseline','outcome','denominator','both','method_only','baseline_only','neither','difference','interpretation',
                'method_pending_reports','baseline_pending_reports','full_population_outcome',
                'observed_both_denominator','observed_both_both','observed_both_method_only',
                'observed_both_baseline_only','observed_both_neither','observed_both_difference']
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        def write_pair(baseline,name,paired):
            flat={key:paired[key] for key in fields if key in paired}
            flat.update({'observed_both_'+key:paired['observed_both'][key] for key in
                         ('denominator','both','method_only','baseline_only','neither','difference')})
            writer.writerow({'baseline':baseline,'outcome':name,**flat})
        for baseline in ['r1_greedy_dynamic','r1_greedy_shared']:
            comparison={}
            for name in ['StockSolved@B','PrecedentCoveredSolved@B']:
                method={r['target_id'] for r in metrics['runs']['r1_mcts_pr88']['targets'] if r[name]}
                control={r['target_id'] for r in metrics['runs'][baseline]['targets'] if r[name]}
                paired=paired_observed_targets(populations['r1_mcts_pr88'],method,control,
                    method_observed=observed['r1_mcts_pr88'],baseline_observed=observed[baseline],
                    bootstrap_draws=config['output']['pairwise_ci_bootstrap_draws'])
                comparison[name]=paired
                write_pair(baseline,name,paired)
            metrics['paired'][baseline]=comparison
            metrics['paired_disconnections'][baseline]=compare_disconnections(
                occurrences['r1_mcts_pr88'],occurrences[baseline],evidence,universe=populations['r1_mcts_pr88'])
        for baseline in ['mech31k_greedy_dynamic','mech31k_greedy_shared']:
            if baseline not in populations:
                continue
            method='mech31k_mcts_pr88';comparison={}
            if set(populations[method])!=set(populations[baseline]):
                raise ValueError('single-reaction baseline target identities differ')
            for name,field in [('AnySourceSupportedReaction','any_supported_targets'),('ExactSourceSupported','exact_supported_targets')]:
                left=metrics['runs'][method]['reaction_source_outcomes'][field]
                right=metrics['runs'][baseline]['reaction_source_outcomes'][field]
                paired=paired_observed_targets(populations[method],left,right,
                    method_observed=observed[method],baseline_observed=observed[baseline],
                    bootstrap_draws=config['output']['pairwise_ci_bootstrap_draws'])
                comparison[name]=paired
                write_pair(baseline,name,paired)
            metrics['paired'][baseline]=comparison
            metrics['paired_disconnections'][baseline]=compare_disconnections(
                occurrences[method],occurrences[baseline],evidence,universe=populations[method])
    with gzip.open(output/'source_passages.jsonl.gz','wt') as handle:
        for record in registry['records']:
            # No full copyrighted text or accumulated long excerpts in export.
            handle.write(json.dumps({k:v for k,v in record.items() if k not in {'document_text','checked_passage'}})+'\n')
    coverage={'unique_reaction_queries':len(evidence),'tier_counts':dict(tiers),**retrieval,
              'source_checked_reactions':sum(r.get('verification_status') in {'source_passage_checked','source_human_reviewed'} for r in evidence.values()),
              'source_supported_reactions':sum(source_supported(r) for r in evidence.values()),
              'primary_registry_sha256':registry['source_manifests'][0]['sha256'] if len(registry['source_manifests'])==1 else None,
              'primary_registry_manifests':registry['source_manifests'],
              'unresolved_means':'no checked supporting primary source under frozen retrieval, not impossible chemistry',
              'remaining_work':['additional primary document checks','analogue policy validation','publication figures']+(
                  [] if program_manifest and program_manifest['unmatched_occurrence_count']==0 else ['authenticated full R2 join'])}
    save(output/'metrics.json',metrics);save(output/'evidence_coverage.json',coverage)
    save(output/'progress.json',{'phase':'structured_retrieval_complete_primary_coverage_explicit',**coverage})
    # Written last: consumers must never treat partial or subsequently edited
    # evidence tables as a completed population analysis.
    producer=[Path(__file__),*sorted((ROOT/'src/mechet/analysis').glob('literature_*.py')),
              ROOT/'src/mechet/analysis/mechanism_consistency.py']
    save(output/'analysis_manifest.json',{
        'schema':'lgrv_analysis_manifest_v1','complete':True,
        'actor_sha256':freeze['actor_sha256'],
        'prediction_freeze_manifest_sha256':file_sha256(args.prediction_freeze_root/'prediction_freeze_manifest.json'),
        'source_corpus_manifest_sha256':file_sha256(args.source_reaction_index.parent/'source_corpus_manifest.json'),
        'retrieval_config_sha256':file_sha256(args.retrieval_config),
        'primary_registry_manifests':registry['source_manifests'],
        'program_join_manifest_sha256':metrics['r2_program_join_manifest_sha256'],
        'native_mapping_enrichment_manifest_sha256':file_sha256(args.native_mapping_view/'mapping_enrichment_manifest.json') if native_mapping else None,
        'producer_files':{str(p.relative_to(ROOT)):file_sha256(p) for p in producer},
        'output_files':{p.name:{'sha256':file_sha256(p)} for p in sorted(output.iterdir()) if p.is_file()}})
    print(json.dumps({'phase':'retrieval_complete','unique_reactions':len(evidence),'tiers':dict(tiers)}),flush=True)


if __name__ == '__main__':
    main()
