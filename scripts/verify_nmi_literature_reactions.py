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
from mechet.analysis.literature_evidence import assess
from mechet.analysis.literature_index import ReactionIndex
from mechet.analysis.literature_reaction_keys import make_reaction
from mechet.analysis.literature_route_metrics import aggregate_routes,disconnection_counts,paired_targets,source_supported
from mechet.analysis.literature_sources import file_sha256
from mechet.analysis.mechanism_consistency import assess_mechanism


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
    parser.add_argument('--primary-source-registry',type=Path)
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
    corpus = json.loads((args.source_reaction_index.parent/'source_corpus_manifest.json').read_bytes())
    if corpus.get('complete') is not True or file_sha256(args.source_reaction_index)!=corpus['index_sha256']:
        raise ValueError('source index incomplete or checksum differs')
    registry = json.loads(args.primary_source_registry.read_bytes()) if args.primary_source_registry else {'records':[]}
    primary = defaultdict(list)
    for record in registry['records']:
        key = make_reaction(record['source_product'],record['source_precursors'])['reaction_key']
        primary[key].append(record)
        if record.get('analogue_query_reaction_key'):
            primary[record['analogue_query_reaction_key']].append(record)
    output = args.output_root;output.mkdir(parents=True,exist_ok=False)
    save(output/'source_corpus_manifest.json',corpus);save(output/'prediction_freeze_manifest.json',freeze)
    evidence = {};tiers=Counter();retrieval=Counter()
    source_index = ReactionIndex(args.source_reaction_index,readonly=True)
    with gzip.open(output/'literature_hits.jsonl.gz','wt') as hits_file, gzip.open(output/'reaction_evidence.jsonl.gz','wt') as evidence_file:
        for number,reaction in enumerate(rows(args.prediction_freeze_root/'unique_reaction_queries.jsonl.gz'),1):
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
               'r2_complete_program_join':'pending','matched_compute_claim':False}
    with gzip.open(output/'route_evidence.jsonl.gz','wt') as handle:
        for run,universe in populations.items():
            result = {'denominator':len(universe),'observed_reports':len(observed[run]),
                'pending_reports':len(set(universe)-observed[run]),
                'supported_disconnections':disconnection_counts(occurrences[run],evidence,universe=universe)}
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
    with (output/'baseline_comparison.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=['baseline','outcome','denominator','both','method_only','baseline_only','neither','difference','interpretation'])
        writer.writeheader()
        for baseline in ['r1_greedy_dynamic','r1_greedy_shared']:
            comparison={}
            for name in ['StockSolved@B','PrecedentCoveredSolved@B']:
                method={r['target_id'] for r in metrics['runs']['r1_mcts_pr88']['targets'] if r[name]}
                control={r['target_id'] for r in metrics['runs'][baseline]['targets'] if r[name]}
                paired=paired_targets(populations['r1_mcts_pr88'],method,control,bootstrap_draws=config['output']['pairwise_ci_bootstrap_draws'])
                comparison[name]=paired
                writer.writerow({'baseline':baseline,'outcome':name,**{key:paired[key] for key in writer.fieldnames if key in paired}})
            metrics['paired'][baseline]=comparison
    with gzip.open(output/'source_passages.jsonl.gz','wt') as handle:
        for record in registry['records']:
            # No full copyrighted text or accumulated long excerpts in export.
            handle.write(json.dumps({k:v for k,v in record.items() if k not in {'document_text','checked_passage'}})+'\n')
    coverage={'unique_reaction_queries':len(evidence),'tier_counts':dict(tiers),**retrieval,
              'source_checked_reactions':sum(source_supported(r) for r in evidence.values()),
              'primary_registry_sha256':file_sha256(args.primary_source_registry) if args.primary_source_registry else None,
              'unresolved_means':'no checked supporting primary source under frozen retrieval, not impossible chemistry',
              'remaining_work':['additional primary document checks','authenticated full R2 join','analogue policy validation','publication figures']}
    save(output/'metrics.json',metrics);save(output/'evidence_coverage.json',coverage)
    save(output/'progress.json',{'phase':'structured_retrieval_complete_primary_coverage_explicit',**coverage})
    print(json.dumps({'phase':'retrieval_complete','unique_reactions':len(evidence),'tiers':dict(tiers)}),flush=True)


if __name__ == '__main__':
    main()
