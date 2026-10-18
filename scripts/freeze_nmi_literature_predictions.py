#!/usr/bin/env python3
"""Freeze every eligible existing PR88 complete reaction before retrieval.

CPU-only analysis, not new model inference. Reference reactants and correctness
flags are intentionally neither queried nor copied into the new evidence pool.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import argparse
import gzip
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
from mechet.analysis.literature_predictions import frozen_reports, route_reactions, single_reactions, resolve_tree_artifact, route_reaction_keys
from mechet.analysis.literature_reaction_keys import canonical_components, make_reaction, stable_hash
from mechet.analysis.literature_sources import file_sha256
from mechet.analysis.tree_records import load_frozen_stocks, assert_stock_identity
from mechet.analysis.tree_metrics import verify_reported_route

ACTOR = '40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'


def dump(path, value):
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(temp,path)


def read_report(path):
    content = path.read_bytes()
    import hashlib
    return json.loads(content), {'report_path':str(path),'report_sha256':hashlib.sha256(content).hexdigest()}


@lru_cache(maxsize=None)
def read_contract(path):
    return read_report(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository-data-root',type=Path,required=True)
    parser.add_argument('--tree-analysis-root',type=Path,required=True)
    parser.add_argument('--progress-snapshot',type=Path,required=True)
    parser.add_argument('--route-assets-manifest',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args = parser.parse_args()
    output = args.output_root.resolve(); output.mkdir(parents=True,exist_ok=False)
    snapshot = json.loads(args.progress_snapshot.read_bytes())
    assets = json.loads(args.route_assets_manifest.read_bytes())
    if file_sha256(assets['full_targets_path']) != assets['full_targets_sha256']:
        raise ValueError('frozen target source checksum mismatch')
    targets = [json.loads(line) for line in Path(assets['full_targets_path']).read_text().splitlines()]
    universe = {row['id']:row['product_unmapped'] for row in targets}
    if len(universe) != 460:
        raise ValueError('R1 full original target denominator must be 460')
    stocks, stock_sources = load_frozen_stocks(args.route_assets_manifest)
    manifest = {'schema':'lgrv_prediction_freeze_v1','complete':False,
        'protocol':'LGRV-v1','actor_sha256':ACTOR,'queries_frozen_before_retrieval':True,
        'no_new_model_inference':True,'reference_precursors_used':False,'ranking_recomputed':False,
        'progress_snapshot_sha256':file_sha256(args.progress_snapshot),
        'route_assets_manifest_sha256':file_sha256(args.route_assets_manifest),
        'prior_analysis_root':str(args.tree_analysis_root.resolve()),
        'prior_analysis_source_manifest_sha256':file_sha256(args.tree_analysis_root/'source_manifest.json'),
        'runs':{},'source_reports':[],'output_files':{}}
    dump(output/'prediction_freeze_manifest.json',manifest)
    single_source = snapshot['sources']['mech31k_evaluation']
    if file_sha256(single_source['path']) != single_source['sha256']:
        raise ValueError('frozen 3120 evaluation checksum differs from PR88')
    single_data = json.loads(Path(single_source['path']).read_bytes())
    single_universe = {row['id']:row['target'] for row in single_data['rows']}
    if len(single_universe) != 3120 or len(single_data['rows']) != 3120:
        raise ValueError('MCTS source does not contain every original 3120 target once')
    report_csv = args.progress_snapshot.parent/'r1_case_results.csv'
    fixed_r1 = frozen_reports(snapshot['sources']['r1_root'],report_csv)
    if len(fixed_r1) != snapshot['r1']['saved_reports'] or len(fixed_r1) != 200:
        raise ValueError('PR88 R1 report population changed')
    batches = [('r1_mcts_pr88','route',fixed_r1,universe)]
    runs = [('r1_greedy_dynamic','route','r1_r2_frozen_06b_dynamic_routes_20261013',universe),
            ('r1_greedy_shared','route','r1_r2_frozen_06b_shared_routes_20261014_05',universe),
            ('mech31k_greedy_dynamic','single','mech_uspto31k_full_dynamic_electron_tree_20261013',single_universe),
            ('mech31k_greedy_shared','single','mech_uspto31k_full_shared_electron_tree_20261014_05',single_universe)]
    for run, mode, folder, population in runs:
        files = sorted((args.repository_data_root/'outputs/eval'/folder/'greedy').glob('rank*/evaluations/*.json'))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            reports = list(pool.map(read_report,files))
        ids = [r['id'] for r,_ in reports]
        if len(ids) != len(population) or set(ids) != set(population):
            raise ValueError('baseline missing/repeated/mismatched target IDs: '+run)
        batches.append((run,mode,reports,population))
    batches.append(('mech31k_mcts_pr88','single',[(row,{
        'report_path':single_source['path'],'report_sha256':single_source['sha256'],
        'report_row_index':i}) for i,row in enumerate(single_data['rows'])],single_universe))
    distinct = {}; counts = Counter(); conversion_failures = []
    with ExitStack() as stack:
        handles = {name:stack.enter_context(gzip.open(output/(name+'.jsonl.gz'),'wt'))
                   for name in ['generated_reaction_steps','frozen_routes','target_population']}
        def line(name,value):
            handles[name].write(json.dumps(value,separators=(',',':'))+'\n'); counts[name] += 1
        for run, mode, reports, population in batches:
            observed = set()
            for report_number,(report,source) in enumerate(reports):
                sid = report['id']; observed.add(sid)
                if sid not in population:
                    raise ValueError('observed target outside original universe')
                if mode=='single' and not report.get('target'):
                    report_path = Path(source['report_path'])
                    tree = resolve_tree_artifact(report,source)
                    report = {**report,'target':json.loads(tree.read_bytes())['target']}
                    source.update({'target_tree_path':str(tree),'target_tree_sha256':file_sha256(tree)})
                if canonical_components(report['target']) != canonical_components(population[sid]):
                    raise ValueError('same-ID target product differs: '+sid)
                if mode=='route':
                    manifest_path = Path(source['report_path']).parent.parent/'manifest.json'
                else:
                    manifest_path = (Path(source['report_path']).parent.parent/'manifest.json')
                    if 'report_row_index' in source:
                        manifest_path = resolve_tree_artifact(report,source).parent.parent/'manifest.json'
                source_contract, contract_source = read_contract(manifest_path)
                if source_contract.get('adapter_sha') != ACTOR:
                    raise ValueError('actor differs from frozen PR89 contract')
                source.update({'manifest_path':str(manifest_path),'manifest_sha256':contract_source['report_sha256'],
                               'actor_sha256':ACTOR,'executor_code_sha':source_contract.get('code_sha')})
                if mode=='route':
                    assert_stock_identity(source_contract,report['cohort'],stock_sources)
                    source['stock_sha256'] = stock_sources['cohort_stock_sha256'][report['cohort']]
                manifest['source_reports'].append({'run_id':run,'target_id':sid,**source})
                records = list((route_reactions if mode=='route' else single_reactions)(report,run_id=run,source=source))
                for reaction in records:
                    line('generated_reaction_steps',reaction)
                    distinct.setdefault(reaction['reaction_key'],reaction)
                if mode=='route':
                    routes = report.get('enumerated_stock_connected_routes',[])
                    if report.get('stock_solved') and not routes:
                        routes = [{'route_key':report.get('route_key'),'complete_reaction_edges':report['complete_reaction_edges'],
                                   'stock_leaves':report['stock_leaves']}]
                    for i,route in enumerate(routes):
                        checked = verify_reported_route({**report,**route},stocks[report['cohort']])
                        if checked['stock_solved'] is not True:
                            raise ValueError('recorded AND route fails frozen stock: '+sid)
                        keys = route_reaction_keys(route,records)
                        line('frozen_routes',{'run_id':run,'target_id':sid,'cohort':report['cohort'],
                             'route_key':stable_hash(keys),'source_route_key':route.get('route_key'),
                             'reaction_keys':keys,'stock_leaves':route['stock_leaves'],'stock_verified':True,
                             'already_in_stock':False,'source':source,'source_pointer':f'/enumerated_stock_connected_routes/{i}',
                             'stock_recheck':checked})
                if (report_number+1)%100==0:
                    progress={'phase':'prediction_freeze','run':run,'reports_done':report_number+1,
                              'reports_total':len(reports),'unique_reactions':len(distinct)}
                    dump(output/'progress.json',progress);print(json.dumps(progress),flush=True)
            for sid,product in population.items():
                line('target_population',{'run_id':run,'target_id':sid,'product':product,
                     'report_status':'observed' if sid in observed else 'pending',
                     'full_denominator':len(population)})
            manifest['runs'][run] = {'mode':mode,'denominator':len(population),'observed_reports':len(observed),
                                    'pending_reports':len(population)-len(observed)}
    with gzip.open(output/'unique_reaction_queries.jsonl.gz','wt') as handle:
        for key,reaction in sorted(distinct.items()):
            handle.write(json.dumps(reaction,separators=(',',':'))+'\n')
    manifest.update({'complete':True,'counts':dict(counts),'unique_reaction_queries':len(distinct),
                     'conversion_failures':conversion_failures,
                     'single_reaction_protocol':'frozen rxn_prod_min proxy DIAGNOSTIC; ten TOTAL MCTS attempts, not a stock-route benchmark'})
    for name in ['generated_reaction_steps','frozen_routes','target_population','unique_reaction_queries']:
        path = output/(name+'.jsonl.gz')
        manifest['output_files'][path.name] = {'sha256':file_sha256(path),'bytes':path.stat().st_size}
    dump(output/'prediction_freeze_manifest.json',manifest)
    dump(output/'progress.json',{'phase':'prediction_freeze_complete','counts':dict(counts),'unique_reactions':len(distinct)})
    print(json.dumps({'phase':'prediction_freeze_complete','counts':dict(counts),'unique_reactions':len(distinct)}),flush=True)


if __name__ == '__main__':
    main()
