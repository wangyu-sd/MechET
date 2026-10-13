#!/usr/bin/env python3
"""Exhaustive route-DAG counts from a completed full460 inference condition.

Only saved reaction edges and frozen stock are read. No model calls, no graph
edits and no fixed enumeration budget. A manifest is sealed only after ALL
targets finish counting; interrupted output stays explicitly partial.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from mechet.route_diversity import iter_complete_routes
from mechet.frozen_route_assets import file_sha256
from scripts.infer_dynamic_electron_trees import load_route_stock,ADAPTER_SHA,REVISION
from scripts.analyze_frozen_route_search import validate_shard_contract
from scripts.prepare_frozen_route_benchmarks import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--mode',choices=['greedy','tree'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--world',type=int,default=8);args=p.parse_args()
    assets=json.loads(args.assets.read_text())
    stock,_=load_route_stock(args.assets,'all',assets['full_targets_sha256'],460)
    expected={'adapter_sha':ADAPTER_SHA,'revision':REVISION,'model':'Qwen/Qwen3-0.6B',
        'source_sha':assets['full_targets_sha256'],'route_assets_sha':file_sha256(args.assets),
        'code_sha':assets['preparation_code_sha'],'full_denominator':460,
        'stock_canonical_shas':{k:v['canonical_sha256'] for k,v in assets['stocks'].items()}}
    source_cohort={identifier:name for name,c in assets['cohorts'].items() for identifier in c['target_ids']}
    shards=[]
    for rank in range(args.world):
        path=args.run/args.mode/f'rank{rank:02d}'/'complete.json'
        if not path.exists():raise ValueError('full inference condition is incomplete; no diversity result sealed')
        value=json.loads(path.read_text());validate_shard_contract(value['manifest'],expected,rank=rank,world=args.world,mode=args.mode)
        shards.append((path,value))
    rows=[row for _,shard in shards for row in shard['rows']]
    if len(rows)!=460 or {r['id'] for r in rows}!=set(source_cohort):raise ValueError('full460 prediction denominator mismatch')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists():raise ValueError('preserve previous diversity result; select a new output')
    counts=[];provenance=[]
    for path,shard in shards:
        for row in shard['rows']:
            cohort=source_cohort[row['id']];count=0
            if row.get('tree_artifact'):
                graph_path=path.parent/row['tree_artifact'];graph=json.loads(graph_path.read_text())
                if graph['case_id']!=row['id']:raise ValueError('saved graph case mismatch')
                for _ in iter_complete_routes(graph['target'],stock[cohort],graph['reaction_edges']):count+=1
                provenance.append({'id':row['id'],'graph_sha256':file_sha256(graph_path)})
            elif row.get('stop_reason')!='ROOT_PARSE_FAILURE':raise ValueError('missing saved route graph')
            if bool(count)!=row['stock_solved']:raise ValueError('exact diversity contradicts saved stock closure')
            counts.append({'id':row['id'],'cohort':cohort,'unique_route_count':count,
                           'route_count_is_exhaustive':True,'already_in_stock':row['already_in_stock']})
            write_json(args.output/'counts.partial.json',{'complete':False,'processed_targets':len(counts),'rows':counts})
            print({'phase':'exact-route-diversity','mode':args.mode,'processed':len(counts),'total':460,
                   'case_id':row['id'],'routes':count},flush=True)
    write_json(args.output/'counts.json',{'complete':True,'denominator':460,'rows':counts})
    write_json(args.output/'manifest.json',{'mode':args.mode,'denominator':460,'model_sha256':ADAPTER_SHA,
        'assets_sha256':file_sha256(args.assets),'counts_sha256':file_sha256(args.output/'counts.json'),
        'source_shards':[{'path':str(path),'sha256':file_sha256(path)} for path,_ in shards],
        'source_graphs':provenance,'route_count_is_exhaustive':True,
        'route_definition':'one_production_choice_per_distinct_nonstock_molecule_DAG_v1',
        'electron_program_alternatives_counted_as_separate_routes':False,'model_called':False})


if __name__=='__main__':main()
