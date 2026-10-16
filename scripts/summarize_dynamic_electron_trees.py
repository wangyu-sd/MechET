#!/usr/bin/env python3
"""Require complete source IDs, then aggregate real frozen-policy outputs."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['tree','greedy'],required=True)
    p.add_argument('--world',type=int,default=8)
    args=p.parse_args()
    ids={json.loads(line)['id'] for line in args.data.open() if line.strip()}
    reports=[]
    manifests=[]
    for rank in range(args.world):
        shard=json.loads((args.output/args.mode/f'rank{rank:02d}'/'complete.json').read_text())
        reports.extend(shard['rows']);manifests.append(shard['manifest'])
    observed=[r['id'] for r in reports]
    if len(observed)!=len(ids) or set(observed)!=ids:raise ValueError('missing/extra/duplicate full-test ID')
    conditions=['adapter_sha','source_sha','full_denominator','mode','raw_unwarped_nll','seed']
    for manifest in manifests:
        if any(manifest[key]!=manifests[0][key] for key in conditions):raise ValueError('mixed shard conditions')
    summary={'mode':args.mode,'denominator':len(ids),'conditions':{k:manifests[0][k] for k in conditions},
             'top1':sum(bool(r['top1']) for r in reports)/len(ids),
             'top5':sum(bool(r['top5']) for r in reports)/len(ids) if args.mode=='tree' else None,
             'top10':sum(bool(r['top10']) for r in reports)/len(ids) if args.mode=='tree' else None,
             'any_endpoint_hit':sum(bool(r['any_endpoint_hit']) for r in reports)/len(ids),
             'terminal_coverage':sum(r['terminal_count']>0 for r in reports)/len(ids),
             'total_output_tokens':sum(r['work']['sampled_tokens'] for r in reports),
             'raw_candidate_count':sum(r['work']['raw_candidates'] for r in reports),
             'stop_reasons':{reason:sum(r['stop_reason']==reason for r in reports) for reason in sorted({r['stop_reason'] for r in reports})},
             'rows':reports}
    target=args.output/args.mode/'evaluation.json'
    target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
