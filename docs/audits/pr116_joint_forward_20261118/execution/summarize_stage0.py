"""Count only published initial-actor outcomes and show explicit denominators for three cohorts."""
import datetime,json
from pathlib import Path
from collections import Counter

def summarize(rows,ids):
    selected=[r for r in rows if r['task_id'] in ids]
    return dict(denominator=len(ids),published=len(selected),
                stock_closed=sum(r['native_stock_closed'] for r in selected),
                reference_verified=sum(r['verified_closed'] for r in selected),
                censored=sum(r['censored'] for r in selected),
                decisions=sum(r['model_decision_requests'] for r in selected))

def current(d):
    folder=d.parents[2]/'outputs/agent/pr116_step0_4h20_512_20261118_01/validation/checkpoint-000000'
    valid={json.loads(s)['task_id'] for s in (d/'clean_valid.jsonl').read_text().splitlines()}
    in_sft=set(json.loads((d/'sft_target_overlap_ids.json').read_text()))
    strict=json.loads((d/'overlap_subsets.json').read_text())
    rows=[r for f in sorted(folder.glob('rank*/batch-*/outcomes.json')) for r in json.loads(f.read_bytes())['outcomes']]
    if len(rows)!=len({r['task_id'] for r in rows}) or any(r['task_id'] not in valid for r in rows):
        raise ValueError('duplicate or out-of-cohort validation outcome')
    return dict(time=datetime.datetime.now().astimezone().isoformat(),all_targets=summarize(rows,valid),
                seen_in_history_sft=summarize(rows,in_sft),unseen_in_history_sft=summarize(rows,valid-in_sft),
                reaction_disjoint=summarize(rows,set(strict['reaction_disjoint'])),
                reaction_and_patent_disjoint=summarize(rows,set(strict['reaction_and_patent_disjoint'])))
if __name__=='__main__':print(json.dumps(current(Path(__file__).resolve().parent),indent=2))
