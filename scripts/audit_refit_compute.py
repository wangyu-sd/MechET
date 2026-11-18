"""Read-only classification of an entire saved round; no model calls or hidden labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def classify(paths):
    totals=Counter();bindings=[];seen=set();edges={}
    for path in paths:
        path=Path(path);raw=path.read_bytes()
        bindings.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        groups=json.loads(raw);del raw
        for group in groups:
            totals['source_row_groups']+=1
            for original in group:
                reports=[original,*original.get('refit',{}).get('suffix_reports',[])]
                totals['original_g4_reports']+=1;totals['extra_suffix_reports']+=len(reports)-1
                retained=set(original.get('refit',{}).get('conditioning',{}))
                for i,report in enumerate(reports):
                    totals['reports']+=1
                    if i:totals['extra_completed_suffixes']+=report['status'] not in ('resource_censored','protected_censored','duplicate_pruned_no_continuation')
                    for event in report.get('refit_path',[]):
                        uid=event['request_id']
                        if uid in seen:continue
                        seen.add(uid);record=report['records'][event['index']]
                        key=(event['node_id'],event['action'])
                        if key in edges:
                            totals['duplicate_semantic_proposals']+=1
                            if record.get('accepted') is True:
                                totals['accepted_duplicate_proposals']+=1
                                totals['repeated_accepted_executor_calls_lower_bound']+=not record.get('executor_skipped')
                            elif record.get('accepted') is False:totals['rejected_duplicate_proposals']+=1
                            # Counts only actual complete observed duplicate-root
                            # continuations; not a counterfactual speedup estimate.
                            if event['name']=='apply_electron_flow' and i:
                                totals['duplicate_flow_events_in_extra_reports']+=1
                                totals['retained_node_duplicate_flow_proposals']+=event['node_id'] in retained
                        else:edges[key]=uid
        del groups
    totals['distinct_exact_condition_action_edges']=len(edges)
    totals['unique_action_receipts']=len(seen)
    return dict(scope='complete saved round read-only exact-conditioning classification',counts=dict(totals),
        input_bindings=bindings,model_calls=0,
        limitations=['Decision identities cover all events, including nodes not retained for Q; these conservative exact-conditioning duplicates are a lower bound on shareable chemical transitions.',
            'Repeated semantic draws are not independent successful route outcomes.',
            'No wall-time speedup or unchanged old/new search distribution is inferred.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',type=Path,required=True)
    p.add_argument('--ranks',type=int,default=4);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();shared=Path('/aaa/fionafyang/buddy1').resolve()
    if not a.output.resolve().is_relative_to(shared):raise ValueError('SHARED_OUTPUT_REQUIRED')
    paths=[a.round/f'rank{i:02d}/rewarded_episodes.json' for i in range(a.ranks)]
    if not all(x.is_file() for x in paths):raise ValueError('FULL_SAVED_ROUND_REQUIRED')
    result=classify(paths);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']))
