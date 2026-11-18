"""Post-collection receipt accounting; never consulted by sampling or credit."""
from collections import Counter,defaultdict
import json
from pathlib import Path

from .external_route_data import signature


def restore_finished_request_labels(broker,version):
    """Finished leases skip CPU replay; join their authenticated scored receipts."""
    for finished in broker.results():
        reports=json.loads(Path(finished['result_path']).read_text())
        broker.register_request_labels(version+'/'+finished['task_id'],request_labels(reports))


def request_labels(roots):
    labels={}
    for root in roots:
        for report in [root,*root.get('refit',{}).get('suffix_reports',[])]:
            events={e['request_id']:e for e in report.get('refit_path',[])
                if e.get('receipt_owner')==report.get('continuation_id')}
            for record in report['records']:
                uid=record['request_id']
                if uid in labels:continue  # copied prefix retains its producer
                event=events.get(uid,{})
                node=event.get('node_id');edge=event.get('action')
                if node is None and record.get('prompt_token_ids'):
                    node=signature([record.get('behavior_version'),record['prompt_token_ids']])
                    from .route_refit import grounded_action_key
                    if record.get('state_before'):
                        edge=grounded_action_key(record['state_before'],record.get('raw',''))
                labels[uid]=dict(lane=event.get('lane',report.get('refit_lane','original_g4')),
                    node_id=node,semantic_edge=edge,accepted=record.get('accepted'))
    return labels


def summarize_requests(requests):
    groups=defaultdict(list);unlabelled=0
    for uid,entry in requests.items():
        label=entry.get('accounting_label')
        if label is None:unlabelled+=1;continue
        value=entry['result'] or {}
        gpu=entry['gpu']
        if gpu is None:gpu=value.get('engine_telemetry',{}).get('physical_gpu_rank','unknown_recovered')
        row=(uid,label,entry,value)
        groups['per_gpu',str(gpu)].append(row);groups['per_target',entry['owner']].append(row)
    result=dict(per_gpu={},per_target={},unlabelled_requests=unlabelled)
    for (kind,key),rows in groups.items():
        duplicate_status=Counter();seen=set()
        for _,label,_,_ in rows:
            edge=(label['node_id'],label['semantic_edge'])
            if None in edge:continue
            if edge in seen:duplicate_status['accepted' if label['accepted'] is True else
                'rejected' if label['accepted'] is False else 'unexecuted']+=1
            seen.add(edge)
        edges=Counter((r[1]['node_id'],r[1]['semantic_edge']) for r in rows
            if r[1]['node_id'] is not None and r[1]['semantic_edge'] is not None)
        result[kind][key]=dict(requests=len(rows),lanes=dict(Counter(r[1]['lane'] for r in rows)),
            completed=sum(r[2]['result'] is not None for r in rows),
            recovered=sum(r[2]['gpu'] is None for r in rows),
            generated_tokens=sum(len(r[3].get('generated_token_ids',[])) for r in rows),
            distinct_decision_nodes=len({r[1]['node_id'] for r in rows if r[1]['node_id'] is not None}),
            distinct_semantic_edges=len(edges),duplicate_semantic_proposals=sum(n-1 for n in edges.values()),
            accepted_duplicate_proposals=duplicate_status['accepted'],
            rejected_duplicate_proposals=duplicate_status['rejected'],
            unexecuted_duplicate_proposals=duplicate_status['unexecuted'])
    return result
