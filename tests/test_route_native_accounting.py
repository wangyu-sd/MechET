import json

import pytest


def test_physical_lane_counts_reconcile_without_copied_prefix(tmp_path):
    from mechet.route_work_broker import WorkBroker
    b=WorkBroker([],actor_version='v',contract_sha='c',journal_path=tmp_path/'claims')
    for uid in ('a','b','c'):
        b.submit(dict(request_id=uid,behavior_version='v'),'target','binding')
    for row in b.take(0,3,0):
        uid=row['request']['request_id']
        b.complete(0,uid,dict(request_id=uid,behavior_version='v',generated_token_ids=[2],behavior_logprobs=[-.2]))
    labels={uid:dict(lane=lane,node_id='node',semantic_edge='edge',accepted=True)
        for uid,lane in zip(('a','b','c'),('original_g4','original_g4','credit'))}
    b.register_request_labels('target',labels)
    s=b.snapshot()['request_accounting']
    assert s['unlabelled_requests']==0
    assert s['per_gpu']['0']['lanes']=={'original_g4':2,'credit':1}
    assert s['per_target']['target']['distinct_decision_nodes']==1
    assert s['per_target']['target']['duplicate_semantic_proposals']==2
    assert s['per_target']['target']['accepted_duplicate_proposals']==2
    assert s['per_target']['target']['rejected_duplicate_proposals']==0
    with pytest.raises(ValueError):b.register_request_labels('wrong',labels)


def test_labels_keep_original_receipt_ownership_in_suffixes():
    from mechet.route_request_accounting import request_labels
    original=dict(records=[dict(request_id='a',accepted=True)],continuation_id='root',
        refit_path=[dict(request_id='a',receipt_owner='root',lane='original_g4',node_id='n',action='e')])
    suffix=dict(records=[dict(request_id='a',accepted=True),dict(request_id='b',accepted=False)],
        continuation_id='fork',refit_path=[*original['refit_path'],
            dict(request_id='b',receipt_owner='fork',lane='credit',node_id='n',action='e')])
    original['refit']={'suffix_reports':[suffix]}
    labels=request_labels([original])
    assert labels['a']['lane']=='original_g4'
    assert labels['b']['lane']=='credit'
    assert len(labels)==2


def test_native_durable_io_metrics_are_actual_bytes_and_fsync_barriers(tmp_path):
    from mechet.route_vllm_backend import DurableGroupCommit
    path=tmp_path/'journal'
    with DurableGroupCommit(path,delay=.01) as writer:
        futures=[writer.append({'receipt':i}) for i in range(3)]
        for f in futures:f.result(timeout=5)
    metrics=writer.metrics()
    assert metrics['rows']==3 and metrics['bytes']==len(path.read_bytes())
    assert 1<=metrics['fsync_barriers']<=3
    assert metrics['fsync_seconds']>=0 and metrics['write_seconds']>=0
    assert [json.loads(line) for line in path.read_text().splitlines()]==[{'receipt':i} for i in range(3)]


def test_finished_target_recovery_restores_physical_lane_labels(tmp_path):
    from mechet.route_work_broker import WorkBroker
    from mechet.route_request_accounting import restore_finished_request_labels
    from mechet.external_route_data import file_sha
    path=tmp_path/'scored.json'
    path.write_text(json.dumps([dict(records=[dict(request_id='a',accepted=True,
        prompt_token_ids=[1],state_before='[CH4:1]',raw='')])]))
    args=dict(actor_version='v',contract_sha='c',journal_path=tmp_path/'claims')
    first=WorkBroker([dict(task_id='target')],**args)
    lease=first.claim(0);first.finish(lease['lease_id'],0,str(path),file_sha(path))
    restarted=WorkBroker([dict(task_id='target')],**args)
    restarted.seed_result(dict(request_id='a',behavior_version='v'),dict(request_id='a',behavior_version='v',
        generated_token_ids=[2],behavior_logprobs=[-.1],engine_telemetry={'physical_gpu_rank':1}))
    restore_finished_request_labels(restarted,'v')
    metrics=restarted.snapshot()['request_accounting']
    assert metrics['unlabelled_requests']==0
    assert metrics['per_target']['v/target']['recovered']==1
    assert metrics['per_gpu']['1']['lanes']=={'original_g4':1}
