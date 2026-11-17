from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest


def broker(tmp_path,rows=None):
    from mechet.route_work_broker import WorkBroker
    return WorkBroker(rows or [{'task_id':str(i),'target':'C'} for i in range(8)],
        actor_version='v1',contract_sha='c1',journal_path=tmp_path/'claims.jsonl')


def test_dynamic_claims_are_unique_and_completed_results_recover(tmp_path):
    from mechet.external_route_data import file_sha
    from mechet.route_work_broker import WorkBroker
    b=broker(tmp_path);claimed=[]
    def worker(rank):
        while True:
            job=b.claim(rank)
            if job is None:return
            claimed.append(job['task']['task_id'])
            p=tmp_path/('result-'+job['task']['task_id']);p.write_text(job['task']['task_id'])
            b.finish(job['lease_id'],rank,str(p),file_sha(p))
    with ThreadPoolExecutor(4) as pool:list(pool.map(worker,range(4)))
    assert sorted(claimed)==list(map(str,range(8))) and b.snapshot()['complete']==8
    recovered=broker(tmp_path);assert recovered.claim(0) is None
    assert len(recovered.results())==8


def test_lease_owner_and_actor_binding_cannot_be_spoofed(tmp_path):
    b=broker(tmp_path);j=b.claim(0)
    with pytest.raises(ValueError,match='LEASE'):b.finish(j['lease_id'],1,'bad','bad')
    r=dict(request_id='r',behavior_version='v1',messages=[],remaining_tokens=7)
    b.submit(r,'owner',prompt_binding='p')
    with pytest.raises(ValueError):b.submit(dict(r,behavior_version='v2'),'owner',prompt_binding='p')
    with pytest.raises(ValueError):b.submit(dict(r,remaining_tokens=8),'owner',prompt_binding='p')


def test_global_requests_keep_original_identity_and_route_back_to_cpu_owner(tmp_path):
    b=broker(tmp_path)
    r=dict(request_id='cpu0/request',behavior_version='v1',remaining_tokens=7)
    b.submit(r,'cpu0',prompt_binding='p');batch=b.take(3,64,.01)
    assert batch[0]['request']==r
    result=dict(request_id=r['request_id'],behavior_version='v1',prompt_token_ids=[1],
                generated_token_ids=[2,3],behavior_logprobs=[-.2,-.4],truncated=False)
    b.complete(3,r['request_id'],result)
    assert b.wait_result('cpu0',r['request_id'],.1)==result
    assert b.wait_result('cpu0',r['request_id'],.1)==result
    with pytest.raises(ValueError):b.wait_result('cpu1',r['request_id'],.1)
    with pytest.raises(ValueError):b.complete(2,r['request_id'],result)


def test_inflight_target_is_reclaimed_only_by_new_broker_epoch(tmp_path):
    b=broker(tmp_path);j=b.claim(0)
    other=broker(tmp_path);j2=other.claim(1)
    assert j2['task']==j['task'] and j2['lease_id']!=j['lease_id']


def test_worker_failure_unblocks_every_waiter(tmp_path):
    b=broker(tmp_path);r=dict(request_id='r',behavior_version='v1')
    b.submit(r,'cpu0',prompt_binding='p');b.fail('worker exploded')
    with pytest.raises(RuntimeError,match='exploded'):b.wait_result('cpu0','r',.1)
    with pytest.raises(RuntimeError,match='exploded'):b.claim(0)


def test_completed_request_can_be_replayed_but_never_rebound(tmp_path):
    b=broker(tmp_path);r=dict(request_id='r',behavior_version='v1')
    v=dict(request_id='r',behavior_version='v1',prompt_token_ids=[1],generated_token_ids=[2],behavior_logprobs=[-.2])
    b.seed_result(r,v)
    b.submit(r,'cpu1',prompt_binding='p')
    assert b.take(2,8,.01)==[] and b.wait_result('cpu1','r',.1)==v
    with pytest.raises(ValueError):b.submit(dict(r,product='other'),'cpu1',prompt_binding='p')


def test_lost_gpu_request_times_out_as_infrastructure_not_chemical_failure(tmp_path):
    import time
    from mechet.route_work_broker import WorkBroker
    b=WorkBroker([],actor_version='v1',contract_sha='c1',journal_path=tmp_path/'j',request_timeout=.01)
    b.submit(dict(request_id='r',behavior_version='v1'),'cpu',prompt_binding='p')
    b.take(2,1,0);time.sleep(.02)
    with pytest.raises(RuntimeError,match='GPU_REQUEST_DEADLINE'):b.wait_result('cpu','r',.1)
    assert b.snapshot()['counts'].get('completed_requests',0)==0
