from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
from types import SimpleNamespace

import pytest


def test_remote_sampler_uses_other_gpu_and_preserves_real_tokens(tmp_path,monkeypatch):
    import torch
    selected=[];monkeypatch.setattr(torch.cuda,'set_device',lambda device:selected.append(device))
    from mechet.route_global_sampler import RemoteSampler,serve_gpu
    from mechet.route_work_broker import WorkBroker
    b=WorkBroker([],actor_version='v1',contract_sha='c',journal_path=tmp_path/'journal')
    class Sampler:
        cfg={'generation_batch_size':4}
        model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=torch.device('cuda:3'))]))
        def generate(self,requests):
            assert selected==[torch.device('cuda:3')]
            return [dict(request_id=r.request_id,behavior_version='v1',prompt_token_ids=[1],
                generated_token_ids=[2,3],behavior_logprobs=[-.2,-.3],raw='real',truncated=False) for r in requests]
    from mechet.route_policy_episode import DecisionRequest
    r=DecisionRequest('cpu0/r','e','rx','C','v1',[],7,'C')
    worker=threading.Thread(target=serve_gpu,args=(b,Sampler(),3),daemon=True);worker.start()
    with RemoteSampler(b,'cpu0',{'generation_batch_size':4},prompt_ids=lambda r:[1]) as client:
        value=client.submit(r).result(timeout=5)
        assert value['generated_token_ids']==[2,3] and value['behavior_logprobs']==[-.2,-.3]
    b.stop();worker.join(5);assert not worker.is_alive()
    assert b.snapshot()['counts']['gpu3_requests']==1


def test_sampler_worker_exception_completes_request_with_error(tmp_path):
    from mechet.route_global_sampler import RemoteSampler,serve_gpu
    from mechet.route_work_broker import WorkBroker
    from mechet.route_policy_episode import DecisionRequest
    b=WorkBroker([],actor_version='v1',contract_sha='c',journal_path=tmp_path/'journal')
    class Broken:
        cfg={'generation_batch_size':4}
        def generate(self,requests):raise RuntimeError('backend failed')
    t=threading.Thread(target=serve_gpu,args=(b,Broken(),1),daemon=True);t.start()
    with RemoteSampler(b,'owner',{'generation_batch_size':4},prompt_ids=lambda r:[1]) as client:
        f=client.submit(DecisionRequest('r','e','rx','C','v1',[],7,'C'))
        with pytest.raises(RuntimeError,match='backend failed'):f.result(timeout=5)
    t.join(5);assert not t.is_alive()


def test_manager_transport_shares_coordinator_without_filesystem_rpc(tmp_path):
    from mechet.route_global_sampler import broker_server,connect_broker
    from mechet.route_work_broker import WorkBroker
    b=WorkBroker([{'task_id':'t','target':'C'}],actor_version='v1',contract_sha='c',journal_path=tmp_path/'j')
    server,address,key=broker_server(b,host='127.0.0.1')
    try:
        proxy=connect_broker(address,key)
        assert proxy.claim(2)['task']['task_id']=='t'
        assert b.snapshot()['unclaimed']==0
    finally:server.stop_event.set()
