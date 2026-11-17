"""The global dispatcher must serve and recover original native engine draws."""
from concurrent.futures import Future
from dataclasses import asdict
import json
import threading
from types import SimpleNamespace

import pytest


def test_full_native_condition_is_accepted_without_enabling_hf_fallback():
    from mechet.route_acceleration import full_acceleration,validate_acceleration
    cfg=dict(acceleration=full_acceleration(refit=False),inference_backend='vllm',
             world_size=4,candidates=4)
    validate_acceleration(cfg)
    assert cfg['acceleration']['sampling_kernel']=='vllm_native_v1'
    with pytest.raises(ValueError):validate_acceleration(dict(cfg,inference_backend='hf'))


def test_global_native_recovery_reuses_exact_original_receipt(tmp_path):
    from mechet.route_gpu_backend import recover_gpu_journal
    from mechet.route_policy_episode import DecisionRequest
    from mechet.route_vllm_protocol import build_request,PROBABILITY_RECEIPT
    from mechet.route_policy_sampling import append_receipt
    request=DecisionRequest('real-request','episode','reaction','CO','round:actor',[],7,'CO')
    wire=build_request(request,[3,4],dict(seed=17,max_action_tokens=7,max_context=32,eos_token_id=9))
    result=dict(request_id='real-request',behavior_version='round:actor',prompt_token_ids=[3,4],
        eos_token_id=9,generated_token_ids=[5,9],behavior_logprobs=[-.123,-.456],
        probability_receipt=PROBABILITY_RECEIPT,finish_reason='stop',raw='native')
    path=tmp_path/'native.jsonl'
    append_receipt(path,dict(event='vllm-submitted',request=wire,decision_request=asdict(request)))
    append_receipt(path,dict(event='vllm-completed',request=wire,receipt=result))
    def forbidden(*_):raise AssertionError('Native recovery recomputed behavior probabilities')
    rows=recover_gpu_journal(path,'round:actor',{'inference_backend':'vllm'},forbidden)
    assert rows[0][0]==asdict(request)
    assert rows[0][1]['generated_token_ids']==[5,9]
    assert rows[0][1]['behavior_logprobs']==[-.123,-.456]
    with pytest.raises(ValueError):recover_gpu_journal(path,'round:other',{'inference_backend':'vllm'},forbidden)


def test_global_native_eos_refills_before_long_neighbor_finishes(tmp_path,monkeypatch):
    from mechet.route_gpu_backend import make_gpu_sampler,close_gpu_sampler
    from mechet.route_global_sampler import RemoteSampler,serve_gpu
    from mechet.route_policy_episode import DecisionRequest
    from mechet.route_work_broker import WorkBroker
    from mechet.route_vllm_protocol import PROBABILITY_RECEIPT
    import mechet.route_vllm_backend as backend
    import mechet.route_policy_protocol as protocol
    monkeypatch.setattr(protocol,'route_tool_prefix',lambda *args:'native-prompt')
    bound=[];monkeypatch.setattr(backend,'bind_rollout_adapter',lambda cfg,version:bound.append(version))
    class Tokenizer:
        eos_token_id=9
        def __call__(self,*args,**kwargs):return {'input_ids':[1,2]}
        def decode(self,ids,**kwargs):return 'generated'
    class Engine:
        timeout=5
        def __init__(self):self.pending={};self.cv=threading.Condition();self.drained=False
        def rpc(self,operation,wire):
            f=Future()
            if operation=='drain':self.drained=True;f.set_result({'inflight':0});return f
            assert operation=='generate'
            with self.cv:self.pending[wire['request_id']]=(wire,f);self.cv.notify_all()
            return f
        def submitted(self,n):
            with self.cv:return self.cv.wait_for(lambda:len(self.pending)>=n,timeout=5)
        def finish(self,uid):
            wire,f=self.pending[uid]
            f.set_result(dict(request_id=uid,behavior_version='round:actor',prompt_token_ids=[1,2],
                eos_token_id=9,generated_token_ids=[7,9],behavior_logprobs=[-.2,-.3],
                probability_receipt=PROBABILITY_RECEIPT,finish_reason='stop'))
    engine=Engine();cfg=dict(inference_backend='vllm',_vllm_process=engine,generation_batch_size=2,
        seed=17,max_action_tokens=8,max_context=32,_sha='contract')
    sampler=make_gpu_sampler(None,Tokenizer(),cfg,tmp_path,'round:actor',0)
    assert bound==['round:actor']
    broker=WorkBroker([],actor_version='round:actor',contract_sha='contract',journal_path=tmp_path/'claims')
    worker=threading.Thread(target=serve_gpu,args=(broker,sampler,0),daemon=True);worker.start()
    def request(uid):return DecisionRequest(uid,uid,'reaction','CO','round:actor',[],8,'CO')
    try:
        with RemoteSampler(broker,'owner',cfg,prompt_ids=lambda r:[1,2]) as owner:
            long=owner.submit(request('long'));short=owner.submit(request('short'))
            assert engine.submitted(2)
            engine.finish('short');assert short.result(timeout=5)['generated_token_ids']==[7,9]
            third=owner.submit(request('third'));assert engine.submitted(3)
            assert not long.done(), 'A completed row must refill while its long neighbor is pending'
            engine.finish('third');assert third.result(timeout=5)['behavior_logprobs']==[-.2,-.3]
            engine.finish('long');assert long.result(timeout=5)['generated_token_ids']==[7,9]
    finally:
        broker.stop();worker.join(5);close_gpu_sampler(sampler,cfg)
    assert not worker.is_alive() and engine.drained
    # Durable evidence retains each original request, including full replay identity.
    events=[json.loads(line) for line in (tmp_path/'gpu-rank00.jsonl').read_text().splitlines()]
    submitted=[e for e in events if e['event']=='vllm-submitted']
    assert {e['decision_request']['request_id'] for e in submitted}=={'long','short','third'}
    assert sum(e['event']=='vllm-completed' for e in events)==3


def test_native_sampler_accepts_saved_namespace_request_and_preserves_identity(tmp_path,monkeypatch):
    from mechet.route_vllm_backend import VllmRouteSampler
    from mechet.route_vllm_protocol import PROBABILITY_RECEIPT
    import mechet.route_policy_protocol as protocol
    monkeypatch.setattr(protocol,'route_tool_prefix',lambda *args:'prompt')
    class Tokenizer:
        eos_token_id=9
        def __call__(self,*a,**kw):return {'input_ids':[1,2]}
        def decode(self,*a,**kw):return 'native'
    class Process:
        timeout=5
        def rpc(self,op,wire):
            assert op=='generate'
            f=Future();f.set_result(dict(request_id=wire['request_id'],behavior_version='round:actor',
                prompt_token_ids=[1,2],eos_token_id=9,generated_token_ids=[7,9],
                behavior_logprobs=[-.2,-.3],probability_receipt=PROBABILITY_RECEIPT,finish_reason='stop'))
            return f
    cfg=dict(seed=17,max_action_tokens=8,max_context=32,generation_batch_size=2)
    request=SimpleNamespace(request_id='saved',episode_id='ep',reaction_id='rx',product='CO',
        public_product='CO',behavior_version='round:actor',messages=[],remaining_tokens=8,tools=None)
    sampler=VllmRouteSampler(Process(),Tokenizer(),cfg,tmp_path/'raw.jsonl')
    try:assert sampler.submit(request).result(timeout=5)['behavior_logprobs']==[-.2,-.3]
    finally:sampler.close()
    event=json.loads((tmp_path/'raw.jsonl').read_text().splitlines()[0])
    assert event['decision_request']==vars(request)
