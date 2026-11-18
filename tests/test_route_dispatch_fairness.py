"""Target fairness and measured work admission without changing policy draws."""
import pytest


def broker(tmp_path,**changes):
    from mechet.route_work_broker import WorkBroker
    dispatch=dict(max_prefill_tokens=100,max_estimated_tokens=1000,max_pending=100,gpu_count=1)
    dispatch.update(changes)
    return WorkBroker([],actor_version='v',contract_sha='c',journal_path=tmp_path/'claims',dispatch=dispatch)


def submit(b,uid,owner,prompt=3,output=2):
    b.submit(dict(request_id=uid,behavior_version='v',remaining_tokens=output),owner,'binding',
             work=dict(prompt_tokens=prompt,max_new_tokens=output))


def test_target_fairness_does_not_let_many_a_requests_starve_b(tmp_path):
    b=broker(tmp_path)
    for i in range(5):submit(b,'A'+str(i),'targetA')
    submit(b,'B','targetB');submit(b,'C','targetC')
    batch=b.take(0,4,0)
    assert [x['request']['request_id'] for x in batch]==['A0','B','C','A1']
    assert b.snapshot()['counts']['policy_requests_submitted']==7


def test_prefill_admission_preserves_long_request_and_refunds_kv_estimate(tmp_path):
    b=broker(tmp_path,max_prefill_tokens=8,max_estimated_tokens=20)
    submit(b,'A','a',prompt=6,output=4);submit(b,'B','b',prompt=6,output=4)
    first=b.take(0,4,0)
    assert [x['request']['request_id'] for x in first]==['A']
    second=b.take(0,4,0)
    assert [x['request']['request_id'] for x in second]==['B']
    assert b.take(0,4,0)==[]
    b.complete(0,'A',dict(request_id='A',behavior_version='v',prompt_token_ids=[1],
        generated_token_ids=[9],behavior_logprobs=[-.2]))
    # An individual long prefill is chunkable by the native engine, not rejected
    # merely because its prompt exceeds the dispatch prefill quantum.
    submit(b,'long','long',prompt=9,output=1)
    assert b.take(0,4,0)[0]['request']['request_id']=='long'


def test_gpu_with_lower_reserved_work_receives_next_request(tmp_path):
    b=broker(tmp_path,gpu_count=2)
    submit(b,'A','a',prompt=8,output=2)
    assert len(b.take(0,1,0))==1
    submit(b,'B','b',prompt=1,output=1)
    assert b.take(0,1,0)==[]
    assert b.take(1,1,0)[0]['request']['request_id']=='B'


def test_work_identity_and_oversized_dispatch_are_explicit(tmp_path):
    b=broker(tmp_path,max_estimated_tokens=10)
    submit(b,'A','a')
    with pytest.raises(ValueError,match='WORK_REBOUND'):submit(b,'A','a',prompt=4)
    with pytest.raises(ValueError,match='DISPATCH_CAPACITY'):submit(b,'large','large',prompt=20)


def test_recovered_draw_is_not_reserved_or_resubmitted_as_new_gpu_work(tmp_path):
    b=broker(tmp_path)
    request=dict(request_id='saved',behavior_version='v',remaining_tokens=2)
    value=dict(request_id='saved',behavior_version='v',prompt_token_ids=[1],
               generated_token_ids=[9],behavior_logprobs=[-.2])
    b.seed_result(request,value)
    b.submit(request,'owner','binding',work=dict(prompt_tokens=1,max_new_tokens=2))
    assert b.take(0,10,0)==[]
    assert b.snapshot()['estimated_gpu_tokens']=={0:0}
    assert b.wait_result('owner','saved',0)['behavior_logprobs']==[-.2]
    assert b.snapshot()['counts']['recovered_request_receipts']==1


def test_overlong_request_does_not_censor_short_neighbor(tmp_path):
    from mechet.route_global_sampler import RemoteSampler
    from mechet.route_policy_episode import DecisionRequest
    b=broker(tmp_path)
    cfg=dict(acceleration={'scheduling':'global_length_fair_v2'},max_action_tokens=8,max_context=5)
    def request(uid):return DecisionRequest(uid,uid,'rx','CO','v',[],8,'CO')
    with RemoteSampler(b,'owner',cfg,prompt_ids=lambda r:[1]*(6 if r.request_id=='long' else 2)) as sampler:
        long=sampler.submit(request('long'));short=sampler.submit(request('short'))
        assert long.result(timeout=2)['resource_error']=='PHYSICAL_CONTEXT_LIMIT'
        assert b.take(0,2,0)[0]['request']['request_id']=='short'
        b.complete(0,'short',dict(request_id='short',behavior_version='v',prompt_token_ids=[1,1],
            generated_token_ids=[9],behavior_logprobs=[-.2]))
        assert short.result(timeout=2)['generated_token_ids']==[9]
    assert b.snapshot()['counts']['policy_requests_submitted']==1
