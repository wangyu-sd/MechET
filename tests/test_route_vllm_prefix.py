"""Version-scoped cache reuse must not substitute a different conditioning."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest


def config():
    return dict(base_path='frozen-base',base_manifest='frozen-tokenizer-manifest',
        _sha='frozen-contract',policy_precision='bf16_autocast',public_observation_contract='native-tools-v1',
        max_context=32768,generation_batch_size=64,acceleration={'prefix_cache':True},
        vllm_runtime=dict(max_num_batched_tokens=8192,kv_cache_memory_bytes=32*1024**3,
                          gpu_memory_utilization=.5))


def test_full_engine_enables_aligned_prefix_cache_and_chunked_prefill():
    from scripts.serve_route_vllm import engine_kwargs
    args=engine_kwargs(config())
    assert args['enable_prefix_caching'] is True
    assert args['mamba_cache_mode']=='align' and args['enable_chunked_prefill'] is True


@pytest.mark.parametrize('field,value',[
    ('base_path','other-model'),('base_manifest','other-tokenizer'),
    ('public_observation_contract','other-tools'),('policy_precision','other-precision'),
    ('_sha','other-frozen-contract')])
def test_prefix_namespace_binds_actor_tokenizer_tools_and_precision(field,value):
    from mechet.route_vllm_cache import prefix_namespace
    cfg=config();a=prefix_namespace(cfg,'round0:sha')
    changed=deepcopy(cfg);changed[field]=value
    assert prefix_namespace(changed,'round0:sha')!=a
    assert prefix_namespace(cfg,'round1:sha')!=a
    assert prefix_namespace(cfg,'round0:other-sha')!=a
    assert prefix_namespace(cfg,'round0:sha')==a


def test_prefix_invalidation_requires_quiescent_engine_and_successful_reset():
    from mechet.route_vllm_cache import invalidate_prefix
    from mechet.route_vllm_protocol import AdapterLease
    class Engine:
        def __init__(self,success):self.success=success;self.calls=0
        async def reset_prefix_cache(self):self.calls+=1;return self.success
    lease=AdapterLease(2);lease.commit('old','sha');lease.begin('request','old')
    engine=Engine(True)
    with pytest.raises(ValueError,match='INFLIGHT'):asyncio.run(invalidate_prefix(engine,lease,True))
    assert engine.calls==0
    lease.finish('request')
    with pytest.raises(ValueError,match='PREFIX_RESET_FAILED'):
        asyncio.run(invalidate_prefix(Engine(False),lease,True))
    assert lease.version is None
    assert asyncio.run(invalidate_prefix(engine,lease,True)) is True


def test_software_fp16_is_explicit_and_never_changes_formal_bf16():
    from scripts.serve_route_vllm import engine_kwargs
    cfg=config();cfg['vllm_runtime']['diagnostic_dtype']='float16'
    with pytest.raises(ValueError,match='SOFTWARE_PROFILE'):engine_kwargs(cfg)
    cfg['vllm_runtime']['software_replay']=True
    args=engine_kwargs(cfg)
    assert args['dtype']==args['lora_dtype']=='float16'
    assert args['attention_config']=={'backend':'TRITON_ATTN'}
    assert engine_kwargs(config())['dtype']=='bfloat16'


def test_actual_engine_iteration_counters_do_not_use_submitted_futures():
    from mechet.route_vllm_metrics import NativeEngineMetrics
    m=NativeEngineMetrics();m.start_version('round:sha')
    scheduler=SimpleNamespace(num_running_reqs=5,num_waiting_reqs=3,kv_cache_usage=.25,
        prefix_cache_stats=SimpleNamespace(queries=100,hits=60,reset=False))
    request=SimpleNamespace(request_id='actual',prefill_time=.2,decode_time=.5,queued_time=.1,e2e_latency=.8)
    iteration=SimpleNamespace(num_generation_tokens=5,
        prompt_token_stats=SimpleNamespace(total=100,computed=40,cached_tokens=60),
        time_to_first_tokens_iter=[.2,.3],inter_token_latencies_iter=[.01,.02,.03],
        finished_requests=[request])
    m.record(scheduler,iteration)
    result=m.snapshot()
    assert result['actual_running_requests_peak']==5
    assert result['actual_decode_batch_peak']==3
    assert result['prefill_tokens_computed']==40 and result['prefill_tokens_cached']==60
    assert result['prefix_tokens_queried']==100 and result['prefix_tokens_hit']==60
    assert result['request_prefill_seconds']==pytest.approx(.2)
    assert result['request_decode_seconds']==pytest.approx(.5)
    assert m.take_finished('actual')['queued_time']==.1
    m.start_version('next:sha');assert m.snapshot()['prefill_tokens_cached']==0


def test_invalid_engine_statistics_are_rejected_without_fabricating_activity():
    from mechet.route_vllm_metrics import NativeEngineMetrics
    m=NativeEngineMetrics()
    with pytest.raises(ValueError):m.record(SimpleNamespace(num_running_reqs=-1),None)


def test_native_prefix_completion_requires_observed_counter_not_inferred():
    from mechet.route_vllm_protocol import build_request,validate_completion,PROBABILITY_RECEIPT
    cfg=dict(config(),seed=17,max_action_tokens=7,eos_token_id=9)
    request=SimpleNamespace(request_id='r',behavior_version='round:sha',remaining_tokens=7)
    wire=build_request(request,[1,2,3],cfg)
    value=dict(request_id='r',behavior_version='round:sha',prompt_token_ids=[1,2,3],
        eos_token_id=9,generated_token_ids=[4,9],behavior_logprobs=[-.2,-.3],finish_reason='stop',
        prefix_namespace=wire['prefix_namespace'],probability_receipt=PROBABILITY_RECEIPT,
        engine_telemetry={'num_cached_tokens':0,'cache_counter_observed':False})
    with pytest.raises(ValueError,match='CACHE_RECEIPT'):validate_completion(value,wire)
    value['engine_telemetry']['cache_counter_observed']=True
    assert validate_completion(value,wire)['behavior_logprobs']==[-.2,-.3]
