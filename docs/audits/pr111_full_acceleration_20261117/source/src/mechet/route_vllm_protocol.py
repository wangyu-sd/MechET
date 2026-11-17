"""Strict native-rollout receipts independent of either Torch runtime."""
import math

from .route_policy_sampling import request_seed


PROBABILITY_RECEIPT = 'vllm_actual_sampled_distribution_v1'


def validate_engine_build(distribution_version,module_version,torch_version,cuda_version):
    # CUDA flavor is wheel/distribution metadata; the public module may expose
    # only the upstream release. Check both, without confusing the two formats.
    if (distribution_version!='0.24.0+cu129'
            or module_version not in ('0.24.0','0.24.0+cu129')
            or torch_version!='2.11.0+cu129' or cuda_version!='12.9'):
        raise ValueError('VLLM_BUILD_VERSION_MISMATCH')


class AdapterLease:
    def __init__(self,capacity):
        self.capacity=capacity;self.version=None;self.sha=None;self.inflight=set();self.seen=set()

    def invalidate(self):
        if self.inflight:raise ValueError('VLLM_ADAPTER_SWITCH_WHILE_INFLIGHT')
        self.version=self.sha=None

    def commit(self,version,sha):
        if self.inflight:raise ValueError('VLLM_ADAPTER_SWITCH_WHILE_INFLIGHT')
        self.version,self.sha=version,sha;self.seen.clear()

    def begin(self,uid,version):
        if self.version is None or version!=self.version:raise ValueError('VLLM_WEIGHT_VERSION_MISMATCH')
        if uid in self.seen:raise ValueError('VLLM_DUPLICATE_REQUEST_ID')
        if len(self.inflight)>=self.capacity:raise ValueError('VLLM_INFLIGHT_CAPACITY')
        self.seen.add(uid);self.inflight.add(uid)

    def finish(self,uid):self.inflight.remove(uid)


def build_request(request, prompt_token_ids, cfg):
    ids=list(prompt_token_ids)
    if not ids or any(type(v) is not int or v < 0 for v in ids):
        raise ValueError('VLLM_PROMPT_IDS_INVALID')
    room=min(request.remaining_tokens,cfg['max_action_tokens'],cfg['max_context']-len(ids))
    if room <= 0:
        raise ValueError('VLLM_CONTEXT_CAPACITY')
    eos=cfg['eos_token_id']
    if type(eos) is not int or eos < 0:
        raise ValueError('VLLM_EOS_ID_INVALID')
    value=dict(request_id=request.request_id,behavior_version=request.behavior_version,
        prompt_token_ids=ids,max_new_tokens=int(room),eos_token_id=eos,
        seed=request_seed(request.request_id,cfg['seed']),
        sampling=dict(temperature=1.,top_p=1.,top_k=-1,repetition_penalty=1.,
            presence_penalty=0.,frequency_penalty=0.,logprobs=0,
            ignore_eos=True,stop_token_ids=[eos],detokenize=False))
    if cfg.get('acceleration',{}).get('prefix_cache'):
        from .route_vllm_cache import prefix_namespace
        value['prefix_namespace']=prefix_namespace(cfg,request.behavior_version)
    return value


def validate_completion(value, request):
    for name in ('request_id','behavior_version','prompt_token_ids','eos_token_id'):
        if value.get(name)!=request[name]:
            raise ValueError('VLLM_COMPLETION_BINDING:'+name)
    if 'prefix_namespace' in request:
        if value.get('prefix_namespace')!=request['prefix_namespace']:
            raise ValueError('VLLM_COMPLETION_PREFIX_BINDING')
        telemetry=value.get('engine_telemetry',{})
        cached=telemetry.get('num_cached_tokens')
        if (telemetry.get('cache_counter_observed') is not True
                or type(cached) is not int or not 0<=cached<=len(request['prompt_token_ids'])):
            raise ValueError('VLLM_COMPLETION_CACHE_RECEIPT')
    ids=value.get('generated_token_ids',[])
    if not ids or any(type(v) is not int or v < 0 for v in ids) or len(ids)>request['max_new_tokens']:
        raise ValueError('VLLM_GENERATED_IDS_INVALID')
    scores=value.get('behavior_logprobs')
    if (value.get('probability_receipt')!=PROBABILITY_RECEIPT or not isinstance(scores,list)
            or len(scores)!=len(ids)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in scores)):
        raise ValueError('VLLM_PROBABILITY_RECEIPT')
    eos=request['eos_token_id']
    reason=value.get('finish_reason')
    if reason not in ('stop','length'):
        raise ValueError('VLLM_FINISH_REASON_INVALID')
    if (eos in ids and (ids[-1]!=eos or ids.count(eos)!=1)) or (reason=='stop' and ids[-1]!=eos):
        raise ValueError('VLLM_EOS_RECEIPT')
    return dict(value,truncated=ids[-1]!=eos)


def validate_probe(value,version,count,tolerance):
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or tolerance<=0:
        raise ValueError('VLLM_PROBE_TOLERANCE')
    if value.get('probe_only') is not True or value.get('behavior_version')!=version:
        raise ValueError('VLLM_PROBE_BINDING')
    if type(count) is not int or count<1:raise ValueError('VLLM_PROBE_PROBABILITY_LENGTH')
    for name in ('base','adapter'):
        scores=value.get(name)
        if (not isinstance(scores,list) or len(scores)!=count
                or any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in scores)):
            raise ValueError('VLLM_PROBE_PROBABILITY_RECEIPT:'+name)
    return value
