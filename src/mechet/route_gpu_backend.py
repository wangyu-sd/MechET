"""GPU backend lifecycle and exact durable recovery for the global dispatcher.

Only the selected backend is constructed. Native probabilities always come from
the original engine completion; interrupted native draws are never regenerated.
"""
from pathlib import Path


def make_gpu_sampler(actor,tokenizer,cfg,directory,version,rank):
    directory=Path(directory);path=directory/f'gpu-rank{rank:02d}.jsonl'
    if cfg.get('inference_backend','hf')=='vllm':
        from .route_vllm_backend import bind_rollout_adapter,VllmRouteSampler
        bind_rollout_adapter(cfg,version)
        return VllmRouteSampler(cfg['_vllm_process'],tokenizer,cfg,path)
    from .external_route_io import ReceiptBatch,BatchedNativeRouteSampler
    writer=ReceiptBatch(directory/f'gpu-prefix-rank{rank:02d}.jsonl',cfg)
    return BatchedNativeRouteSampler(actor,tokenizer,cfg,path,writer)


def close_gpu_sampler(sampler,cfg):
    if cfg.get('inference_backend','hf')=='vllm':
        sampler.close()
        cfg['_last_native_journal_metrics']=sampler.writer.metrics()
        process=cfg['_vllm_process']
        result=process.rpc('drain',{}).result(timeout=process.timeout)
        if result.get('inflight')!=0:raise ValueError('GLOBAL_NATIVE_REQUESTS_NOT_DRAINED')
        if 'engine_metrics' in result:cfg['_last_native_engine_metrics']=result['engine_metrics']
    else:
        sampler.writer.flush();sampler.execution_writer.flush()


def recover_gpu_journal(path,version,cfg,recompute):
    from .route_policy_sampling import read_owned_journal,recover_behavior_receipts
    events=read_owned_journal(path);requests={}
    native=cfg.get('inference_backend','hf')=='vllm'
    if native:
        from .route_vllm_backend import recover_native_receipts
        recovered=recover_native_receipts(path,version,events=events)
        for event in events:
            if event['event']=='vllm-submitted':
                decision=event.get('decision_request');wire=event['request']
                if (not isinstance(decision,dict) or decision.get('request_id')!=wire['request_id']
                        or decision.get('behavior_version')!=version):
                    raise ValueError('GLOBAL_NATIVE_RECOVERY_DECISION_BINDING')
                prior=requests.get(wire['request_id'])
                if prior is not None and prior!=decision:
                    raise ValueError('GLOBAL_NATIVE_RECOVERY_DECISION_REBOUND')
                requests[wire['request_id']]=decision
    else:
        recovered=recover_behavior_receipts(path,version,recompute)
        for event in events:
            if event['event']=='gpu-requested':
                for request in event['requests']:requests[request['request_id']]=request
    if set(recovered)-set(requests):raise ValueError('GLOBAL_RECOVERY_MISSING_ORIGINAL_REQUEST')
    return [(requests[uid],value) for uid,value in recovered.items()]
