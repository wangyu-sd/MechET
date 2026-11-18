"""Persistent isolated native-ID vLLM server; JSONL protocol on original stdout."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time


def engine_kwargs(cfg):
    """Eager dispatch with Triton GDN; no torch.compile or graph capture."""
    runtime=cfg['vllm_runtime']
    if runtime.get('execution_profile','eager_triton_v1')!='eager_triton_v1':
        raise ValueError('VLLM_UNRECOGNIZED_EXECUTION_PROFILE')
    dtype=runtime.get('diagnostic_dtype','bfloat16')
    if dtype!='bfloat16' and (not runtime.get('software_replay') or dtype!='float16'):
        raise ValueError('VLLM_EXPLICIT_SOFTWARE_PROFILE_REQUIRED')
    prefix=cfg.get('acceleration',{}).get('prefix_cache',False)
    args=dict(model=cfg['base_path'],dtype=dtype,tensor_parallel_size=1,
        max_model_len=cfg['max_context'],max_num_seqs=cfg['generation_batch_size'],
        max_num_batched_tokens=runtime['max_num_batched_tokens'],
        kv_cache_memory_bytes=runtime['kv_cache_memory_bytes'],
        gpu_memory_utilization=runtime['gpu_memory_utilization'],
        enable_lora=True,max_lora_rank=16,max_loras=1,max_cpu_loras=1,lora_dtype=dtype,
        generation_config='vllm',logprobs_mode='processed_logprobs',
        language_model_only=True,enable_prefix_caching=prefix,
        enforce_eager=True,additional_config={'gdn_prefill_backend':'triton'},
        worker_extension_cls='scripts.route_vllm_worker.RouteVllmAuditWorker')
    if prefix:args.update(mamba_cache_mode='align',enable_chunked_prefill=True)
    if dtype=='float16':args['attention_config']={'backend':'TRITON_ATTN'}
    return args


async def serve(cfg,output):
    import vllm
    import torch
    from importlib.metadata import version
    from mechet.route_vllm_protocol import validate_engine_build
    validate_engine_build(version('vllm'),vllm.__version__,torch.__version__,torch.version.cuda)
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.metrics.loggers import StatLoggerBase
    from mechet.route_vllm_metrics import NativeEngineMetrics
    from mechet.route_vllm_cache import prefix_namespace,invalidate_prefix
    from mechet.route_vllm_protocol import (PROBABILITY_RECEIPT,validate_completion,
                                           validate_probe,AdapterLease)
    from mechet.route_vllm_adapter import sha256
    runtime=cfg['vllm_runtime']
    metrics=NativeEngineMetrics()
    class Logger(StatLoggerBase):
        def __init__(self,vllm_config,engine_index=0):self.engine_index=engine_index
        def record(self,scheduler_stats,iteration_stats,mm_cache_stats=None,engine_idx=0):
            metrics.record(scheduler_stats,iteration_stats)
        def log_engine_initialized(self):pass
        def log(self):pass
    args=AsyncEngineArgs(**engine_kwargs(cfg))
    engine=AsyncLLM.from_engine_args(args,stat_loggers=[Logger])
    adapter=None;counter=0;lease=AdapterLease(cfg['generation_batch_size'])
    output_lock=asyncio.Lock()
    async def reply(uid,*,result=None,error=None):
        value=dict(rpc_id=uid)
        value.update(error=error) if error else value.update(result=result)
        def write():output.write(json.dumps(value)+'\n');output.flush()
        # The engine event loop continues processing input/decoding even when
        # the parent briefly applies backpressure to a large completion receipt.
        async with output_lock:await asyncio.to_thread(write)
    async def generate(message):
        wire=message['payload'];uid=message['rpc_id']
        began=time.monotonic()
        try:
            params=SamplingParams(**wire['sampling'],seed=wire['seed'],
                max_tokens=wire['max_new_tokens'],output_kind=RequestOutputKind.FINAL_ONLY)
            result=None
            prompt=dict(prompt_token_ids=wire['prompt_token_ids'])
            if cfg.get('acceleration',{}).get('prefix_cache'):
                if wire.get('prefix_namespace')!=prefix_namespace(cfg,wire['behavior_version']):
                    raise ValueError('VLLM_PREFIX_NAMESPACE_MISMATCH')
                prompt['cache_salt']=wire['prefix_namespace']
            async for result in engine.generate(prompt,
                    params,wire['request_id'],lora_request=adapter):pass
            if result is None or not result.finished or len(result.outputs)!=1:
                raise ValueError('VLLM_INCOMPLETE_OUTPUT')
            completion=result.outputs[0];ids=list(completion.token_ids)
            if completion.logprobs is None or len(ids)!=len(completion.logprobs):
                raise ValueError('VLLM_NATIVE_LOGPROBS_MISSING')
            logps=[float(row[token].logprob) for row,token in zip(completion.logprobs,ids)]
            value=dict(request_id=wire['request_id'],
                behavior_version=wire['behavior_version'],prompt_token_ids=wire['prompt_token_ids'],
                eos_token_id=wire['eos_token_id'],generated_token_ids=ids,behavior_logprobs=logps,
                probability_receipt=PROBABILITY_RECEIPT,finish_reason=completion.finish_reason)
            cached=getattr(result,'num_cached_tokens',None)
            timings=metrics.take_finished(wire['request_id'])
            value['engine_telemetry']=dict(num_cached_tokens=0 if cached is None else cached,
                cache_counter_observed=cached is not None,
                elapsed_seconds=time.monotonic()-began,request_timings=timings,
                physical_gpu_rank=cfg.get('physical_gpu_rank',0))
            if 'prefix_namespace' in wire:value['prefix_namespace']=wire['prefix_namespace']
            receipt=validate_completion(value,wire)
            await reply(uid,result=receipt)
        except Exception as exc:await reply(uid,error=type(exc).__name__+':'+str(exc))
        finally:lease.finish(wire['request_id'])
    tasks=set()
    while True:
        line=await asyncio.to_thread(sys.stdin.readline)
        if not line:break
        message=json.loads(line);uid=message['rpc_id'];op=message['operation'];payload=message['payload']
        try:
            if op=='generate':
                lease.begin(payload['request_id'],payload['behavior_version'])
                task=asyncio.create_task(generate(message));tasks.add(task);task.add_done_callback(tasks.discard)
                continue
            if op=='ready':result=dict(vllm_version=vllm.__version__,ready=True)
            elif op=='load_adapter':
                if lease.inflight:raise ValueError('VLLM_ADAPTER_SWITCH_WHILE_INFLIGHT')
                path=Path(payload['path']);mapping=json.loads((path/'mapping.json').read_text())
                if mapping['source_sha256']!=payload['source_sha256']:raise ValueError('VLLM_ADAPTER_SOURCE_SHA')
                if sha256(path/'adapter_model.safetensors')!=mapping['mapped_sha256']:
                    raise ValueError('VLLM_ADAPTER_MAPPED_SHA')
                if lease.version!=payload['version'] or lease.sha!=payload['source_sha256']:
                    await invalidate_prefix(engine,lease,args.enable_prefix_caching)
                    if adapter:await engine.remove_lora(adapter.lora_int_id)
                    from mechet.external_route_data import signature
                    counter+=1;adapter=LoRARequest('route-'+signature([payload['version'],payload['source_sha256']])[:24]
                                                  +'-'+str(counter),counter,str(path))
                    if not await engine.add_lora(adapter):raise ValueError('VLLM_ADAPTER_LOAD_FAILED')
                    result=await engine.collective_rpc('verify_route_lora',args=(counter,str(path)))
                    lease.commit(payload['version'],payload['source_sha256'])
                    metrics.start_version(payload['version'])
                else:result=dict(already_loaded=True,source_sha256=lease.sha)
            elif op=='drain':
                if tasks:await asyncio.gather(*tasks)
                result=dict(inflight=len(lease.inflight),behavior_version=lease.version,engine_metrics=metrics.snapshot())
            elif op=='metrics':result=metrics.snapshot()
            elif op=='probe_logprobs':
                # Numerical replay only: teacher-force the SAME saved next token
                # with adapter disabled/enabled. Never return a policy receipt.
                if not runtime.get('software_replay') or lease.inflight:
                    raise ValueError('VLLM_SOFTWARE_PROBE_NOT_ENABLED_OR_BUSY')
                if payload['behavior_version']!=lease.version:
                    raise ValueError('VLLM_WEIGHT_VERSION_MISMATCH')
                ids=payload['prompt_token_ids']+payload['continuation_token_ids']
                if len(ids)<2 or len(ids)>=cfg['max_context']:raise ValueError('VLLM_PROBE_CONTEXT')
                scores={}
                for name,lora in (('base',None),('adapter',adapter)):
                    params=SamplingParams(temperature=0.,max_tokens=1,prompt_logprobs=0,
                        ignore_eos=True,detokenize=False,output_kind=RequestOutputKind.FINAL_ONLY)
                    counter+=1;probe=None
                    async for probe in engine.generate(dict(prompt_token_ids=ids),params,
                            'software-probe-'+str(counter),lora_request=lora):pass
                    if probe is None or probe.prompt_logprobs is None:
                        raise ValueError('VLLM_PROBE_LOGPROBS_MISSING')
                    rows=probe.prompt_logprobs[len(payload['prompt_token_ids']):]
                    if len(rows)!=len(payload['continuation_token_ids']):
                        raise ValueError('VLLM_PROBE_PROBABILITY_LENGTH')
                    scores[name]=[float(row[token].logprob) for row,token in
                                  zip(rows,payload['continuation_token_ids'])]
                result=validate_probe(dict(probe_only=True,behavior_version=lease.version,**scores),
                    lease.version,len(payload['continuation_token_ids']),1.)
            else:raise ValueError('VLLM_UNKNOWN_OPERATION')
            await reply(uid,result=result)
        except Exception as exc:await reply(uid,error=type(exc).__name__+':'+str(exc))
    if tasks:await asyncio.gather(*tasks)
    engine.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    parsed=parser.parse_args()
    # Keep engine/import/native prints off the protocol FD, including children.
    protocol=os.fdopen(os.dup(sys.stdout.fileno()),'w',buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    root=Path(__file__).resolve().parents[1]
    import types
    package=types.ModuleType('mechet');package.__path__=[str(root/'src/mechet')]
    sys.modules['mechet']=package
    from mechet.route_vllm_environment import engine_import_paths
    cfg=json.loads(parsed.config.read_text())
    sys.path[:]=engine_import_paths(sys.path,root,cfg['vllm_runtime'].get('bootstrap'))
    asyncio.run(serve(cfg,protocol))
