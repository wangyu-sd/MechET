"""Authorized small real-engine software admission; never an RL or route result.

The T4 profile is explicitly FP16/unfused HF versus FP16 native. Formal H20
admission must separately use the production BF16 forward and frozen checkpoint.
"""
import argparse
from concurrent.futures import as_completed
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

ROOT=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs(count):
    path=ROOT/'outputs/agent/pr109_standard_d3_refit_20261116_02/round-000009/rank00/raw_generation.jsonl'
    from mechet.route_policy_episode import DecisionRequest
    with path.open() as source:
        row=next(json.loads(s) for s in source if '"gpu-requested"' in s)
    indices=[4*i for i in range(count)]
    if max(indices)>=len(row['requests']):raise ValueError('DIAGNOSTIC_INPUT_COUNT')
    return [DecisionRequest(**row['requests'][i]) for i in indices],\
        [row['prompt_token_ids'][i] for i in indices],dict(source=str(path),indices=indices,
            source_event_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest())


def checkpoint(contract):
    cfg=json.loads(contract.read_text());pointer=json.loads((Path(cfg['output_dir'])/'latest.json').read_text())
    p=Path(pointer['path']);state=json.loads((p/'complete.json').read_text())
    if sha(p/'complete.json')!=pointer['complete_sha256'] or state['contract_sha256']!=sha(contract):
        raise ValueError('SOURCE_COMPLETE_BINDING_CHANGED')
    if sha(p/'adapter_model.safetensors')!=state['payload_sha256']['adapter_model.safetensors']:
        raise ValueError('SOURCE_ACTOR_BYTES_CHANGED')
    return p,dict(checkpoint=str(p),adapter_sha256=state['payload_sha256']['adapter_model.safetensors'],
        complete_sha256=pointer['complete_sha256'],contract_sha256=sha(contract),
        optimizer_step=state['optimizer_step'],next_round=state['next_round'])


def run(args):
    from mechet.route_training_contract import atomic_json
    output=args.output.resolve()
    if not output.is_relative_to('/aaa/fionafyang/buddy1') or output.exists():
        raise ValueError('NEW_OWNED_CEPH_DIAGNOSTIC_DIRECTORY_REQUIRED')
    output.mkdir(parents=True)
    report=dict(status='RUNNING',scope='user_authorized_small_native_engine_software_admission',
        production_updates=0,route_evaluation=False,hardware_profile=args.profile,
        legal_checks={},timings={},source_bindings={})
    def event(stage,**values):
        report.update(stage=stage,**values);atomic_json(output/'admission.json',report)
        print(json.dumps(dict(stage=stage,**values)),flush=True)
    process=None
    try:
        cfg=json.loads((args.candidate_contract or args.source_contract).read_text())
        t4=args.profile=='t4_fp16'
        if not t4:
            from mechet.route_runtime_efficiency import configure_profile
            configure_profile(cfg,Path(__file__).resolve().parents[1])
        from mechet.route_transformers_import import import_transformers_indexed
        event('import-learner-runtime');import_transformers_indexed(cfg)
        import torch
        from transformers.utils import import_utils
        if t4:
            import_utils.is_flash_linear_attention_available=lambda:False
            import_utils.is_causal_conv1d_available=lambda:False
        from transformers import AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
        if t4:
            for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                         'causal_conv1d_fn','causal_conv1d_update'):setattr(upstream,name,None)
        from peft import PeftModel
        from mechet.qwen35_protocol import load_language_model,OFFICIAL_REVISION
        from mechet.route_acceleration import full_acceleration,DEFAULT_ACCELERATION,dispatch_settings
        from mechet.route_vllm_contract import configure_runtime
        from mechet.route_vllm_backend import start_vllm_process
        from mechet.route_gpu_backend import make_gpu_sampler,close_gpu_sampler
        from mechet.route_global_sampler import RemoteSampler,serve_gpu,LockedTokenizer
        from mechet.route_work_broker import WorkBroker
        from mechet.route_policy_protocol import route_tool_prefix
        from mechet.route_action_feedback import tools_for_request
        from mechet.external_route_io import BatchedNativeRouteSampler,ReceiptBatch
        from scripts.pr96_earho_contract import token_logps_batch
        torch.cuda.set_device(0);torch.set_num_threads(4)
        required_gpu='T4' if t4 else 'H20'
        if required_gpu not in torch.cuda.get_device_name(0):raise ValueError('ADMISSION_HARDWARE_PROFILE_MISMATCH')
        dtype=torch.float16 if t4 else torch.bfloat16
        precision='legacy_fp32_lora' if t4 else cfg['policy_precision']
        if not t4:
            from mechet.route_precision import configure_precision
            configure_precision(cfg)
        report['hardware']=torch.cuda.get_device_name(0)
        report['precision']=precision
        report['candidate_contract_sha256']=sha(args.candidate_contract) if args.candidate_contract else None
        first,binding=checkpoint(args.source_contract);second,other=checkpoint(args.second_contract)
        report['source_bindings']={'first':binding,'second':other}
        requests,prompt_ids,source=inputs(args.count*args.waves);report['input_binding']=source
        old=json.loads((ROOT/'outputs/submissions/external_route_vllm_full_20261110_04/contract.json').read_text())
        runtime=deepcopy(cfg.get('vllm_runtime',old['vllm_runtime']))
        if t4:runtime['cache_root']=str(output/'cache')
        if args.reuse_cache:
            prior=args.reuse_cache.resolve()
            if (not prior.is_relative_to(ROOT/'outputs/diagnostics/pr111_full_acceleration_20261117_01')
                    or json.loads((prior/'admission.json').read_text()).get('status')!='PASS'):
                raise ValueError('ONLY_OWN_PRIOR_PASSED_CACHE_REUSE_ALLOWED')
            runtime['cache_root']=str(prior/'cache')
            report['cache_reuse']=dict(source=str(prior),prior_receipt_sha256=sha(prior/'admission.json'))
        if t4:runtime.pop('shared_triton_cache_root',None)
        runtime.update(kv_cache_memory_bytes=1024**3 if t4 else runtime['kv_cache_memory_bytes'],
            max_num_batched_tokens=8192,gpu_memory_utilization=.45,request_timeout_seconds=600,startup_timeout_seconds=3600)
        cfg.update(output_dir=str(output),_sha=sha(args.source_contract),generation_batch_size=args.count,world_size=1,
            max_context=8192 if t4 else cfg['max_context'],policy_precision=precision,
            acceleration=full_acceleration(refit=bool(cfg.get('refit'))),
            _rollout_adapter_path=str(first))
        configure_runtime(cfg,runtime)
        if t4:cfg['vllm_runtime'].update(software_replay=True,diagnostic_dtype='float16')
        event('load-real-actor',checkpoint=str(first))
        tokenizer=AutoTokenizer.from_pretrained(cfg['base_path'],revision=OFFICIAL_REVISION,local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        actor,_=load_language_model(cfg['base_path'],dtype=dtype)
        actor=PeftModel.from_pretrained(actor,first,is_trainable=False).cuda().eval()
        reference=None
        if not t4:
            reference,_=load_language_model(cfg['base_path'],dtype=dtype)
            reference=PeftModel.from_pretrained(reference,cfg['parent'],is_trainable=False).cuda().eval()
            report['reference_residency']=dict(adapter_sha256=sha(Path(cfg['parent'])/'adapter_model.safetensors'),
                allocated_bytes=torch.cuda.memory_allocated())
        tokenizer=LockedTokenizer(tokenizer)
        def ids(request):
            return tokenizer(route_tool_prefix(tokenizer,request.messages,tools_for_request(request),cfg),
                             add_special_tokens=False)['input_ids']
        for request,want in zip(requests,prompt_ids):
            if ids(request)!=want:raise ValueError('SAVED_PROMPT_TOOL_TOKENIZER_PARITY')
        event('start-real-native-engine')
        started=time.monotonic();process=start_vllm_process(cfg)
        report['engine_start_seconds']=time.monotonic()-started
        report['phase_metrics']={}
        code=Path(__file__).resolve().parents[1]
        from mechet.route_acceleration_continuation import COMPUTE_FILES
        names=set(COMPUTE_FILES)|{'src/mechet/route_gpu_backend.py','src/mechet/route_vllm_backend.py',
            'src/mechet/route_vllm_protocol.py','src/mechet/route_vllm_cache.py',
            'src/mechet/route_vllm_metrics.py','src/mechet/route_vllm_environment.py',
            'src/mechet/route_reference_cache.py','src/mechet/route_request_accounting.py','scripts/serve_route_vllm.py',
            'scripts/validate_pr111_native_runtime.py'}
        report['validated_compute_files_sha256']={name:sha(code/name) for name in sorted(names)}
        version='software-native:'+binding['adapter_sha256']
        def native(tag,rows,version):
            directory=output/tag;directory.mkdir()
            sampler=make_gpu_sampler(None,tokenizer,cfg,directory,version,0)
            broker=WorkBroker([],actor_version=version,contract_sha=cfg['_sha'],journal_path=directory/'claims.jsonl',
                dispatch=dispatch_settings(cfg))
            worker=threading.Thread(target=serve_gpu,args=(broker,sampler,0),daemon=True);worker.start()
            began=time.monotonic();results=[]
            try:
                with ExitStack() as stack:
                    remotes={r.episode_id:stack.enter_context(RemoteSampler(broker,tag+'/'+r.episode_id,cfg,prompt_ids=ids))
                        for r in rows}
                    pending={remotes[r.episode_id].submit(r):i for i,r in enumerate(rows)}
                    for f in as_completed(pending):results.append((pending[f],f.result(timeout=600)))
            finally:
                broker.stop();worker.join(30)
                if worker.is_alive():raise RuntimeError('DIAGNOSTIC_NATIVE_WORKER_STILL_RUNNING')
                close_gpu_sampler(sampler,cfg)
            report['phase_metrics'][tag]=dict(engine=deepcopy(cfg.get('_last_native_engine_metrics')),
                broker=broker.snapshot())
            atomic_json(directory/'metrics.json',report['phase_metrics'][tag])
            return time.monotonic()-began,[v for _,v in sorted(results)]
        def rows(tag,version,count=None):
            return [replace(r,request_id='software/'+tag+'/'+str(i),behavior_version=version)
                    for i,r in enumerate(requests[:count])]
        event('native-global-warmup')
        warm_seconds,warm=native('native-warmup',rows('warm',version),version)
        atomic_json(output/'native_warm.json',warm)
        hf_cfg=dict(cfg,inference_backend='hf',acceleration=dict(DEFAULT_ACCELERATION,
            duplicate_suffix_policy='disabled',sampling_kernel='private_pinned_argmax_v1'))
        hf=BatchedNativeRouteSampler(actor,tokenizer,hf_cfg,output/'hf.jsonl',ReceiptBatch(output/'hf-tools.jsonl'))
        event('hf-independent-warmup')
        with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
            hf.generate(rows('hf-warmup',version))
        torch.cuda.synchronize()
        native_timings=[];hf_timings=[];all_native=[]
        for repeat in range(args.repeats):
            group=rows('paired-'+str(repeat),version)
            event('paired-complete-actions',repeat=repeat)
            def timed_hf():
                torch.cuda.synchronize();began=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):values=hf.generate(group)
                torch.cuda.synchronize();return time.monotonic()-began,values
            if repeat%2:
                seconds,native_values=native('native-paired-'+str(repeat),group,version)
                hf_seconds,hf_values=timed_hf()
            else:
                hf_seconds,hf_values=timed_hf()
                seconds,native_values=native('native-paired-'+str(repeat),group,version)
            hf_timings.append(hf_seconds);native_timings.append(seconds);all_native.extend(native_values)
            if any(v['truncated'] for v in hf_values+native_values):raise ValueError('DIAGNOSTIC_ACTION_TRUNCATED')
        differences=[]
        for value in all_native:
            with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
                expected=token_logps_batch(actor,[(value['prompt_token_ids'],value['generated_token_ids'])],
                                          tokenizer.pad_token_id,precision=precision)[0]
            differences.extend(abs(a-b) for a,b in zip(expected.tolist(),value['behavior_logprobs']))
        atomic_json(output/'native_outputs.json',all_native)
        report['max_native_logprob_difference']=max(differences)
        if max(differences)>.25:raise ValueError('NATIVE_PROBABILITY_PARITY_UNCHANGED_025_GATE')
        if not any(v['engine_telemetry']['num_cached_tokens']>0 for v in all_native):
            raise ValueError('ACTUAL_PREFIX_CACHE_HIT_NOT_OBSERVED')
        report['legal_checks'].update(native_probability='PASS',prompt_ids='PASS',
            complete_actions='PASS',native_cache_hit='PASS',global_async_receipts='PASS')
        report['timings']=dict(hf_complete_action_seconds=hf_timings,native_complete_action_seconds=native_timings,
            requests_per_backend_per_repeat=len(requests),maximum_inflight=args.count,waves=args.waves,
            median_speedup=statistics.median(hf_timings)/statistics.median(native_timings),
            qualification='same prompts and completed-request count; independently sampled responses; '+args.profile)
        event('actual-native-eos-refill-probe')
        probe=[replace(requests[i%len(requests)],request_id='software/refill/'+str(i),
            episode_id='software-refill-owner-'+str(i),behavior_version=version,
            remaining_tokens=1 if i==0 else requests[i%len(requests)].remaining_tokens)
            for i in range(args.count+2)]
        _,refilled=native('native-refill',probe,version)
        events=[json.loads(line) for line in (output/'native-refill/gpu-rank00.jsonl').read_text().splitlines()]
        submitted=[(i,e['request']['request_id']) for i,e in enumerate(events) if e['event']=='vllm-submitted']
        completed={e['request']['request_id']:i for i,e in enumerate(events) if e['event']=='vllm-completed'}
        refill_position=submitted[args.count][0]
        long_initial=[uid for _,uid in submitted[:args.count] if uid!=probe[0].request_id]
        if not any(refill_position<completed[uid] for uid in long_initial):
            raise ValueError('REAL_NATIVE_REFILL_NOT_OBSERVED_BEFORE_LONG_NEIGHBOR')
        if len(refilled)!=len(probe) or any(v.get('truncated') or v.get('resource_error') for v in refilled[1:]):
            raise ValueError('REAL_NATIVE_REFILL_NEIGHBOR_CENSORED')
        report['refill_probe']=dict(status='PASS',requests=len(probe),dedicated_short_request_budget=1,
            short_request_truncated=refilled[0].get('truncated'),
            refill_event_position=refill_position,initial_long_completion_positions=[completed[u] for u in long_initial],
            scope='explicit one-token resource-boundary software probe; excluded from complete-action timings')
        report['legal_checks']['actual_native_refill_before_long_neighbor']='PASS'
        cfg['_rollout_adapter_path']=str(second);version2='software-switch:'+other['adapter_sha256']
        event('switch-real-actor-and-reset-cache')
        _,switched=native('native-after-switch',rows('switch',version2,1),version2)
        if switched[0]['engine_telemetry']['num_cached_tokens']!=0:
            raise ValueError('OLD_ACTOR_PREFIX_SURVIVED_SWITCH')
        _,cached=native('native-after-switch-repeat',rows('switch-repeat',version2,1),version2)
        if cached[0]['engine_telemetry']['num_cached_tokens']<=0:
            raise ValueError('NEW_ACTOR_PREFIX_NOT_REUSED')
        report['legal_checks']['actor_switch_cache_invalidation']='PASS'
        report['native_engine_metrics']=cfg.get('_last_native_engine_metrics')
        report['status']='PASS';event('software-admission-passed')
    except BaseException as exc:
        report.update(status='FAIL',error=type(exc).__name__+':'+str(exc));event('software-admission-failed')
        raise
    finally:
        if process is not None:process.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-contract',type=Path,default=ROOT/'outputs/submissions/pr111_d3_accelerated_resume_20261117_01/contract.json')
    parser.add_argument('--second-contract',type=Path,default=ROOT/'outputs/submissions/pr111_d1_accelerated_resume_20261117_01/contract.json')
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--waves',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=4)
    parser.add_argument('--reuse-cache',type=Path)
    parser.add_argument('--profile',choices=['t4_fp16','h20_bf16'],default='t4_fp16')
    parser.add_argument('--candidate-contract',type=Path)
    run(parser.parse_args())
