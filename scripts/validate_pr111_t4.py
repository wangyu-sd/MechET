"""User-authorized small T4 legality and paired-work timing, never a route benchmark."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path('/aaa/fionafyang/buddy1/whaleywang/MechET')
WORK=Path(__file__).resolve().parents[1]
COMPUTE_FILES=['src/mechet/'+n+'.py' for n in (
    'route_policy_sampling','route_counter_sampling','route_execution_cache','route_acceleration',
    'route_distributed_collection','route_global_sampler','route_work_broker','route_async_collection',
    'route_policy_episode','route_refit_runtime','route_refit_credit','route_policy_objective',
    'external_route_rl','external_route_io','route_precision','route_training_batches')]+['scripts/train_external_route_rl.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    from mechet.route_training_contract import atomic_json
    atomic_json(path,value)


def saved_inputs(count,stride=1):
    path=ROOT/'outputs/agent/pr109_standard_d3_refit_20261116_02/round-000009/rank00/raw_generation.jsonl'
    requested=None;returned=None
    with path.open() as source:
        for line in source:
            row=json.loads(line)
            if row['event']=='gpu-requested' and requested is None:requested=row
            if row['event']=='gpu-returned':returned=row;break
    if requested is None or returned is None:raise ValueError('SAVED_PAIRED_INPUTS_MISSING')
    assert [r['request_id'] for r in requested['requests']]==[r['request_id'] for r in returned['requests']]
    indices=list(range(0,count*stride,stride))
    return [requested['prompt_token_ids'][i] for i in indices],[returned['generated_token_ids'][i] for i in indices],dict(
        source=str(path),requested_event_sha=hashlib.sha256(json.dumps(requested,sort_keys=True).encode()).hexdigest(),
        returned_event_sha=hashlib.sha256(json.dumps(returned,sort_keys=True).encode()).hexdigest(),request_indices=indices)


def run(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not out.is_relative_to(ROOT):raise ValueError('SHARED_OUTPUT_REQUIRED')
    report=dict(status='RUNNING',scope='user_authorized_T4_small_software_legality_and_fixed_work_timing',
        production_updates=0,corpus_evaluation=False,legal_checks={},paired_benchmarks=[],source_task_bindings={},
        validated_compute_files_sha256={p:sha(WORK/p) for p in COMPUTE_FILES})
    def event(stage,**kw):
        report.update(stage=stage,**kw);write(out/'admission.json',report)
        print(json.dumps(dict(stage=stage,**kw)),flush=True)
    try:
        cfg=json.loads((ROOT/'outputs/submissions/pr109_standard_d3_refit_20261116_02/contract.json').read_text())
        event('import-pinned-runtime')
        from mechet.route_transformers_import import import_transformers_indexed
        import_transformers_indexed(cfg)
        import torch
        from transformers.utils import import_utils
        # T4 (SM75) has no native BF16. Both sides use identical FP32 weights
        # and the same unfused Qwen CUDA path; production H20 keeps BF16/FLA.
        import_utils.is_flash_linear_attention_available=lambda:False
        import_utils.is_causal_conv1d_available=lambda:False
        from transformers import AutoTokenizer,GenerationConfig,LogitsProcessorList
        from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
        for name in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule',
                     'causal_conv1d_fn','causal_conv1d_update'):
            setattr(upstream,name,None)
        from peft import PeftModel,set_peft_model_state_dict
        from safetensors.torch import load_file
        from mechet.qwen35_protocol import load_language_model,OFFICIAL_REVISION
        from mechet.route_policy_sampling import TokenLogprobs,PrivateSampling
        from mechet.route_counter_sampling import CounterCategorical
        from scripts.pr96_earho_contract import token_logps_batch
        from mechet.route_policy_objective import action_loss
        from mechet.route_training_memory import checkpointed_policy_forward
        torch.cuda.set_device(0);torch.set_num_threads(4)
        if 'T4' not in torch.cuda.get_device_name(0):raise ValueError('LOCAL_T4_REQUIRED')
        report.update(hardware=torch.cuda.get_device_name(0),precision='legacy_fp32_lora',
                      torch_version=torch.__version__,cuda=torch.version.cuda,
                      hardware_limit='FP32/unfused T4 timing is not H20 BF16/FLA timing')
        prompts,responses,source=saved_inputs(4);report['input_binding']=source
        write(out/'fixed_inputs.json',dict(prompts=prompts,responses=responses,source=source))
        bindings={k:json.loads((args.bindings/(k+'_actor_binding.json')).read_text()) for k in ('d3','d1')}
        for key,binding in bindings.items():
            checkpoint=Path(binding['checkpoint']);expected=binding['payload_sha256']['adapter_model.safetensors']
            if sha(checkpoint/'adapter_model.safetensors')!=expected:raise ValueError('ACTOR_BYTES_CHANGED')
            report['source_task_bindings'][key]=dict(contract_sha256=binding['contract_sha256'],
                checkpoint=str(checkpoint),adapter_sha256=expected,optimizer_step=binding['optimizer_step'],
                complete_sha256=binding['complete_sha256'])
        tokenizer=AutoTokenizer.from_pretrained(cfg['base_path'],revision=OFFICIAL_REVISION,local_files_only=True)
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        event('load-actual-D3-actor',checkpoint=bindings['d3']['checkpoint'])
        actor,_=load_language_model(cfg['base_path'],dtype=torch.float32)
        actor=PeftModel.from_pretrained(actor,bindings['d3']['checkpoint'],is_trainable=True).cuda().eval()
        for module in actor.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        encoded=tokenizer.pad(dict(input_ids=prompts),padding=True,return_tensors='pt')
        encoded={k:v.cuda() for k,v in encoded.items()};width=encoded['input_ids'].shape[1]
        uids=['fixed-real-request-'+str(i) for i in range(len(prompts))]
        def processor(kind,capture):
            if kind=='counter_categorical_v1':return CounterCategorical(uids,17,'cuda:0',capture)
            return PrivateSampling(uids,17,'cuda:0',finished=lambda:capture.done,prompt_width=width)
        class Replay:
            def __call__(self,ids,scores):
                step=ids.shape[1]-width
                chosen=[r[step] if step<len(r) else tokenizer.eos_token_id for r in responses]
                return torch.full_like(scores,-float('inf')).scatter(1,torch.tensor(chosen,device=scores.device)[:,None],0.)
        def fixed_generate(kind):
            capture=TokenLogprobs(tokenizer.eos_token_id)
            generation=GenerationConfig(do_sample=kind=='legacy_private_v1',top_k=0,top_p=1.,temperature=1.,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            torch.cuda.synchronize();t=time.monotonic()
            with torch.inference_mode():
                result=actor.generate(**encoded,generation_config=generation,max_new_tokens=max(map(len,responses)),
                    logits_processor=LogitsProcessorList([capture,processor(kind,capture),Replay()]))
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            capture.finish(result[:,-1]);values=capture.values()
            for i,expected in enumerate(responses):
                if result[i,width:width+len(expected)].tolist()!=expected:raise ValueError('FIXED_REPLAY_OUTPUT_MISMATCH')
            return elapsed,values
        def timed_pair(task):
            event('paired-fixed-replay-'+task)
            kinds=['legacy_private_v1','counter_categorical_v1','private_pinned_argmax_v1'];timings={k:[] for k in kinds};values={}
            for kind in kinds:fixed_generate(kind)
            for order in [kinds,list(reversed(kinds))]:
                for kind in order:
                    seconds,values[kind]=fixed_generate(kind);timings[kind].append(seconds)
            baseline=statistics.median(timings[kinds[0]])
            best=min(kinds[1:],key=lambda k:statistics.median(timings[k]))
            for kind in kinds:
                if max(abs(a-b) for row,other in zip(values[kind],values[kinds[0]]) for a,b in zip(row,other))>.01:
                    raise ValueError('REPLAY_LOGPROB_KERNEL_CHANGED')
            record=dict(task=task,stage='same_prompt_same_complete_response_fixed_replay_generation',
                baseline_kernel=kinds[0],candidate_kernel=best,raw_seconds=timings,
                speedup=baseline/statistics.median(timings[best]),requests=4,
                response_lengths=list(map(len,responses)),not_on_policy_samples=True)
            report['paired_benchmarks'].append(record)
            report['legal_checks'][task+'_fixed_replay_tokens_and_probabilities']='PASS'
            event('paired-fixed-replay-complete',last_benchmark=record)
            return best
        reused=None
        if args.reuse_generation:
            reused=json.loads(args.reuse_generation.read_text())
            if (reused['validated_compute_files_sha256']!=report['validated_compute_files_sha256']
                    or reused['source_task_bindings']!=report['source_task_bindings']):
                raise ValueError('REUSED_TIMING_BINDINGS_CHANGED')
            report['paired_benchmarks']=deepcopy(reused['paired_benchmarks'])
            report['generation_timing_reused_from']=dict(path=str(args.reuse_generation),sha256=sha(args.reuse_generation))
        selected='private_pinned_argmax_v1'
        if not reused:timed_pair('d3')
        # Validate fresh actual model draws separately from controlled timing.
        def legal_generation(task):
            capture=TokenLogprobs(tokenizer.eos_token_id)
            gc=GenerationConfig(do_sample=selected=='legacy_private_v1',top_k=0,top_p=1.,temperature=1.,
                pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id,use_cache=True)
            with torch.inference_mode():
                seq=actor.generate(**encoded,generation_config=gc,max_new_tokens=256,
                    logits_processor=LogitsProcessorList([capture,processor(selected,capture)]))
            capture.finish(seq[:,-1]);lp=capture.values();receipts=[]
            for i,row in enumerate(seq):
                ids=row[width:].tolist()
                complete=tokenizer.eos_token_id in ids
                if complete:ids=ids[:ids.index(tokenizer.eos_token_id)+1]
                if not ids or len(ids)!=len(lp[i]) or any(not math.isfinite(x) or x>0 for x in lp[i]):
                    raise ValueError('ACTUAL_SAMPLING_PROBABILITY_RECEIPT_INVALID')
                with torch.no_grad():expected=token_logps_batch(actor,[(prompts[i],ids)],tokenizer.pad_token_id)[0].tolist()
                drift=max(abs(a-b) for a,b in zip(lp[i],expected))
                if drift>.25:raise ValueError('ACTUAL_SAMPLING_TEACHERFORCE_PARITY')
                receipts.append(dict(prompt_token_ids=prompts[i],generated_token_ids=ids,behavior_logprobs=lp[i],
                    raw=tokenizer.decode(ids,skip_special_tokens=False),complete=complete,max_logprob_drift=drift))
            write(out/(task+'_actual_draws.json'),receipts)
            report['legal_checks'][task+'_actual_draw_probabilities']='PASS'
            event('actual-draws-valid-'+task,complete=sum(r['complete'] for r in receipts),max_drift=max(r['max_logprob_drift'] for r in receipts))
        legal_generation('d3')
        event('load-actual-D1-adapter')
        set_peft_model_state_dict(actor,load_file(str(Path(bindings['d1']['checkpoint'])/'adapter_model.safetensors')),adapter_name='default')
        if not reused:timed_pair('d1')
        for row in report['paired_benchmarks']:
            row['candidate_kernel']=selected;row['speedup']=statistics.median(row['raw_seconds']['legacy_private_v1'])/statistics.median(row['raw_seconds'][selected])
            row['interpretation']='generation stage approximately unchanged; no positive total-training speedup inferred'
        legal_generation('d1')
        event('learner-batch-parity-and-speed')
        gp,gr,gs=saved_inputs(4,stride=4)
        pairs=[(p,r[:32]) for p,r in zip(gp,gr)]
        report['learner_input_binding']=dict(gs,prompt_lengths=list(map(len,gp)),
            response_prefix_lengths=[len(r) for _,r in pairs],test_advantages=[.5,-.5,.2,-.2],
            interpretation='signed numerical test covectors over four distinct saved target inputs; not routeRL training data')
        report['learner_benchmarks']={}
        for task in ['d1','d3']:
            if task=='d3':set_peft_model_state_dict(actor,load_file(str(Path(bindings['d3']['checkpoint'])/'adapter_model.safetensors')),adapter_name='default')
            actor.eval()
            with torch.no_grad():old=[v.detach().clone() for v in token_logps_batch(actor,pairs,tokenizer.pad_token_id)]
            grads={};latency={};losses={}
            for size in [1,4]:
                measurements=[]
                for iteration in range(3):
                    actor.zero_grad(set_to_none=True);torch.cuda.synchronize();started=time.monotonic();total=0.
                    with checkpointed_policy_forward(actor,dict(training_gradient_checkpointing=True)):
                        for start in range(0,len(pairs),size):
                            news=token_logps_batch(actor,pairs[start:start+size],tokenizer.pad_token_id)
                            loss=None
                            for i,new in enumerate(news,start):
                                part=action_loss(new,old[i],old[i], [.5,-.5,.2,-.2][i],clip=.2,beta=.02,normalizer=4*1024.)['total']
                                loss=part if loss is None else loss+part
                            total+=float(loss.detach());loss.backward()
                    torch.cuda.synchronize()
                    if iteration:measurements.append(time.monotonic()-started)
                flat=torch.cat([p.grad.detach().float().reshape(-1) for p in actor.parameters() if p.requires_grad and p.grad is not None])
                if not torch.isfinite(flat).all() or flat.norm()==0:raise ValueError('ACTUAL_ADAPTER_GRADIENT_INVALID')
                grads[size]=flat.cpu();latency[size]=measurements;losses[size]=total
            relative=float((grads[1]-grads[4]).norm()/grads[1].norm().clamp_min(1e-12))
            row=dict(task=task,stage='same4distinct_saved_inputs_actor_forward_backward',candidate_kernel=selected,
                raw_seconds=latency,gradient_relative_difference=relative,gradient_norms={k:float(v.norm()) for k,v in grads.items()},
                losses=losses,speedup=statistics.median(latency[1])/statistics.median(latency[4]),optimizer_updates=0,
                scope='actor_forward_backward_only_reference_excluded')
            report['learner_benchmarks'][task]=row;event('learner-'+task+'-complete',learner_benchmark=row)
            if relative>.02:raise ValueError('ACTUAL_ADAPTER_BATCH_GRADIENT_MISMATCH')
            actor.zero_grad(set_to_none=True);report['legal_checks'][task+'_real_adapter_batch1_vs4_gradient']='PASS'
            report['paired_benchmarks'].append(row)
        report['admitted_settings']=dict(sampling_kernel=selected,loss_batch_size=4)
        if not all(v['speedup']>=.95 for v in report['paired_benchmarks'] if v['stage'].startswith('same_prompt')):
            raise ValueError('SELECTED_GENERATION_KERNEL_REGRESSION')
        if not all(v['speedup']>1 for v in report['learner_benchmarks'].values()):
            raise ValueError('LEARNER_HAS_NO_PAIRED_SPEEDUP')
        report['status']='PASS';event('complete',peak_device_allocated=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        import traceback
        report.update(status='FAIL',error=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc())
        write(out/'admission.json',report);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--reuse-generation',type=Path)
    run(p.parse_args())
