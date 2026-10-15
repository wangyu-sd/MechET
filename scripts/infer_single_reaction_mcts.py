#!/usr/bin/env python3
"""Full3120 single-reaction MCTS with continuous GPU batching and CPU overlap."""
from __future__ import annotations
import argparse,json,math,os,sys,time,threading
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO),str(REPO/'src')]
from scripts.infer_dynamic_electron_trees import ADAPTER_SHA,REVISION,sha256,dump,load_source,evaluate,ForwardCallCounter,HostMemoryGuard
from scripts.single_reaction_mcts import SingleReactionMCTS
from scripts.mcts_runtime_io import FastSingleReactionMCTSStore,DurablePolicyWriter
from scripts.rolling_vllm_search import ContinuousBatchEngine
from scripts.run_natural_language_value_search import policy_prompt
from scripts.build_natural_language_event_sft import SYSTEM,TOOLS
from scripts.eval_natural_language_event_local import prediction_call
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.dynamic_electron_tree import Proposal,available_host_memory
from mechet.shared_electron_search import SharedSearchStore,SearchJournal,fingerprint,sampling_seed

SOURCE_SHA='819215c5d40720afd998aaf77b7b938045d8171a5487b26bf929897f65a957c4'


def policy_cache_contract():
    return {'model':'Qwen/Qwen3-0.6B','revision':REVISION,'adapter_sha':ADAPTER_SHA,
        'history_contract':'natural_language_electron_event_history_v2','dtype':'bfloat16',
        'backend':'vllm_0.8.5_V0_TP1','executor_code_sha':fingerprint({
            str(path.relative_to(REPO)):sha256(path) for path in sorted([
                *(REPO/'src/mechet').glob('*.py'),REPO/'scripts/run_natural_language_value_search.py',
                REPO/'scripts/build_natural_language_event_sft.py',REPO/'scripts/eval_natural_language_event_local.py'])}),
        'sampling_seed_contract':'case_independent_native_context_v1'}


def make_controller(row,**kwargs):
    return SingleReactionMCTS(row['id'],row['product_unmapped'],**kwargs)


def search_is_complete(reports,expected):
    return len(reports)==expected and not any(any(word in r.get('stop_reason','')
        for word in ('MEMORY','RESOURCE','CONTEXT_BUDGET')) for r in reports)


def completion_hit_context_limit(request,proposals,max_model_len):
    return bool(proposals) and all(g.truncated for g in proposals) and (
        request['max_tokens']>=max_model_len-len(request['prompt_ids']))


def decode_request(result,tokenizer,parse=prediction_call):
    proposals=[]
    for generated in result.outputs:
        ids=list(generated.token_ids);raw=tokenizer.decode(ids,skip_special_tokens=False)
        name,arguments,error=parse(raw,tokenizer)
        lp=generated.cumulative_logprob
        if lp is None or not math.isfinite(lp):raise RuntimeError('missing/nonfinite unwarped NLL')
        proposals.append(asdict(Proposal(name,arguments,raw,float(lp),len(ids),str(error or ''),
            generated.finish_reason=='length',ids)))
    return {'request_id':str(result.request_id),'cached_prefix_tokens':getattr(result,'num_cached_tokens',None),
        'generations':proposals,'engine_finish_reasons':[g.finish_reason for g in result.outputs]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world',type=int,default=8);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--program-goal',type=int,default=10)
    p.add_argument('--inflight-requests',type=int,default=128)
    p.add_argument('--cpu-workers',type=int,default=16)
    p.add_argument('--io-workers',type=int,default=2)
    p.add_argument('--max-model-len',type=int,default=8192)
    p.add_argument('--shared-cache',type=Path,required=True)
    p.add_argument('--mcts-shared-root',type=Path,required=True)
    args=p.parse_args()
    if not 0<=args.rank<args.world or min(args.program_goal,args.cpu_workers,args.io_workers,args.inflight_requests)<1:
        p.error('invalid shard or concurrency')
    if sha256(args.data)!=SOURCE_SHA:raise ValueError('frozen full3120 source SHA mismatch')
    if sha256(args.adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('frozen actor SHA mismatch')
    lineage=json.loads((args.adapter/'data_contract.json').read_text())
    if lineage['environment_revision']!='natural_language_electron_event_history_v2':raise ValueError('history protocol mismatch')
    if os.environ.get('VLLM_USE_V1','0')!='0':raise ValueError('pinned V0 engine required')
    os.environ['VLLM_USE_V1']='0'
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1:raise ValueError('one GPU per worker')
    rows=load_source(args.data,3120);selected=rows[args.rank::args.world];references={r['id']:r for r in selected}
    root=args.output/'tree'/f'rank{args.rank:02d}'
    if root.exists():raise ValueError('output exists: preserve evidence; choose a fresh attempt')
    root.mkdir(parents=True)
    import vllm,psutil
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    if vllm.__version__.split('+')[0]!='0.8.5':raise ValueError('pinned vLLM0.8.5 required')
    engine=LLM(model='Qwen/Qwen3-0.6B',revision=REVISION,tokenizer='Qwen/Qwen3-0.6B',
        trust_remote_code=True,dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.82,
        max_model_len=args.max_model_len,max_num_seqs=256,enable_prefix_caching=True,enable_lora=True,
        max_lora_rank=16,enforce_eager=True,num_scheduler_steps=1,seed=args.seed+args.rank)
    forwards=ForwardCallCounter(engine);tokenizer=engine.get_tokenizer()
    process=psutil.Process();guard=HostMemoryGuard(world=args.world,initial_rss=process.memory_info().rss,available=available_host_memory())
    cache=SharedSearchStore(args.shared_cache,policy_cache_contract())
    planning=FastSingleReactionMCTSStore(args.mcts_shared_root,{'policy_executor_namespace':cache.namespace,
        'max_model_len':args.max_model_len,'sampling_seed':args.seed,'program_goal':args.program_goal})
    manifest={'model':'Qwen/Qwen3-0.6B','revision':REVISION,'adapter_sha':ADAPTER_SHA,'source_sha':SOURCE_SHA,
        'full_denominator':3120,'shard_rows':len(selected),'rank':args.rank,'world':args.world,'mode':'tree',
        'seed':args.seed,'raw_unwarped_nll':True,'search_algorithm':'single_reaction_mcts',
        'evaluation_unit':'single_reaction_endpoint','source_protocol':'frozen_rxn_prod_min_proxy_DIAGNOSTIC',
        'code_sha':os.environ.get('MECHET_TREE_CODE_SHA','not_supplied'),
        'history_contract':lineage['environment_revision'],'shared_cache_namespace':cache.namespace,
        'shared_cache_path':str(cache.root),'mcts_shared_root':str(planning.root),'program_goal':args.program_goal,
        'reward_contract':'productive_strict_finish_v1_NOT_gold_endpoint_correctness',
        'reference_visible_to_policy':False,'ranking':'unwarped_assistant_mean_NLL',
        'batching':'continuous_public_add_request_step_single_GPU_owner_v1',
        'inflight_requests_per_GPU':args.inflight_requests,'cpu_workers_per_GPU':args.cpu_workers,
        'samples_per_expansion':16,'max_num_seqs':256,'fixed_depth_or_total_branch_cap':False,
        'rollout_liveness':'process_owned_lease_fastpath_v1',
        'policy_persistence':'bounded_async_commit_before_execute_v1',
        'io_workers_per_GPU':args.io_workers,'max_pending_policy_writes':args.inflight_requests,
        'stopping_contract':'ten_trajectory_attempts_including_invalid_and_censored_not_ten_successes',
        'hard_token_or_wall_cap':False,'max_model_len':args.max_model_len,
        'stock_or_recursive_reaction_search':False,'fresh_attempt_no_old_graph_resume':True}
    dump(root/'manifest.json',manifest)
    controllers={};journals={};reports=[]
    for row in selected:
        try:c=make_controller(row,mcts_store=planning,shared_store=cache,
            checkpoint_path=root/'mcts'/(fingerprint(row['id'])+'.json'),program_goal=args.program_goal)
        except ValueError as exc:
            reports.append({'id':row['id'],'top1':False,'top5':False,'top10':False,'any_endpoint_hit':False,
                'terminal_count':0,'stop_reason':'ROOT_PARSE_FAILURE','error':str(exc),'work':{'sampled_tokens':0,'raw_candidates':0}})
            continue
        controllers[row['id']]=c
        j=SearchJournal(root/'journals',row['id'],manifest);j.initialize(c);journals[row['id']]=j
    ready=deque(controllers);preparing={};executing={};gpu={};attempted=set()
    costs={k:{'policy_request_batches':0,'prompt_tokens':0,'shared_policy_cache_hits':0,'reused_output_tokens':0,
              'cpu_executor_wall_seconds':0.,'cpu_executor_thread_seconds':0.,
              'policy_persistence_wall_seconds':0.} for k in controllers}
    total_tokens=0;started=time.monotonic();last_progress=0.;sequence=0;peak_gpu=0;aborted_gpu=0
    writer=DurablePolicyWriter(root/'raw_generations.jsonl',cache.namespace,
        workers=args.io_workers,max_pending=args.inflight_requests)
    prompts=(root/'policy_contexts.jsonl').open('a');prompt_lock=threading.Lock()
    waves=(root/'work_waves.jsonl').open('a');prompt_seen=set();resource_reason=''

    def progress(force=False):
        nonlocal last_progress,sequence
        if not force and time.monotonic()-last_progress<5:return
        last_progress=time.monotonic();sequence+=1
        limited=sum('MEMORY' in r.get('stop_reason','') or 'RESOURCE' in r.get('stop_reason','') for r in reports)
        value={'algorithm':'single_reaction_mcts','denominator':len(selected),'initialized':len(selected),
            'attempted':len(attempted),'attempted_percent':100*len(attempted)/len(selected),
            'finished':len(reports)-limited,'finished_percent':100*(len(reports)-limited)/len(selected),
            'resource_limited':limited,'pending':len(selected)-len(reports)+limited,
            'output_tokens':total_tokens,'model_forward_calls':forwards.total_calls,'elapsed_seconds':last_progress-started,
            'ready_states':len(ready),'CPU_planning_requests':len(preparing),'GPU_inflight_requests':len(gpu),
            'CPU_executor_requests':len(executing),'peak_GPU_inflight_requests':peak_gpu,
            'GPU_aborted_requests':aborted_gpu,
            'completed_mcts_rollouts':sum(c.completed_rollouts for c in controllers.values())+sum(r.get('mcts',{}).get('completed_rollouts',0) for r in reports),
            'collected_terminal_programs':sum(len(c.completed_programs) for c in controllers.values())+sum(r.get('collected_programs',0) for r in reports),
            'shared_policy_cache_hits':sum(v['shared_policy_cache_hits'] for v in costs.values()),
            'policy_persistence':writer.metrics(),'rollout_liveness':planning.liveness_metrics(),
            'full_result_complete':len(reports)==len(selected) and not limited,
            'boundary':'rolling_observation_not_synchronized_generation_wave_v1'}
        dump(root/'progress.json',value);waves.write(json.dumps(value)+'\n');waves.flush()
        print({'phase':'rolling-MCTS','rank':args.rank,**value},flush=True)

    def finish(identifier,reason):
        c=controllers.pop(identifier);c.stop_reason=reason;c.checkpoint()
        graph=root/'trees'/(fingerprint(identifier)+'.json');dump(graph,c.to_dict())
        # FIRST and only access to this target's reference endpoint during search.
        result=evaluate(c,references[identifier]);result.update(c.report());result['work'].update(costs[identifier])
        result['tree_artifact']=str(graph.relative_to(root))
        dump(root/'evaluations'/(fingerprint(identifier)+'.json'),result);reports.append(result);c.close()

    def prepare(identifier):
        c=controllers[identifier];ctx=c.next_context()
        for parent,edges in c.drain_cached_events():journals[identifier].record(c,parent,edges)
        if ctx is None:return identifier,None,None,None,None
        messages=[{'role':'system','content':SYSTEM},{'role':'user','content':policy_prompt(
            ctx.node.target,ctx.node.state,include_inventory=True,actions=ctx.node.actions,compact_history=True)}]
        prompt=render_qwen_sft_tool_prefix(tokenizer,messages,tools=TOOLS)
        ids=tokenizer.encode(prompt,add_special_tokens=False);room=args.max_model_len-len(ids)
        if room<2:
            ctx.resource_closed='CONTEXT_MEMORY_BUDGET';journals[identifier].record(c,ctx.key,[])
            c.censor_selected('CONTEXT_MEMORY_BUDGET');return identifier,None,None,None,None
        request={'context':ctx.key,'prompt_ids':ids,'n':16,'max_tokens':min(ctx.max_new_tokens,room),
            'temperature':1,'top_p':1,'top_k':-1,'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,
            'skip_special_tokens':False,'logprobs':0,'min_tokens':0,'seed':sampling_seed(args.seed,ctx.key,ctx.stats.batches)}
        # Prompt I/O belongs to the CPU preparation stage, not the GPU-owner
        # scheduler. Serialize appends and publish seen only after flush.
        prompt_key=(identifier,ctx.key)
        with prompt_lock:
            if prompt_key not in prompt_seen:
                prompts.write(json.dumps({'case_id':identifier,'context':ctx.key,
                    'text':prompt,'token_ids':ids},ensure_ascii=False)+'\n');prompts.flush()
                prompt_seen.add(prompt_key)
        lease,cached=cache.acquire_request(request)
        return identifier,request,lease,cached,prompt

    def execute(identifier,request,record):
        c=controllers[identifier];key=request['context'];wall=time.monotonic();cpu=time.thread_time()
        proposals=[Proposal(**g) for g in record['generations']]
        edges=c.incorporate(key,proposals)
        ctx=c.contexts[key]
        if completion_hit_context_limit(request,proposals,args.max_model_len):
            ctx.resource_closed='COMPLETION_CONTEXT_BUDGET'
        journals[identifier].record(c,key,edges)
        return time.monotonic()-wall,time.thread_time()-cpu

    def account(identifier,request,record,cache_hit):
        # Account observed GPU output at admission, not after asynchronous I/O:
        # a resource stop must not erase already generated, subsequently drained tokens.
        nonlocal total_tokens
        cost=costs[identifier];tokens=sum(g['tokens'] for g in record['generations'])
        if cache_hit:cost['shared_policy_cache_hits']+=1;cost['reused_output_tokens']+=tokens
        else:
            total_tokens+=tokens;cost['policy_request_batches']+=1;cost['prompt_tokens']+=len(request['prompt_ids'])

    def accept(identifier,request,record,pool):
        future=pool.submit(execute,identifier,request,record);executing[future]=identifier

    def persist_gpu_output(done):
        rid,result=done;identifier,request,lease=gpu.pop(rid)
        try:
            record=decode_request(result,tokenizer)
            writer.submit(identifier,request,record,lease=lease)
        except BaseException:
            lease.close();raise
        # Ownership transferred successfully. An accounting error must not
        # close the lease concurrently with its persistence worker.
        account(identifier,request,record,False)
        forwards.by_request.pop(rid,None)

    def record_persistence(persisted):
        costs[persisted.case_id]['policy_persistence_wall_seconds']+=persisted.wall_seconds

    def params(request):return SamplingParams(**{k:v for k,v in request.items() if k not in {'context','prompt_ids'}})
    pool=ThreadPoolExecutor(max_workers=args.cpu_workers)
    pipe=ContinuousBatchEngine(engine.llm_engine,params_factory=params,
        lora=LoRARequest('frozen_flowER_stage2',1,str(args.adapter.resolve())),max_inflight=args.inflight_requests)
    try:
        progress(True)
        while ready or preparing or executing or gpu or writer.has_pending:
            if guard.exhausted(rss=process.memory_info().rss,available=available_host_memory()):
                resource_reason='TASK_HOST_MEMORY_BUDGET';break
            while True:
                persisted=writer.poll()
                if persisted is None:break
                record_persistence(persisted)
                accept(persisted.case_id,persisted.request,persisted.record,pool)
            # Admit independent CASES, never concurrent writes to one controller.
            while ready and len(preparing)+len(executing)<2*args.cpu_workers and len(gpu)+len(preparing)<args.inflight_requests:
                identifier=ready.popleft();attempted.add(identifier)
                preparing[pool.submit(prepare,identifier)]=identifier
            for future in [f for f in preparing if f.done()]:
                identifier=preparing[future]
                _,request,lease,cached,_=future.result();c=controllers[identifier]
                if request is not None and cached is not None and not writer.available:
                    continue  # retain the completed preparation until I/O capacity returns
                preparing.pop(future)
                if request is None:
                    if c.has_pending:ready.append(identifier)
                    else:finish(identifier,c.stop_reason or 'TERMINAL_PROGRAM_GOAL_REACHED')
                    continue
                if cached is not None:
                    writer.submit(identifier,request,cached,cache_hit=True)
                    account(identifier,request,cached,True)
                elif lease is None:ready.append(identifier)
                else:
                    rid=pipe.submit(request);gpu[rid]=(identifier,request,lease);peak_gpu=max(peak_gpu,len(gpu))
            while writer.available:
                done=pipe.poll()
                if done is None:break
                persist_gpu_output(done)
            for future in [f for f in executing if f.done()]:
                identifier=executing.pop(future);wall,cpu=future.result()
                costs[identifier]['cpu_executor_wall_seconds']+=wall;costs[identifier]['cpu_executor_thread_seconds']+=cpu
                ready.append(identifier)
            progress();time.sleep(.002)
    finally:
        try:
            pipe.close()
            if resource_reason:
                # The owner has stopped, so this queue can no longer grow.
                # Backpressure may have left FINISHED GPU outputs unpolled:
                # persist/count them without executing chemistry after a stop.
                while True:
                    if not writer.available:
                        persisted=writer.poll()
                        if persisted is not None:record_persistence(persisted)
                        else:time.sleep(.002)
                        continue
                    done=pipe.poll()
                    if done is None:break
                    persist_gpu_output(done)
                while writer.has_pending:
                    persisted=writer.poll()
                    if persisted is not None:record_persistence(persisted)
                    else:time.sleep(.002)
        finally:
            try:
                pool.shutdown(wait=True)
            finally:
                try:
                    writer.close()  # drain admitted writes; errors prevent complete.json
                finally:
                    aborted_gpu+=len(gpu)
                    for _,_,lease in gpu.values():lease.close()
                    gpu.clear()
                    for future in preparing:
                        if future.done() and future.exception() is None:
                            lease=future.result()[2]
                            if lease:lease.close()
                    prompts.close()
                    for c in controllers.values():
                        try:c.checkpoint()
                        finally:c.close()
    if resource_reason:
        for identifier in list(controllers):finish(identifier,resource_reason)
    if {r['id'] for r in reports}!=set(references):raise RuntimeError('full3120 shard lost IDs')
    progress(True);waves.close()
    dump(root/'complete.json',{'manifest':manifest,'rows':reports,'processing_complete':True,
        'search_complete':search_is_complete(reports,len(selected)),'resource_stop':resource_reason,
        'tokens':total_tokens,'model_forward_calls':forwards.total_calls,'seconds':time.monotonic()-started})
    return 0


if __name__=='__main__':raise SystemExit(main())
