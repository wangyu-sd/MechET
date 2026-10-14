#!/usr/bin/env python3
"""Full-cohort frozen-Qwen vLLM search with adaptive breadth/depth and journals.

Eight independent one-GPU workers; batched prompts from different molecules;
CPU-parallel executor work. No fixed chemical-depth/beam/import cap. Only the
policy-support exhaustion and model/memory limits stop exploration. Token/time
values are renewable accounting checkpoints, not forced stopping ceilings.
"""
from __future__ import annotations

import argparse
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO))
from mechet.dynamic_electron_tree import ElectronTree, Proposal, DynamicWorkCheckpoints, validate_probability_contract, chemical_key, available_host_memory
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.reverse_route_network import RouteNetwork
from mechet.endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from scripts.run_natural_language_value_search import policy_prompt
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.eval_natural_language_event_local import prediction_call
from mechet.shared_electron_search import (
    SharedSearchStore, SearchJournal, sampling_seed, resolve_policy_requests, fingerprint,
)

ADAPTER_SHA = '40204c7a059d16976fc4b24b315c3bdde7bf61c705d24bbde515071757e9de75'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(16<<20),b''):result.update(block)
    return result.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    with temporary.open('w') as handle:
        for chunk in json.JSONEncoder(ensure_ascii=False,separators=(',',':')).iterencode(data):
            handle.write(chunk)
        handle.write('\n')
    os.replace(temporary,path)


def continue_empty_wave(*, has_queue, budget_blocked):
    return has_queue and not budget_blocked


def record_work_wave(*,wave,elapsed,tokens,forward_calls,runtime_work,trees,seen_solutions):
    """Observed post-executor boundary; inclusive cost of every generated proposal.

    Forward calls are engine-global, NOT the sum of request participation.
    Prompt tokens are logical request length, not uncached GPU prefill/FLOPs.
    A route is credited only after its whole generation/executor wave finishes.
    """
    newly=sorted(identifier for identifier,tree in trees.items()
                 if getattr(tree,'first_solution_work',None) is not None and identifier not in seen_solutions)
    seen_solutions.update(newly)
    return {'wave':wave,'inference_elapsed_seconds':elapsed,'output_tokens':tokens,
        'model_forward_calls':forward_calls,
        'prompt_tokens':sum(cost['prompt_tokens'] for cost in runtime_work.values()),
        'policy_request_batches':sum(cost['policy_request_batches'] for cost in runtime_work.values()),
        'cpu_executor_thread_seconds':sum(cost.get('cpu_executor_thread_seconds',0.) for cost in runtime_work.values()),
        'shared_policy_cache_hits':sum(cost.get('shared_policy_cache_hits',0) for cost in runtime_work.values()),
        'reused_output_tokens':sum(cost.get('reused_output_tokens',0) for cost in runtime_work.values()),
        'new_stock_solved_ids':newly,'credited_stock_solved_targets':len(seen_solutions),
        'boundary':'post_executor_whole_generation_wave_v1','actual_GPU_FLOPs_claim':False}


def select_context(controller, mode, forced_key):
    if isinstance(controller, RouteNetwork):
        return controller.next_context()
    if mode == 'greedy':
        context = controller.contexts[forced_key]
        return None if context.node.terminal or context.resource_closed else context
    return controller.next_context()


def load_route_stock(manifest_path, cohort, data_sha, expected_rows):
    assets = json.loads(Path(manifest_path).read_text())
    if not assets['stock_preparation_complete']:
        raise ValueError('stock canonicalization is still pending')
    sources = assets['cohorts'] if cohort=='all' else {cohort:assets['cohorts'][cohort]}
    source_sha = assets['full_targets_sha256'] if cohort=='all' else sources[cohort]['targets_sha256']
    source_rows = assets['full_target_rows'] if cohort=='all' else sources[cohort]['rows']
    if source_sha != data_sha or source_rows != expected_rows:
        raise ValueError('full frozen route cohort mismatch')
    stock_sets={}
    for name in {source['stock'] for source in sources.values()}:
        stock=assets['stocks'][name]
        if sha256(stock['canonical_path']) != stock['canonical_sha256']:
            raise ValueError('frozen stock SHA mismatch')
        with Path(stock['canonical_path']).open() as handle:
            values={line.strip() for line in handle if line.strip()}
        if len(values)!=stock['normalized_rows']:raise ValueError('frozen stock cardinality mismatch')
        stock_sets[name]=values
    return {name:stock_sets[source['stock']] for name,source in sources.items()}, {
        'route_assets_sha':sha256(manifest_path),'route_cohort':cohort,
        'stock_source_shas':{name:assets['stocks'][name]['sha256'] for name in stock_sets},
        'stock_canonical_shas':{name:assets['stocks'][name]['canonical_sha256'] for name in stock_sets},
        'stock_canonical_sha_by_cohort':{
            name:assets['stocks'][source['stock']]['canonical_sha256'] for name,source in sources.items()},
        'stock_normalization':assets['normalization']}


def verify_saved_run(saved,expected):
    for key,value in expected.items():
        if saved.get(key)!=value:
            raise ValueError(f'saved run contract mismatch: {key}')


class ForwardCallCounter:
    """Instrument the pinned vLLM V0 TP1 executor, not API request counts."""
    def __init__(self,engine):
        self.total_calls=0
        self.by_request=Counter()
        executor=engine.llm_engine.model_executor
        original=executor.execute_model
        def execute(*args,**kwargs):
            request=kwargs.get('execute_model_req') if not args else args[0]
            groups=request.seq_group_metadata_list if request is not None else []
            result=original(*args,**kwargs)
            if groups:
                self.total_calls+=1
                for identifier in {str(group.request_id) for group in groups}:
                    self.by_request[identifier]+=1
            return result
        executor.execute_model=execute


class HostMemoryGuard:
    """RAM-only guard; no chemical branch/depth or state-count cutoff."""
    def __init__(self, *, world, initial_rss, available):
        self.initial_rss = initial_rss
        self.growth_allowance = available * .60 / world
        self.reserve = available * .10

    def exhausted(self, *, rss, available):
        return rss - self.initial_rss >= self.growth_allowance or available <= self.reserve


def evaluate(tree, reference):
    expected=reference_structural_precursor(reference)
    root=tree.contexts[tree.root_key].node
    unique={}
    for context in tree.terminals():
        parts=split_precursor_endpoints(context.node.state,root.state)
        key=chemical_key(parts.structural)
        if key not in unique or context.node.policy_score>unique[key]['score']:
            unique[key]={'context':context.key,'structural_precursor':key,'full_mixture':context.node.state,
                         'structural_precursor_mapped':parts.structural,
                         'score':context.node.policy_score,'mean_nll':-context.node.policy_score,
                         'matches_reference':structural_exact(key,expected)}
    ranked=sorted(unique.values(),key=lambda r:r['score'],reverse=True)
    return {'id':tree.case_id,'expected_structural_precursor':expected,'ranked_candidates':ranked,
            'top1':any(r['matches_reference'] for r in ranked[:1]),
            'top5':any(r['matches_reference'] for r in ranked[:5]),
            'top10':any(r['matches_reference'] for r in ranked[:10]),
            'any_endpoint_hit':any(r['matches_reference'] for r in ranked),
            'terminal_count':len(ranked),'work':dict(tree.work),'stop_reason':tree.stop_reason}


def load_source(path, expected_rows):
    rows=[json.loads(line) for line in path.open() if line.strip()]
    ids=[row['id'] for row in rows]
    if len(rows)!=expected_rows or len(set(ids))!=expected_rows:
        raise ValueError('full source denominator/IDs mismatch')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--source-sha',required=True)
    p.add_argument('--expected-rows',type=int,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3-0.6B')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['tree','greedy'],default='tree')
    p.add_argument('--rank',type=int,required=True)
    p.add_argument('--world',type=int,default=8)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--initial-token-checkpoint','--task-token-budget',dest='task_token_budget',
                   type=int,default=80000000,help='initial shared work checkpoint, NOT a token ceiling')
    p.add_argument('--initial-wall-checkpoint-hours','--wall-hours',dest='wall_hours',
                   type=float,default=24,help='initial elapsed-work checkpoint, NOT a time ceiling')
    p.add_argument('--batch-prompts',type=int,default=32)
    p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--max-model-len',type=int,default=8192)
    p.add_argument('--shared-cache',type=Path,
                   help='shared durable state/policy/execution cache; defaults beside output')
    p.add_argument('--route-assets-manifest',type=Path)
    p.add_argument('--route-cohort')
    args=p.parse_args()
    if args.world<1 or args.task_token_budget<args.world or not math.isfinite(args.wall_hours) or args.wall_hours<=0:
        p.error('positive finite checkpoints and world size required')
    if bool(args.route_assets_manifest) != bool(args.route_cohort):
        p.error('route manifest and cohort must be provided together')
    route_mode = args.route_assets_manifest is not None
    if sha256(args.data)!=args.source_sha:raise ValueError('source SHA mismatch')
    if sha256(args.adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('Stage-II weight SHA mismatch')
    if not 0<=args.rank<args.world:raise ValueError('invalid shard')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1:
        raise ValueError('each vLLM worker needs exactly one visible GPU')
    validate_probability_contract(temperature=1,top_p=1,repetition_penalty=1)
    os.environ.setdefault('VLLM_USE_V1','0')
    if os.environ['VLLM_USE_V1']!='0':raise ValueError('pinned V0 engine required for exact executor accounting')
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    if vllm.__version__.split('+')[0]!='0.8.5':raise ValueError('pinned vLLM0.8.5 required')
    rows=load_source(args.data,args.expected_rows)
    stock,route_contract = load_route_stock(args.route_assets_manifest,args.route_cohort,args.source_sha,args.expected_rows) if route_mode else (None,{})
    selected=[r for i,r in enumerate(rows) if i%args.world==args.rank]
    root_dir=args.output/args.mode/f'rank{args.rank:02d}'
    root_dir.mkdir(parents=True,exist_ok=True)
    expected_contract={'model':args.model,'revision':REVISION,'adapter_sha':ADAPTER_SHA,
        'source_sha':args.source_sha,'mode':args.mode,'rank':args.rank,'world':args.world,
        'full_denominator':args.expected_rows,'seed':args.seed,'task_token_budget':args.task_token_budget,
        'wall_hours':args.wall_hours,'task_budget_mode':'renewable_checkpoints_no_token_or_time_cap_v1',
        'max_model_len':args.max_model_len,'raw_unwarped_nll':True,
        'evaluation_unit':'stock_connected_AND_route' if route_mode else 'single_reaction_endpoint',
        'persistence_contract':'shared_cache_incremental_journal_v1',
        'sampling_seed_contract':'case_independent_native_context_v1',
        'code_sha':os.environ.get('MECHET_TREE_CODE_SHA','not_supplied'),**route_contract}
    if (root_dir/'complete.json').exists():
        verify_saved_run(json.loads((root_dir/'complete.json').read_text())['manifest'],expected_contract)
        print({'phase':'already-complete','rank':args.rank,'mode':args.mode},flush=True);return 0
    # Resume never overwrites a partial raw history under a changed run.
    if (root_dir/'raw_generations.jsonl').exists():
        raise ValueError('partial shard exists; preserve it and select a new attempt directory')
    lineage=json.loads((args.adapter/'data_contract.json').read_text())
    if lineage.get('environment_revision')!='natural_language_electron_event_history_v2':
        raise ValueError('not the Stage-II compressed-history protocol')
    engine=LLM(model=args.model,revision=REVISION,tokenizer=args.model,trust_remote_code=True,
               dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.82,
               max_model_len=args.max_model_len,max_num_seqs=256,enable_prefix_caching=True,
               enable_lora=True,max_lora_rank=16,enforce_eager=True,num_scheduler_steps=1,seed=args.seed+args.rank)
    forwards=ForwardCallCounter(engine)
    tokenizer=engine.get_tokenizer()
    import psutil
    process = psutil.Process()
    memory_guard = HostMemoryGuard(world=args.world, initial_rss=process.memory_info().rss,
                                   available=available_host_memory())
    lora=LoRARequest('frozen_flowER_stage2',1,str(args.adapter.resolve()))
    manifest={'model':args.model,'revision':REVISION,'adapter':str(args.adapter),'adapter_sha':ADAPTER_SHA,
              'source':str(args.data),'source_sha':args.source_sha,'full_denominator':args.expected_rows,
              'shard_rows':len(selected),'rank':args.rank,'world':args.world,'mode':args.mode,
              'sampling_temperature':0 if args.mode=='greedy' else 1,'top_p':1,'raw_unwarped_nll':True,
              'dynamic_depth':True,'dynamic_breadth':True,'fixed_beam_pruning':False,
              'task_token_budget':args.task_token_budget,'wall_hours':args.wall_hours,
              'task_budget_mode':'renewable_checkpoints_no_token_or_time_cap_v1',
              'max_model_len':args.max_model_len,'reference_visible_to_policy':False,
              'self_improvement':False,'seed':args.seed,'history_contract':lineage['environment_revision']}
    manifest.update(route_contract)
    manifest['evaluation_unit'] = 'stock_connected_AND_route' if route_mode else 'single_reaction_endpoint'
    manifest['code_sha'] = os.environ.get('MECHET_TREE_CODE_SHA','not_supplied')
    manifest['host_memory_growth_allowance_bytes'] = memory_guard.growth_allowance
    manifest['host_memory_persistence_reserve_bytes'] = memory_guard.reserve
    manifest['work_wave_contract'] = 'post_executor_whole_generation_wave_v1'
    # Do not include source IDs/stocks/labels in the reusable cache namespace.
    # Hash executable dependencies conservatively; a changed executor must not
    # reuse old chemical decisions, even if a caller forgets a code SHA.
    cache_contract={'model':args.model,'revision':REVISION,'adapter_sha':ADAPTER_SHA,
        'history_contract':lineage['environment_revision'],'dtype':'bfloat16',
        'backend':'vllm_0.8.5_V0_TP1','executor_code_sha':fingerprint({
            str(path.relative_to(REPO)):sha256(path) for path in sorted([
                *(REPO/'src/mechet').glob('*.py'),
                REPO/'scripts/run_natural_language_value_search.py',
                REPO/'scripts/build_natural_language_event_sft.py',
                REPO/'scripts/eval_natural_language_event_local.py'])}),
        'sampling_seed_contract':'case_independent_native_context_v1'}
    shared_store=SharedSearchStore(args.shared_cache or args.output.parent/'shared_electron_search_v1',cache_contract)
    manifest.update(shared_cache_namespace=shared_store.namespace,shared_cache_path=str(shared_store.root),
        persistence_contract='shared_cache_incremental_journal_v1',
        sampling_seed_contract='case_independent_native_context_v1')
    dump(root_dir/'manifest.json',manifest)
    references={r['id']:r for r in selected}
    runtime_work={r['id']:{'policy_request_batches':0,'prompt_tokens':0,
                          'candidate_prompt_token_equivalents':0,'model_forward_batches_participated':0,
                          'shared_policy_cache_hits':0,'reused_output_tokens':0,
                          'cpu_executor_wall_seconds':0.0,'cpu_executor_thread_seconds':0.0} for r in selected}
    trees={}
    reports=[]
    for row in selected:
        try:
            # ONLY unmapped PRODUCT is exposed to the tree/prompt builder.
            trees[row['id']]=(RouteNetwork(row['id'],row['product_unmapped'],stock[row['cohort']],
                                          mode=args.mode,shared_store=shared_store)
                               if route_mode else ElectronTree(row['id'],row['product_unmapped'],shared_store=shared_store))
        except Exception as exc:
            reports.append({'id':row['id'],'top1':False,'top5':False,'top10':False,
                            'any_endpoint_hit':False,'terminal_count':0,'stop_reason':'ROOT_PARSE_FAILURE',
                            'error':str(exc),'stock_solved':False,'already_in_stock':False,
                            'work':{'sampled_tokens':0,'raw_candidates':0}})
    journals={}
    for identifier,controller in trees.items():
        case_contract=dict(manifest)
        if route_mode:
            cohort=references[identifier]['cohort']
            case_contract.update(case_cohort=cohort,
                case_stock_sha=route_contract['stock_canonical_sha_by_cohort'][cohort])
        journal=SearchJournal(root_dir/'journals',identifier,case_contract)
        journal.initialize(controller)
        journals[identifier]=journal
    queue=deque(trees)
    started=time.monotonic()
    total_tokens=0
    checkpoints=DynamicWorkCheckpoints(initial_tokens=args.task_token_budget//args.world,
                                       initial_seconds=args.wall_hours*3600)
    checkpoint_events=[]
    raw_file=(root_dir/'raw_generations.jsonl').open('a')
    prompt_file=(root_dir/'policy_contexts.jsonl').open('a')
    wave_file=(root_dir/'work_waves.jsonl').open('a')
    wave_index=0
    credited_solutions=set()
    def save_wave():
        value=record_work_wave(wave=wave_index,elapsed=time.monotonic()-started,tokens=total_tokens,
            forward_calls=forwards.total_calls,runtime_work=runtime_work,trees=trees,seen_solutions=credited_solutions)
        wave_file.write(json.dumps(value,ensure_ascii=False)+'\n');wave_file.flush()
        return value
    save_wave()
    seen_prompts=set()
    forced={identifier:tree.root_key for identifier,tree in trees.items()} if not route_mode else {}
    pool=ThreadPoolExecutor(max_workers=args.cpu_workers)
    def finish(identifier,reason):
        tree=trees.pop(identifier)
        tree.stop_reason=reason
        stem=hashlib.sha256(identifier.encode()).hexdigest() if route_mode else identifier
        dump(root_dir/'trees'/f'{stem}.json',tree.to_dict())
        # Endpoint labels are attached AFTER inference; never read by tree selection.
        report=tree.report() if route_mode else evaluate(tree,references[identifier])
        report['work'].update(runtime_work[identifier])
        if route_mode:report['cohort']=references[identifier]['cohort']
        report.update(tree_artifact=f'trees/{stem}.json',evaluation_artifact=f'evaluations/{stem}.json')
        dump(root_dir/'evaluations'/f'{stem}.json',report)
        reports.append(report)
    try:
        resource_stop = ''
        while queue:
            if memory_guard.exhausted(rss=process.memory_info().rss, available=available_host_memory()):
                resource_stop = 'TASK_HOST_MEMORY_BUDGET'
                break
            event=checkpoints.update(tokens=total_tokens,seconds=time.monotonic()-started,active_roots=len(queue))
            if event:
                checkpoint_events.append(event)
                dump(root_dir/'work_checkpoints.json',checkpoint_events)
                print({**event,'rank':args.rank,'mode':args.mode},flush=True)
            batch=[]
            budget_blocked = False
            visits=len(queue)
            while queue and len(batch)<args.batch_prompts and visits:
                visits-=1
                identifier=queue.popleft()
                tree=trees[identifier]
                context=select_context(tree,args.mode,forced.get(identifier))
                if context is None:
                    finish(identifier,'ROUTE_SUPPORT_STABLE_OR_RESOURCE_CLOSED' if route_mode else
                           ('GREEDY_TERMINAL_OR_INVALID' if args.mode=='greedy' else 'ADAPTIVE_SUPPORT_STABLE_OR_RESOURCE_CLOSED'))
                    continue
                messages=[{'role':'system','content':SYSTEM}, {'role':'user','content':policy_prompt(
                    context.node.target,context.node.state,include_inventory=True,
                    actions=context.node.actions,compact_history=True)}]
                prompt=render_qwen_sft_tool_prefix(tokenizer,messages,tools=TOOLS)
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                room=args.max_model_len-len(ids)
                if room<2:
                    context.resource_closed='CONTEXT_MEMORY_BUDGET'
                    journals[identifier].record(tree,context.key,[])
                    queue.append(identifier);continue
                allowance=min(context.max_new_tokens,room)
                n=1 if args.mode=='greedy' else context.stats.next_batch
                n=checkpoints.candidate_count(n,completion_room=allowance)
                if n<1:
                    budget_blocked = True
                    queue.appendleft(identifier);break
                params=SamplingParams(n=n,temperature=0 if args.mode=='greedy' else 1,top_p=1,top_k=-1,
                    repetition_penalty=1,presence_penalty=0,frequency_penalty=0,min_tokens=0,
                    max_tokens=allowance,logprobs=0,skip_special_tokens=False,
                    seed=sampling_seed(args.seed,context.key,context.stats.batches))
                batch.append((identifier,context.key,ids,params,n,allowance))
                prompt_key=f'{identifier}:{context.key}'
                if prompt_key not in seen_prompts:
                    seen_prompts.add(prompt_key)
                    prompt_file.write(json.dumps({'case_id':identifier,'context':context.key,'text':prompt,
                                                  'token_ids':ids},ensure_ascii=False)+'\n')
            if not batch:
                if continue_empty_wave(has_queue=bool(queue), budget_blocked=budget_blocked):
                    continue
                break
            generation_start=time.monotonic()
            requests=[{'context':x[1],'prompt_ids':x[2],'n':x[4],'max_tokens':x[5],
                'temperature':0 if args.mode=='greedy' else 1,'top_p':1,'top_k':-1,
                'repetition_penalty':1,'presence_penalty':0,'frequency_penalty':0,
                'skip_special_tokens':False,'logprobs':0,'min_tokens':0,
                'seed':x[3].seed} for x in batch]
            def generate_missing(missing):
                outputs=engine.generate([{'prompt_token_ids':r['prompt_ids']} for r in missing],
                    [SamplingParams(**{k:v for k,v in r.items() if k not in {'context','prompt_ids'}})
                     for r in missing],lora_request=lora,use_tqdm=False)
                records=[]
                for result in outputs:
                    proposals=[]
                    for generated in result.outputs:
                        tokens=list(generated.token_ids)
                        raw=tokenizer.decode(tokens,skip_special_tokens=False)
                        name,arguments,error=prediction_call(raw,tokenizer)
                        value=generated.cumulative_logprob
                        if value is None or not math.isfinite(value):
                            raise RuntimeError('missing/nonfinite unwarped model NLL')
                        proposals.append(asdict(Proposal(name,arguments,raw,float(value),len(tokens),
                            str(error or ''),generated.finish_reason=='length',tokens)))
                    records.append({'request_id':str(result.request_id),
                        'cached_prefix_tokens':getattr(result,'num_cached_tokens',None),
                        'generations':proposals,'engine_finish_reasons':[x.finish_reason for x in result.outputs]})
                return records
            resolved=resolve_policy_requests(shared_store,requests,generate_missing)
            generation_seconds=time.monotonic()-generation_start
            pending=[]
            for item,request,result in zip(batch,requests,resolved,strict=True):
                identifier,key,_,_,_,allowance=item
                if result is None:
                    queue.append(identifier)  # another worker owns the shared request
                    continue
                record,cache_hit=result
                proposals=[Proposal(**value) for value in record['generations']]
                tree=trees[identifier]
                cost=runtime_work[identifier]
                if cache_hit:
                    cost['shared_policy_cache_hits']+=1
                    cost['reused_output_tokens']+=sum(x.tokens for x in proposals)
                else:
                    total_tokens+=sum(x.tokens for x in proposals)
                    cost['policy_request_batches']+=1
                    cost['prompt_tokens']+=len(item[2])
                    cost['candidate_prompt_token_equivalents']+=len(item[2])*len(proposals)
                    cost['model_forward_batches_participated']+=forwards.by_request.pop(record['request_id'],0)
                raw_file.write(json.dumps({'case_id':identifier,'context':key,'prompt_tokens':len(item[2]),
                            'request_id':None if cache_hit else record['request_id'],
                            'cached_prefix_tokens':None if cache_hit else record['cached_prefix_tokens'],
                            'shared_cache_hit':cache_hit,'shared_request_key':fingerprint(request),
                            'shared_cache_namespace':shared_store.namespace,
                            'generations':[asdict(x) for x in proposals],
                            'engine_finish_reasons':record['engine_finish_reasons']},ensure_ascii=False)+'\n')
                def incorporate_timed(controller,key,proposals):
                    wall=time.monotonic();cpu=time.thread_time()
                    result=controller.incorporate(key,proposals)
                    return result,time.monotonic()-wall,time.thread_time()-cpu
                pending.append((identifier,key,proposals,pool.submit(incorporate_timed,tree,key,proposals)))
            cpu_start=time.monotonic()
            for identifier,key,proposals,future in pending:
                edges,cpu_wall,cpu_thread=future.result()
                runtime_work[identifier]['cpu_executor_wall_seconds']+=cpu_wall
                runtime_work[identifier]['cpu_executor_thread_seconds']+=cpu_thread
                tree=trees[identifier]
                context=tree.contexts[key]
                if context.max_new_tokens>args.max_model_len-len(next(x[2] for x in batch if x[0]==identifier)):
                    context.max_new_tokens=args.max_model_len-len(next(x[2] for x in batch if x[0]==identifier))
                    if proposals and all(x.truncated for x in proposals):context.resource_closed='COMPLETION_CONTEXT_BUDGET'
                if args.mode=='greedy' and not route_mode:
                    accepted=[e for e in edges if e.get('child_context')]
                    if accepted:forced[identifier]=accepted[0]['child_context']
                    elif not any(x.truncated for x in proposals):context.resource_closed='GREEDY_INVALID_ACTION'
                # Durable native contexts, accepted/rejected edges and complete
                # outer reaction state BEFORE scheduling another expansion.
                journals[identifier].record(tree,key,edges)
                queue.append(identifier)
            if not pending:
                time.sleep(.05)  # shared requests busy; avoid a CPU spin loop
            raw_file.flush();prompt_file.flush()
            wave_index+=1
            wave_work=save_wave()
            print({'phase':'dynamic-search','rank':args.rank,'mode':args.mode,'finished':len(reports),
                   'denominator':len(selected),'active_roots':len(queue),'output_tokens':total_tokens,
                   'model_forward_calls':wave_work['model_forward_calls'],
                   'stock_solved_targets':wave_work['credited_stock_solved_targets'],
                   'generation_seconds':round(generation_seconds,3),'cpu_join_seconds':round(time.monotonic()-cpu_start,3)},flush=True)
        for identifier in list(trees):
            reason=resource_stop or 'NO_EXECUTABLE_CONTEXT_AVAILABLE'
            finish(identifier,reason)
    finally:
        pool.shutdown(wait=True)
        raw_file.close();prompt_file.close();wave_file.close()
    if {r['id'] for r in reports}!={r['id'] for r in selected}:raise RuntimeError('lost full-cohort rows')
    dump(root_dir/'complete.json',{'manifest':manifest,'rows':reports,'tokens':total_tokens,
                                 'work_checkpoint_events':checkpoint_events,
                                 'work_waves_artifact':'work_waves.jsonl',
                                 'work_waves_sha256':sha256(root_dir/'work_waves.jsonl'),
                                 'model_forward_calls':forwards.total_calls,'seconds':time.monotonic()-started})
    print({'phase':'dynamic-search-complete','rank':args.rank,'mode':args.mode,'rows':len(reports),'tokens':total_tokens},flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
