#!/usr/bin/env python3
"""Full-cohort frozen-Qwen vLLM search with adaptive breadth/depth and journals.

Eight independent one-GPU workers; batched prompts from different molecules;
CPU-parallel executor work. No fixed chemical-depth/beam/import cap. Only the
declared job-level token/wall budget and model/memory limits stop exploration.
"""
from __future__ import annotations

import argparse
from collections import deque
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
from mechet.dynamic_electron_tree import ElectronTree, Proposal, validate_probability_contract, chemical_key, available_host_memory
from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from scripts.run_natural_language_value_search import policy_prompt
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.eval_natural_language_event_local import prediction_call

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
    p.add_argument('--task-token-budget',type=int,default=80000000)
    p.add_argument('--wall-hours',type=float,default=24)
    p.add_argument('--batch-prompts',type=int,default=32)
    p.add_argument('--cpu-workers',type=int,default=8)
    p.add_argument('--max-model-len',type=int,default=8192)
    args=p.parse_args()
    if sha256(args.data)!=args.source_sha:raise ValueError('source SHA mismatch')
    if sha256(args.adapter/'adapter_model.safetensors')!=ADAPTER_SHA:raise ValueError('Stage-II weight SHA mismatch')
    if not 0<=args.rank<args.world:raise ValueError('invalid shard')
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=1:
        raise ValueError('each vLLM worker needs exactly one visible GPU')
    validate_probability_contract(temperature=1,top_p=1,repetition_penalty=1)
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    if vllm.__version__.split('+')[0]!='0.8.5':raise ValueError('pinned vLLM0.8.5 required')
    rows=load_source(args.data,args.expected_rows)
    selected=[r for i,r in enumerate(rows) if i%args.world==args.rank]
    root_dir=args.output/args.mode/f'rank{args.rank:02d}'
    root_dir.mkdir(parents=True,exist_ok=True)
    if (root_dir/'complete.json').exists():
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
               enable_lora=True,max_lora_rank=16,enforce_eager=True,seed=args.seed+args.rank)
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
              'max_model_len':args.max_model_len,'reference_visible_to_policy':False,
              'self_improvement':False,'seed':args.seed,'history_contract':lineage['environment_revision']}
    manifest['host_memory_growth_allowance_bytes'] = memory_guard.growth_allowance
    manifest['host_memory_persistence_reserve_bytes'] = memory_guard.reserve
    dump(root_dir/'manifest.json',manifest)
    references={r['id']:r for r in selected}
    trees={}
    reports=[]
    for row in selected:
        try:
            # ONLY unmapped PRODUCT is exposed to the tree/prompt builder.
            trees[row['id']]=ElectronTree(row['id'],row['product_unmapped'])
        except Exception as exc:
            reports.append({'id':row['id'],'top1':False,'top5':False,'top10':False,
                            'any_endpoint_hit':False,'terminal_count':0,'stop_reason':'ROOT_PARSE_FAILURE',
                            'error':str(exc),'work':{'sampled_tokens':0,'raw_candidates':0}})
    queue=deque(trees)
    started=time.monotonic()
    total_tokens=0
    token_budget=args.task_token_budget//args.world
    raw_file=(root_dir/'raw_generations.jsonl').open('a')
    prompt_file=(root_dir/'policy_contexts.jsonl').open('a')
    seen_prompts=set()
    forced={identifier:tree.root_key for identifier,tree in trees.items()}
    pool=ThreadPoolExecutor(max_workers=args.cpu_workers)
    def finish(identifier,reason):
        tree=trees.pop(identifier)
        tree.stop_reason=reason
        dump(root_dir/'trees'/f'{identifier}.json',tree.to_dict())
        # Endpoint labels are attached AFTER inference; never read by tree selection.
        report=evaluate(tree,references[identifier])
        dump(root_dir/'evaluations'/f'{identifier}.json',report)
        reports.append(report)
    try:
        resource_stop = ''
        while queue and total_tokens<token_budget and time.monotonic()-started<args.wall_hours*3600:
            if memory_guard.exhausted(rss=process.memory_info().rss, available=available_host_memory()):
                resource_stop = 'TASK_HOST_MEMORY_BUDGET'
                break
            batch=[]
            budget_blocked = False
            visits=len(queue)
            while queue and len(batch)<args.batch_prompts and visits:
                visits-=1
                identifier=queue.popleft()
                tree=trees[identifier]
                if args.mode=='greedy':
                    context=tree.contexts[forced[identifier]]
                    if context.node.terminal or context.resource_closed:
                        finish(identifier,'GREEDY_TERMINAL' if context.node.terminal else context.resource_closed);continue
                else:
                    context=tree.next_context()
                    if context is None:
                        finish(identifier,'ADAPTIVE_SUPPORT_STABLE_OR_RESOURCE_CLOSED');continue
                messages=[{'role':'system','content':SYSTEM}, {'role':'user','content':policy_prompt(
                    context.node.target,context.node.state,include_inventory=True,
                    actions=context.node.actions,compact_history=True)}]
                prompt=render_qwen_sft_tool_prefix(tokenizer,messages,tools=TOOLS)
                ids=tokenizer.encode(prompt,add_special_tokens=False)
                room=args.max_model_len-len(ids)
                if room<2:
                    context.resource_closed='CONTEXT_MEMORY_BUDGET'
                    queue.append(identifier);continue
                allowance=min(context.max_new_tokens,room)
                n=1 if args.mode=='greedy' else context.stats.next_batch
                remaining=token_budget-total_tokens-sum(x[4]*x[5] for x in batch)
                n=min(n,remaining//allowance)
                if n<1:
                    budget_blocked = True
                    queue.appendleft(identifier);break
                params=SamplingParams(n=n,temperature=0 if args.mode=='greedy' else 1,top_p=1,top_k=-1,
                    repetition_penalty=1,presence_penalty=0,frequency_penalty=0,min_tokens=0,
                    max_tokens=allowance,logprobs=0,skip_special_tokens=False,
                    seed=int(hashlib.sha256(f'{args.seed}:{identifier}:{context.key}:{context.stats.batches}'.encode()).hexdigest()[:8],16))
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
            outputs=engine.generate([{'prompt_token_ids':x[2]} for x in batch],[x[3] for x in batch],
                                    lora_request=lora,use_tqdm=False)
            generation_seconds=time.monotonic()-generation_start
            pending=[]
            for item,result in zip(batch,outputs,strict=True):
                identifier,key,_,_,_,allowance=item
                proposals=[]
                for generated in result.outputs:
                    tokens=list(generated.token_ids)
                    raw=tokenizer.decode(tokens,skip_special_tokens=False)
                    name,arguments,error=prediction_call(raw,tokenizer)
                    truncated=generated.finish_reason=='length'
                    value=generated.cumulative_logprob
                    if value is None or not math.isfinite(value):
                        raise RuntimeError('missing/nonfinite unwarped model NLL')
                    proposals.append(Proposal(name,arguments,raw,float(value),len(tokens),str(error or ''),truncated,tokens))
                    total_tokens+=len(tokens)
                tree=trees[identifier]
                raw_file.write(json.dumps({'case_id':identifier,'context':key,'prompt_tokens':len(item[2]),
                            'generations':[asdict(x) for x in proposals], 'engine_finish_reasons':[x.finish_reason for x in result.outputs]},ensure_ascii=False)+'\n')
                pending.append((identifier,key,proposals,pool.submit(tree.incorporate,key,proposals)))
            cpu_start=time.monotonic()
            for identifier,key,proposals,future in pending:
                edges=future.result()
                tree=trees[identifier]
                context=tree.contexts[key]
                if context.max_new_tokens>args.max_model_len-len(next(x[2] for x in batch if x[0]==identifier)):
                    context.max_new_tokens=args.max_model_len-len(next(x[2] for x in batch if x[0]==identifier))
                    if proposals and all(x.truncated for x in proposals):context.resource_closed='COMPLETION_CONTEXT_BUDGET'
                if args.mode=='greedy':
                    accepted=[e for e in edges if e.get('child_context')]
                    if accepted:forced[identifier]=accepted[0]['child_context']
                    elif not any(x.truncated for x in proposals):context.resource_closed='GREEDY_INVALID_ACTION'
                queue.append(identifier)
            raw_file.flush();prompt_file.flush()
            print({'phase':'dynamic-search','rank':args.rank,'mode':args.mode,'finished':len(reports),
                   'denominator':len(selected),'active_roots':len(queue),'output_tokens':total_tokens,
                   'generation_seconds':round(generation_seconds,3),'cpu_join_seconds':round(time.monotonic()-cpu_start,3)},flush=True)
        for identifier in list(trees):
            reason=resource_stop or ('TASK_TOKEN_BUDGET' if total_tokens>=token_budget else ('TASK_WALL_BUDGET' if time.monotonic()-started>=args.wall_hours*3600 else 'INSUFFICIENT_REMAINING_TOKEN_ROOM'))
            finish(identifier,reason)
    finally:
        pool.shutdown(wait=True)
        raw_file.close();prompt_file.close()
    if {r['id'] for r in reports}!={r['id'] for r in selected}:raise RuntimeError('lost full-cohort rows')
    dump(root_dir/'complete.json',{'manifest':manifest,'rows':reports,'tokens':total_tokens,'seconds':time.monotonic()-started})
    print({'phase':'dynamic-search-complete','rank':args.rank,'mode':args.mode,'rows':len(reports),'tokens':total_tokens},flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
