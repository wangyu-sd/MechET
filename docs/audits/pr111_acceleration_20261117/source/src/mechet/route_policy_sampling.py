"""Rolling-batch native Qwen3.5 sampling with private per-request RNG streams."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .route_phase_profile import phase


def request_seed(request_id, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{request_id}'.encode()).digest()[:8],'big') % (2**63-1)


class TokenLogprobs:
    def __init__(self,eos_id):
        self.eos_id=eos_id
        self.previous=self.done=None
        self.steps=[]

    def finish(self,tokens):
        import torch
        if self.previous is None:
            return
        active=~self.done
        chosen=self.previous.gather(1,tokens[:,None]).squeeze(1)
        self.steps.append((chosen,active))
        self.done=self.done | (tokens==self.eos_id)
        self.previous=None

    def __call__(self,input_ids,scores):
        import torch
        self.finish(input_ids[:,-1])
        if self.done is None:
            self.done=torch.zeros(scores.shape[0],dtype=torch.bool,device=scores.device)
        self.previous=scores.float().log_softmax(-1)
        return scores

    def values(self):
        import torch
        if not self.steps:
            return []
        scores=torch.stack([v for v,a in self.steps],dim=1).tolist()
        masks=torch.stack([a for v,a in self.steps],dim=1).tolist()
        return [[v for v,a in zip(row,mask) if a] for row,mask in zip(scores,masks)]


class PrivateSampling:
    """Exact categorical draw from original logits, independent of batch order.

    The final HF processor pins that already sampled token; TokenLogprobs must
    run FIRST and records the original categorical distribution, not the pin.
    No top-k/top-p/temperature/penalty or constrained-action mask is introduced.
    """
    def __init__(self,uids,seed,device,*,finished=None,on_failure=None,prompt_width=0):
        import torch
        self.request_ids=list(uids)
        self.finished,self.on_failure,self.prompt_width=finished,on_failure,prompt_width
        self.decode_step=0
        self.generators=[torch.Generator(device=device).manual_seed(request_seed(u,seed)) for u in self.request_ids]

    def __call__(self,input_ids,scores):
        import torch
        probabilities=scores.float().softmax(-1)
        valid=torch.isfinite(probabilities).all(-1) & (probabilities.sum(-1)>0)
        if not bool(valid.all()):
            bad=(~valid).nonzero().flatten().tolist()
            from .route_sampling_diagnostics import (InvalidPolicyDistribution,
                                                     describe_distribution_failure)
            report=dict(decode_step=self.decode_step,invalid_rows=bad,
                        request_ids=self.request_ids)
            try:
                report=describe_distribution_failure(scores,input_ids,bad_rows=bad,
                    request_ids=self.request_ids,
                    finished=self.finished() if self.finished is not None else None,
                    decode_step=self.decode_step,prompt_width=self.prompt_width)
            except Exception as diagnostic_error:
                report['diagnostic_error']=str(diagnostic_error)
            error=InvalidPolicyDistribution(bad,report)
            if self.on_failure is not None:
                try:self.on_failure(dict(event='policy-distribution-failure',**report))
                except Exception as write_error:
                    error.add_note('Failed to persist sampling diagnostics: '+str(write_error))
            raise error
        tokens=torch.stack([torch.multinomial(row,1,generator=g)[0]
                            for row,g in zip(probabilities,self.generators)])
        result=torch.full_like(scores,-float('inf'))
        self.decode_step+=1
        return result.scatter(1,tokens[:,None],0.)


def append_receipt(path,value,cfg=None):
    with Path(path).open('a') as sink:
        with phase(cfg,'io.journal_write',rows=1):
            sink.write(json.dumps(value,ensure_ascii=False)+'\n')
        with phase(cfg,'io.journal_flush'):
            sink.flush()
            os.fsync(sink.fileno())


def read_owned_journal(path):
    """Recover only an uncommitted final tail; preserve it in a sibling file.

    Caller must hold the worker's exclusive lease. Interior corruption fails.
    This is never called on source datasets, only this run's owned journals.
    """
    path=Path(path)
    if not path.exists():
        return []
    rows=[]; offset=0
    with path.open('rb') as reader:
        for line in reader:
            try:
                value=json.loads(line)
            except (json.JSONDecodeError,UnicodeDecodeError):
                if line.endswith(b'\n') or reader.read(1):
                    raise ValueError('CORRUPT_INTERIOR_JOURNAL:'+str(path))
                backup=path.with_name(path.name+'.incomplete-'+str(time.time_ns()))
                with backup.open('xb') as sink:
                    sink.write(line); sink.flush(); os.fsync(sink.fileno())
                with path.open('r+b') as sink:
                    sink.truncate(offset); sink.flush(); os.fsync(sink.fileno())
                print(json.dumps(dict(event='recover-uncommitted-journal-tail',path=str(path),
                    preserved_tail=str(backup),committed_bytes=offset)),flush=True)
                break
            rows.append(value); offset+=len(line)
            if not line.endswith(b'\n'):
                with path.open('ab') as sink:
                    sink.write(b'\n'); sink.flush(); os.fsync(sink.fileno())
    return rows


def recover_behavior_receipts(path,version,recompute):
    """Reuse original token draws, never silently generate another candidate."""
    started,returned,receipts={},{},{}
    for event in read_owned_journal(path):
        kind=event['event']
        if kind in ('gpu-requested','gpu-returned'):
            for i,request in enumerate(event['requests']):
                if request['behavior_version']!=version:
                    raise ValueError('RECOVERY_BEHAVIOR_VERSION_MISMATCH')
                uid=request['request_id']
                if kind=='gpu-requested':
                    started[uid]=dict(request_id=uid,behavior_version=version,
                                      prompt_token_ids=event['prompt_token_ids'][i])
                else:
                    record=dict(request_id=uid,behavior_version=version,
                        prompt_token_ids=event['prompt_token_ids'][i],
                        generated_token_ids=event['generated_token_ids'][i],raw=event['raw'][i],
                        truncated=not event['generated_token_ids'][i] or
                            event['generated_token_ids'][i][-1]!=event['eos_token_id'])
                    if uid in returned and record!=returned[uid]:
                        raise ValueError('CONFLICTING_REPEATED_POLICY_DRAW')
                    returned[uid]=record
        elif kind in ('behavior-receipt','recovered-behavior-receipt'):
            record={k:v for k,v in event.items() if k!='event'}
            if record['behavior_version']!=version:
                raise ValueError('RECOVERY_BEHAVIOR_VERSION_MISMATCH')
            uid=record['request_id']
            if uid in receipts and record!=receipts[uid]:
                raise ValueError('CONFLICTING_BEHAVIOR_RECEIPT')
            receipts[uid]=record
    for uid,original in returned.items():
        if uid in receipts:
            for key in ('prompt_token_ids','generated_token_ids','raw','truncated'):
                if receipts[uid][key]!=original[key]:
                    raise ValueError('RAW_BEHAVIOR_RECEIPT_DISAGREEMENT')
            continue
        # Same immutable behavior actor, original sampled IDs. No new sample.
        scores=recompute(original['prompt_token_ids'],original['generated_token_ids'])
        if len(scores)!=len(original['generated_token_ids']):
            raise ValueError('RECOVERED_LOGPROB_LENGTH_MISMATCH')
        receipts[uid]=dict(original,behavior_logprobs=scores,
            probability_receipt='recomputed_frozen_behavior_after_interruption')
        append_receipt(path,dict(event='recovered-behavior-receipt',**receipts[uid]))
    for uid,original in started.items():
        if uid not in receipts:
            receipts[uid]=dict(original,raw='',generated_token_ids=[],behavior_logprobs=[],
                              resource_error='INTERRUPTED_GENERATION_NO_OUTPUT')
            append_receipt(path,dict(event='recovered-behavior-receipt',**receipts[uid]))
    return receipts


class NativeRouteSampler:
    def __init__(self,model,tokenizer,cfg,raw_path,*,receipt_writer=None):
        self.model,self.tokenizer,self.cfg,self.raw_path=model,tokenizer,cfg,Path(raw_path)
        from .route_generation_capacity import validate_model_capacity
        validate_model_capacity(model,cfg)
        self.raw_path.parent.mkdir(parents=True,exist_ok=True)
        self.receipt_writer=receipt_writer or (lambda value:append_receipt(self.raw_path,value,cfg=self.cfg))
        tokenizer.padding_side='left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token=tokenizer.eos_token

    def generate(self,requests):
        import torch
        from transformers import GenerationConfig,LogitsProcessorList
        from .route_policy_protocol import route_tool_prefix
        from .route_action_feedback import tools_for_request
        results=[None]*len(requests)
        if not requests:
            return results
        if len({r.behavior_version for r in requests})!=1:
            raise ValueError('MIXED_BEHAVIOR_VERSIONS')
        # Equal completion room avoids charging any request another's budget.
        # Measure unpadded histories FIRST: an overlong request must not censor
        # its shorter batch neighbors or reduce their available output room.
        with phase(self.cfg,'sampling.render',batch_size=len(requests)):
            all_prompts=[route_tool_prefix(self.tokenizer,r.messages,tools_for_request(r),self.cfg) for r in requests]
        with phase(self.cfg,'sampling.encode',batch_size=len(requests)):
            all_ids=self.tokenizer(all_prompts,padding=False,add_special_tokens=False)['input_ids']
        with phase(self.cfg,'sampling.batch_plan',batch_size=len(requests)):
            bins={}
            for i,r in enumerate(requests):
                room=min(r.remaining_tokens,self.cfg['max_action_tokens'],self.cfg['max_context']-len(all_ids[i]))
                if room<=0:
                    results[i]=dict(request_id=r.request_id,behavior_version=r.behavior_version,
                        prompt_token_ids=all_ids[i],raw='',generated_token_ids=[],behavior_logprobs=[],
                        resource_error='PHYSICAL_CONTEXT_LIMIT')
                else:
                    bins.setdefault(int(room),[]).append((i,r))
        for allowed,items in sorted(bins.items()):
            if self.cfg.get('acceleration',{}).get('hf_length_bucketing'):
                items.sort(key=lambda item:(len(all_ids[item[0]]),item[0]))
            from .route_generation_capacity import generation_batches,capacity_kwargs
            with phase(self.cfg,'sampling.batch_plan',batch_size=len(items)):
                parts=generation_batches([len(all_ids[i]) for i,_ in items],
                    max_batch_size=self.cfg['generation_batch_size'],**capacity_kwargs(self.cfg))
            for indices in parts:
                batch=[items[j] for j in indices]
                prompts=[all_prompts[i] for i,_ in batch]
                prompt_ids=[all_ids[i] for i,_ in batch]
                with phase(self.cfg,'sampling.encode',batch_size=len(batch)):
                    encoded=self.tokenizer.pad(dict(input_ids=prompt_ids),padding=True,return_tensors='pt')
                width=encoded['input_ids'].shape[1]
                room=min(allowed,self.cfg['max_context']-width)
                version={r.behavior_version for _,r in batch}
                if len(version)!=1:
                    raise ValueError('MIXED_BEHAVIOR_VERSIONS')
                common=[dict(request_id=r.request_id,behavior_version=r.behavior_version,
                             prompt_token_ids=p) for (_,r),p in zip(batch,prompt_ids)]
                if room<=0:
                    for (i,r),v in zip(batch,common):
                        results[i]=dict(v,raw='',generated_token_ids=[],behavior_logprobs=[],
                                        resource_error='PHYSICAL_CONTEXT_LIMIT')
                    continue
                device=next(self.model.parameters()).device
                with phase(self.cfg,'sampling.encode',batch_size=len(batch)):
                    encoded={k:v.to(device) for k,v in encoded.items()}
                capture=TokenLogprobs(self.tokenizer.eos_token_id)
                def sampling_failure(event):
                    self.receipt_writer(dict(event,behavior_version=next(iter(version)),
                        contract_sha256=self.cfg.get('_sha'),
                        prompt_lengths=[len(p) for p in prompt_ids]))
                private=PrivateSampling([r.request_id for _,r in batch],self.cfg['seed'],device,
                    finished=lambda:capture.done,on_failure=sampling_failure,prompt_width=width)
                counter=self.cfg.get('acceleration',{}).get('sampling_kernel')=='counter_categorical_v1'
                if counter:
                    from .route_counter_sampling import CounterCategorical
                    private=CounterCategorical([r.request_id for _,r in batch],self.cfg['seed'],device,capture,
                        on_failure=sampling_failure,prompt_width=width)
                generation=GenerationConfig(do_sample=not counter,top_k=0,top_p=1.,temperature=1.,
                    repetition_penalty=1.,num_beams=1,renormalize_logits=False,
                    pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True)
                before=time.monotonic()
                self.model.eval()
                self.receipt_writer(dict(event='gpu-requested',
                    requests=[vars(r) for _,r in batch],prompt_token_ids=prompt_ids,room=room))
                request_journal_seconds=time.monotonic()-before
                generation_began=time.monotonic()
                from .route_precision import policy_autocast,precision_mode
                # Host elapsed only: the backend exposes no device prefill/decode split.
                with phase(self.cfg,'sampling.generate',batch_size=len(batch),
                           prompt_tokens=sum(map(len,prompt_ids)),padded_tokens=width*len(batch)):
                    with torch.inference_mode(),policy_autocast(self.model,precision_mode(self.cfg)):
                        sequences=self.model.generate(**encoded,generation_config=generation,max_new_tokens=room,
                            logits_processor=LogitsProcessorList([capture,private]))
                with phase(self.cfg,'sampling.decode_receipts',batch_size=len(batch)):
                    ids=[]
                    for seq in sequences:
                        tokens=seq[width:].tolist()
                        if self.tokenizer.eos_token_id in tokens:
                            tokens=tokens[:tokens.index(self.tokenizer.eos_token_id)+1]
                        ids.append(tokens)
                generation_seconds=time.monotonic()-generation_began
                # GPU output is durable BEFORE execution/probability finalization.
                return_journal_began=time.monotonic()
                with phase(self.cfg,'sampling.decode_receipts',batch_size=len(batch),
                           generated_tokens=sum(map(len,ids))):
                    self.receipt_writer(dict(event='gpu-returned',requests=[vars(r) for _,r in batch],
                        prompts=prompts,prompt_token_ids=prompt_ids,generated_token_ids=ids,
                        raw=[self.tokenizer.decode(v,skip_special_tokens=False) for v in ids],
                        seconds=time.monotonic()-before,room=room,eos_token_id=self.tokenizer.eos_token_id))
                return_journal_seconds=time.monotonic()-return_journal_began
                self.receipt_writer(dict(event='gpu-batch-timing',requests=len(batch),
                    generation_seconds=generation_seconds,request_journal_seconds=request_journal_seconds,
                    return_journal_seconds=return_journal_seconds))
                with phase(self.cfg,'sampling.decode_receipts',batch_size=len(batch)):
                    capture.finish(sequences[:,-1])
                    logps=capture.values()
                    for (i,r),v,tokens,lp in zip(batch,common,ids,logps):
                        if len(tokens)!=len(lp):
                            raise ValueError('BEHAVIOR_LOGPROB_LENGTH_MISMATCH')
                        results[i]=dict(v,raw=self.tokenizer.decode(tokens,skip_special_tokens=False),
                            generated_token_ids=tokens,behavior_logprobs=lp,
                            truncated=not tokens or tokens[-1]!=self.tokenizer.eos_token_id)
                        self.receipt_writer(dict(event='behavior-receipt',**results[i]))
                print(json.dumps(dict(event='route-policy-batch',sequences=len(batch),
                    generated_tokens=sum(map(len,ids)),seconds=time.monotonic()-before)),flush=True)
        return results


def collect_route_groups(episodes,sampler,*,workers,journal,snapshot=None,recovered=None,on_completed_action=None):
    """GPU owner rolls across independent episodes while CPU executes others."""
    cfg=getattr(sampler,'cfg',None)
    coalesce=float(sampler.cfg.get('executor_coalesce_seconds',.25))
    if not math.isfinite(coalesce) or coalesce<0:
        raise ValueError('INVALID_EXECUTOR_COALESCING_DURATION')
    metrics=dict(dispatch_batches=0,dispatch_requests=0,single_request_dispatches=0,
                 coalesce_wait_seconds=0.,cpu_completed_calls=0)
    sampler.scheduler_metrics=metrics
    def next_decision(ep):
        with phase(cfg,'executor.next_decision'):
            return ep.next_decision()
    def accept(ep,receipt):
        with phase(cfg,'executor.accept'):
            return ep.accept(receipt)
    all_episodes=list(episodes)
    ready=deque(e for e in all_episodes if next_decision(e) is not None)
    jobs={}
    def harvest(done):
        for job in done:
            ep=jobs.pop(job)
            job.result()
            metrics['cpu_completed_calls']+=1
            journal(ep.records[-1])
            if on_completed_action:
                added=on_completed_action(ep)
                all_episodes.extend(added)
                ready.extend(e for e in added if next_decision(e) is not None)
            if snapshot:
                with phase(cfg,'collection.snapshot'):
                    snapshot(ep)
            if next_decision(ep) is not None:
                ready.append(ep)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while ready or jobs:
            # Only CPU successors that are finished can become GPU requests.
            done=[job for job in jobs if job.done()]
            if not ready and jobs and not done:
                with phase(cfg,'scheduler.wait',ready=len(ready),inflight=len(jobs)):
                    done=list(wait(jobs,return_when=FIRST_COMPLETED)[0])
            harvest(done)
            # CPU successors can complete while journal/snapshot I/O runs.
            # Re-drain them before dispatch; wait only a bounded interval for
            # outstanding work, never indefinitely behind a slow obligation.
            deadline=time.monotonic()+coalesce
            while jobs and len(ready)<sampler.cfg['generation_batch_size']:
                done=[job for job in jobs if job.done()]
                if not done:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:
                        break
                    began=time.monotonic()
                    with phase(cfg,'scheduler.wait',ready=len(ready),inflight=len(jobs)):
                        done=list(wait(jobs,timeout=remaining,return_when=FIRST_COMPLETED)[0])
                    metrics['coalesce_wait_seconds']+=time.monotonic()-began
                    if not done:
                        break
                harvest(done)
            if not ready:
                continue
            batch=[ready.popleft() for _ in range(min(len(ready),sampler.cfg['generation_batch_size']))]
            requests=[next_decision(ep) for ep in batch]
            outputs=[(recovered or {}).get(r.request_id) for r in requests]
            missing=[i for i,v in enumerate(outputs) if v is None]
            if missing:
                metrics['dispatch_batches']+=1
                metrics['dispatch_requests']+=len(missing)
                metrics['single_request_dispatches']+=len(missing)==1
            if missing:
                with phase(cfg,'scheduler.dispatch',ready=len(ready),inflight=len(jobs),batch_size=len(missing)):
                    generated=sampler.generate([requests[i] for i in missing])
            else:
                generated=[]
            if len(generated)!=len(missing):
                raise ValueError('SAMPLER_DROPPED_NEW_REQUEST')
            for i,value in zip(missing,generated):
                outputs[i]=value
            if len(outputs)!=len(batch):
                raise ValueError('SAMPLER_DROPPED_REQUEST')
            for ep,receipt in zip(batch,outputs):
                jobs[pool.submit(accept,ep,receipt)]=ep
    return [ep.finish_report() for ep in all_episodes]
