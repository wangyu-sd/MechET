"""Round-local CPU ownership and shared GPU requests; no per-token disk RPC."""
from collections import Counter,deque,OrderedDict,defaultdict
from copy import deepcopy
import math
from pathlib import Path
from threading import Condition
import time
import uuid

from .external_route_data import file_sha,signature
from .route_policy_sampling import append_receipt,read_owned_journal


class WorkBroker:
    def __init__(self,rows,*,actor_version,contract_sha,journal_path,max_pending=4096,request_timeout=1800.,dispatch=None):
        self.cv=Condition();self.rows={r['task_id']:deepcopy(r) for r in rows}
        if len(self.rows)!=len(rows):raise ValueError('DUPLICATE_TARGET_GROUP')
        self.version=actor_version;self.contract=contract_sha;self.path=Path(journal_path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.epoch=uuid.uuid4().hex;self.leases={};self.done={};self.error='';self.stopped=False
        self.requests={};self.ready=deque();self.max_pending=max_pending;self.counts=Counter()
        self.dispatch=deepcopy(dispatch)
        self.owner_ready={};self.owner_turns=deque();self.gpu_load={}
        self.owner_stats=defaultdict(Counter);self.gpu_stats=defaultdict(Counter)
        self.queue_samples=defaultdict(lambda:deque(maxlen=1024))
        if dispatch is not None:
            if (set(dispatch)!={'max_prefill_tokens','max_estimated_tokens','max_pending','gpu_count'}
                    or any(type(v) is not int or v<1 for v in dispatch.values())):
                raise ValueError('GLOBAL_DISPATCH_SETTINGS_INVALID')
            self.ready=OrderedDict();self.max_pending=dispatch['max_pending']
            self.gpu_load={i:0 for i in range(dispatch['gpu_count'])}
        if not math.isfinite(request_timeout) or request_timeout<=0:raise ValueError('GPU_REQUEST_TIMEOUT_INVALID')
        self.request_timeout=request_timeout
        binding=dict(event='binding',actor_version=actor_version,contract_sha=contract_sha,
                     task_ids_sha=signature(rows))
        if self.dispatch is not None:binding['dispatch']=self.dispatch
        old=read_owned_journal(self.path)
        if old and old[0]!=binding:raise ValueError('WORK_BROKER_BINDING_CHANGED')
        if not old:append_receipt(self.path,binding)
        for row in old[1:]:
            if row['event']=='finished':
                if row['task_id'] not in self.rows:raise ValueError('UNEXPECTED_FINISHED_TARGET')
                p=self._owned_path(row['result_path'])
                if file_sha(p)!=row['result_sha']:raise ValueError('GROUP_RESULT_HASH_CHANGED')
                previous=self.done.get(row['task_id'])
                if previous and previous['result_sha']!=row['result_sha']:raise ValueError('DOUBLE_GROUP_COMMIT')
                self.done[row['task_id']]=row
        self.unclaimed=deque(k for k in self.rows if k not in self.done)

    def _check(self):
        if self.error:raise RuntimeError('ROUND_BROKER_FAILED:'+self.error)

    def _request_deadline(self,entry):
        if entry['result'] is None and entry.get('deadline',float('inf'))<time.monotonic():
            self.error='GPU_REQUEST_DEADLINE:'+entry['request']['request_id'];self.cv.notify_all();self._check()

    def _owned_path(self,path):
        p=Path(path).resolve()
        if not p.is_relative_to(self.path.parent.resolve()):raise ValueError('GROUP_RESULT_OUTSIDE_ROUND')
        return p

    def claim(self,rank):
        with self.cv:
            self._check()
            if not self.unclaimed:return None
            key=self.unclaimed.popleft();lease=signature([self.epoch,key,rank])
            row=dict(event='claimed',task_id=key,rank=rank,lease_id=lease,
                     actor_version=self.version,contract_sha=self.contract)
            append_receipt(self.path,row);self.leases[lease]=row
            return dict(task=deepcopy(self.rows[key]),lease_id=lease)

    def finish(self,lease,rank,path,digest):
        with self.cv:
            self._check();owner=self.leases.get(lease)
            if not owner or owner['rank']!=rank:raise ValueError('GROUP_LEASE_OWNER_MISMATCH')
            if owner['task_id'] in self.done:raise ValueError('DOUBLE_GROUP_COMMIT')
            if file_sha(self._owned_path(path))!=digest:raise ValueError('GROUP_RESULT_HASH_CHANGED')
            row=dict(owner,event='finished',result_path=str(Path(path).resolve()),result_sha=digest)
            append_receipt(self.path,row);self.done[owner['task_id']]=row;self.cv.notify_all()

    def results(self):
        with self.cv:
            self._check()
            return [deepcopy(self.done[k]) for k in self.rows if k in self.done]

    def snapshot(self):
        with self.cv:
            result=dict(expected=len(self.rows),complete=len(self.done),unclaimed=len(self.unclaimed),
                        ready=len(self.ready),counts=dict(self.counts),error=self.error,stopped=self.stopped)
            if any('accounting_label' in r for r in self.requests.values()):
                from .route_request_accounting import summarize_requests
                result['request_accounting']=summarize_requests(self.requests)
            if self.dispatch is not None:
                result.update(dispatch=self.dispatch,estimated_gpu_tokens=dict(self.gpu_load),
                    gpu_metrics={str(k):dict(v) for k,v in self.gpu_stats.items()},
                    target_metrics={k:dict(v) for k,v in self.owner_stats.items()})
                for owner,values in self.queue_samples.items():
                    samples=sorted(values)
                    result['target_metrics'][owner]['queue_wait_p95_seconds']=samples[int(.95*(len(samples)-1))] if samples else 0.
                    result['target_metrics'][owner]['queue_quantile_sample_limit']=1024
            return result

    def register_request_labels(self,owner,labels):
        """Join harvested receipt ownership only; labels cannot affect dispatch."""
        with self.cv:
            physical=missing=0
            for uid,label in labels.items():
                if (set(label)!={'lane','node_id','semantic_edge','accepted'}
                        or label['lane'] not in ('original_g4','credit','exploration')):
                    raise ValueError('GLOBAL_ACCOUNTING_LABEL_INVALID')
                entry=self.requests.get(uid)
                if entry is None:
                    missing+=1;continue  # CPU-side physical context limit, never dispatched
                if entry['owner'] not in (None,owner):raise ValueError('GLOBAL_ACCOUNTING_OWNER_MISMATCH')
                if 'accounting_label' in entry and entry['accounting_label']!=label:
                    raise ValueError('GLOBAL_ACCOUNTING_LABEL_REBOUND')
                entry.update(owner=owner,accounting_label=deepcopy(label));physical+=1
            return dict(physical_requests=physical,undispatched_capacity_attempts=missing)

    def is_stopped(self):
        with self.cv:self._check();return self.stopped

    def wait_collection(self,timeout):
        with self.cv:
            self._check()
            if len(self.done)<len(self.rows):self.cv.wait(timeout)
            self._check()
            return len(self.done)==len(self.rows)

    def _valid_result(self,request,value,binding=None):
        if value.get('request_id')!=request['request_id'] or value.get('behavior_version')!=self.version:
            raise ValueError('GLOBAL_RECEIPT_IDENTITY_MISMATCH')
        ids=value.get('generated_token_ids',[]);lp=value.get('behavior_logprobs',[])
        if len(ids)!=len(lp) or any(not math.isfinite(x) or x>0 for x in lp):
            raise ValueError('GLOBAL_RECEIPT_PROBABILITY_INVALID')
        if binding and binding.startswith('ids:') and binding!='ids:'+signature(value.get('prompt_token_ids')):
            raise ValueError('GLOBAL_RECEIPT_PROMPT_MISMATCH')

    def seed_result(self,request,value):
        with self.cv:
            self._valid_result(request,value)
            uid=request['request_id']
            if uid in self.requests:raise ValueError('DUPLICATE_RECOVERED_GLOBAL_REQUEST')
            self.requests[uid]=dict(request=deepcopy(request),owner=None,binding=None,
                                    result=deepcopy(value),gpu=None,work=None)
            self.counts['recovered_request_receipts']+=1

    def _work(self,work):
        if self.dispatch is None:return None
        if (not isinstance(work,dict) or set(work)!={'prompt_tokens','max_new_tokens'}
                or type(work['prompt_tokens']) is not int or work['prompt_tokens']<1
                or type(work['max_new_tokens']) is not int or work['max_new_tokens']<0):
            raise ValueError('GLOBAL_REQUEST_WORK_REQUIRED')
        if work['prompt_tokens']+work['max_new_tokens']>self.dispatch['max_estimated_tokens']:
            raise ValueError('GLOBAL_REQUEST_EXCEEDS_DISPATCH_CAPACITY')
        return deepcopy(work)

    def submit(self,request,owner,prompt_binding,work=None):
        with self.cv:
            self._check()
            if self.stopped or request.get('behavior_version')!=self.version:raise ValueError('GLOBAL_POLICY_VERSION_MISMATCH')
            uid=request['request_id'];prior=self.requests.get(uid)
            work=self._work(work)
            if prior:
                if prior['request']!=request or prior['owner'] not in (None,owner):raise ValueError('GLOBAL_REQUEST_REBOUND')
                if prior['binding'] not in (None,prompt_binding):raise ValueError('GLOBAL_REQUEST_PROMPT_REBOUND')
                if prior.get('work') not in (None,work):raise ValueError('GLOBAL_REQUEST_WORK_REBOUND')
                prior.update(owner=owner,binding=prompt_binding)
                prior['work']=work
                if prior['result'] is not None:self._valid_result(request,prior['result'],prompt_binding)
                return
            pending=sum(r['result'] is None for r in self.requests.values())
            if pending>=self.max_pending:raise RuntimeError('GLOBAL_QUEUE_ADMISSION_CAPACITY')
            self.requests[uid]=dict(request=deepcopy(request),owner=owner,binding=prompt_binding,
                                    result=None,gpu=None,submitted=time.monotonic(),work=work)
            if self.dispatch is None:
                self.ready.append(uid);self.counts['physical_requests_submitted']+=1
            else:
                self.ready[uid]=None;queue=self.owner_ready.setdefault(owner,deque())
                if not queue:self.owner_turns.append(owner)
                queue.append(uid);self.counts['policy_requests_submitted']+=1
                self.owner_stats[owner]['submitted_requests']+=1
            self.cv.notify_all()

    def _fair_uids(self,gpu,limit):
        if gpu not in self.gpu_load:raise ValueError('GLOBAL_UNKNOWN_GPU')
        if self.gpu_load[gpu]>min(self.gpu_load.values()):return []
        idle=sum(v==0 for v in self.gpu_load.values())
        if idle>1:limit=min(limit,max(1,math.ceil(len(self.ready)/idle)))
        selected=[];prefill=0;width=0;reserved=self.gpu_load[gpu];misses=0
        while self.owner_turns and len(selected)<limit:
            owner=self.owner_turns.popleft();queue=self.owner_ready[owner];uid=queue[0]
            work=self.requests[uid]['work'];length=work['prompt_tokens']
            quantum=min(length,self.dispatch['max_prefill_tokens']);cost=length+work['max_new_tokens']
            fits=(prefill+quantum<=self.dispatch['max_prefill_tokens']
                  and reserved+cost<=self.dispatch['max_estimated_tokens']
                  and max(width,length)*(len(selected)+1)<=self.dispatch['max_estimated_tokens'])
            if fits:
                queue.popleft();self.ready.pop(uid);selected.append(uid)
                prefill+=quantum;reserved+=cost;width=max(width,length);misses=0
                if queue:self.owner_turns.append(owner)
            else:
                self.owner_turns.append(owner);misses+=1
                if misses>=len(self.owner_turns):break
        self.gpu_load[gpu]=reserved
        stats=self.gpu_stats[gpu]
        stats['estimated_tokens_peak']=max(stats['estimated_tokens_peak'],reserved)
        stats['dispatch_prefill_quanta']+=prefill
        stats['estimated_padded_tokens']+=width*len(selected)
        return selected

    def take(self,gpu,limit,timeout):
        deadline=time.monotonic()+timeout
        with self.cv:
            self._check()
            while len(self.ready)<limit and not self.stopped:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                self.cv.wait(remaining);self._check()
            self._check();batch=[]
            selected=(self._fair_uids(gpu,limit) if self.dispatch is not None else
                      [self.ready.popleft() for _ in range(min(len(self.ready),limit))])
            for uid in selected:
                entry=self.requests[uid];entry['gpu']=gpu;entry['dispatched']=time.monotonic()
                entry['deadline']=entry['dispatched']+self.request_timeout
                waited=entry['dispatched']-entry['submitted']
                self.counts['queue_wait_microseconds']+=int(waited*1e6)
                if self.dispatch is not None:
                    self.queue_samples[entry['owner']].append(waited)
                    self.owner_stats[entry['owner']]['queue_wait_seconds']+=waited
                    self.gpu_stats[gpu]['assigned_requests']+=1
                batch.append(dict(request=deepcopy(entry['request']),prompt_binding=entry['binding']))
            self.counts['gpu'+str(gpu)+'_requests']+=len(batch)
            return batch

    def complete(self,gpu,uid,value):
        with self.cv:
            self._check();entry=self.requests[uid]
            if entry['gpu']!=gpu:raise ValueError('GLOBAL_COMPLETION_OWNER_MISMATCH')
            self._valid_result(entry['request'],value,entry['binding'])
            if entry['result'] is not None:raise ValueError('GLOBAL_DUPLICATE_COMPLETION')
            if self.dispatch is not None:
                work=entry['work'];self.gpu_load[gpu]-=work['prompt_tokens']+work['max_new_tokens']
                self.owner_stats[entry['owner']]['completed_requests']+=1
                self.owner_stats[entry['owner']]['dispatch_to_return_seconds']+=time.monotonic()-entry['dispatched']
                self.gpu_stats[gpu]['completed_requests']+=1
            entry['result']=deepcopy(value);self.counts['completed_requests']+=1;self.cv.notify_all()

    def wait_result(self,owner,uid,timeout):
        deadline=time.monotonic()+timeout
        with self.cv:
            while True:
                self._check();entry=self.requests[uid]
                self._request_deadline(entry)
                if entry['owner']!=owner:raise ValueError('GLOBAL_RESULT_OWNER_MISMATCH')
                if entry['result'] is not None:return deepcopy(entry['result'])
                remaining=deadline-time.monotonic()
                if remaining<=0:return None
                self.cv.wait(remaining)

    def wait_many(self,owner,uids,timeout):
        deadline=time.monotonic()+timeout
        with self.cv:
            while True:
                self._check();values=[]
                for uid in uids:
                    entry=self.requests[uid]
                    self._request_deadline(entry)
                    if entry['owner']!=owner:raise ValueError('GLOBAL_RESULT_OWNER_MISMATCH')
                    if entry['result'] is not None:values.append((uid,deepcopy(entry['result'])))
                if values:return values
                remaining=deadline-time.monotonic()
                if remaining<=0:return []
                self.cv.wait(remaining)

    def fail(self,message):
        with self.cv:self.error=str(message);self.cv.notify_all()

    def stop(self):
        with self.cv:self.stopped=True;self.cv.notify_all()
