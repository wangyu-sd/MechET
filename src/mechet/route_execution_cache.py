"""Share deterministic FLOW chemistry, never sampled history or route returns."""
from collections import Counter, OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import RLock

from .external_route_data import signature


class ExecutionCache:
    def __init__(self, capacity=8192):
        if type(capacity) is not int or capacity<1:raise ValueError('EXECUTION_CACHE_CAPACITY')
        self.capacity=capacity;self.lock=RLock();self.entries=OrderedDict();self.counts=Counter()
        self.counts['cache_reconstruction_failures']=0
        self.receipt_provenance={}

    def provenance(self,receipt_id):
        with self.lock:return deepcopy(self.receipt_provenance[receipt_id])

    def persist_provenance(self,path):
        """Round barrier snapshot; GPU/execution journals remain crash authority."""
        from .route_training_contract import atomic_json
        from .external_route_data import file_sha
        with self.lock:records=deepcopy(self.receipt_provenance)
        atomic_json(path,records)
        return dict(path=str(path),sha256=file_sha(path),receipts=len(records),
            scope='completed round cache producer/witness mapping; not a replacement for execution journals')

    def reconstruction_failure(self):
        with self.lock:self.counts['cache_reconstruction_failures']+=1

    def metrics(self):
        with self.lock:return dict(self.counts)

    def resolve(self,key,receipt_id,compute):
        with self.lock:
            future=self.entries.get(key)
            owner=future is None
            if owner:
                future=Future();self.entries[key]=future
                self.counts['deterministic_execute_calls']+=1
            else:
                self.entries.move_to_end(key)
                self.counts['inflight_waiters']+=not future.done()
        if owner:
            try:
                value=deepcopy(compute())
                with self.lock:
                    error=value.get('error','')
                    cacheable=not error or error.startswith(('ValueError:','KeyError:','TypeError:'))
                    if not cacheable:
                        self.entries.pop(key,None)
                    witness=dict(execution_key=key,producer_receipt_id=receipt_id,transition=value)
                    future.set_result(dict(witness,witness_sha=signature(witness)))
                    # Never evict in-flight entries and create a second producer.
                    for old in list(self.entries):
                        if len(self.entries)<=self.capacity:break
                        if self.entries[old].done():del self.entries[old]
            except BaseException as exc:
                with self.lock:self.entries.pop(key,None);future.set_exception(exc)
                raise
        entry=deepcopy(future.result())
        try:
            witness={k:entry[k] for k in ('execution_key','producer_receipt_id','transition')}
            if entry['execution_key']!=key or signature(witness)!=entry['witness_sha']:
                raise ValueError('CACHE_TRANSITION_WITNESS_MISMATCH')
        except Exception:
            self.reconstruction_failure()
            raise
        with self.lock:
            self.receipt_provenance[receipt_id]=dict(request_id=receipt_id,execution_key=key,
                producer_receipt_id=entry['producer_receipt_id'],witness_sha=entry['witness_sha'])
        value=entry['transition']
        if not owner and (not value.get('error') or value['error'].startswith(('ValueError:','KeyError:','TypeError:'))):
            with self.lock:
                label='rejected_execution_cache_hits' if value.get('error') else 'accepted_execution_cache_hits'
                self.counts[label]+=1
        return value


class CachedChemistry:
    """First version caches FLOW only. IMPORT memory and FINISH guards stay live.

    Only state/next_map/imported/result are cached. Each caller reconstructs
    actions/scores/visited from its own node and real proposal metadata.
    """
    def __init__(self,base,cache,contracts):
        self.base,self.cache,self.contracts=base,cache,deepcopy(contracts)

    def __getattr__(self,name):return getattr(self.base,name)

    def step(self,node,output):
        from .pr96_evaluation import parsed
        from .route_refit import grounded_action_key
        from scripts.run_natural_language_value_search import visible,Node
        name,arguments,error=parsed(output)
        if error or name!='apply_electron_flow':return self.base.step(node,output)
        key=signature(dict(state=node.state,target=node.target,next_map=node.next_map,
            imported=node.imported,action=grounded_action_key(node.state,output['raw']),
            contracts=self.contracts))
        def compute():
            pure=deepcopy(node);pure.actions=[];pure.visited=set();pure.logprob=0.;pure.tokens=0;pure.pointer_score=0.
            child,error=self.base.step(pure,output)
            if child is None:return dict(error=error)
            if child.terminal:raise ValueError('CACHE_FLOW_TERMINAL_RECONSTRUCTION')
            return dict(error='',state=child.state,next_map=child.next_map,imported=child.imported,
                        result=child.actions[-1]['result'])
        value=self.cache.resolve(key,output['request_id'],compute)
        if value.get('error'):return None,value['error']
        try:
            if visible(value['state']) in node.visited:return None,'ValueError:STATE_CYCLE'
            record=dict(state_before=node.state,name=name,arguments=deepcopy(arguments),result=deepcopy(value['result']))
            child=Node(target=node.target,state=value['state'],next_map=value['next_map'],
                actions=node.actions+[record],visited=set(node.visited)|{visible(value['state'])},
                imported=deepcopy(value['imported']),logprob=node.logprob+sum(output['behavior_logprobs']),
                tokens=node.tokens+len(output['generated_token_ids']),pointer_score=node.pointer_score,
                terminal=False)
            if (child.state!=value['state'] or child.next_map!=value['next_map'] or
                    child.imported!=value['imported'] or child.actions[-1]!=record or child.terminal):
                raise ValueError('CACHE_FLOW_RECONSTRUCTION_MISMATCH')
            return child,''
        except Exception:
            self.cache.reconstruction_failure()
            raise
