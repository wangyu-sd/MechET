"""Exact-conditioning ReFIT primitives and replayable independent suffix forks.

No action masks, forced FLOW proposals or inferred behavior probabilities.
Canonical route occurrence IDs survive forks; new draws have new request IDs.
"""
from collections import OrderedDict,defaultdict
from copy import deepcopy
from dataclasses import dataclass,field
import math

from .external_route_data import signature
from .external_route_rl import TargetOnlyRouteEpisode


def decision_identity(*,target_id,tree_id,occurrence,prompt_token_ids,tools,
                      accepted_history,route_context,behavior_version,
                      executor_contract,mapped_state):
    if not prompt_token_ids or not behavior_version or not executor_contract:
        raise ValueError('REFIT_CONDITIONING_INCOMPLETE')
    return signature(dict(target_id=target_id,tree_id=tree_id,occurrence=occurrence,
        prompt_token_ids=prompt_token_ids,tools=tools,accepted_history=accepted_history,
        route_context=route_context,behavior_version=behavior_version,
        executor_contract=executor_contract,mapped_state=mapped_state))


def grounded_action_key(mapped_state,raw):
    from .route_action_feedback import semantic_action
    parsed=semantic_action(raw)
    if parsed is None:return signature(['malformed',raw])
    action=parsed[1]
    if action['name']=='apply_electron_flow':
        from .natural_language_electron_flow import compile_event_arguments
        try:
            moves=compile_event_arguments(mapped_state,action['arguments'])
            return signature(['grounded_FLOW',moves])
        except (ValueError,KeyError,TypeError,AttributeError):pass
    return signature(['parsed',action])


@dataclass
class DecisionNode:
    identity:str
    max_expansions:int
    max_duplicates:int
    edges:dict=field(default_factory=dict)
    censored_receipts:list=field(default_factory=list)
    duplicate_draws:int=0

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.max_expansions,self.max_duplicates)):
            raise ValueError('REFIT_FINITE_NODE_BUDGET_REQUIRED')

    @property
    def expansions(self):return len(self.edges)

    @property
    def status(self):
        if self.expansions>=self.max_expansions or self.duplicate_draws>=self.max_duplicates:
            return 'EXHAUSTED'
        return 'OPEN_WITH_REJECTIONS' if any(e['status']=='REJECTED_TERMINAL' for e in self.edges.values()) else 'OPEN'

    def known_rejection(self,action):
        edge=self.edges.get(action)
        return edge['error'] if edge and edge['status']=='REJECTED_TERMINAL' else None

    def add(self,action,receipt_id,*,successor=None,error='',resource=False,executor_skipped=False):
        if resource:
            self.censored_receipts.append(receipt_id)
            return
        if not error and successor is None:
            raise ValueError('REFIT_ACCEPTED_SUCCESSOR_REQUIRED')
        if error and successor is not None:
            raise ValueError('REFIT_REJECTED_EDGE_HAS_NO_CHILD')
        if action in self.edges:
            edge=self.edges[action]
            if edge['error']!=error or edge['successor']!=successor:
                raise ValueError('REFIT_NONDETERMINISTIC_EXECUTOR')
            if receipt_id in edge['receipt_ids']:
                raise ValueError('REFIT_DUPLICATE_RECEIPT')
            self.duplicate_draws+=1;edge['receipt_ids'].append(receipt_id)
            return
        if executor_skipped:raise ValueError('REFIT_UNKNOWN_REJECTION_CACHE')
        self.edges[action]=dict(status='REJECTED_TERMINAL' if error else 'ACCEPTED',
            error=error,successor=successor,receipt_ids=[receipt_id])


class RefitRouteEpisode(TargetOnlyRouteEpisode):
    def next_decision(self):
        request=super().next_decision()
        if request is None:return None
        for entry in self.cfg.get('refit_request_namespaces',[]):
            if len(self.records)>=entry['start']:
                request.request_id=entry['namespace']+'/r0/a'+str(len(self.records))
                request.episode_id=entry['namespace']
        return request

    def fork(self,namespace):
        if self.next_decision() is None:raise ValueError('REFIT_FORK_REQUIRES_OPEN_PARENT')
        if namespace==self.uid or any(e['namespace']==namespace for e in self.cfg.get('refit_request_namespaces',[])):
            raise ValueError('REFIT_FORK_NAMESPACE_REUSED')
        result=self.clone()
        result.cfg.setdefault('refit_request_namespaces',[]).append(
            dict(start=len(self.records),namespace=namespace))
        return result

    def clone(self):
        # Copy executor-owned state, not the quarter-million-entry frozen stock.
        # Recovery still reruns and validates the complete receipt history.
        result=object.__new__(type(self))
        result.__dict__.update(deepcopy({k:v for k,v in vars(self).items() if k not in ('stock','chemistry')}))
        result.stock=self.stock;result.chemistry=self.chemistry
        return result

    def set_rejections(self,errors):
        self.cfg.setdefault('refit_rejections',{})[str(len(self.records))]=dict(errors)

    def _prior_rejection(self,receipt):
        key=grounded_action_key(self.node.state,receipt.get('raw',''))
        errors=self.cfg.get('refit_rejections',{}).get(str(len(self.records)),{})
        return dict(error=errors[key],count=1) if key in errors else None


def branch_advantages(rows,*,matched_continuations=1):
    """Equal semantic-action baseline. Duplicate proposals get no extra vote.

    First M declared continuations per semantic action define Q. Extra duplicate
    draws are recorded and may receive the same action credit, but do not add
    baseline weight. Any censored draw masks the whole conditional group.
    """
    if type(matched_continuations) is not int or matched_continuations<1:
        raise ValueError('REFIT_MATCHED_CONTINUATIONS_REQUIRED')
    groups=defaultdict(list)
    for row in rows:
        value=row.get('value')
        if (row.get('proposal')!='original_policy' or row.get('complete') is not True or
                value is None or not math.isfinite(value)):
            return None
        groups[row['action']].append(row)
    if len(groups)<2 or any(len(g)<matched_continuations for g in groups.values()):return None
    values={key:math.fsum(r['value'] for r in group[:matched_continuations])/matched_continuations
            for key,group in groups.items()}
    mean=math.fsum(values.values())/len(values)
    return {r['receipt_id']:values[key]-mean for key,group in groups.items() for r in group}


def unique_credit(fallback,branch):
    """A real receipt enters loss once; branch credit replaces G4 credit."""
    records=OrderedDict()
    for record,advantage in fallback:
        key=record['request_id']
        if key in records:
            old,old_advantage=records[key]
            if old!=record or old_advantage!=advantage:
                raise ValueError('REFIT_RECEIPT_CONFLICT')
        else:records[key]=(record,advantage)
    result=[]
    for key,(record,advantage) in records.items():
        advantage=branch.get(key,advantage)
        if advantage is None:continue
        if not math.isfinite(advantage):raise ValueError('REFIT_NONFINITE_ADVANTAGE')
        p,r,b=(record.get(k) for k in ('prompt_token_ids','generated_token_ids','behavior_logprobs'))
        if not p or not r or b is None or len(b)!=len(r):
            raise ValueError('REFIT_POLICY_RECEIPT_REQUIRED')
        result.append((record,advantage))
    return result
