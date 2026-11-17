"""Independent, occurrence-preserving AND-route environment.

No candidates, teacher suffixes, or MCTS statistics enter this state machine.
FINISH commits one reaction, not an entire synthesis route.
"""
from dataclasses import asdict, dataclass
import copy
import json

from .route_training_contract import chemical_identity,public_molecule
from .endpoints import ROUTE_ENDPOINT_CONTRACT
from .route_native_observation import NATIVE_SFT_ROUTE


FULL_ROUTE_HISTORY = 'full_route_history_feedback_v2'
from .route_event_history import (EVENT_ROUTE_HISTORY, SCOPED_ROUTE_HISTORY, NATIVE_ROUTE_HISTORY,
    NATIVE_LAST_ROUTE_HISTORY, PASSIVE_ROUTE_HISTORY, ACTION_ROUTE_HISTORY, NATIVE_SYSTEM_HEADER, RouteEventHistory,
    HISTORY_HEADER, ACTIVE_HEADER, LOCAL_HEADER, public_json)

EVENT_ROUTE_OBSERVATIONS = (EVENT_ROUTE_HISTORY, SCOPED_ROUTE_HISTORY, NATIVE_ROUTE_HISTORY,
                            NATIVE_LAST_ROUTE_HISTORY,PASSIVE_ROUTE_HISTORY,ACTION_ROUTE_HISTORY,
                            NATIVE_SFT_ROUTE)
AGENTIC_ROUTE_OBSERVATIONS = (FULL_ROUTE_HISTORY, *EVENT_ROUTE_OBSERVATIONS)


def import_work_bytes(state,arguments):
    """Conservative graph/matrix memory estimate, BEFORE count expansion."""
    from rdkit import Chem
    params=Chem.SmilesParserParams(); params.removeHs=False
    initial=Chem.MolFromSmiles(state,params)
    if initial is None:
        raise ValueError('INVALID_CURRENT_STATE')
    atoms=initial.GetNumAtoms()
    if not isinstance(arguments,dict):
        raise ValueError('IMPORT_SCHEMA_INVALID')
    fragments=arguments.get('fragments',[])
    if not isinstance(fragments,list) or not fragments:
        raise ValueError('IMPORT_SCHEMA_INVALID')
    for item in fragments:
        if not isinstance(item,dict):
            raise ValueError('IMPORT_SCHEMA_INVALID')
        try:
            count=int(item.get('count',0))
        except (TypeError,ValueError,OverflowError):
            raise ValueError('IMPORT_COUNT_INVALID')
        fragment=Chem.MolFromSmiles(str(item.get('smiles','')),params)
        if count<1:
            raise ValueError('IMPORT_COUNT_INVALID')
        if fragment is None or fragment.GetNumAtoms()==0:
            raise ValueError('IMPORT_FRAGMENT_INVALID')
        atoms+=count*fragment.GetNumAtoms()
    return 64*atoms*atoms+4096*atoms


class NativeChemistry:
    def __init__(self,memory_divisor=32):
        self.memory_divisor=memory_divisor

    def root(self, product):
        from .pr96_evaluation import public_root
        return public_root({'target_smiles':product})

    def messages(self, node):
        from .pr96_evaluation import messages_for
        return messages_for(node, True)

    def step(self, node, output):
        from .pr96_evaluation import parsed
        from scripts.run_natural_language_value_search import Action, execute
        name, arguments, error = parsed(output)
        if error:
            return None, error
        if name=='import_fragments':
            from .dynamic_electron_tree import available_host_memory
            try:
                memory=import_work_bytes(node.state,arguments)
            except ValueError as exc:
                return None,'ValueError:'+str(exc)
            if memory>available_host_memory()/self.memory_divisor:
                return None,'RESOURCE_CENSORED:IMPORT_MEMORY_ESTIMATE'
        return execute(node, Action(name, arguments, output['raw'],
                       sum(output['behavior_logprobs']),len(output['generated_token_ids'])), max_imports=float('inf'))

    def precursors(self, node, initial):
        from .endpoints import split_route_precursor_endpoints
        return split_route_precursor_endpoints(node.state,initial.state).structural


@dataclass
class DecisionRequest:
    request_id: str
    episode_id: str
    reaction_id: str
    product: str
    behavior_version: str
    messages: list
    remaining_tokens: int
    public_product: str
    tools: list | None = None


class RouteEpisode:
    def __init__(self, contract, public_root, stock, uid, seed, *, chemistry=None):
        self.cfg, self.stock = dict(contract), stock
        self.cfg.setdefault('route_endpoint_contract',ROUTE_ENDPOINT_CONTRACT)
        if self.cfg['route_endpoint_contract'] != ROUTE_ENDPOINT_CONTRACT:
            raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
        from .route_materials import material_contract
        material_contract(self.cfg)
        from .route_failure_policy import check_failure_policy, FIRST_REJECTION
        failure=check_failure_policy(self.cfg.get('route_failure_policy'))
        if failure==FIRST_REJECTION and (not self.agentic_feedback or
                self.cfg.get('max_consecutive_recoveries')!=1):
            raise ValueError('FIRST_REJECTION_REQUIRES_NATIVE_TERMINAL_CONTRACT')
        self.uid, self.seed = uid, seed
        self.chemistry = chemistry or NativeChemistry(contract.get('concurrent_executors',32))
        self.public_root = public_molecule(public_root)
        self.root = chemical_identity(public_root)
        self.protected = set(contract.get('protected_identities', []))
        self.pending = [dict(product=self.root, public_product=self.public_root,
                             occurrence='0', ancestors=[], level=0)]
        self.current = self.node = self.initial = None
        self.records, self.reactions, self.stock_leaves = [], [], []
        self.status = self.error = self.reward = None
        self.used_tokens, self.depth = 0, 0
        self.full_history = []
        self.event_history = (RouteEventHistory() if
            self.cfg.get('public_observation_contract') in EVENT_ROUTE_OBSERVATIONS else None)
        self.consecutive_recoveries = 0
        if self.agentic_feedback:
            ceiling=self.cfg.get('max_consecutive_recoveries',8)
            if type(ceiling) is not int or ceiling < 1:
                raise ValueError('INVALID_RECOVERY_CEILING')
        if self.root in stock:
            self.pending.clear()
            self.stock_leaves.append(self.root)
            self._end('already_stock',0.)

    def _end(self, status, reward, error=''):
        self.status, self.reward, self.error = status, reward, error

    @property
    def agentic_feedback(self):
        return self.cfg.get('public_observation_contract') in AGENTIC_ROUTE_OBSERVATIONS

    def _messages(self):
        messages=copy.deepcopy(self.chemistry.messages(self.node))
        if not self.agentic_feedback:
            return messages
        if self.cfg.get('public_observation_contract')==NATIVE_SFT_ROUTE:
            from .route_native_observation import native_messages
            return native_messages(self, messages)
        if self.event_history is not None:
            rid=self.uid+'/r'+str(len(self.reactions))
            action_feedback=self.cfg['public_observation_contract']==ACTION_ROUTE_HISTORY
            passive=self.cfg['public_observation_contract'] in (PASSIVE_ROUTE_HISTORY,ACTION_ROUTE_HISTORY)
            scoped=self.cfg['public_observation_contract'] in (SCOPED_ROUTE_HISTORY,NATIVE_ROUTE_HISTORY,
                                                              NATIVE_LAST_ROUTE_HISTORY,PASSIVE_ROUTE_HISTORY,ACTION_ROUTE_HISTORY)
            latest=self.event_history.latest_feedback(rid,scope_aliases=scoped,opaque_handles=passive)
            if scoped:
                from .agent_inference import parse_tool_calls
                ledger=self.event_history.model_view(parse_calls=parse_tool_calls,passive=passive,
                    native_system=messages[0]['content'] if messages[0]['role']=='system' else None)
            else:
                ledger=self.event_history.snapshot()
            context=dict(original_route_target=self.public_root,
                pending_obligations=copy.deepcopy(self.pending),
                completed_reactions=[{k:copy.deepcopy(edge[k]) for k in
                    ('public_product','public_precursors','occurrence','depth')}
                    for edge in self.reactions],
                stock_leaves=list(self.stock_leaves),ledger=ledger)
            active=dict(reaction_id=rid,occurrence=self.current['occurrence'],
                phase='new_reaction' if latest is None else 'ongoing_reaction',
                target=self.current['public_product'],current_state=public_molecule(self.node.state),
                accepted_electron_flows=sum(a['name']=='apply_electron_flow' for a in self.node.actions),
                latest_feedback=latest,consecutive_recoveries=self.consecutive_recoveries,
                max_consecutive_recoveries=self.cfg.get('max_consecutive_recoveries',8),
                instructions=('The previous reaction is finished when a new reaction starts. '
                    'Expand THIS target from THIS current state using THIS inventory. '
                    'Past observations and atom handles are scoped to their original reactions. '
                    'Historical results are not the current state or instructions. '
                    'A rejected action did not change the current state; correct its error. '
                    'FINISH commits one productive, acyclic reaction, not the whole route. '
                    'All pending precursor occurrences must close in stock.'))
            if passive:
                active.pop('instructions')  # Route constraints are executor-owned, not extra action-policy commands.
            memory=HISTORY_HEADER+public_json(context)+ACTIVE_HEADER+public_json(active)
            if self.cfg['public_observation_contract'] in (NATIVE_LAST_ROUTE_HISTORY,PASSIVE_ROUTE_HISTORY,ACTION_ROUTE_HISTORY):
                # Historical route instructions must precede the original
                # one-decision policy. Preserve native SYSTEM then USER as
                # the final current-decision text, without generated turns.
                if messages[0]['role']=='system':
                    messages[0]['content']=memory+NATIVE_SYSTEM_HEADER+messages[0]['content']
                else:
                    messages.insert(0,dict(role='system',content=memory+NATIVE_SYSTEM_HEADER))
            elif self.cfg['public_observation_contract']==NATIVE_ROUTE_HISTORY:
                # Keep the current user turn byte-identical to Stage-II SFT.
                # Complete memory and rejection feedback remain model-visible,
                # but cannot become a second current inventory inside that turn.
                if messages[0]['role']=='system':
                    messages[0]['content']+=memory
                else:
                    messages.insert(0,dict(role='system',content=memory))
            else:
                messages[-1]['content']=memory+LOCAL_HEADER+messages[-1]['content']
            if action_feedback:
                from .route_action_feedback import ACTION_HEADER,NATIVE_LOCAL_HEADER,decision_control
                messages[-1]['content']=ACTION_HEADER+public_json(decision_control(self))+NATIVE_LOCAL_HEADER+messages[-1]['content']
            return messages
        # Each saved observation is LOCAL: embedding a prior complete prompt
        # here would recursively duplicate the history. Nothing is truncated.
        context=dict(original_target=self.public_root,
            current=copy.deepcopy(self.current),pending=copy.deepcopy(self.pending),
            reactions=copy.deepcopy(self.reactions),stock_leaves=list(self.stock_leaves),
            attempts=self.full_history,
            consecutive_recoveries=self.consecutive_recoveries,
            max_consecutive_recoveries=self.cfg.get('max_consecutive_recoveries',8),
            instructions=('Use the complete route history. All pending precursor occurrences '
                'must close in stock. Never recreate an ancestor on the current branch. '
                'Past Axx/Bxx handles belong only to their original observations; use '
                'the current inventory for a new action. A rejected action or FINISH '
                'does not change state: read its error and choose a different action. '
                'FINISH commits one productive, acyclic reaction, not the whole route.'))
        serialized=json.dumps(context,ensure_ascii=False,separators=(',',':'))
        # Raw model output contains EOS/chat markers. Preserve their bytes in
        # the JSON value without introducing real chat delimiters inside USER.
        serialized=serialized.replace('<|','\\u003c|')
        messages[-1]['content']+='\nFULL ROUTE HISTORY AND EXECUTOR FEEDBACK\n'+serialized
        return messages

    def _finish_guard_error(self, child):
        if not any(a['name']=='apply_electron_flow' for a in child.actions):
            return 'NONPRODUCTIVE_FINISH'
        if self.cfg.get('route_failure_policy'):
            from .route_action_feedback import semantic_action
            from .route_flow_credit import features
            productive=False
            for record in self.records:
                if record.get('accepted') is not True or record.get('occurrence')!=self.current['occurrence']:
                    continue
                action=semantic_action(record.get('raw',''))
                if action and action[1]['name']=='apply_electron_flow':
                    if features(record['state_before'])!=features(record['state_after']):
                        productive=True;break
            if not productive:return 'NONPRODUCTIVE_FINISH'
        try:
            structural=chemical_identity(self.chemistry.precursors(child,self.initial))
        except ValueError as exc:
            return 'FINISH_ENDPOINT_INVALID:'+str(exc)
        if structural==self.current['product']:
            return 'NONPRODUCTIVE_FINISH'
        from .route_materials import material_contract, project_material_endpoints
        children=project_material_endpoints(self.chemistry.precursors(child,self.initial),
            self.stock,material_contract(self.cfg))['precursors']
        ancestors=set(self.current['ancestors']) | {self.current['product']}
        cyclic=sorted(set(children) & ancestors)
        return 'ROUTE_CYCLE:'+','.join(cyclic) if cyclic else ''

    def _recover(self, error):
        self.consecutive_recoveries+=1
        from .route_failure_policy import FIRST_REJECTION
        if self.cfg.get('route_failure_policy')==FIRST_REJECTION:
            self._end('rejected_action',0.,error)
            return
        if self.consecutive_recoveries >= self.cfg.get('max_consecutive_recoveries',8):
            self._end('recovery_exhausted',0.,error)

    def next_decision(self):
        if self.status is not None:
            return None
        while self.current is None:
            if not self.pending:
                self._end('stock_solved',1. if self.reactions else 0.)
                return None
            self.pending.sort(key=lambda v:(v['product'],v['occurrence']))
            obligation=self.pending.pop(0)
            product=obligation['product']
            if product in obligation['ancestors']:
                self._end('cycle',0.)
                return None
            if product in self.stock:
                self.stock_leaves.append(product)
                continue
            if self.used_tokens >= self.cfg['episode_token_budget']:
                self.pending.insert(0,obligation)
                self._end('budget_exhausted',0.)
                return None
            if product in self.protected:
                self._end('resource_censored',None,'PROTECTED_EVAL_IDENTITY')
                return None
            self.current=obligation
            try:
                self.node=self.chemistry.root(obligation['public_product'])
            except ValueError as exc:
                # E.g. an inseparable non-stock multi-component root. Do not
                # silently strip the counterion or invent a compatible prompt.
                self._end('resource_censored',None,'ROOT_PROTOCOL:'+str(exc))
                return None
            self.initial=copy.deepcopy(self.node)
            if self.event_history is not None:
                self.event_history.start(self.uid+'/r'+str(len(self.reactions)),
                    self.current,public_molecule(self.node.state))
        if self.used_tokens >= self.cfg['episode_token_budget']:
            self._end('budget_exhausted',0.)
            return None
        rid=self.uid+'/r'+str(len(self.reactions))
        attempt=len(self.records) if self.agentic_feedback else len(self.node.actions)
        request_id=(self.uid+'/r0' if self.agentic_feedback else rid)+'/a'+str(attempt)
        tools=None
        if self.cfg.get('public_observation_contract')==ACTION_ROUTE_HISTORY:
            from .route_action_feedback import tools_for_episode
            tools=tools_for_episode(self)
        return DecisionRequest(request_id,self.uid,rid,
            self.current['product'],self.cfg['behavior_version'],self._messages(),
            self.cfg['episode_token_budget']-self.used_tokens,self.current['public_product'],tools)

    def _prior_rejection(self,receipt):
        if self.cfg.get('public_observation_contract')==ACTION_ROUTE_HISTORY:
            from .route_action_feedback import previous_rejection
            return previous_rejection(self,receipt)
        return None

    def accept(self, receipt):
        request=self.next_decision()
        if request is None:
            raise ValueError('EPISODE_ALREADY_FINISHED')
        if receipt['behavior_version'] != request.behavior_version:
            raise ValueError('BEHAVIOR_VERSION_MISMATCH')
        if receipt['request_id'] != request.request_id:
            raise ValueError('REQUEST_UID_MISMATCH')
        if 'messages' in receipt and receipt['messages']!=request.messages:
            raise ValueError('BEHAVIOR_OBSERVATION_MISMATCH')
        before=self.node.state
        action_feedback=self.cfg.get('public_observation_contract')==ACTION_ROUTE_HISTORY
        prior_rejection=self._prior_rejection(receipt)
        if 'state_before' in receipt and receipt['state_before']!=before:
            raise ValueError('RECOVERED_EXECUTION_STATE_BEFORE_MISMATCH')
        record=dict(receipt, messages=request.messages, reaction_id=request.reaction_id,
                    occurrence=self.current['occurrence'],state_before=before)
        local_observation=copy.deepcopy(self.chemistry.messages(self.node)) if self.agentic_feedback else None
        accepted=False
        self.records.append(record)  # even rejected/censored output is evidence
        self.used_tokens+=len(receipt.get('generated_token_ids',[]))
        if self.used_tokens>self.cfg['episode_token_budget']:
            raise ValueError('SAMPLER_EXCEEDED_EPISODE_BUDGET')
        if receipt.get('resource_error'):
            self._end('resource_censored',None,receipt['resource_error'])
        elif receipt.get('truncated'):
            if self.used_tokens==self.cfg['episode_token_budget']:
                self._end('budget_exhausted',0.)
            else:
                self._end('resource_censored',None,'ACTION_OUTPUT_CONTEXT_LIMIT')
        else:
            # Do not allow an unsuccessful executor or premature terminal to
            # mutate the last valid state, imports, visited states or history.
            candidate=copy.deepcopy(self.node) if self.agentic_feedback else self.node
            if prior_rejection is not None:
                child,error=None,prior_rejection['error']
                record['executor_skipped']='repeated_rejected_action'
            else:
                child,error=self.chemistry.step(candidate,receipt)
            if child is not None and child.terminal and self.agentic_feedback:
                error=self._finish_guard_error(child)
                if error:
                    child=None
            if 'error' in receipt and receipt['error']!=error:
                raise ValueError('RECOVERED_EXECUTION_ERROR_MISMATCH')
            if child is not None and 'state_after' in receipt and child.state!=receipt['state_after']:
                raise ValueError('RECOVERED_EXECUTION_STATE_AFTER_MISMATCH')
            record['error']=error
            if child is None:
                if error.startswith('RESOURCE_CENSORED:'):
                    self._end('resource_censored',None,error)
                elif self.agentic_feedback:
                    self._recover(error)
                else:
                    self._end('rejected_action',0.,error)
            else:
                self.node=child
                accepted=True
                self.consecutive_recoveries=0
                record['state_after']=child.state
                if child.terminal:
                    self._finish_reaction()
        if self.agentic_feedback:
            error=record.get('error',self.error or '')
            if not accepted and 'state_after' in receipt and receipt['state_after']!=before:
                raise ValueError('RECOVERED_EXECUTION_STATE_AFTER_MISMATCH')
            state_after=record['state_after'] if accepted else before
            feedback=dict(accepted=accepted,error=error,
                state_changed=state_after!=before,current_state=public_molecule(state_after),
                instruction=('Continue from the executor-owned state.' if accepted else
                    'State was not committed. Correct the error and choose another action.'))
            if self.cfg.get('route_failure_policy') and self.status=='rejected_action':
                feedback['instruction']='Sampled rollout terminated. State was not committed.'
            if action_feedback:
                feedback.update(repeated_rejected_action=prior_rejection is not None,
                    failure_repeat_count=(prior_rejection['count']+1 if prior_rejection else 1) if not accepted else 0)
            if 'feedback' in receipt and receipt['feedback']!=feedback:
                raise ValueError('RECOVERED_EXECUTION_FEEDBACK_MISMATCH')
            if 'accepted' in receipt and receipt['accepted']!=accepted:
                raise ValueError('RECOVERED_EXECUTION_ACCEPTANCE_MISMATCH')
            record['accepted']=accepted
            record['feedback']=feedback
            if self.event_history is not None:
                self.event_history.record(record,local_observation,public_molecule(before),
                    feedback['current_state'],reaction_committed=accepted and self.current is None)
            else:
                self.full_history.append(dict(request_id=request.request_id,
                    reaction_id=request.reaction_id,occurrence=record['occurrence'],
                    observation=local_observation,model_output=receipt.get('raw',''),feedback=feedback))
        # Resolve stock children and detect cycles before reporting success.
        if self.status is None:
            self.next_decision()
        record['episode_status_after']=self.status

    def _finish_reaction(self):
        public_structural=public_molecule(self.chemistry.precursors(self.node,self.initial))
        structural=chemical_identity(public_structural)
        has_flow=any(a['name']=='apply_electron_flow' for a in self.node.actions)
        if not has_flow or structural==self.current['product']:
            self._end('nonproductive_finish',0.)
            return
        from .route_materials import material_contract, project_material_endpoints, EXACT_MATERIAL
        material=material_contract(self.cfg)
        projection=project_material_endpoints(public_structural,self.stock,material)
        children,public_children=projection['precursors'],projection['public_precursors']
        if material != EXACT_MATERIAL and any(p in self.current['ancestors']+[self.current['product']] for p in children):
            self._end('cycle',0.)
            return
        depth=self.current['level']+1
        self.depth=max(self.depth,depth)
        self.reactions.append(dict(product=self.current['product'],precursors=children,
            public_product=self.current['public_product'],public_precursors=public_children,
            occurrence=self.current['occurrence'],depth=depth,final_state=self.node.state,
            initial_state=self.initial.state,actions=copy.deepcopy(self.node.actions)))
        if material != EXACT_MATERIAL:
            self.reactions[-1]['endpoint_materials']=projection['materials']
        ancestors=self.current['ancestors']+[self.current['product']]
        self.pending.extend(dict(product=p,public_product=public_children[i],
                                 occurrence=self.current['occurrence']+'.'+str(i),
                                 ancestors=ancestors,level=depth) for i,p in enumerate(children))
        self.current=self.node=self.initial=None

    def finish_report(self):
        if self.status is None:
            raise ValueError('INCOMPLETE_ROUTE_EPISODE')
        report=dict(episode_id=self.uid,root=self.root,public_root=self.public_root,seed=self.seed,
            route_endpoint_contract=self.cfg['route_endpoint_contract'],
            behavior_version=self.cfg['behavior_version'],status=self.status,error=self.error,
            reward=self.reward,generated_tokens=self.used_tokens,depth=self.depth,
            reactions=self.reactions,stock_leaves=self.stock_leaves,records=self.records,
            pending=self.pending,gt_used_for_generation_or_stopping=False)
        if self.agentic_feedback:
            report['public_observation_contract']=self.cfg['public_observation_contract']
            report['max_consecutive_recoveries']=self.cfg.get('max_consecutive_recoveries',8)
        if 'stock_material_contract' in self.cfg:
            report['stock_material_contract']=self.cfg['stock_material_contract']
        if 'route_failure_policy' in self.cfg:
            report['route_failure_policy']=self.cfg['route_failure_policy']
        return report

    def snapshot(self, *, include_stock=True):
        result={k:copy.deepcopy(v) for k,v in self.__dict__.items()
                if k not in ('chemistry','stock','node','initial','protected','event_history')}
        if self.event_history is not None:
            result['event_history']=self.event_history.snapshot()
        result['protected']=sorted(self.protected)
        if include_stock:
            result['stock']=sorted(self.stock)
        for key in ('node','initial'):
            node=getattr(self,key)
            result[key]=None if node is None else copy.deepcopy(vars(node))
            if result[key] and isinstance(result[key].get('visited'),set):
                result[key]['visited']=sorted(result[key]['visited'])
        return result

    def _validate_restored_event_history(self):
        """A valid reference graph must also retain every original receipt."""
        data=self.event_history.data
        attempts=[e for e in data['events'] if e['type']=='attempt']
        starts=[e for e in data['events'] if e['type']=='reaction_started']
        expected=[dict(reaction_id=self.uid+'/r'+str(i),occurrence=e['occurrence'],
            target=e['public_product'],state=public_molecule(e['initial_state']),level=e['depth']-1)
            for i,e in enumerate(self.reactions)]
        if self.current is not None and self.initial is not None:
            expected.append(dict(reaction_id=self.uid+'/r'+str(len(self.reactions)),
                occurrence=self.current['occurrence'],target=self.current['public_product'],
                state=public_molecule(self.initial.state),level=self.current['level']))
        if len(attempts)!=len(self.records) or len(starts)!=len(expected):
            raise ValueError('HISTORY_RECEIPT_COUNT_MISMATCH')
        for actual,wanted in zip(starts,expected):
            if (any(actual[k]!=wanted[k] for k in ('reaction_id','occurrence','target','level'))
                    or data['state_texts'][actual['initial_state_ref']]!=wanted['state']):
                raise ValueError('HISTORY_REACTION_START_MISMATCH')
        completed={e['reaction_id'] for e in expected[:len(self.reactions)]}
        last={r['reaction_id']:r['request_id'] for r in self.records}
        for event,record in zip(attempts,self.records):
            observation=copy.deepcopy(record['messages'])
            from .route_action_feedback import ACTION_HEADER,NATIVE_LOCAL_HEADER
            if observation[-1]['content'].startswith(ACTION_HEADER):
                if NATIVE_LOCAL_HEADER not in observation[-1]['content']:
                    raise ValueError('HISTORY_CURRENT_OBSERVATION_MISSING')
                observation[-1]['content']=observation[-1]['content'].split(NATIVE_LOCAL_HEADER,1)[1]
            if LOCAL_HEADER in observation[-1]['content']:
                observation[-1]['content']=observation[-1]['content'].split(LOCAL_HEADER,1)[1]
            elif (observation[0]['role']=='system'
                    and observation[0]['content'].startswith(HISTORY_HEADER)
                    and NATIVE_SYSTEM_HEADER in observation[0]['content']):
                native_system=observation[0]['content'].split(NATIVE_SYSTEM_HEADER,1)[1]
                original=self.event_history.observation(event['observation_ref'])
                if native_system or original[0]['role']=='system':
                    observation[0]['content']=native_system
                else:
                    observation.pop(0)
            elif observation[0]['role']=='system' and HISTORY_HEADER in observation[0]['content']:
                # Saved prefixes can mix v3/v4 user wrappers with v5 system
                # appendices. Recover each record's actual native observation,
                # then keep the same byte-for-byte receipt validation below.
                native_system=observation[0]['content'].split(HISTORY_HEADER,1)[0]
                original=self.event_history.observation(event['observation_ref'])
                if native_system or original[0]['role']=='system':
                    observation[0]['content']=native_system
                else:
                    observation.pop(0)
            else:
                raise ValueError('HISTORY_LOCAL_OBSERVATION_MISSING')
            after=record.get('state_after',record['state_before'])
            commit=record['reaction_id'] in completed and record['request_id']==last[record['reaction_id']]
            if (any(event[k]!=record[k] for k in ('request_id','reaction_id','occurrence'))
                    or data['output_texts'][event['output_ref']]!=record.get('raw','')
                    or data['state_texts'][event['before_state_ref']]!=public_molecule(record['state_before'])
                    or data['state_texts'][event['after_state_ref']]!=public_molecule(after)
                    or event['feedback']!=RouteEventHistory.scoped_feedback(record['feedback'])
                    or self.event_history.observation(event['observation_ref'])!=observation
                    or event['reaction_committed']!=commit):
                raise ValueError('HISTORY_RECEIPT_CONTENT_MISMATCH')

    @classmethod
    def restore(cls, snapshot, *, stock=None, chemistry=None):
        from types import SimpleNamespace
        if snapshot.get('cfg',{}).get('route_endpoint_contract') != ROUTE_ENDPOINT_CONTRACT:
            raise ValueError('UNRECOGNIZED_ROUTE_ENDPOINT_CONTRACT')
        from .route_materials import material_contract, EXACT_MATERIAL
        material=material_contract(snapshot['cfg'])
        native=snapshot.get('cfg',{}).get('public_observation_contract')==NATIVE_SFT_ROUTE
        if native or material!=EXACT_MATERIAL:
            from .route_native_observation import restore_native_episode
            return restore_native_episode(cls, snapshot, stock=stock, chemistry=chemistry,
                error_prefix='SFT_HISTORY' if native else 'MATERIAL_HISTORY')
        result=object.__new__(cls)
        result.__dict__.update(copy.deepcopy(snapshot))
        result.event_history=(RouteEventHistory(snapshot['event_history'])
            if result.cfg.get('public_observation_contract') in EVENT_ROUTE_OBSERVATIONS else None)
        result.stock=stock if stock is not None else set(snapshot['stock'])
        result.protected=set(snapshot['protected'])
        result.chemistry=chemistry or NativeChemistry(result.cfg.get('concurrent_executors',32))
        for key in ('node','initial'):
            state=snapshot[key]
            if state is None:
                continue
            state=copy.deepcopy(state)
            if 'visited' in state:
                state['visited']=set(state['visited'])
            if chemistry is None:
                from scripts.run_natural_language_value_search import Node
                state=Node(**state)
            else:
                state=SimpleNamespace(**state)
            setattr(result,key,state)
        if result.event_history is not None:
            result._validate_restored_event_history()
        return result
