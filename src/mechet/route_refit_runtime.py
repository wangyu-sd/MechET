"""Batched independent ReFIT suffixes with deterministic journal-only recovery.

The CPU harvester owns tree mutation; worker threads only execute their episode.
The policy always samples its original complete tool distribution. Budgets bound
additional branching, never invent a chemical failure for an unfinished rollout.
"""
from collections import Counter
from copy import deepcopy
from threading import RLock

from .route_refit import DecisionNode,decision_identity,grounded_action_key

DEFAULTS=dict(max_nodes_per_tree=8,max_expansions=2,max_duplicates=2,
              max_forks_per_tree=16,matched_continuations=1)


def validate_refit(cfg):
    if set(cfg)!=set(DEFAULTS) or any(type(v) is not int or v<1 for v in cfg.values()):
        raise ValueError('REFIT_EXPLICIT_FINITE_BUDGETS_REQUIRED')
    if cfg['matched_continuations']!=1:
        raise ValueError('REFIT_RUNTIME_V1_MATCHED_M1_ONLY')
    return cfg


def unique_collection_counts(episodes):
    from .route_action_feedback import semantic_action
    records={r['request_id']:r for ep in episodes for r in ep.records}
    commits=0
    for r in records.values():
        action=semantic_action(r.get('raw',''))
        commits+=bool(r.get('accepted') is True and action and action[1]['name']=='finish_trace')
    return dict(model_decision_requests=len(records),
        executor_calls=sum(not r.get('executor_skipped') and not r.get('resource_error')
                           and not r.get('truncated') for r in records.values()),
        completed_reaction_programs=commits)


class ObservedEpisode:
    def __init__(self,ep,controller,path=(),reserved_from=None,*,continuation_id=None,
                 lane='original_g4',creation_order=0):
        self.ep,self.controller=ep,controller
        self.path=deepcopy(list(path));self.completed_action=None
        self.reserved_from=reserved_from
        self.continuation_id=ep.uid if continuation_id is None else continuation_id
        self.lane,self.creation_order=lane,creation_order

    def __getattr__(self,key):return getattr(self.ep,key)

    def accept(self,receipt):
        from .route_action_feedback import semantic_action,tools_for_request
        request=self.ep.next_decision()
        if request is None:raise ValueError('REFIT_CLOSED_ROLLOUT')
        conditioning=dict(target_id=self.ep.root,tree_id=self.ep.uid,
            occurrence=self.ep.current['occurrence'],prompt_token_ids=receipt['prompt_token_ids'],
            tools=tools_for_request(request),
            accepted_history=[dict(raw=r['raw'],state_before=r['state_before'],
                state_after=r.get('state_after')) for r in self.ep.records if r.get('accepted') is True],
            route_context=dict(current=self.ep.current,pending=self.ep.pending,reactions=self.ep.reactions,
                               stock_leaves=self.ep.stock_leaves),
            behavior_version=request.behavior_version,
            executor_contract=[self.ep.cfg['route_endpoint_contract'],
                               self.ep.cfg.get('stock_material_contract','exact_material_stock_v1')],
            mapped_state=self.ep.node.state)
        # FINISH mutates pending/reactions in place. Freeze the complete hash
        # preimage before execution, not aliases to the live obligation lists.
        conditioning=deepcopy(conditioning)
        key=decision_identity(**conditioning)
        action=grounded_action_key(self.ep.node.state,receipt.get('raw',''))
        with self.controller.lock:
            node=self.controller.nodes.get(key)
            errors=({k:e['error'] for k,e in node.edges.items() if e['status']=='REJECTED_TERMINAL'} if node else {})
            retain_parent=node is not None or self.controller.can_add_node(self.ep.uid)
        if errors:self.ep.set_rejections(errors)
        parent=self.ep.clone() if retain_parent else None
        index=len(self.ep.records);parsed=semantic_action(receipt.get('raw',''))
        self.ep.accept(receipt)
        self.completed_action=dict(node_id=key,action=action,index=index,
            name=parsed[1]['name'] if parsed else 'malformed',request_id=receipt['request_id'],
            parent=parent,conditioning=conditioning)


class RefitCollection:
    def __init__(self,episodes,cfg,*,duplicate_policy=None,lane_config=None):
        if duplicate_policy not in (None,'skip_extra_duplicates_v1','skip_extra_duplicates_v2'):
            raise ValueError('REFIT_DUPLICATE_POLICY')
        self.duplicate_policy=duplicate_policy;self.duplicate_pruned=0
        self.cfg=dict(validate_refit(cfg));self.nodes={};self.parents={};self.node_trees={}
        self.v2=duplicate_policy=='skip_extra_duplicates_v2'
        if self.v2:
            if (not isinstance(lane_config,dict) or
                    set(lane_config)!={'credit_forks_per_tree','exploration_forks_per_tree'} or
                    any(type(v) is not int or v<0 for v in lane_config.values()) or
                    lane_config['credit_forks_per_tree']<1 or
                    sum(lane_config.values())>min(16,self.cfg['max_forks_per_tree'])):
                raise ValueError('REFIT_EXPLICIT_FINITE_LANE_BUDGETS_REQUIRED')
        elif lane_config is not None:
            raise ValueError('REFIT_LANE_CONFIG_REQUIRES_V2')
        self.lane_config=deepcopy(lane_config)
        self.lane_forks=Counter();self.receipt_owner={};self.completed_eligible={}
        self.harvest_order=0;self.harvested_status={}
        self.conditions={};self.lock=RLock()
        self.forks=Counter();self.initial=[ObservedEpisode(ep,self,creation_order=i) for i,ep in enumerate(episodes)]
        self.pending=Counter();self.draw_serials=Counter()
        self.all=list(self.initial)
        self.initial_ids={id(e) for e in self.initial}
        if self.v2:
            # Initial terminal states predate executor ownership and have no
            # sampled receipt to harvest or to promote as M1 coverage.
            self.harvested_status.update((e.continuation_id,e.ep.status) for e in self.initial)

    def can_add_node(self,tree):
        return (self.forks[tree]<self.cfg['max_forks_per_tree'] and
                sum(v==tree for v in self.node_trees.values())<self.cfg['max_nodes_per_tree'])

    def owner(self,request_id):
        matches=[e for e in self.all if e.next_decision() is not None and e.next_decision().request_id==request_id]
        if len(matches)!=1:raise ValueError('REFIT_REQUEST_OWNER_MISMATCH:'+request_id)
        return matches[0]

    def available(self,key):
        node=self.nodes[key];pending=self.pending[key]
        return (node.status!='EXHAUSTED' and
                node.expansions+pending<node.max_expansions and
                node.duplicate_draws+pending<node.max_duplicates)

    def completed(self,episode):
        with self.lock:return self._completed(episode)

    def _completed(self,episode):
        event=episode.completed_action
        if event is None:raise ValueError('REFIT_COMPLETED_ACTION_REQUIRED')
        episode.completed_action=None
        record=episode.ep.records[-1];tree=episode.ep.uid;key=event['node_id']
        if episode.reserved_from is not None:
            origin=episode.reserved_from
            if origin!=key or self.pending[origin]<1:
                raise ValueError('REFIT_PENDING_PARENT_BINDING_MISMATCH')
            self.pending[origin]-=1;episode.reserved_from=None
        parent=event.pop('parent');conditioning=event.pop('conditioning');node=self.nodes.get(key)
        resource=episode.ep.status in ('resource_censored','protected_censored')
        # FLOW siblings are primary; a rejected IMPORT/FINISH also opens a
        # backtracking point, but never becomes a FLOW comparison by relabeling.
        if (node is None and parent is not None and self.can_add_node(tree) and
                (event['name']=='apply_electron_flow' or record.get('accepted') is False)):
            node=DecisionNode(key,self.cfg['max_expansions'],self.cfg['max_duplicates'])
            self.nodes[key]=node;self.parents[key]=(parent,deepcopy(episode.path))
            self.node_trees[key]=tree
            self.conditions[key]=conditioning
        event['proposal']='original_policy'
        if self.v2:
            if event['request_id'] in self.receipt_owner:
                raise ValueError('REFIT_DUPLICATE_OWNED_RECEIPT')
            self.receipt_owner[event['request_id']]=episode.continuation_id
            event.update(receipt_owner=episode.continuation_id,lane=episode.lane,
                         declaration_order=episode.creation_order,harvest_order=self.harvest_order)
            self.harvest_order+=1
        episode.path.append(event)
        accepted_duplicate=(node is not None and event['action'] in node.edges and
                            node.edges[event['action']]['status']=='ACCEPTED' and record.get('accepted') is True)
        if node is not None:
            successor=None if not record.get('accepted') else record.get('state_after')
            node.add(event['action'],event['request_id'],successor=successor,
                error=record.get('error','') if not record.get('accepted') else '',
                resource=resource,executor_skipped=bool(record.get('executor_skipped')))
        if self.v2:
            # Only this harvested continuation may become eligible. Worker
            # statuses elsewhere in all are deliberately never scanned here.
            self.harvested_status[episode.continuation_id]=episode.ep.status
            self._promote_completed(episode)
        if ((self.duplicate_policy=='skip_extra_duplicates_v1' or
                (self.v2 and episode.lane=='credit' and episode.ep.status is None and
                 len(self.coverage(key,event['action']))>=self.cfg['matched_continuations']))
                and accepted_duplicate and id(episode) not in self.initial_ids
                and event['name']=='apply_electron_flow'):
            from .route_acceleration import SKIPPED
            episode.ep._end(SKIPPED,None)
            record['episode_status_after']=SKIPPED
            self.duplicate_pruned+=1
            if self.v2:self.harvested_status[episode.continuation_id]=SKIPPED
            return []
        if resource or self.forks[tree]>=self.cfg['max_forks_per_tree']:return []
        lane=self._next_lane(tree) if self.v2 else 'credit'
        if lane is None:return []
        candidate=key if node is not None and self.available(key) else None
        if candidate is None and episode.ep.status=='rejected_action':
            candidate=next((e['node_id'] for e in reversed(episode.path[:-1])
                if e['node_id'] in self.nodes and self.available(e['node_id'])),None)
        if candidate is None:return []
        node=self.nodes[candidate];parent,path=self.parents[candidate]
        # Reserve before publishing a fork: pending draws may become either a
        # distinct edge or a duplicate. Both ceilings must hold conservatively.
        # Journal replay follows this same serialized reservation/release path.
        self.pending[candidate]+=1;self.draw_serials[candidate]+=1
        draw=self.draw_serials[candidate]
        from .external_route_data import signature
        namespace=tree+'/fork-'+signature([candidate,draw])[:20]
        fork=ObservedEpisode(parent.fork(namespace),self,path,reserved_from=candidate,
            continuation_id=namespace,lane=lane,creation_order=len(self.all))
        self.forks[tree]+=1
        if self.v2:self.lane_forks[tree,lane]+=1
        self.all.append(fork)
        return [fork]

    def _next_lane(self,tree):
        for lane in ('credit','exploration'):
            if self.lane_forks[tree,lane]<self.lane_config[lane+'_forks_per_tree']:
                return lane
        return None

    def _promote_completed(self,episode):
        if (episode.lane=='exploration' or episode.ep.status in
                (None,'resource_censored','protected_censored','duplicate_pruned_no_continuation')):
            return
        for event in episode.path:
            if (event['name']!='apply_electron_flow' or
                    event['receipt_owner']!=episode.continuation_id):continue
            key=(event['node_id'],event['action'])
            ledger=self.completed_eligible.setdefault(key,[])
            if any(r['request_id']==event['request_id'] for r in ledger):continue
            ledger.append(dict(request_id=event['request_id'],continuation_id=episode.continuation_id,
                lane=episode.lane,tree_id=episode.ep.uid,declaration_order=event['declaration_order'],
                harvest_order=event['harvest_order'],completion_order=self.harvest_order-1))
            ledger.sort(key=lambda r:(r['declaration_order'],r['harvest_order']))

    def coverage(self,node_id,action):
        with self.lock:return deepcopy(self.completed_eligible.get((node_id,action),[]))

    def lane_metrics(self,tree=None):
        """Harvested counters only: executor threads may still own live states."""
        with self.lock:
            counts={lane:dict(attempts=0,forks=0,complete=0,failed=0,censored=0,skipped=0,
                completed_eligible_coverage=0) for lane in ('original_g4','credit','exploration')}
            for ep in self.all:
                if tree is not None and ep.ep.uid!=tree:continue
                row=counts[ep.lane]
                row['forks']+=ep.lane!='original_g4'
                row['attempts']+=sum(e.get('receipt_owner')==ep.continuation_id for e in ep.path)
                status=self.harvested_status.get(ep.continuation_id)
                if status in ('resource_censored','protected_censored'):row['censored']+=1
                elif status=='duplicate_pruned_no_continuation':row['skipped']+=1
                elif status is not None:
                    row['complete']+=1
                    row['failed']+=status not in ('stock_solved','already_stock')
            # This is online eligibility, never a claim about the post-score
            # estimator's actual M1 use (which may still censor-mask a group).
            for ledger in self.completed_eligible.values():
                for entry in ledger:
                    if tree is None or entry['tree_id']==tree:
                        counts[entry['lane']]['completed_eligible_coverage']+=1
            return counts

    def replay(self,receipts):
        for receipt in receipts:
            owner=self.owner(receipt['request_id']);owner.accept(deepcopy(receipt));self.completed(owner)

    def pack(self,reports):
        if len(reports)!=len(self.all):raise ValueError('REFIT_REPORT_DENOMINATOR')
        for report,ep in zip(reports,self.all):
            report['refit_path']=deepcopy(ep.path)
            if self.v2:
                report.update(refit_lane=ep.lane,continuation_id=ep.continuation_id,
                              continuation_declaration_order=ep.creation_order)
        originals=reports[:len(self.initial)]
        for report,ep in zip(originals,self.initial):
            report['refit']=dict(version='exact_state_flow_siblings_v1',config=self.cfg,
                suffix_reports=[r for r,e in zip(reports[len(self.initial):],self.all[len(self.initial):])
                                if e.ep.uid==ep.ep.uid],
                node_status={k:n.status for k,n in self.nodes.items() if self.node_trees[k]==ep.ep.uid})
            report['refit']['conditioning']={k:v for k,v in self.conditions.items() if self.node_trees[k]==ep.ep.uid}
            report['refit']['node_censored_receipts']={k:list(n.censored_receipts) for k,n in self.nodes.items()
                if self.node_trees[k]==ep.ep.uid and n.censored_receipts}
            report['refit']['reserved_draws']={k:self.pending[k] for k in self.nodes if self.node_trees[k]==ep.ep.uid}
            if self.duplicate_policy:
                report['refit']['duplicate_suffix_policy']=self.duplicate_policy
            if self.v2:
                exploration_ids={e['request_id'] for continuation in self.all
                    if continuation.ep.uid==ep.ep.uid for e in continuation.path
                    if e['lane']=='exploration'}
                # Exploration outcomes remain visible, but only original/credit
                # censoring may mask the conditional-credit estimator.
                censored=report['refit']['node_censored_receipts']
                report['refit']['exploration_node_censored_receipts']={k:[r for r in ids if r in exploration_ids]
                    for k,ids in censored.items() if any(r in exploration_ids for r in ids)}
                report['refit']['node_censored_receipts']={k:[r for r in ids if r not in exploration_ids]
                    for k,ids in censored.items() if any(r not in exploration_ids for r in ids)}
                report['refit'].update(lane_config=deepcopy(self.lane_config),
                    lane_counts=self.lane_metrics(ep.ep.uid),
                    completed_eligible_coverage=[dict(node_id=node,action=action,receipts=deepcopy(ledger))
                        for (node,action),ledger in self.completed_eligible.items()
                        if ledger[0]['tree_id']==ep.ep.uid],
                    receipt_owner={e['request_id']:e['receipt_owner'] for continuation in self.all
                        if continuation.ep.uid==ep.ep.uid for e in continuation.path})
        return originals
