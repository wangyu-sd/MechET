"""Shared sampled-action PUCT with real electronic/AND-route rollouts.

No learned dynamics/value and no gold feedback. Planning statistics bind the
stock and the FULL unresolved state, not merely the current molecule. This
module deliberately lives outside the unchanged executor/cache dependency set.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

from mechet.dynamic_electron_tree import Proposal, SearchContext, chemical_key
from mechet.reverse_route_network import RouteNetwork
from mechet.shared_electron_search import durable_json, read_record, fingerprint, context_record
from scripts.run_natural_language_value_search import Node

ALGORITHM = 'shared_sampled_puct_mcts_v1'


def planning_edge_id(parent,child,proposal,error):
    def payload(value):
        if isinstance(value,dict):return {k:payload(v) for k,v in value.items() if k!='instruction'}
        if isinstance(value,list):return [payload(v) for v in value]
        return value
    arguments=(payload(proposal.arguments) if proposal.name in {'apply_electron_flow','import_fragments'}
               else proposal.arguments)
    return fingerprint([parent,child,proposal.name,arguments,error])


class RolloutLease:
    def __init__(self, uid, handle):
        self.uid, self.handle = uid, handle

    def close(self):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close(); self.handle = None


class SharedMCTSStore:
    def __init__(self, root, contract):
        self.contract = {'schema': ALGORITHM, 'c_puct': 1.5,
                         'planner_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**contract}
        self.root = Path(root) / fingerprint(self.contract)
        self.root.mkdir(parents=True, exist_ok=True)
        self.recover_backups()

    def path(self, sid):
        return self.root / 'nodes' / sid[:2] / (sid + '.json')

    @contextmanager
    def edit(self, sid):
        p = self.path(sid); p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix('.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            value = read_record(p) if p.exists() else None
            yield value
            if value is not None: durable_json(p, value)

    def intern(self, state):
        # Native payload is evidence/rehydration metadata, not a substitute for
        # the exact electronic context key (which binds history and inventory).
        identity = {k:v for k,v in state.items() if k != 'native'}
        sid = fingerprint(identity); p = self.path(sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix('.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            if not p.exists():
                durable_json(p, {'state': state, 'visits': 0, 'value_sum': 0., 'selections': 0,
                    'edges': {}, 'observations': [], 'policy_batches': 0,'last_expansion_selection':-1,
                    'reservations': {}, 'backed_up': {}})
        return sid

    def alive(self, uid):
        p = self.root / 'rollouts' / (uid + '.lock')
        if not p.exists(): return False
        with p.open('a+') as handle:
            try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return True
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False

    def snapshot(self, sid):
        value = read_record(self.path(sid))
        stale = [uid for uid in value['reservations'] if not self.alive(uid)]
        if stale:
            with self.edit(sid) as value:
                for uid in stale:
                    if not self.alive(uid): value['reservations'].pop(uid, None)
        return value

    def start_rollout(self):
        uid = uuid.uuid4().hex
        p = self.root / 'rollouts' / (uid + '.lock'); p.parent.mkdir(parents=True, exist_ok=True)
        handle = p.open('a+'); fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return RolloutLease(uid, handle)

    def reserve(self, uid, sid, edge):
        with self.edit(sid) as value:
            if edge not in value['edges']: raise ValueError('unknown MCTS edge')
            if uid not in value['reservations']: value['selections'] += 1
            value['reservations'][uid] = edge

    def claim_expansion(self, sid):
        p = self.path(sid).with_suffix('.expansion.lock')
        handle = p.open('a+')
        try: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close(); return None
        return RolloutLease(sid, handle)

    def add_edges(self, sid, edges, *, observation):
        with self.edit(sid) as value:
            if observation in value['observations']: return
            for edge in edges:
                old = value['edges'].get(edge['id'])
                if old is None:
                    value['edges'][edge['id']] = {**edge, 'n': 0, 'w': 0.}
                else: old['count'] += edge['count']
            value['observations'].append(observation); value['policy_batches'] += 1
            value['last_expansion_selection']=value['selections']

    def select_edge(self, sid, *, excluded=()):
        value = self.snapshot(sid)
        width = 1 + math.isqrt(value['selections'])
        edges = sorted((e for e in value['edges'].values() if e.get('child') not in excluded),
                       key=lambda e:(-e['count'], e['id']))[:width]
        if not edges: return None
        total = sum(e['count'] for e in value['edges'].values())
        virtual = list(value['reservations'].values())
        visits = value['visits'] + len(virtual)
        def score(edge):
            n = edge['n']; q = edge['w']/n if n else 0.
            return q + self.contract['c_puct'] * (edge['count']/max(1,total)) * math.sqrt(visits+1) / (1+n+virtual.count(edge['id']))
        return max(edges, key=lambda e:(score(e), e['id']))

    def needs_expansion(self, sid):
        value = self.snapshot(sid)
        if any(edge['w']>0 for edge in value['edges'].values()):return False
        return not value['edges'] or (1 + math.isqrt(value['selections']) > len(value['edges'])
                and value['selections']>value.get('last_expansion_selection',-1))

    def backup(self, uid, path, reward):
        if reward not in (0.,1.): raise ValueError('binary completed-rollout return required')
        path = [list(pair) for pair in path]
        if len({sid for sid,_ in path}) != len(path): raise ValueError('cyclic MCTS backup path')
        p = self.root / 'backups' / (uid + '.json'); p.parent.mkdir(parents=True, exist_ok=True)
        with p.with_suffix('.lock').open('a+') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            record = {'uid':uid,'path':path,'reward':reward}
            if p.exists():
                if read_record(p) != record: raise ValueError('conflicting duplicate backup')
            else: durable_json(p, record)
            self._apply_backup(record)

    def _apply_backup(self, record):
        for sid, eid in record['path']:
            with self.edit(sid) as value:
                if record['uid'] not in value['backed_up']:
                    edge = value['edges'][eid]
                    edge['n'] += 1; edge['w'] += record['reward']
                    value['visits'] += 1; value['value_sum'] += record['reward']
                    value['backed_up'][record['uid']] = [eid,record['reward']]
                value['reservations'].pop(record['uid'],None)
        done = self.root / 'backup_done' / (record['uid'] + '.json')
        if not done.exists(): durable_json(done, {'uid':record['uid']})

    def recover_backups(self):
        for p in (self.root/'backups').glob('*.json'):
            if not (self.root/'backup_done'/p.name).exists():
                with p.with_suffix('.lock').open('a+') as handle:
                    fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
                    self._apply_backup(read_record(p))


def finish_obligations(product, precursors, pending, ancestors, stock):
    next_ancestors = tuple(sorted(set(ancestors) | {product}))
    remaining = [(m,tuple(a)) for m,a in pending]
    for precursor in precursors:
        if precursor in next_ancestors: raise ValueError('synthesis cycle')
        if precursor not in stock: remaining.append((precursor,next_ancestors))
    return sorted(remaining)


class MCTSRouteNetwork(RouteNetwork):
    def __init__(self, case_id, product, stock, *, mcts_store, shared_store=None, checkpoint_path=None):
        super().__init__(case_id,product,stock,mode='tree',shared_store=shared_store)
        self.mcts_store = mcts_store; self.checkpoint_path = checkpoint_path
        self.rollouts = deque(); self.selected = None; self.cached_events = []
        self.applied = set(); self.blocked = set(); self.seen_states = set()
        self.completed_rollouts = 0; self.positive_backups = 0; self.censored_rollouts = 0
        self.has_pending = not self.already_in_stock
        self.mcts_root = None if self.already_in_stock else self._state([],self.target,self.trees[self.target].root_key,())

    @classmethod
    def restore(cls,journal,*,stock,mcts_store,shared_store=None,checkpoint_path=None):
        base=journal.restore(stock=stock,shared_store=shared_store)
        controller=cls(base.case_id,base.target,stock,mcts_store=mcts_store,
                       shared_store=shared_store,checkpoint_path=checkpoint_path)
        root=controller.mcts_root
        controller.__dict__.update(base.__dict__)
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            saved=read_record(checkpoint_path)
            if (saved['case_id']!=base.case_id or saved['root']!=root
                    or saved.get('planning_contract')!=mcts_store.contract):
                raise ValueError('MCTS checkpoint contract mismatch')
            if any(mcts_store.alive(r['uid']) for r in saved['pending_rollouts']):
                raise ValueError('MCTS checkpoint still has a live rollout owner')
            for name in ('completed_rollouts','positive_backups','censored_rollouts'):
                setattr(controller,name,saved['metrics'][name])
            controller.censored_rollouts+=len(saved['pending_rollouts'])
            controller.seen_states=set(saved['seen_states']);controller.blocked=set(saved['blocked'])
            controller.first_solution_work=saved['first_solution']
            controller.stop_reason=saved.get('stop_reason','')
        controller.has_pending=not (controller.already_in_stock or controller.first_solution_work is not None)
        # Restart fresh rollouts through the shared graph; interrupted paths are
        # censored, never backed up as chemical failure or resumed twice.
        controller.checkpoint()
        return controller

    def _register_terminal(self,molecule,context):
        # The backend's union graph may already close via a DIFFERENT path.
        # It is evidence for route reporting, not a reward for this rollout.
        prior = self.first_solution_work
        super()._register_terminal(molecule,context)
        self.first_solution_work = prior

    def _state(self,pending,molecule,key,ancestors):
        payload = {'pending':pending,'molecule':molecule,'active':key,'ancestors':sorted(ancestors),
                   'solved':molecule is None}
        if molecule is not None: payload['native'] = context_record(self.contexts[key])
        sid = self.mcts_store.intern(payload); self.seen_states.add(sid)
        return sid

    def _hydrate(self,state):
        molecule = state['molecule']
        if molecule is None: return
        self._add_obligation(molecule)
        record = state['native']; key = record['id']
        if key not in self.contexts:
            data = dict(record['node']); data['visited'] = set(data['visited'])
            context = SearchContext(key,Node(**data),record['chemical_id'],max_new_tokens=record['max_new_tokens'])
            tree = self.trees[molecule]; tree.contexts[key] = context
            tree.chemical_nodes.setdefault(context.chemical_id,{'id':context.chemical_id,
                'state_unmapped':chemical_key(context.node.state),'private_state':context.node.state})
            self._index_contexts(molecule)

    def _transition(self,state,native_edge):
        child = native_edge.get('child_context')
        if not child: return None, native_edge.get('error','invalid action')
        context = self.contexts[child]
        molecule = state['molecule']
        if not context.node.terminal:
            return self._state(state['pending'],molecule,child,state['ancestors']), ''
        tree = self.trees[molecule]
        from mechet.endpoints import split_precursor_endpoints
        parts = split_precursor_endpoints(context.node.state,tree.contexts[tree.root_key].node.state)
        structural = chemical_key(parts.structural) if parts.structural else ''
        if (not structural or structural == molecule or
                not any(a['name']=='apply_electron_flow' for a in context.node.actions)):
            return None, 'NONPRODUCTIVE_REACTION'
        precursors = [structural] if structural in self.stock else sorted(structural.split('.'))
        try: remaining = finish_obligations(molecule,precursors,state['pending'],state['ancestors'],self.stock)
        except ValueError as error: return None,str(error)
        if not remaining: return self._state([],None,None,()), ''
        active,ancestry = remaining[0]; self._add_obligation(active)
        return self._state(remaining[1:],active,self.trees[active].root_key,ancestry), ''

    def _ensure_transition(self,sid,edge):
        identity = (sid,edge['id'])
        if identity in self.applied: return
        state = self.mcts_store.snapshot(sid)['state']; self._hydrate(state)
        native = super().incorporate(state['active'],[Proposal(**edge['proposal'])])
        child,error = self._transition(state,native[0])
        if child != edge['child']: raise ValueError('shared MCTS transition differs from native replay')
        self.applied.add(identity); self.cached_events.append((state['active'],native))

    def _complete(self,rollout,reward):
        if reward and self._certificate() is None:
            raise ValueError('MCTS success lacks stock-connected certificate')
        self.mcts_store.backup(rollout['lease'].uid,rollout['path'],reward)
        rollout['lease'].close(); self.completed_rollouts += 1
        if reward:
            self.positive_backups += 1
            self.first_solution_work = {'work':self.work,'wall_seconds':time.monotonic()-self.started,
                                       'rollout_path':rollout['path'],'return':1.}
            self.has_pending = False

    def next_context(self):
        if not self.has_pending: return None
        if self.selected is not None: return self.contexts[self.selected['context']]
        # Multiple live rollouts prevent one deep electronic continuation from
        # consuming a target forever before an alternative gets explored.
        if len(self.rollouts)<8:
            self.rollouts.append({'lease':self.mcts_store.start_rollout(), 'sid':self.mcts_root,'path':[]})
        rollout = self.rollouts.popleft()
        for _ in range(64):  # cooperative CPU quantum, NOT a chemical horizon
            sid = rollout['sid']; self.seen_states.add(sid)
            value = self.mcts_store.snapshot(sid); state = value['state']
            if state['solved']:
                self._complete(rollout,1.); self.checkpoint(); return None
            self._hydrate(state); context = self.contexts[state['active']]
            if context.resource_closed or sid in self.blocked:
                self.blocked.add(sid)
                if sid==self.mcts_root:
                    self.has_pending=False;self.stop_reason=context.resource_closed or 'ROOT_RESOURCE_CLOSED'
                self.censored_rollouts += 1; rollout['lease'].close(); self.checkpoint(); return None
            if self.mcts_store.needs_expansion(sid):
                claim = self.mcts_store.claim_expansion(sid)
                if claim is not None:
                    value=self.mcts_store.snapshot(sid)
                    if not self.mcts_store.needs_expansion(sid):
                        claim.close();continue
                    # Global per-planning-node batch indices; exact native
                    # requests still use the unchanged shared policy cache.
                    context.stats.batches = value['policy_batches']
                    self.selected = {'rollout':rollout,'sid':sid,'claim':claim,'context':context.key,
                                     'batch':value['policy_batches']}
                    self.checkpoint(); return context
            edge = self.mcts_store.select_edge(sid,excluded=self.blocked)
            if edge is None:
                # All admitted continuations resource-closed is not evidence
                # that the policy has no other valid action: ask for more.
                claim=self.mcts_store.claim_expansion(sid)
                if claim is not None:
                    value=self.mcts_store.snapshot(sid);context.stats.batches=value['policy_batches']
                    self.selected={'rollout':rollout,'sid':sid,'claim':claim,'context':context.key,
                                   'batch':value['policy_batches']}
                    self.checkpoint();return context
                self.rollouts.append(rollout); self.checkpoint(); return None
            self.mcts_store.reserve(rollout['lease'].uid,sid,edge['id'])
            rollout['path'].append((sid,edge['id']))
            self._ensure_transition(sid,edge)
            if edge['child'] is None:
                self._complete(rollout,0.)
                self.checkpoint(); return None
            if edge['child'] in {x for x,_ in rollout['path']}:
                self._complete(rollout,0.); self.checkpoint(); return None
            rollout['sid'] = edge['child']
        self.rollouts.append(rollout); self.checkpoint(); return None

    def incorporate(self,parent_key,proposals):
        if self.selected is None or self.selected['context'] != parent_key:
            raise ValueError('MCTS expansion without matching selected node')
        selected = self.selected
        try:
            state = self.mcts_store.snapshot(selected['sid'])['state']
            native = super().incorporate(parent_key,proposals)
            additions = []
            for p,edge in zip(proposals,native,strict=True):
                error_text=str(edge.get('error','')).upper()
                if p.truncated or any(word in error_text for word in
                        ('RESOURCE','MEMORY','ALLOCATION','TIMEOUT','TEMPORARY')):
                    continue  # physical/incomplete output is not a failed chemical action
                child,error = self._transition(state,edge)
                eid = planning_edge_id(selected['sid'],child,p,error)
                additions.append({'id':eid,'child':child,'error':error,'count':1,'proposal':asdict(p)})
                self.applied.add((selected['sid'],eid))
            self.mcts_store.add_edges(selected['sid'],additions,
                observation=fingerprint([selected['sid'],selected['batch']]))
            self.rollouts.append(selected['rollout']); return native
        finally:
            selected['claim'].close(); self.selected = None; self.checkpoint()

    def proposal_count(self,context): return 16

    def drain_cached_events(self):
        events,self.cached_events = self.cached_events,[]; return events

    def censor_selected(self,reason):
        if self.selected is None: return
        self.blocked.add(self.selected['sid'])
        if self.selected['sid']==self.mcts_root:
            self.has_pending=False;self.stop_reason=reason
        self.censored_rollouts += 1
        self.selected['rollout']['lease'].close(); self.selected['claim'].close()
        self.selected = None; self.checkpoint()

    def mcts_metrics(self):
        root = self.mcts_store.snapshot(self.mcts_root) if self.mcts_root else {'visits':0,'value_sum':0.}
        return {'algorithm':ALGORITHM,'completed_rollouts':self.completed_rollouts,
                'positive_backups':self.positive_backups,'censored_rollouts':self.censored_rollouts,
                'root_visits':root['visits'],'root_value_sum':root['value_sum'],
                'planning_states_observed':len(self.seen_states),'shared_statistics_root':str(self.mcts_store.root)}

    def checkpoint(self):
        if self.checkpoint_path is None: return
        rollouts = list(self.rollouts) + ([self.selected['rollout']] if self.selected else [])
        durable_json(self.checkpoint_path, {'case_id':self.case_id,'root':self.mcts_root,
            'planning_contract':self.mcts_store.contract,'stop_reason':self.stop_reason,
            'metrics':self.mcts_metrics(),'seen_states':sorted(self.seen_states),'blocked':sorted(self.blocked),
            'pending_rollouts':[{'uid':r['lease'].uid,'sid':r['sid'],'path':r['path']} for r in rollouts],
            'first_solution':self.first_solution_work,
            'search_complete':self.already_in_stock or self.first_solution_work is not None})

    def close(self):
        # Resource shutdown is censoring. Do not invent completed failure trials.
        if self.selected:
            self.selected['claim'].close(); self.selected['rollout']['lease'].close()
            self.selected = None
        for rollout in self.rollouts: rollout['lease'].close()
        self.rollouts.clear()

    def report(self):
        result = super().report(); result['mcts'] = self.mcts_metrics()
        result['search_algorithm'] = ALGORITHM; return result

    def to_dict(self):
        result = super().to_dict(); result['mcts'] = self.mcts_metrics()
        result['mcts_graph'] = {sid:self.mcts_store.snapshot(sid) for sid in sorted(self.seen_states)}
        return result
