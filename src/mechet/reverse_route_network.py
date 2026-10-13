"""Recursive reverse-electron AND/OR network over a frozen purchasable stock.

No reference routes, target precursors or learned critic enter this class.
Executable reaction endpoints are recursively pursued by the SAME policy.
Stock closure is graph reachability, not an experimental chemistry guarantee.
"""
from __future__ import annotations
from collections import Counter
import math
import time
from .dynamic_electron_tree import ElectronTree, SearchContext, chemical_key, digest
from .endpoints import split_precursor_endpoints


def stock_closure(stock_nodes: set[str], edges: list[dict]) -> dict[str,dict]:
    """Least fixed point, with positive-cost acyclic best certificates.

    Each hyperedge requires ALL precursors, including repeated multiplicities.
    Positive model negative-loglikelihood costs prevent a chemical cycle from
    replacing a stock-connected best certificate. No synthesis-depth limit.
    """
    closure={node:{'cost':0.0,'tokens':0,'edge':None} for node in stock_nodes}
    changed=True
    while changed:
        changed=False
        for edge in edges:
            children=edge['precursors']
            if not children or not all(child in closure for child in children):continue
            if edge['product'] in stock_nodes:continue
            cost=max(1e-9,float(edge['cost']))+sum(closure[child]['cost'] for child in children)
            previous=closure.get(edge['product'])
            if previous is None or cost<previous['cost']-1e-9:
                closure[edge['product']]={'cost':cost,'tokens':edge['tokens']+sum(closure[child]['tokens'] for child in children),
                                         'edge':edge['id']}
                changed=True
    return closure


class RouteNetwork:
    def __init__(self, case_id: str, product: str, stock: set[str], *, mode: str):
        if mode not in {'tree','greedy'}:raise ValueError('invalid route policy mode')
        self.case_id=case_id
        self.target=chemical_key(product)
        self.stock=stock  # shared immutable stock set, not copied per target
        self.mode=mode
        self.already_in_stock=self.target in stock
        self.trees:dict[str,ElectronTree]={}
        self.contexts:dict[str,SearchContext]={}
        self.owners:dict[str,str]={}
        self.forced:dict[str,str]={}
        self.reaction_edges:list[dict]=[]
        self.edge_index:dict[str,dict]={}
        self.rejected_route_terminals:list[dict]=[]
        self.failed_obligations:list[dict]=[]
        self.first_solution_work=None
        self.stop_reason=''
        self.started=time.monotonic()
        self._closure_cache=None
        if not self.already_in_stock:self._add_obligation(self.target)

    @property
    def work(self):
        work=Counter()
        for tree in self.trees.values():work.update(tree.work)
        for key in ['sampled_tokens','raw_candidates','executor_calls','execution_cache_hits','node_expansions']:
            work.setdefault(key,0)
        work['outer_precursor_expansions']=len(self.trees)+len(self.failed_obligations)
        work['unique_structural_precursor_obligations']=len(self.trees)
        work['unique_chemical_successors']=sum(max(0,len(tree.chemical_nodes)-1) for tree in self.trees.values())
        return dict(work)

    def _add_obligation(self,molecule):
        if molecule in self.stock or molecule in self.trees:return
        try:
            tree=ElectronTree(f'{self.case_id}:{digest(molecule)[:16]}',molecule)
        except Exception as exc:
            self.failed_obligations.append({'molecule':molecule,'error':str(exc)})
            return
        self.trees[molecule]=tree
        self.forced[molecule]=tree.root_key
        self._index_contexts(molecule)

    def _index_contexts(self,molecule):
        tree=self.trees[molecule]
        for key,context in tree.contexts.items():
            self.contexts[key]=context
            self.owners[key]=molecule

    def next_context(self):
        candidates=[]
        for molecule,tree in self.trees.items():
            if self.mode=='greedy':
                context=tree.contexts[self.forced[molecule]]
                if context.node.terminal or context.resource_closed:continue
            else:context=tree.next_context()
            if context is not None:
                score=(context.node.policy_score+math.log(max(1e-9,context.stats.unseen_upper))
                       -math.log1p(tree.work['node_expansions']))
                candidates.append((score,context.key,context))
        return max(candidates,key=lambda x:(x[0],x[1]))[2] if candidates else None

    def incorporate(self,parent_key,proposals):
        molecule=self.owners[parent_key]
        tree=self.trees[molecule]
        edges=tree.incorporate(parent_key,proposals)
        self._index_contexts(molecule)
        for edge in edges:
            key=edge.get('child_context')
            if key and tree.contexts[key].node.terminal:
                self._register_terminal(molecule,tree.contexts[key])
        if self.mode=='greedy':
            accepted=[edge for edge in edges if edge.get('child_context')]
            if accepted:self.forced[molecule]=accepted[0]['child_context']
            elif not any(p.truncated for p in proposals):self.contexts[parent_key].resource_closed='GREEDY_INVALID_ACTION'
        return edges

    def _register_terminal(self,molecule,context):
        tree=self.trees[molecule]
        parts=split_precursor_endpoints(context.node.state,tree.contexts[tree.root_key].node.state)
        structural=chemical_key(parts.structural) if parts.structural else ''
        # Keep all reaction terminals in their electronic trees, even when
        # they cannot establish nontrivial route progress.
        if (not any(action['name']=='apply_electron_flow' for action in context.node.actions)
                or not structural or structural==molecule):
            self.rejected_route_terminals.append({'molecule':molecule,'context':context.key,
                                                 'reason':'NONPRODUCTIVE_REACTION'})
            return
        # Preserve a purchasable whole salt if the COMPLETE structural mixture
        # is itself an exact stock entry. Otherwise every connected component
        # is an obligation; individual ions are not invented as stock entries.
        precursors=[structural] if structural in self.stock else sorted(structural.split('.'))
        identifier=digest([molecule,precursors])
        cost=max(1e-9,-context.node.logprob)
        existing=self.edge_index.get(identifier)
        program={'terminal_context':context.key,'full_terminal_state':context.node.state,
                 'structural_terminal_mapped':parts.structural,'auxiliary_fragments':list(parts.auxiliary),
                 'actions':context.node.actions,'logprob':context.node.logprob,'tokens':context.node.tokens,
                 'mean_nll':-context.node.policy_score}
        if existing:
            existing['terminal_contexts'].add(context.key)
            if cost<existing['cost']:
                existing.update(cost=cost,tokens=context.node.tokens,program=program)
                self._closure_cache=None
        else:
            edge={'id':identifier,'product':molecule,'precursors':precursors,'cost':cost,
                  'tokens':context.node.tokens,'program':program,'auxiliary_fragments':list(parts.auxiliary),
                  'terminal_contexts':{context.key}}
            self.edge_index[identifier]=edge
            self.reaction_edges.append(edge)
            self._closure_cache=None
            for precursor in precursors:self._add_obligation(precursor)
        if self.first_solution_work is None and self.target in self._closure():
            self.first_solution_work={'work':self.work,'wall_seconds':time.monotonic()-self.started}

    def _closure(self):
        if self._closure_cache is None:
            # Never allocate a dictionary for the entire 300k stock per target.
            discovered={self.target}|{p for edge in self.reaction_edges for p in edge['precursors']}
            self._closure_cache=stock_closure(self.stock&discovered,self.reaction_edges)
        return self._closure_cache

    def _certificate(self, root_edge=None):
        closure=self._closure()
        if self.target not in closure:return None
        pending=[(self.target,frozenset(),root_edge)]
        edges=[];leaves=[]
        while pending:
            molecule,ancestors,forced=pending.pop()
            if molecule in ancestors:return None
            if molecule in self.stock:
                leaves.append(molecule);continue
            if molecule not in closure:return None
            edge=forced or self.edge_index[closure[molecule]['edge']]
            edges.append(edge)
            pending.extend((child,ancestors|{molecule},None) for child in edge['precursors'])
        key=digest(sorted({digest([edge['product'],edge['precursors']]) for edge in edges}))
        return {'route_key':key,'complete_reaction_edges':[self._serial_edge(edge) for edge in edges],
                'stock_leaves':sorted(leaves),'negative_loglikelihood':sum(edge['cost'] for edge in edges),
                'action_tokens':sum(edge['tokens'] for edge in edges)}

    @staticmethod
    def _serial_edge(edge):
        return {**edge,'terminal_contexts':sorted(edge['terminal_contexts'])}

    def report(self):
        best=self._certificate()
        nontrivial=bool(best and best['complete_reaction_edges'] and not self.already_in_stock)
        unique={}
        if nontrivial:
            for edge in self.reaction_edges:
                if edge['product']!=self.target:continue
                candidate=self._certificate(edge)
                if candidate:unique[candidate['route_key']]=candidate
        return {'id':self.case_id,'target':self.target,'already_in_stock':self.already_in_stock,
                'generated_reaction_graph':[{'product':e['product'],'precursors':e['precursors']} for e in self.reaction_edges],
                'stock_solved':nontrivial,'route_key':best['route_key'] if nontrivial else None,
                'complete_reaction_edges':best['complete_reaction_edges'] if nontrivial else [],
                'stock_leaves':best['stock_leaves'] if nontrivial else [],
                'enumerated_stock_connected_routes':list(unique.values()),
                'unique_route_count_lower_bound':len(unique),'route_enumeration_exhaustive':False,
                'first_solution':self.first_solution_work,'work':self.work,'stop_reason':self.stop_reason,
                'wall_seconds':time.monotonic()-self.started,'chemical_plausibility':'not_independently_verified'}

    def to_dict(self):
        return {'case_id':self.case_id,'target':self.target,'mode':self.mode,
                'already_in_stock':self.already_in_stock,'reaction_edges':[self._serial_edge(e) for e in self.reaction_edges],
                'reaction_trees':{key:tree.to_dict() for key,tree in self.trees.items()},
                'failed_obligations':self.failed_obligations,'rejected_route_terminals':self.rejected_route_terminals,
                'work':self.work,'stop_reason':self.stop_reason}
