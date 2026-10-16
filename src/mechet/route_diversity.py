"""Exhaustive acyclic AND-route DAG enumeration in an already saved graph.

One production choice per distinct nonstock molecule; repeated precursor
occurrences retain stoichiometry but reuse that choice. Electron-program
alternatives for the same product/precursor multiset are NOT separate outer
routes. No new chemistry or fixed depth/width/count stopping ceiling.
Enumeration can be exponential: streaming avoids storing every route, not the
inherent combinatorial cost. Unfinished enumeration must never be called exact.
"""
from __future__ import annotations
from collections import defaultdict
from .dynamic_electron_tree import digest
from .reverse_route_network import stock_closure


def _new_cycle(chosen,product,stock):
    pending=list(chosen[product]['precursors']);seen=set()
    while pending:
        node=pending.pop()
        if node in stock:continue
        if node==product:return True
        if node in seen:continue
        seen.add(node)
        if node in chosen:pending.extend(chosen[node]['precursors'])
    return False


def iter_complete_routes(target,stock,edges):
    """Yield every unique stock-connected production-choice DAG, no recursion."""
    if target in stock:return
    identities={};unique={}
    for edge in edges:
        if not edge['precursors']:continue
        key=(edge['product'],tuple(sorted(edge['precursors'])))
        previous=identities.setdefault(edge['id'],key)
        if previous!=key:raise ValueError('saved reaction ID has conflicting chemistry')
        if key not in unique or edge['id']<unique[key]['id']:unique[key]=edge
    edges=list(unique.values())
    nodes={target}|{e['product'] for e in edges}|{p for e in edges for p in e['precursors']}
    stock=stock&nodes
    supported=stock_closure(stock,edges)
    if target not in supported:return
    options=defaultdict(list)
    for edge in edges:
        if all(child in supported for child in edge['precursors']):options[edge['product']].append(edge)
    for choices in options.values():choices.sort(key=lambda edge:edge['id'])
    pending=[({target},{})]
    while pending:
        obligations,chosen=pending.pop()
        if not obligations:
            selected=list(chosen.values())
            route_key=digest(sorted(digest([e['product'],sorted(e['precursors'])]) for e in selected))
            yield {'route_key':route_key,'reaction_edge_ids':[e['id'] for e in selected],
                   'complete_reaction_edges':selected,
                   'definition':'one_production_choice_per_distinct_nonstock_molecule_DAG_v1'}
            continue
        product=min(obligations)
        for edge in reversed(options.get(product,[])):
            next_chosen={**chosen,product:edge}
            if _new_cycle(next_chosen,product,stock):continue
            children={child for child in edge['precursors'] if child not in stock and child not in next_chosen}
            pending.append(((obligations-{product})|children,next_chosen))
