"""Frozen-policy electronic state graph with adaptive (not fixed-depth) search.

The graph pools equivalent chemistry for display/accounting, but keeps distinct
history/visited-set policy contexts. No gold labels or trained value model are
available to this controller. The caller owns a GLOBAL compute budget.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from scripts.run_natural_language_value_search import (
    Action, Node, compact_history_from_actions, execute, product_only_private_state, visible,
)
from .in_place_grounded_flow import mapped_atom_numbers
from .proof_program import _canonical_unmapped


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def chemical_key(state: str) -> str:
    return '.'.join(sorted(_canonical_unmapped(x) for x in state.split('.')))


def available_host_memory() -> int:
    import psutil
    available = psutil.virtual_memory().available
    for maximum, current in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                             ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            cap = Path(maximum).read_text().strip()
            if cap != 'max':
                available = min(available, max(0, int(cap)-int(Path(current).read_text())))
        except (OSError, ValueError):
            pass
    return available


def context_key(node: Node) -> str:
    # Preserve native Kekule state: a canonical roundtrip can change the next
    # executor transition even when display chemistry is equivalent.
    return digest([node.target, node.state, node.next_map, node.imported,
                   compact_history_from_actions(node.actions).render(), sorted(node.visited)])


def validate_probability_contract(*, temperature: float, top_p: float, repetition_penalty: float) -> None:
    # vLLM0.8.5's sampler computes reported logprobs AFTER warping. At these
    # settings no warp is applied, so cumulative_logprob is genuine model NLL.
    if temperature not in (0, 1) or top_p != 1 or repetition_penalty != 1:
        raise ValueError('unwarped NLL requires temperature1 (or greedy), top_p1 and no penalties')


@dataclass
class NoveltyStats:
    draws: int = 0
    counts: Counter = field(default_factory=Counter)
    batches: int = 0

    def observe(self, key: str | None) -> None:
        self.draws += 1
        if key is not None:
            self.counts[key] += 1

    @property
    def unseen_upper(self) -> float:
        # Good--Turing singleton mass plus a conservative rule-of-three margin.
        # This is a sampling novelty heuristic, NOT a chemical completeness proof.
        singletons = sum(n == 1 for n in self.counts.values())
        return min(1.0, (singletons + 3) / max(1, self.draws))

    @property
    def needs_more_sampling(self) -> bool:
        return self.unseen_upper > .10

    @property
    def next_batch(self) -> int:
        # A request microbatch may be split by the backend; this is not a
        # maximum branch count. Every accepted distinct successor is retained.
        return max(4, int(math.ceil(16 * self.unseen_upper)))


@dataclass
class Proposal:
    name: str
    arguments: dict[str, Any]
    raw: str
    logprob: float
    tokens: int
    error: str = ''
    truncated: bool = False
    token_ids: list[int] = field(default_factory=list)


@dataclass
class SearchContext:
    key: str
    node: Node
    chemical_id: str
    stats: NoveltyStats = field(default_factory=NoveltyStats)
    resource_closed: str = ''
    max_new_tokens: int = 384
    greedy_done: bool = False


class ElectronTree:
    def __init__(self, case_id: str, product: str):
        mapped = product_only_private_state(product)
        target = visible(mapped)
        root = Node(target, mapped, max(mapped_atom_numbers(mapped), default=0)+1, visited={target})
        self.case_id = case_id
        self.target = target
        self.root_key = context_key(root)
        ck = chemical_key(mapped)
        self.contexts = {self.root_key: SearchContext(self.root_key, root, digest(ck))}
        self.chemical_nodes = {digest(ck): {'id': digest(ck), 'state_unmapped': ck, 'private_state': mapped}}
        self.edges: list[dict[str, Any]] = []
        self.execution_cache: dict[str, tuple[Node | None, str]] = {}
        self.work = {'sampled_tokens': 0, 'raw_candidates': 0, 'node_expansions': 0,
                     'executor_calls': 0, 'execution_cache_hits': 0}
        self.stop_reason = ''

    def next_context(self) -> SearchContext | None:
        eligible = [c for c in self.contexts.values() if not c.node.terminal and not c.resource_closed and c.stats.needs_more_sampling]
        if not eligible:
            return None
        # No fixed beam pruning. Every deferred context remains eligible.
        return max(eligible, key=lambda c: (c.node.policy_score + math.log(max(1e-9, c.stats.unseen_upper))
                    - math.log1p(c.stats.batches), -len(c.node.actions), c.key))

    def incorporate(self, parent_key: str, proposals: list[Proposal]) -> list[dict[str, Any]]:
        parent = self.contexts[parent_key]
        parent.stats.batches += 1
        self.work['node_expansions'] += 1
        new_edges = []
        any_truncated = False
        for proposal in proposals:
            self.work['sampled_tokens'] += proposal.tokens
            self.work['raw_candidates'] += 1
            action = Action(proposal.name, proposal.arguments, proposal.raw, proposal.logprob, proposal.tokens)
            error = proposal.error
            child = None
            if proposal.truncated:
                error = 'ACTION_TOKEN_BUDGET_TRUNCATED'
                any_truncated = True
            elif not error:
                # Strip inert natural-language instructions for execution-cache
                # lookup, not for saved model output or probabilistic scoring.
                def executable_payload(x):
                    if isinstance(x, dict): return {k: executable_payload(v) for k,v in x.items() if k != 'instruction'}
                    if isinstance(x, list): return [executable_payload(v) for v in x]
                    return x
                cache_key = digest([parent_key, action.name, executable_payload(action.arguments)])
                cached = self.execution_cache.get(cache_key)
                if cached is None:
                    # No arbitrary max-imports or chemical path-depth cap. Huge
                    # allocations are rejected as RESOURCE failures before execution.
                    if action.name == 'import_fragments':
                        try:
                            # String length upper-bounds atom count and accounts
                            # for long fragments before allocating copies. Room
                            # covers native molecules, matrix/state serialization.
                            allocation = sum(max(0, int(x.get('count', 0))) *
                                max(1, len(str(x.get('smiles', '')))) * 16384
                                for x in (action.arguments.get('fragments') or []) if isinstance(x, dict))
                        except (ValueError, TypeError, OverflowError):
                            allocation = 0  # authoritative executor rejects malformed schemas
                        available_bytes = available_host_memory()
                        # Conservative host-memory allowance shared among GPU
                        # drivers and concurrent executor threads, not chemistry.
                        if allocation > available_bytes // 128:
                            error = 'STATE_ALLOCATION_RESOURCE_FAILURE'
                    if not error:
                        child, error = execute(parent.node, action, max_imports=math.inf)
                        self.work['executor_calls'] += 1
                        self.execution_cache[cache_key] = (child, error)
                else:
                    self.work['execution_cache_hits'] += 1
                    cached_child, error = cached
                    if cached_child is not None:
                        record = dict(cached_child.actions[-1])
                        record['arguments'] = action.arguments
                        child = Node(target=cached_child.target, state=cached_child.state, next_map=cached_child.next_map,
                                     actions=parent.node.actions + [record], visited=set(cached_child.visited),
                                     imported=dict(cached_child.imported), logprob=parent.node.logprob+action.logprob,
                                     tokens=parent.node.tokens+action.tokens, terminal=cached_child.terminal)
            edge = {'id': f'E{len(self.edges)}', 'parent_context': parent_key,
                    'parent_chemical': parent.chemical_id, 'action': asdict(proposal),
                    'status': 'rejected', 'error': error, 'decision_depth': len(parent.node.actions)+1}
            if child is not None:
                child_key = context_key(child)
                chem = chemical_key(child.state)
                chemical_id = digest(chem)
                self.chemical_nodes.setdefault(chemical_id, {'id': chemical_id, 'state_unmapped': chem,
                                                             'private_state': child.state})
                existed = child_key in self.contexts
                if not existed:
                    self.contexts[child_key] = SearchContext(child_key, child, chemical_id)
                elif child.policy_score > self.contexts[child_key].node.policy_score:
                    self.contexts[child_key].node = child
                edge.update(child_context=child_key, child_chemical=chemical_id,
                            status='duplicate_successor' if existed else ('terminal' if child.terminal else 'accepted'),
                            terminal=child.terminal)
                parent.stats.observe(child_key)
            else:
                # Truncation is incomplete sampling, not evidence of low chemical
                # support. Retry with more per-call token room when possible.
                if not proposal.truncated:
                    parent.stats.observe(None)
            self.edges.append(edge)
            new_edges.append(edge)
        if any_truncated:
            parent.max_new_tokens *= 2
        return new_edges

    def terminals(self) -> list[SearchContext]:
        # Pool only AFTER structural endpoint extraction. Full-mixture equality
        # alone does not preserve product-atom ancestry.
        return sorted((c for c in self.contexts.values() if c.node.terminal),
                      key=lambda c: c.node.policy_score, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        return {'case_id': self.case_id, 'target': self.target, 'root_context': self.root_key,
                'chemical_nodes': list(self.chemical_nodes.values()), 'edges': self.edges,
                'contexts': [{'id': c.key, 'chemical_id': c.chemical_id, 'state': c.node.state,
                              'actions': c.node.actions, 'terminal': c.node.terminal,
                              'mean_nll': -c.node.policy_score, 'depth': len(c.node.actions),
                              'sampling_draws': c.stats.draws, 'unseen_support_heuristic': c.stats.unseen_upper,
                              'resource_closed': c.resource_closed} for c in self.contexts.values()],
                'terminal_contexts': [c.key for c in self.terminals()],
                'work': dict(self.work), 'stop_reason': self.stop_reason}
