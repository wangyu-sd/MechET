"""One logical electron-event search over within-reaction states and route obligations.

The chemical decisions are executor-grounded IMPORT preparation and coupled FLOW
events. FINISH is a *boundary observation*, never a PUCT child/action. An
executed FINISH creates AND precursor obligations in the same search graph.
No gold precursor, reference mechanism, value network or RSI is consulted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .dynamic_electron_tree import ElectronTree, Proposal
from .pr96_evaluation import messages_for, parsed
from .reverse_route_network import RouteNetwork
from .shared_electron_search import SearchJournal, durable_json, read_record


@dataclass
class ElectronicFrontier:
    """Progressively widen *chemical successors*, not tool strings.

    Each distinct policy history remains separate even when two paths execute
    to the same unmapped chemical state. A FINISH marker contributes no child
    to UCT or the widening count. An IMPORT must still be sampled and charged.
    """
    c_puct: float = 1.25
    widen_k: float = 1.5
    widen_alpha: float = 0.5
    visits: dict[str, int] = field(default_factory=dict)
    rewards: dict[str, float] = field(default_factory=dict)
    parents: dict[str, str] = field(default_factory=dict)
    credited_routes: set[str] = field(default_factory=set)

    def __post_init__(self):
        if self.c_puct <= 0 or self.widen_k <= 0 or not 0 < self.widen_alpha < 1:
            raise ValueError("PUCT_AND_WIDENING_PARAMETERS_MUST_BE_POSITIVE")

    @staticmethod
    def _action(edge: dict) -> str:
        return edge.get("action", {}).get("name", "")

    @classmethod
    def successors(cls, tree: ElectronTree, key: str) -> set[tuple[str, str]]:
        """Different IMPORT states and FLOW states are valid distinct choices."""
        return {(cls._action(e), e["child_chemical"])
                for e in tree.edges
                if e.get("parent_context") == key
                and cls._action(e) in {"apply_electron_flow", "import_fragments"}
                and e.get("child_chemical")}

    @classmethod
    def _incoming_score(cls, tree: ElectronTree, key: str) -> float:
        for edge in reversed(tree.edges):
            if edge.get("child_context") == key and cls._action(edge) != "finish_trace":
                a = edge.get("action", {})
                return float(a.get("logprob", -100.0)) / max(1, int(a.get("tokens", 1)))
        return 0.0

    def _prior(self, tree: ElectronTree, key: str) -> float:
        parent = self.parents.get(key)
        if parent is None:
            return 1.0
        siblings = {e["child_context"] for e in tree.edges
                    if e.get("parent_context") == parent and e.get("child_context")
                    and self._action(e) != "finish_trace"}
        if not siblings:
            return 1.0
        values = {s: self._incoming_score(tree, s) for s in siblings}
        peak = max(values.values())
        probs = {s: math.exp(max(-30.0, values[s]-peak)) for s in siblings}
        return probs.get(key, 0.0) / max(sum(probs.values()), 1e-9)

    def _ancestors(self, key: str) -> list[str]:
        chain, seen = [], set()
        while key and key not in seen:
            seen.add(key)
            chain.append(key)
            key = self.parents.get(key, "")
        return chain

    def observe(self, parent: str, edges: list[dict]) -> None:
        """Record one *actor* evaluation and the actual state transitions."""
        for edge in edges:
            child = edge.get("child_context")
            if child and self._action(edge) != "finish_trace":
                self.parents.setdefault(child, parent)
        for ancestor in self._ancestors(parent):
            self.visits[ancestor] = self.visits.get(ancestor, 0) + 1

    def select(self, trees: dict[str, ElectronTree]) -> tuple[str, Any] | None:
        """PUCT-style selection across all electronic states, no outer reaction action."""
        candidates = []
        for molecule, tree in trees.items():
            for key, ctx in tree.contexts.items():
                if ctx.node.terminal or ctx.resource_closed or not ctx.stats.needs_more_sampling:
                    continue
                n = self.visits.get(key, 0)
                maximum = max(1, math.ceil(self.widen_k * (n + 1) ** self.widen_alpha))
                if len(self.successors(tree, key)) >= maximum:
                    continue
                parent = self.parents.get(key)
                parent_n = self.visits.get(parent, n if parent is None else 0)
                prior = self._prior(tree, key)
                q = self.rewards.get(key, 0.0) / max(1, n)
                u = self.c_puct * prior * math.sqrt(parent_n + 1) / (1+n)
                # A small sampling-cost penalty prevents repeatedly probing a
                # root whose proposals all collapse to the same successor.
                priority = q + u - 0.02 * math.log1p(ctx.stats.batches)
                candidates.append((priority, ctx.node.policy_score, -len(ctx.node.actions),
                                   molecule, key, ctx))
        if not candidates:
            return None
        _, _, _, molecule, key, ctx = max(candidates,
                                          key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        return molecule, ctx

    def credit_complete_route(self, certificate: dict) -> bool:
        """Sparse *positive* backup only after all AND obligations reach stock.

        A budget-exhausted branch is censored, not assigned a false chemical
        negative. Credit each electron history on a confirmed stock-closed route.
        """
        if not certificate or not certificate.get("complete_reaction_edges"):
            return False
        identity = certificate.get("route_key")
        if not identity or identity in self.credited_routes:
            return False
        self.credited_routes.add(identity)
        for reaction in certificate["complete_reaction_edges"]:
            terminal = reaction["program"]["terminal_context"]
            for key in self._ancestors(terminal):
                # FINISH has no branch in parents, so the terminal's parent
                # must be recovered from the verified reaction's last event.
                self.rewards[key] = self.rewards.get(key, 0.0) + 1.0
        return True

    def state(self) -> dict:
        return {"visits": self.visits, "rewards": self.rewards,
                "parents": self.parents, "credited_routes": sorted(self.credited_routes),
                "c_puct": self.c_puct, "widen_k": self.widen_k,
                "widen_alpha": self.widen_alpha}

    @classmethod
    def load(cls, record: dict):
        obj = cls(c_puct=record["c_puct"], widen_k=record["widen_k"],
                  widen_alpha=record["widen_alpha"])
        obj.visits = dict(record["visits"])
        obj.rewards = dict(record["rewards"])
        obj.parents = dict(record["parents"])
        obj.credited_routes = set(record["credited_routes"])
        return obj


def proposal_from_actor(actor, context, *, budget: int, seed: int) -> tuple[Proposal, dict]:
    """Actual Stage-II generation; gold references never appear in messages."""
    output = actor.generate(messages_for(context.node, True),
                            max_new_tokens=budget, seed=seed)
    name, arguments, error = parsed(output)
    proposal = Proposal(name=name, arguments=arguments, raw=output.get("raw", ""),
                        logprob=float(output.get("logprob", 0.0)),
                        tokens=len(output.get("generated_token_ids", [])),
                        error=error, truncated=bool(output.get("truncated", False)),
                        token_ids=output.get("generated_token_ids", []))
    if proposal.tokens <= 0:
        raise ValueError("ZERO_GENERATED_TOKENS")
    return proposal, output


def deterministic_seed(seed: int, case_id: str, context: str, count: int) -> int:
    payload = json.dumps([seed, case_id, context, count], ensure_ascii=False)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


class UnifiedRouteSearch:
    """One event decision frontier; RouteNetwork only maintains AND obligations.

    The RouteNetwork.next_context heuristic is intentionally not invoked.
    Reaction boundaries are not separate planning actions; only executor-derived
    precursor obligations are created after an accepted FINISH tool marker.
    """

    def __init__(self, *, case_id: str, product: str, stock: set[str],
                 stock_sha: str, actor, actor_sha: str, output: Path,
                 max_calls: int=128, max_tokens: int=12000, max_new_tokens: int=512,
                 seed: int=17, c_puct: float=1.25, widen_k: float=1.5,
                 widen_alpha: float=0.5, resume: bool=False):
        if min(max_calls, max_tokens, max_new_tokens) <= 0:
            raise ValueError("SEARCH_BUDGET_MUST_BE_POSITIVE")
        self.case_id, self.actor, self.actor_sha = case_id, actor, actor_sha
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.max_new_tokens, self.seed = max_new_tokens, seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        contract = {"actor_sha": actor_sha, "stock_sha": stock_sha,
                    "search": "unified_event_puct_v1", "seed": seed,
                    "max_calls": max_calls, "max_tokens": max_tokens,
                    "max_new_tokens": max_new_tokens,
                    "c_puct": c_puct, "widen_k": widen_k, "widen_alpha": widen_alpha}
        self.journal = SearchJournal(self.output / "journal", case_id, contract)
        self.state_path = self.output / "search_state.json"
        if self.journal.sequence:
            if not resume:
                raise ValueError("EXISTING_JOURNAL_REQUIRES_RESUME")
            record = read_record(self.state_path)
            if record["journal_sequence"] != self.journal.sequence:
                raise ValueError("MCTS_STATS_JOURNAL_MISMATCH_RECONCILIATION_REQUIRED")
            self.frontier = ElectronicFrontier.load(record["frontier"])
            self.counters = dict(record["counters"])
            self.network = self.journal.restore(stock=stock)
            from .dynamic_electron_tree import chemical_key
            if self.network.target != chemical_key(product):
                raise ValueError("ROUTE_TARGET_MISMATCH_ON_RESUME")
        else:
            self.network = RouteNetwork(case_id, product, stock, mode="tree")
            self.frontier = ElectronicFrontier(c_puct, widen_k, widen_alpha)
            self.counters = {"calls":0, "tokens":0, "electron_events":0,
                             "import_preparations":0, "finish_markers":0,
                             "rejected_calls":0, "successful_route_backups":0}
            self.journal.initialize(self.network)
            self._persist()

    def _persist(self):
        durable_json(self.state_path, {"journal_sequence": self.journal.sequence,
                    "frontier": self.frontier.state(), "counters": self.counters})

    def run(self, *, stop_on_first: bool=False) -> dict:
        reason = "BUDGET_EXHAUSTED"
        while self.counters["calls"] < self.max_calls and self.counters["tokens"] < self.max_tokens:
            if self.network.already_in_stock:
                reason = "ALREADY_IN_STOCK"
                break
            chosen = self.frontier.select(self.network.trees)
            if chosen is None:
                reason = "FRONTIER_EXHAUSTED"
                break
            molecule, ctx = chosen
            room = min(self.max_new_tokens, self.max_tokens-self.counters["tokens"])
            seed = deterministic_seed(self.seed, self.case_id, ctx.key, ctx.stats.draws)
            proposal, output = proposal_from_actor(self.actor, ctx, budget=room, seed=seed)
            edges = self.network.incorporate(ctx.key, [proposal])
            self.frontier.observe(ctx.key, edges)
            self.counters["calls"] += 1
            self.counters["tokens"] += proposal.tokens
            accepted = any(e.get("child_context") for e in edges)
            if not accepted:
                self.counters["rejected_calls"] += 1
            elif proposal.name == "finish_trace":
                self.counters["finish_markers"] += 1
                # No separate MCTS branch/visit is created for FINISH.
                # Connect the verified terminal to its last real event.
                for edge in edges:
                    terminal = edge.get("child_context")
                    if terminal:
                        self.frontier.parents.setdefault(terminal, ctx.key)
            elif proposal.name == "apply_electron_flow":
                self.counters["electron_events"] += 1
            elif proposal.name == "import_fragments":
                self.counters["import_preparations"] += 1
            certificate = self.network._certificate()
            if certificate and self.frontier.credit_complete_route(certificate):
                self.counters["successful_route_backups"] += 1
            self.journal.record(self.network, ctx.key, edges)
            self._persist()
            with (self.output / "calls.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_id":self.case_id, "index":self.counters["calls"],
                    "molecule":molecule, "context":ctx.key, "action":proposal.name,
                    "accepted":accepted, "boundary_only":proposal.name=="finish_trace",
                    "generated_tokens":proposal.tokens, "executor_edges":edges,
                    "root_stock_solved":certificate is not None},ensure_ascii=False)+"\n")
            if stop_on_first and certificate is not None:
                reason = "STOCK_SOLVED"
                break
        self.network.stop_reason = reason
        report = self.network.report()
        report.update({"search_kind":"unified_event_puct_v1", "boundary_semantics":"FINISH_IS_MARKER",
                       "actor_sha":self.actor_sha, "real_executor":True,
                       "counters":self.counters, "budget":{
                           "max_calls":self.max_calls, "max_tokens":self.max_tokens},
                       "first_solution":self.network.first_solution_work,
                       "puct_statistics":self.frontier.state(),
                       "model_output_is_gold_free":True})
        durable_json(self.output / "report.json", report)
        return report


class OneStepElectronicSearch:
    """Reuse the same event-frontier logic, without inventing route-stock labels."""

    def __init__(self, *, case_id: str, product: str, actor, max_calls: int,
                 max_tokens: int, max_new_tokens: int=512, seed: int=17):
        self.case_id, self.actor, self.seed = case_id, actor, seed
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.max_new_tokens = max_new_tokens
        self.tree = ElectronTree(case_id, product)
        self.frontier = ElectronicFrontier()

    def run(self) -> dict:
        calls, tokens, finish = 0,0,0
        while calls < self.max_calls and tokens < self.max_tokens:
            choice = self.frontier.select({"single": self.tree})
            if choice is None:
                break
            _, ctx = choice
            p, _ = proposal_from_actor(self.actor, ctx,
                 budget=min(self.max_new_tokens,self.max_tokens-tokens),
                 seed=deterministic_seed(self.seed,self.case_id,ctx.key,ctx.stats.draws))
            edges = self.tree.incorporate(ctx.key,[p])
            self.frontier.observe(ctx.key,edges)
            calls += 1
            tokens += p.tokens
            if p.name == "finish_trace":
                for edge in edges:
                    if edge.get("child_context"):
                        self.frontier.parents.setdefault(edge["child_context"],ctx.key)
                        finish += 1
        from .endpoints import split_precursor_endpoints
        candidates={}
        for ctx in self.tree.terminals():
            parts=split_precursor_endpoints(ctx.node.state,
                                           self.tree.contexts[self.tree.root_key].node.state)
            if parts.structural:
                from .dynamic_electron_tree import chemical_key
                structure=chemical_key(parts.structural)
                record={"precursor":structure,"score":ctx.node.policy_score,
                        "electron_actions":sum(x["name"]=="apply_electron_flow"
                                               for x in ctx.node.actions),
                        "context":ctx.key}
                if structure not in candidates or record["score"]>candidates[structure]["score"]:
                    candidates[structure]=record
        ranked=sorted(candidates.values(),key=lambda x:(-x["score"],x["precursor"]))
        return {"id":self.case_id,"target":self.tree.target,"candidates":ranked,
                "model_calls":calls,"generated_tokens":tokens,
                "finish_markers":finish,"unique_structural_precursors":len(ranked),
                "gold_not_used_for_generation":True}
