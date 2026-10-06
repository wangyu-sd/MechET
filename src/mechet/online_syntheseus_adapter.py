"""On-demand Syntheseus adapter for product-only MechET electron-flow rollouts.

Each planner query starts a fresh reverse-electron-flow episode from that
query's molecule. No reference precursor or cross-reaction state is supplied.
K>1 uses K independent, deterministically seeded stochastic episodes; K=1
uses the frozen greedy protocol. Only strictly terminal executor outcomes are
offered as planner edges, with replayable action certificates attached.
"""
from __future__ import annotations

from copy import copy
import hashlib
from typing import Any, Sequence

from .endpoints import split_precursor_endpoints
from .proof_program import sides_equal
from .syntheseus_adapter import (
    MechETBackwardReactionModel, PoolCandidate, canonical_unmapped,
    normalized_probabilities,
)

try:
    from syntheseus import BackwardReactionModel
except ImportError:  # optional planning extra
    BackwardReactionModel = object  # type: ignore[assignment,misc]


def _seed(base_seed: int, target: str, index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{canonical_unmapped(target)}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _candidate_from_search(target: str, searched: dict[str, Any], args: Any) -> PoolCandidate | None:
    from scripts.run_natural_language_value_search import visible

    top = searched["top"]
    if not top.terminal or not top.actions or top.actions[-1]["name"] != "finish_trace":
        return None
    if not any(record["name"] == "apply_electron_flow" for record in top.actions):
        return None
    endpoints = split_precursor_endpoints(top.state, searched["target_mapped"])
    if not endpoints.structural:
        return None
    structural = visible(endpoints.structural)
    if sides_equal(structural, target, ignore_maps=True):
        return None
    trace_certificate = {
        "protocol": "mechet_nl_reverse_et_v2",
        "observation": "compact_history" if args.compact_history else "current_state",
        "product": searched["target"],
        "actions": [
            {"name": str(record["name"]), "arguments": dict(record["arguments"])}
            for record in top.actions
        ],
        "max_imports": int(args.max_imports),
        "reject_target_retained_finish": bool(args.reject_target_retained_finish),
    }
    return PoolCandidate(
        target=target,
        precursor=structural,
        score=float(top.policy_score),
        metadata={
            "trace_certificate": trace_certificate,
            "full_executor_precursor": visible(top.state),
            "strict_trace_execute_ok": True,
        },
    )


class OnlineMechETBackwardReactionModel(BackwardReactionModel):  # type: ignore[misc]
    """Query one frozen MechET policy instead of a precomputed candidate pool."""

    def __init__(
        self, runtime: Any, rollout_args: Any, *, seed: int = 17,
        max_candidates: int = 10,
    ) -> None:
        if BackwardReactionModel is object:
            raise ImportError("install mechet[planning] to use online Syntheseus search")
        if not bool(getattr(rollout_args, "product_only_remap", False)):
            raise ValueError("planning requires product-only private remapping")
        if not bool(getattr(rollout_args, "matched_v2", False)):
            raise ValueError("planning requires the SFT-aligned v2 tool-call prefix")
        if not bool(getattr(rollout_args, "reject_target_retained_finish", False)):
            raise ValueError("planning must reject unchanged-target finish")
        if int(rollout_args.max_decisions) != 40 or int(rollout_args.max_imports) != 32:
            raise ValueError("planning requires frozen 40-decision/32-import budgets")
        if any(int(getattr(rollout_args, key)) != 1 for key in (
            "branching", "early_beam", "late_beam",
        )):
            raise ValueError("each planning episode must be a single policy trajectory")
        if abs(float(getattr(rollout_args, "value_weight", 0))) > 1e-12:
            raise ValueError("matched planning forbids a value critic")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self.runtime = runtime
        self.rollout_args = copy(rollout_args)
        self.seed = int(seed)
        self.max_candidates = int(max_candidates)
        self.episode_count = 0
        self.policy_decision_count = 0
        self.rejected_decision_count = 0
        self.nonterminal_count = 0
        self.unadmitted_episode_count = 0
        super().__init__(default_num_results=max_candidates, use_cache=True)

    def reset(self, use_cache=None) -> None:
        super().reset(use_cache=use_cache)
        self.episode_count = 0
        self.policy_decision_count = 0
        self.rejected_decision_count = 0
        self.nonterminal_count = 0
        self.unadmitted_episode_count = 0

    def _get_reactions(self, inputs: list[Any], num_results: int) -> list[Sequence[Any]]:
        from scripts.run_natural_language_value_search import search_unlabeled

        if num_results < 1 or num_results > self.max_candidates:
            raise ValueError("requested planning candidate budget exceeds frozen maximum")
        outputs: list[Sequence[Any]] = []
        for product in inputs:
            candidates: dict[str, PoolCandidate] = {}
            for index in range(num_results):
                args = copy(self.rollout_args)
                args.planning_sample = num_results > 1
                self.runtime.torch.manual_seed(_seed(self.seed, product.smiles, index))
                searched = search_unlabeled(self.runtime, product.smiles, args)
                self.episode_count += 1
                self.policy_decision_count += len(searched["attempts"])
                self.rejected_decision_count += sum(
                    not attempt["accepted"] for attempt in searched["attempts"]
                )
                candidate = _candidate_from_search(product.smiles, searched, args)
                if candidate is None:
                    self.unadmitted_episode_count += 1
                    self.nonterminal_count += not searched["top"].terminal
                    continue
                key = canonical_unmapped(candidate.precursor)
                current = candidates.get(key)
                if current is None or candidate.score > current.score:
                    candidates[key] = candidate
            ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
            probabilities = normalized_probabilities(ranked)
            outputs.append([
                MechETBackwardReactionModel._reaction(product, candidate, probability)
                for candidate, probability in zip(ranked, probabilities, strict=True)
            ])
        return outputs
