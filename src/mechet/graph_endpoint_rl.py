"""Verified-group reward and policy-gradient surrogate for graph-only endpoint RL.

Critically, an all-negative group never makes its least bad member a positive
example. Reference replay is used only as a supervised fallback on training
states; it is never supplied to product-start inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class VerifiedBranch:
    """Executor-observed branch; absence of a verified outcome is not impossibility."""
    accepted: bool
    exact_terminal: bool = False
    reference_successor: bool = False

    @property
    def reward(self) -> float:
        if self.exact_terminal:
            if not self.accepted:
                raise ValueError("an exact terminal must be executed")
            return 1.0
        if self.reference_successor:
            if not self.accepted:
                raise ValueError("a reference successor must be executed")
            return 0.20
        return 0.0 if self.accepted else -0.20

    @property
    def verified_positive(self) -> bool:
        return self.exact_terminal or self.reference_successor


def success_gated_advantages(branches: Sequence[VerifiedBranch]) -> tuple[float, ...]:
    """Return strictly signed sibling advantages, or zero for unsupported groups.

    Endpoint winners receive greater credit than locally matched reference
    successors. A wrong/unknown branch is not elevated when all actions fail.
    Groups with no contrast receive no policy-gradient update.
    """
    if not branches:
        raise ValueError("empty sibling group")
    valid = [b.verified_positive for b in branches]
    if not any(valid) or all(valid):
        return (0.0,) * len(branches)
    positive_total = sum(b.reward for b in branches if b.verified_positive)
    negatives = sum(not ok for ok in valid)
    return tuple(
        b.reward / positive_total if b.verified_positive else -1.0 / negatives
        for b in branches
    )


def graph_policy_gradient_loss(
    candidate_nlls, advantages: Sequence[float], reference_nll, *,
    reference_replay_weight: float = 0.10,
):
    """Signed group policy gradient + training-only verified reference replay.

    candidate_nlls come from CRN-JEPA's differentiable source/sink/commit
    decoder under teacher forcing, computed on sampled graph candidates.
    This is a truncated-slate policy-gradient SURROGATE, not unbiased GRPO/PPO:
    CRN-JEPA's bounded candidate beam and mean-per-decision NLL differ from
    exact unconstrained trajectory log-probabilities.
    """
    import torch

    if len(candidate_nlls) != len(advantages) or not candidate_nlls:
        raise ValueError("candidate losses/advantages mismatch")
    if not 0.0 <= reference_replay_weight <= 1.0:
        raise ValueError("invalid reference replay coefficient")
    if any(not -1.000001 <= float(a) <= 1.000001 for a in advantages):
        raise ValueError("advantages exceed bounded group contract")
    policy_term = sum(
        nll * float(weight) for nll, weight in zip(candidate_nlls, advantages, strict=True)
    ) / len(candidate_nlls)
    if reference_nll is not None:
        policy_term = policy_term + reference_replay_weight * reference_nll
    if not torch.isfinite(policy_term).all():
        raise FloatingPointError("nonfinite graph policy-gradient loss")
    return policy_term
