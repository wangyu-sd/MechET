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



def grounded_event_log_probability(
    policy, graph, inventory, moves, nodes, context, initial_prefix,
):
    """Exact differentiable decoder log-probability for ONE ordered FLOW event.

    Uses the actual CRN source->conditional sink->continue/commit categorical
    heads. Unlike the previous teacher-forced *mean* CE, this is the SUM of
    log-probabilities of the sampled event's decisions (within the graph's
    source/sink candidate inventory). Unordered chemical equivalence is handled
    separately by grouping executed successors, not by pretending an ordered
    token sequence is permutation invariant.
    """
    import torch
    import torch.nn.functional as F

    if not 1 <= len(moves) <= 6:
        raise ValueError("FLOW event requires 1-6 grounded moves")
    sources = tuple(inventory.sources)
    source_embeds = policy.encode_containers(nodes, graph.atom_maps, sources)
    prefix = initial_prefix
    parts = []
    for k, move in enumerate(moves):
        if move.source not in sources:
            raise ValueError("sampled FLOW source not grounded in current state")
        source_index = sources.index(move.source)
        source_logits = policy.source_logits(source_embeds, context, prefix)
        parts.append(F.log_softmax(source_logits, dim=-1)[source_index])
        sinks = tuple(inventory.compatible_sinks(move.source))
        if move.sink not in sinks:
            raise ValueError("sampled FLOW sink not compatible with chosen source")
        sink_index = sinks.index(move.sink)
        sink_embeds = policy.encode_containers(nodes, graph.atom_maps, sinks)
        sink_logits = policy.sink_logits(
            sink_embeds, source_embeds[source_index], context, prefix,
        )
        parts.append(F.log_softmax(sink_logits, dim=-1)[sink_index])
        move_embed = policy.encode_move(
            source_embeds[source_index], sink_embeds[sink_index],
        )
        prefix = policy.update_prefix(prefix, move_embed)
        choice_logits = policy.decision_logits(context, prefix)
        parts.append(F.log_softmax(choice_logits, dim=-1)[
            1 if k == len(moves) - 1 else 0
        ])
    result = torch.stack(parts).sum()
    if not torch.isfinite(result):
        raise FloatingPointError("nonfinite graph event log probability")
    return result


def successor_class_policy_loss(
    candidate_log_probs, class_member_indices, class_advantages,
    reference_nll, *, reference_weight: float = 0.1,
):
    """Optimize unique executed successors, not duplicated event strings.

    Event-mass probability is logsumexp of all ordered generated actions
    reaching the same executor successor. This is a truncated-slate objective;
    its candidate set was model-proposed and is not an unbiased full-policy
    GRPO or PPO estimator. The negative sign gives correct policy-gradient
    directions for signed group advantages.
    """
    import torch

    if (
        not candidate_log_probs or not class_member_indices
        or len(class_member_indices) != len(class_advantages)
        or not 0 <= reference_weight <= 1
    ):
        raise ValueError("invalid successor-class policy loss inputs")
    assigned = []
    for members in class_member_indices:
        if not members or len(set(members)) != len(members):
            raise ValueError("empty/duplicate equivalent action class")
        if any(not 0 <= index < len(candidate_log_probs) for index in members):
            raise ValueError("out-of-range candidate index")
        assigned.extend(members)
    if sorted(assigned) != list(range(len(candidate_log_probs))):
        raise ValueError("successor groups must partition sampled candidates")
    terms = [
        torch.logsumexp(torch.stack([
            candidate_log_probs[i] for i in group
        ]), dim=0) * float(advantage)
        for group, advantage in zip(class_member_indices, class_advantages, strict=True)
    ]
    actor = -torch.stack(terms).sum() / len(class_member_indices)
    imitation = (
        reference_nll * float(reference_weight)
        if reference_nll is not None else actor.new_zeros(())
    )
    total = actor + imitation
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("nonfinite successor-class policy loss")
    return total
