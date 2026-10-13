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



def truncated_slate_reference_kl(student_log_probs, frozen_log_probs):
    """Frozen CRN -> adapted CRN KL on IDENTICAL sampled electron events.

    This is a conservative drift regularizer on the observed bounded event
    slate. It does not estimate full-policy KL: the slate is top-K filtered
    and may miss important events. Reference scores must come from the
    ORIGINAL frozen pretrained graph actor, never the last adapted actor.
    """
    import torch
    import torch.nn.functional as F

    if len(student_log_probs) != len(frozen_log_probs) or not student_log_probs:
        raise ValueError("student/frozen sampled graph slates differ")
    student = torch.stack(tuple(student_log_probs))
    original = torch.stack(tuple(frozen_log_probs)).detach()
    if not bool(torch.isfinite(original).all() and torch.isfinite(student).all()):
        raise FloatingPointError("nonfinite graph policy slate KL inputs")
    reference = F.softmax(original, dim=0)
    student_log = F.log_softmax(student, dim=0)
    return (reference * (F.log_softmax(original, dim=0) - student_log)).sum()


def slate_log_probabilities(event_log_probs, *, temperature: float = 1.0):
    """Normalize the COMPLETE support frozen before an individual update."""
    import math
    import torch
    scores = (event_log_probs if isinstance(event_log_probs, torch.Tensor)
              else torch.stack(tuple(event_log_probs)))
    if (scores.ndim != 1 or not scores.numel() or not torch.isfinite(scores).all()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("invalid fixed-support probabilities or temperature")
    return (scores / temperature).log_softmax(dim=0)


def iid_slate_indices(normalized_log_probs, count: int, rng):
    """Independent draws WITH replacement; Python RNG is resume-checkpointed."""
    import bisect
    import math
    import torch
    values = torch.as_tensor(normalized_log_probs).detach().double().cpu()
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all() or count < 2:
        raise ValueError("invalid IID behavior distribution or draw count")
    weights = values.exp().tolist()
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("behavior probabilities are not normalized over their full support")
    cumulative, total = [], 0.0
    for weight in weights:
        total += weight
        cumulative.append(total)
    return tuple(min(bisect.bisect_right(cumulative, rng.random() * total), len(weights) - 1)
                 for _ in range(count))


def leave_one_out_advantages(endpoint_returns):
    """Other IID returns are an action-independent baseline given s and S."""
    import torch
    values = torch.as_tensor(endpoint_returns)
    if not values.is_floating_point():
        values = values.float()
    values = values.detach()
    if (values.ndim != 1 or values.numel() < 2 or not torch.isfinite(values).all()
            or not ((values == 0) | (values == 1)).all()):
        raise ValueError("endpoint returns must be at least two binary observations")
    return values - (values.sum() - values) / (values.numel() - 1)


def conditional_endpoint_policy_loss(event_log_probs, draw_indices, endpoint_returns, *,
                                     temperature: float = 1.0, actor_enabled: bool = True):
    """Score-function loss for fixed-support endpoint value with a frozen follower.

    The complete pool is normalized once. Each IID draw keeps its own
    multiplicity even if execution/continuation was shared by equivalent
    states. No sampled-class mass approximation or group sign/std gate is used.
    This is a conditional frontier objective, not full product-start PG.
    """
    import torch
    logq = slate_log_probabilities(event_log_probs, temperature=temperature)
    indices = torch.as_tensor(draw_indices, device=logq.device, dtype=torch.long)
    returns = torch.as_tensor(endpoint_returns, device=logq.device, dtype=logq.dtype)
    if (indices.ndim != 1 or indices.numel() < 2 or returns.shape != indices.shape
            or (indices < 0).any() or (indices >= logq.numel()).any()):
        raise ValueError("IID draws and their endpoint returns differ")
    advantage = leave_one_out_advantages(returns)
    if not actor_enabled:
        # Scheduling is independent of observed returns; short H is replay-only.
        return logq.sum() * 0.0
    return -(advantage * logq[indices]).mean()


def sample_grounded_event(policy, graph, inventory, nodes, context, initial_prefix, rng):
    """Ancestral source/sink/commit exploration, without beam/top-width pruning.

    Draws construct the pre-update support; they are not themselves actor
    update samples. Incomplete events are omitted, never forced to COMMIT or
    assigned a fabricated complete-event log probability.
    """
    import math
    import torch
    from crn_jepa.grounded_actions import GroundedMove

    def categorical(logits):
        if logits.ndim != 1 or not logits.numel() or not torch.isfinite(logits).all():
            raise ValueError("invalid native categorical logits")
        logp = logits.log_softmax(dim=-1)
        weights = logp.detach().double().cpu().exp().tolist()
        draw = rng.random() * sum(weights)
        total = 0.0
        for index, weight in enumerate(weights):
            total += weight
            if draw < total:
                return index, float(logp[index])
        return len(weights) - 1, float(logp[-1])

    sources = tuple(inventory.sources)
    if not sources:
        return None
    source_embeddings = policy.encode_containers(nodes, graph.atom_maps, sources)
    prefix, moves, total_logp = initial_prefix, [], 0.0
    for _ in range(policy.max_event_moves):
        source_index, source_logp = categorical(policy.source_logits(source_embeddings, context, prefix))
        source = sources[source_index]
        sinks = tuple(inventory.compatible_sinks(source))
        if not sinks:
            return None
        sink_embeddings = policy.encode_containers(nodes, graph.atom_maps, sinks)
        sink_index, sink_logp = categorical(policy.sink_logits(
            sink_embeddings, source_embeddings[source_index], context, prefix))
        move = GroundedMove(source, sinks[sink_index])
        move_embedding = policy.encode_move(source_embeddings[source_index], sink_embeddings[sink_index])
        prefix = policy.update_prefix(prefix, move_embedding)
        decision, decision_logp = categorical(policy.decision_logits(context, prefix))
        moves.append(move)
        total_logp += source_logp + sink_logp + decision_logp
        if decision == 1:
            if not math.isfinite(total_logp):
                raise FloatingPointError("nonfinite ancestral event probability")
            return tuple(moves), total_logp
    return None
