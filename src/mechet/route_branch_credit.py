"""Opt-in, same-state conditional route continuations for electronic-flow RL.

This is NOT the original G=4 target-only Route-Dr.GRPO collector.  It consumes
preselected *on-policy* decision checkpoints from an external route episode,
restores the same executor-owned state for each sibling, collects complete
independent suffixes, and learns only from suffix assistant tokens.

The route-episode implementation is deliberately injected through an adapter:
there is no multi-reaction TargetOnlyRouteEpisode committed on MechET/main yet.
No model, reward, action-space or inference search change is implied here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol, Sequence


OBSERVED = "observed_terminal"
RESOURCE_CENSORED = "resource_censored"
PROTECTED_CENSORED = "protected_censored"
PENDING = "pending"
CENSORING = {RESOURCE_CENSORED, PROTECTED_CENSORED}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _strict_real(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class PolicyAction:
    """One *actual* sampled native assistant tool call (including malformed calls).

    Every generated token, including terminal token if sampled, is included.
    Tool/environment output tokens and prompt tokens are not in this sequence.
    """

    action_name: str
    arguments: Mapping[str, Any]
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    behavior_version: str

    def __post_init__(self) -> None:
        if not self.behavior_version or not self.action_name:
            raise ValueError("missing behavior version or native tool action name")
        if not self.prompt_token_ids or not self.generated_token_ids:
            raise ValueError("empty prompt/completion tokens are not valid on-policy samples")
        if len(self.generated_token_ids) != len(self.behavior_logprobs):
            raise ValueError("behavior logprob count must equal generated assistant token count")
        if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in self.prompt_token_ids + self.generated_token_ids):
            raise ValueError("token IDs must be nonnegative integers")
        for lp in self.behavior_logprobs:
            if _strict_real(lp, "behavior logprob") > 1e-6:
                raise ValueError("log probability cannot be positive")
        _canonical(self.arguments)  # required for a reproducible execution receipt


@dataclass(frozen=True)
class ForkAnchor:
    """A *pre-reward* eligible FLOW decision from an independent root rollout.

    The snapshot is private executor state for restore only.  The sampler must
    receive env.public_observation(), not the snapshot or reference information.
    """

    target_id: str
    source_rollout_id: str
    decision_index: int
    behavior_version: str
    snapshot: Mapping[str, Any]
    public_observation_hash: str
    accepted: bool = True
    next_action_kind: str = "FLOW"
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.target_id or not self.source_rollout_id or not self.behavior_version:
            raise ValueError("missing anchor identity or frozen behavior version")
        if self.decision_index < 0 or not self.public_observation_hash:
            raise ValueError("invalid decision index or observation fingerprint")
        if not self.accepted or self.next_action_kind != "FLOW":
            raise ValueError("fork only at a preselected accepted FLOW decision, never FINISH")
        object.__setattr__(self, "anchor_id", digest([
            self.target_id, self.source_rollout_id, self.decision_index,
            self.behavior_version, self.public_observation_hash, self.snapshot,
        ]))


def choose_anchor(anchors: Sequence[ForkAnchor], seed: int) -> ForkAnchor:
    """Select an on-policy FLOW state before *any* hidden route evaluation.

    Fixed hash ordering avoids reward-aware / hindsight branch selection.
    """
    if not anchors:
        raise ValueError("no eligible on-policy FLOW anchors")
    versions = {a.behavior_version for a in anchors}
    if len(versions) != 1:
        raise ValueError("do not mix behavior-policy versions")
    return min(anchors, key=lambda a: digest([int(seed), a.anchor_id]))


class RouteEnvironment(Protocol):
    """Interface to the existing/private target-only multi-reaction episode.

    Must restore CURRENT state, accepted-history, pending precursor occurrences,
    frozen stock, scheduling state and route graph. No hidden route labels in
    public_observation. outcome_status must be PENDING, OBSERVED or a censor.
    """

    def restore(self, snapshot: Mapping[str, Any]) -> None: ...
    def public_observation(self) -> Mapping[str, Any]: ...
    def step(self, action: PolicyAction) -> None: ...
    def outcome_status(self) -> str: ...


@dataclass(frozen=True)
class SuffixRollout:
    sibling_index: int
    status: str
    reward: float | None
    actions: tuple[PolicyAction, ...]
    seed: int

    @property
    def generated_tokens(self) -> int:
        return sum(len(a.generated_token_ids) for a in self.actions)


@dataclass(frozen=True)
class ConditionalGroup:
    anchor: ForkAnchor
    siblings: tuple[SuffixRollout, ...]

    def __post_init__(self) -> None:
        if len(self.siblings) < 2:
            raise ValueError("a conditional group requires at least two independent siblings")
        if tuple(s.sibling_index for s in self.siblings) != tuple(range(len(self.siblings))):
            raise ValueError("sibling indices must be contiguous and ordered")
        if len({s.seed for s in self.siblings}) != len(self.siblings):
            raise ValueError("siblings must have independent sampling seeds")
        for sibling in self.siblings:
            if not sibling.actions:
                raise ValueError("all sibling continuations must contain sampled actions")
            if any(a.behavior_version != self.anchor.behavior_version for a in sibling.actions):
                raise ValueError("behavior policy changed within collection")
            if sibling.status == OBSERVED:
                if sibling.reward is None or not (0.0 <= _strict_real(sibling.reward, "reward") <= 1.0):
                    raise ValueError("observed terminal needs a finite [0,1] route reward")
            elif sibling.status in CENSORING:
                if sibling.reward is not None:
                    raise ValueError("censored route must not receive a reward")
            else:
                raise ValueError("not an observed terminal or declared censor")

    @property
    def censored(self) -> bool:
        return any(s.status in CENSORING for s in self.siblings)

    @property
    def n_calls(self) -> int:
        return sum(len(s.actions) for s in self.siblings)

    @property
    def n_tokens(self) -> int:
        return sum(s.generated_tokens for s in self.siblings)


def conditional_advantages(group: ConditionalGroup) -> tuple[float, ...] | None:
    """Fixed-scale conditional Dr.GRPO advantages; no per-group std factor."""
    if group.censored:
        return None  # exclude entire group, not only the censored sibling
    rewards = [float(s.reward) for s in group.siblings]
    mean = sum(rewards) / len(rewards)
    return tuple(value - mean for value in rewards)


def _sibling_seed(base_seed: int, anchor_id: str, sibling: int) -> int:
    return int(digest([base_seed, anchor_id, sibling])[:8], 16)


def collect_conditional_group(
    anchor: ForkAnchor,
    *,
    env_factory: Callable[[], RouteEnvironment],
    sample_action: Callable[[Mapping[str, Any], int], PolicyAction],
    score_after_termination: Callable[[RouteEnvironment], float],
    group_size: int = 2,
    max_decisions: int = 128,
    base_seed: int = 17,
) -> ConditionalGroup:
    """On-policy independent continuations from one already accepted state.

    Caller must collect anchor BEFORE post-stop hidden reward access and freeze
    the behavior actor throughout this call. Never resume a checkpoint under a
    different actor version. A decision budget hit is censoring, not R=0.
    """
    if group_size < 2 or max_decisions < 1:
        raise ValueError("need >=2 siblings and positive resource decision budget")
    siblings: list[SuffixRollout] = []
    for index in range(group_size):
        env = env_factory()
        env.restore(copy.deepcopy(anchor.snapshot))
        observation = env.public_observation()
        if digest(observation) != anchor.public_observation_hash:
            raise ValueError("restored public observation differs from selected FLOW checkpoint")
        seed = _sibling_seed(base_seed, anchor.anchor_id, index)
        actions: list[PolicyAction] = []
        for turn in range(max_decisions):
            status = env.outcome_status()
            if status != PENDING:
                break
            # Only the public observation reaches the actor (never snapshot/GT).
            action = sample_action(copy.deepcopy(env.public_observation()),
                                   _sibling_seed(seed, anchor.anchor_id, turn))
            if action.behavior_version != anchor.behavior_version:
                raise ValueError("sampler is not using frozen behavior policy")
            actions.append(action)
            # Even malformed/rejected actions count as policy-generated tokens.
            env.step(action)
        status = env.outcome_status()
        if status == PENDING:
            status = RESOURCE_CENSORED
        if status not in ({OBSERVED} | CENSORING):
            raise ValueError("environment returned unrecognized terminal status")
        if not actions:
            raise ValueError("fork is terminal; cannot collect a conditional continuation")
        # Hidden reference evaluator first consulted *after* observed termination.
        reward = _strict_real(score_after_termination(env), "terminal reward") if status == OBSERVED else None
        siblings.append(SuffixRollout(index, status, reward, tuple(actions), seed))
    return ConditionalGroup(anchor, tuple(siblings))


def conditional_drgrpo_loss(
    groups: Sequence[ConditionalGroup],
    current_logprobs: Mapping[tuple[int, int, int], Any],
    reference_logprobs: Mapping[tuple[int, int, int], Any] | None = None,
    *,
    epsilon: float = 0.2,
    beta: float = 0.02,
    fixed_scale: float = 1024.0,
):
    """Torch loss for suffix-only, within-fork group comparisons.

    Per-action logprobs must be recomputed on that action's ORIGINAL prompt IDs,
    with shape exactly equal to its generated assistant IDs (incl. EOS).
    Behavior logprobs are recorded frozen sampling probabilities, not detached
    learner logprobs; reference logprobs come from the frozen SFT policy.

    Return (loss, audit). Normalize by N * G * fixed_scale as in route Dr.GRPO.
    This is an optional *separate* conditional objective, not root G=4 RL.
    """
    import torch

    if not 0 <= beta or not (0 < epsilon < 1) or fixed_scale <= 0:
        raise ValueError("invalid loss parameters")
    if not groups:
        raise ValueError("empty conditional groups")
    gsize = len(groups[0].siblings)
    if any(len(g.siblings) != gsize for g in groups):
        raise ValueError("mixed sibling counts; cannot use a common G denominator")
    usable = [(i, g, conditional_advantages(g)) for i, g in enumerate(groups) if not g.censored]
    if not usable:
        raise ValueError("all conditional groups censored; skip optimizer step")
    if beta > 0 and reference_logprobs is None:
        raise ValueError("frozen reference logprobs required for KL")
    total = None
    count = 0
    nonzero = 0
    for i, group, advantages in usable:
        for j, sibling in enumerate(group.siblings):
            advantage = float(advantages[j])
            nonzero += int(abs(advantage) > 1e-12)
            for k, action in enumerate(sibling.actions):
                key = (i, j, k)
                if key not in current_logprobs:
                    raise ValueError(f"missing learner logprobs for {key}")
                cur = current_logprobs[key]
                if not isinstance(cur, torch.Tensor) or cur.ndim != 1 or cur.numel() != len(action.generated_token_ids):
                    raise ValueError("learner logprob tensor shape must match generated tokens")
                if not bool(torch.isfinite(cur.detach()).all()):
                    raise ValueError("nonfinite learner logprobs")
                old = torch.tensor(action.behavior_logprobs, dtype=cur.dtype, device=cur.device)
                delta = cur - old
                if not bool((delta.detach().abs() < 40).all()):
                    raise ValueError("unstable importance ratios; inspect behavior/learner identity")
                ratio = torch.exp(delta)
                a = torch.tensor(advantage, dtype=cur.dtype, device=cur.device)
                pg = torch.minimum(ratio * a, torch.clamp(ratio, 1-epsilon, 1+epsilon) * a)
                term = -pg
                if beta > 0:
                    if key not in reference_logprobs:
                        raise ValueError(f"missing frozen reference logprobs for {key}")
                    ref = reference_logprobs[key]
                    if not isinstance(ref, torch.Tensor) or ref.shape != cur.shape:
                        raise ValueError("reference logprob shape must match generated tokens")
                    if not bool(torch.isfinite(ref.detach()).all()):
                        raise ValueError("nonfinite reference logprobs")
                    d = ref.detach() - cur
                    term = term + beta * ratio * (torch.exp(d) - d - 1.0)
                aggregate = term.sum()
                total = aggregate if total is None else total + aggregate
                count += int(cur.numel())
    if total is None:
        raise ValueError("no policy-generated suffix tokens in usable groups")
    loss = total / (len(usable) * gsize * fixed_scale)
    return loss, {
        "usable_groups": len(usable),
        "censored_groups": len(groups)-len(usable),
        "siblings_per_group": gsize,
        "policy_tokens": count,
        "nonzero_advantage_siblings": nonzero,
        "raw_model_calls": sum(g.n_calls for g in groups),
        "raw_generated_tokens": sum(g.n_tokens for g in groups),
        "objective": "conditional_dr_grpo_suffix_only_v1",
    }


def summarize_groups(groups: Sequence[ConditionalGroup]) -> dict[str, Any]:
    """Audit reward variation, censoring and charged sampling work."""
    usable = [g for g in groups if not g.censored]
    variable = sum(len({round(float(s.reward), 12) for s in g.siblings}) > 1 for g in usable)
    return {
        "groups": len(groups),
        "usable": len(usable),
        "censored": len(groups)-len(usable),
        "variable_reward_groups": variable,
        "variable_reward_fraction": variable / len(usable) if usable else None,
        "calls": sum(g.n_calls for g in groups),
        "generated_tokens": sum(g.n_tokens for g in groups),
        "branch_policy": "conditional_fork_only_not_independent_target_rollouts",
    }