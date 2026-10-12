"""Executable graph-policy EARHO primitives (no model or RDKit imports).

A full product-start run identifies the first actual chemical divergence.
A graph-only optimizer may update ONLY if that first divergence belongs to
FLOW on both the reference and predicted controller routes. Later gold
states are legal train-only resets but are NEVER used to define inference.
Successor-equivalent action strings contribute a SINGLE branch outcome.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


FLOW = "apply_electron_flow"


@dataclass(frozen=True)
class Transition:
    name: str
    accepted: bool
    successor_key: str | None
    terminal: bool = False
    error: str = ""


@dataclass(frozen=True)
class Divergence:
    index: int | None
    reason: str
    graph_correctable: bool
    aligned_decisions: int


def first_executed_divergence(
    reference: Sequence[Transition],
    predicted: Sequence[Transition],
) -> Divergence:
    """Compare actual executed successors, never text/action identity alone."""
    if not reference:
        raise ValueError("empty reference trajectory")
    for index, gold in enumerate(reference):
        if index >= len(predicted):
            return Divergence(index, "MODEL_NO_ACTION", False, index)
        trial = predicted[index]
        different = (
            not trial.accepted
            or gold.successor_key != trial.successor_key
            or gold.terminal != trial.terminal
        )
        if not different:
            continue
        correctable = gold.name == FLOW and trial.name == FLOW
        if not trial.accepted:
            reason = (
                "GRAPH_FLOW_REJECTED" if correctable else
                "CONTROLLER_OR_ENVIRONMENT_REJECTED"
            )
        elif not correctable:
            reason = "QWEN_ROUTING_IMPORT_OR_FINISH"
        else:
            reason = "GRAPH_EXECUTED_SUCCESSOR_DIVERGED"
        return Divergence(index, reason, correctable, index)
    if len(predicted) > len(reference):
        return Divergence(
            len(reference), "EXTRA_ACTION_AFTER_REFERENCE",
            False, len(reference),
        )
    return Divergence(None, "REFERENCE_ALIGNED", False, len(reference))


@dataclass(frozen=True)
class AnchorChoice:
    decision_index: int
    horizon: int
    full_product: bool
    earliest_divergence: int
    future_decisions: int


def select_receding_flow_anchor(
    *,
    divergence: Divergence,
    reference_names: Sequence[str],
    horizon: int,
    full_product: bool = False,
) -> AnchorChoice | None:
    """Start near the endpoint, move backwards toward actual divergence as H grows.

    Never train graph FLOW when the first observed error is a Qwen IMPORT/FINISH
    or a wrong tool routing. Gold reset states are train-only.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if divergence.index is None or not divergence.graph_correctable:
        return None
    first = divergence.index
    if first >= len(reference_names) or reference_names[first] != FLOW:
        raise ValueError("first graph-correctable action disagrees with replay")
    eligible = [
        i for i in range(first, len(reference_names))
        if reference_names[i] == FLOW
    ]
    if not eligible:
        raise ValueError("no replayable electronic frontier")
    if full_product and first == 0:
        return AnchorChoice(
            decision_index=0,
            horizon=horizon,
            full_product=True,
            earliest_divergence=first,
            future_decisions=len(reference_names),
        )
    within = [
        i for i in eligible if len(reference_names) - i <= horizon
    ]
    chosen = min(within) if within else max(eligible)
    return AnchorChoice(
        decision_index=chosen,
        horizon=horizon,
        full_product=False,
        earliest_divergence=first,
        future_decisions=len(reference_names) - chosen,
    )


@dataclass(frozen=True)
class ExecutedAlternative:
    candidate_index: int
    successor_key: str | None
    accepted: bool
    logprob: float
    error: str = ""


@dataclass(frozen=True)
class SuccessorClass:
    key: str
    member_indices: tuple[int, ...]
    representative_index: int
    accepted: bool


def pool_executed_successors(
    alternatives: Sequence[ExecutedAlternative],
) -> tuple[SuccessorClass, ...]:
    """Pool only accepted and chemically identical executed successors.

    Failure texts are *not* equivalent chemical successors and never pool
    together. A candidate order change within one event is not a separate
    chemical outcome if the executor leads to the same canonical state.
    """
    if not alternatives:
        raise ValueError("empty electron event slate")
    if len({item.candidate_index for item in alternatives}) != len(alternatives):
        raise ValueError("duplicate candidate index")
    groups: dict[str, list[ExecutedAlternative]] = {}
    for entry in alternatives:
        if entry.accepted:
            if not entry.successor_key:
                raise ValueError("accepted event missing canonical successor")
            key = "ACCEPTED:" + entry.successor_key
        else:
            key = f"INVALID:{entry.candidate_index}"
        groups.setdefault(key, []).append(entry)
    return tuple(
        SuccessorClass(
            key=key,
            member_indices=tuple(member.candidate_index for member in members),
            representative_index=max(
                members, key=lambda item: item.logprob
            ).candidate_index,
            accepted=members[0].accepted,
        )
        for key, members in groups.items()
    )


@dataclass(frozen=True)
class VerifiedOutcome:
    executed: bool
    endpoint_exact: bool = False
    reference_successor: bool = False

    @property
    def positive(self) -> bool:
        if (self.endpoint_exact or self.reference_successor) and not self.executed:
            raise ValueError("invalid candidate cannot have positive evidence")
        return self.endpoint_exact or self.reference_successor

    @property
    def reward(self) -> float:
        if self.endpoint_exact:
            return 1.0
        if self.reference_successor:
            return 0.2
        return 0.0 if self.executed else -0.2


def successor_group_advantages(
    outcomes: Sequence[VerifiedOutcome],
) -> tuple[float, ...]:
    """Success-gated relative credit across *unique successor states*.

    With no positive evidence, every policy advantage is zero. If every
    branch is supported but only some reach a complete endpoint, endpoint
    winners still get relative credit over local reference-only support.
    """
    if not outcomes:
        raise ValueError("empty executed branch group")
    support = [item.positive for item in outcomes]
    if not any(support):
        return (0.0,) * len(outcomes)
    rewards = [item.reward for item in outcomes]
    if not all(support):
        total = sum(rew for ok, rew in zip(support, rewards) if ok)
        nneg = sum(not ok for ok in support)
        return tuple(
            rew / total if ok else -1.0 / nneg
            for ok, rew in zip(support, rewards)
        )
    if max(rewards) == min(rewards):
        return (0.0,) * len(outcomes)
    mean = sum(rewards) / len(rewards)
    centered = [value - mean for value in rewards]
    scale = sum(max(value, 0.0) for value in centered)
    return tuple(value / scale for value in centered)


@dataclass(frozen=True)
class AdaptiveHorizon:
    current: int = 2
    maximum: int = 12
    min_effective_groups: int = 4
    min_productive_fraction: float = 0.1
    full_episode_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not 1 <= self.current <= self.maximum:
            raise ValueError("invalid horizon range")
        if self.min_effective_groups < 1:
            raise ValueError("effective-group threshold must be positive")
        if not 0 <= self.min_productive_fraction <= 1:
            raise ValueError("invalid productive threshold")
        if not 0 <= self.full_episode_fraction <= 1:
            raise ValueError("invalid full-episode fraction")

    def advance(
        self, *, total_groups: int, effective_groups: int,
        productive_groups: int,
    ) -> tuple["AdaptiveHorizon", bool]:
        if not 0 <= productive_groups <= total_groups or not 0 <= effective_groups <= total_groups:
            raise ValueError("invalid curriculum evidence")
        promoted = (
            total_groups > 0
            and effective_groups >= self.min_effective_groups
            and productive_groups / total_groups >= self.min_productive_fraction
            and self.current < self.maximum
        )
        return (
            AdaptiveHorizon(
                current=self.current + int(promoted),
                maximum=self.maximum,
                min_effective_groups=self.min_effective_groups,
                min_productive_fraction=self.min_productive_fraction,
                full_episode_fraction=self.full_episode_fraction,
            ),
            promoted,
        )
