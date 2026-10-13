"""Executable graph-policy EARHO primitives (no model or RDKit imports).

A full product-start run identifies the first actual chemical divergence.
A graph-only optimizer may update ONLY if that first divergence belongs to
FLOW on both the reference and predicted controller routes. Later gold
states are legal train-only resets but are NEVER used to define inference.
Successor-equivalent action strings contribute a SINGLE branch outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
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
    """Anchor at the FIRST autonomous graph-correctable deviation.

    H bounds the FORWARD continuation from this error; it never replaces
    the model-induced prefix with an increasingly long reference prefix.
    A full-product reset is eligible only if the first deviation is t=0.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if divergence.index is None or not divergence.graph_correctable:
        return None
    first = divergence.index
    if first >= len(reference_names) or reference_names[first] != FLOW:
        raise ValueError("first graph-correctable action disagrees with replay")
    return AnchorChoice(
        decision_index=first,
        horizon=horizon,
        full_product=bool(full_product and first == 0),
        earliest_divergence=first,
        future_decisions=len(reference_names) - first,
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
    wrong_terminal: bool = False
    target_retained_noop: bool = False
    exhausted: bool = False

    def __post_init__(self) -> None:
        if (self.endpoint_exact or self.reference_successor) and not self.executed:
            raise ValueError("invalid candidate cannot have positive evidence")
        if self.endpoint_exact and (self.wrong_terminal or self.target_retained_noop):
            raise ValueError("exact endpoint cannot also be a wrong terminal")
        if not self.executed and (self.wrong_terminal or self.exhausted):
            raise ValueError("rejected action has no continuation")
        if self.wrong_terminal and self.exhausted:
            raise ValueError("terminal failure cannot be an incomplete search")

    @property
    def positive(self) -> bool:
        return self.endpoint_exact or self.reference_successor

    @property
    def certified_negative(self) -> bool:
        # Executor rejection is definitive. Wrong terminal means wrong for
        # the recorded endpoint benchmark, not chemically impossible.
        return (not self.positive) and (not self.executed or self.wrong_terminal or self.target_retained_noop)

    @property
    def unresolved(self) -> bool:
        return self.executed and not self.positive and not self.certified_negative

    @property
    def reward(self) -> float:
        if self.endpoint_exact:
            return 1.0
        if self.reference_successor:
            return 0.2
        if not self.executed:
            return -0.2
        if self.target_retained_noop:
            return -0.2
        if self.wrong_terminal:
            return -0.1
        return 0.0


def successor_group_advantages(
    outcomes: Sequence[VerifiedOutcome],
    *, endpoint_only: bool = False,
) -> tuple[float, ...]:
    """Assign signed credit only from supported chemical observations.

    An accepted successor unresolved in finite continuation search is
    CENSORED, not a negative example: its actor advantage remains zero.
    No group without any verified positive receives a relative update.
    """
    if not outcomes:
        raise ValueError("empty executed branch group")
    if endpoint_only:
        # Local successor labels remain replay supervision, not positive
        # endpoint evidence. Unknown finite-horizon outcomes stay censored.
        outcomes = tuple(replace(o, reference_successor=False) for o in outcomes)
    if not any(o.positive for o in outcomes):
        return (0.0,) * len(outcomes)
    if all(o.positive for o in outcomes) and len({o.reward for o in outcomes}) == 1:
        return (0.0,) * len(outcomes)
    positives = sum(o.reward for o in outcomes if o.positive)
    negatives = sum(o.certified_negative for o in outcomes)
    if negatives:
        return tuple(
            o.reward / positives if o.positive
            else -1.0 / negatives if o.certified_negative
            else 0.0
            for o in outcomes
        )
    if all(o.positive for o in outcomes):
        average = sum(o.reward for o in outcomes) / len(outcomes)
        weights = [o.reward - average for o in outcomes]
        divisor = sum(max(w, 0.0) for w in weights)
        return tuple(w / divisor for w in weights)
    # Positive + censored is not a known negative comparison.
    return tuple(o.reward / positives if o.positive else 0.0 for o in outcomes)

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
        productive_groups: int, endpoint_groups: int | None = None,
    ) -> tuple["AdaptiveHorizon", bool]:
        if not 0 <= productive_groups <= total_groups or not 0 <= effective_groups <= total_groups:
            raise ValueError("invalid curriculum evidence")
        if endpoint_groups is not None and not 0 <= endpoint_groups <= productive_groups:
            raise ValueError("invalid endpoint curriculum evidence")
        # None preserves the historical local-support ablation API. Main
        # training always supplies independently counted endpoint evidence.
        promotion_support = productive_groups if endpoint_groups is None else endpoint_groups
        promoted = (
            total_groups > 0
            and effective_groups >= self.min_effective_groups
            and promotion_support / total_groups >= self.min_productive_fraction
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
