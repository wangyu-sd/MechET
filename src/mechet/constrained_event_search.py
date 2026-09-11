"""Executor-constrained event-level search primitives.

This module deliberately separates three concerns:

1. the language model proposes one complete electron-flow event and supplies its
   own likelihood score;
2. the deterministic MechET executor decides whether that event can be applied;
3. search keeps only executable, non-cyclic, non-duplicate successor states and
   ranks them by model likelihood.

Formal executability is therefore a hard feasibility gate, not a positive
search score.  The module does not read a gold precursor and does not introduce
an auxiliary value model or forward critic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rdkit import Chem

from .forward_expert import verify_electron_step


StateValidator = Callable[[str], tuple[bool, str]]


def canonical_state_key(smiles: str) -> str:
    """Return a deterministic mapped-state key without changing atom identity.

    The key normalizes component and traversal order but preserves atom maps,
    stereochemistry, charge and bond identity.  It is intentionally conservative:
    symmetry-equivalent states with different map assignments are not collapsed
    in this first implementation.
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError(f"invalid molecular state: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@dataclass(frozen=True)
class EventProposal:
    """One model-proposed elementary electron-flow event.

    ``logprob_sum`` and ``token_count`` must describe only the generated event
    span.  Search uses their ratio so formal-validity heuristics never outrank
    the model's own probability distribution.
    """

    label: str
    moves: tuple[dict[str, Any], ...]
    logprob_sum: float
    token_count: int = 1

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("event proposal requires a label")
        if self.token_count < 1:
            raise ValueError("token_count must be >= 1")

    @property
    def normalized_logprob(self) -> float:
        return float(self.logprob_sum) / int(self.token_count)


@dataclass(frozen=True)
class SearchNode:
    """One surviving partial trajectory in event-level search."""

    state_smiles: str
    trace_labels: tuple[str, ...] = ()
    logprob_sum: float = 0.0
    token_count: int = 0
    visited_state_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        key = canonical_state_key(self.state_smiles)
        if not self.visited_state_keys:
            object.__setattr__(self, "visited_state_keys", (key,))
        elif key != self.visited_state_keys[-1]:
            raise ValueError("current state must be the last visited state key")

    @property
    def normalized_logprob(self) -> float:
        if self.token_count <= 0:
            return 0.0
        return float(self.logprob_sum) / int(self.token_count)


@dataclass(frozen=True)
class RejectedBranch:
    parent_trace: tuple[str, ...]
    proposal_label: str
    code: str
    message: str
    successor_smiles: str = ""
    normalized_logprob: float = 0.0


@dataclass(frozen=True)
class BeamStepResult:
    selected: tuple[SearchNode, ...]
    executor_rejected: tuple[RejectedBranch, ...]
    duplicate_pruned: tuple[RejectedBranch, ...]
    beam_pruned: tuple[RejectedBranch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": [
                {
                    "state_smiles": node.state_smiles,
                    "trace_labels": list(node.trace_labels),
                    "normalized_logprob": node.normalized_logprob,
                }
                for node in self.selected
            ],
            "executor_rejected": [branch.__dict__ for branch in self.executor_rejected],
            "duplicate_pruned": [branch.__dict__ for branch in self.duplicate_pruned],
            "beam_pruned": [branch.__dict__ for branch in self.beam_pruned],
        }


def _evaluate_event(
    node: SearchNode,
    proposal: EventProposal,
    *,
    state_validator: StateValidator | None,
) -> tuple[SearchNode | None, RejectedBranch | None]:
    if not proposal.moves:
        return None, RejectedBranch(
            parent_trace=node.trace_labels,
            proposal_label=proposal.label,
            code="EMPTY_EVENT",
            message="event contains no electron moves",
            normalized_logprob=proposal.normalized_logprob,
        )

    result = verify_electron_step(node.state_smiles, proposal.moves)
    if not result.get("ok"):
        return None, RejectedBranch(
            parent_trace=node.trace_labels,
            proposal_label=proposal.label,
            code=str(result.get("code") or "EXECUTOR_REJECTED"),
            message=str(result.get("message") or result.get("code") or "executor rejected event"),
            normalized_logprob=proposal.normalized_logprob,
        )

    successor = str(result.get("state_smiles") or "")
    if not successor:
        return None, RejectedBranch(
            parent_trace=node.trace_labels,
            proposal_label=proposal.label,
            code="EMPTY_SUCCESSOR",
            message="executor accepted event but returned no state",
            normalized_logprob=proposal.normalized_logprob,
        )

    successor_key = canonical_state_key(successor)
    if successor_key == node.visited_state_keys[-1]:
        return None, RejectedBranch(
            parent_trace=node.trace_labels,
            proposal_label=proposal.label,
            code="NO_OP",
            message="event leaves the authoritative molecular state unchanged",
            successor_smiles=successor,
            normalized_logprob=proposal.normalized_logprob,
        )
    if successor_key in node.visited_state_keys:
        return None, RejectedBranch(
            parent_trace=node.trace_labels,
            proposal_label=proposal.label,
            code="STATE_CYCLE",
            message="event returns to an ancestor molecular state",
            successor_smiles=successor,
            normalized_logprob=proposal.normalized_logprob,
        )

    if state_validator is not None:
        accepted, reason = state_validator(successor)
        if not accepted:
            return None, RejectedBranch(
                parent_trace=node.trace_labels,
                proposal_label=proposal.label,
                code="CHEMISTRY_SUPPORT_REJECTED",
                message=str(reason or "successor outside frozen chemistry support"),
                successor_smiles=successor,
                normalized_logprob=proposal.normalized_logprob,
            )

    child = SearchNode(
        state_smiles=successor,
        trace_labels=node.trace_labels + (proposal.label,),
        logprob_sum=node.logprob_sum + float(proposal.logprob_sum),
        token_count=node.token_count + int(proposal.token_count),
        visited_state_keys=node.visited_state_keys + (successor_key,),
    )
    return child, None


def advance_event_beam(
    expansions: Sequence[tuple[SearchNode, Sequence[EventProposal]]],
    *,
    beam_width: int,
    state_validator: StateValidator | None = None,
) -> BeamStepResult:
    """Execute one search layer and keep the highest-likelihood unique states.

    Every proposal is executed immediately against its parent state.  Invalid,
    cyclic and optional chemistry-support failures are removed before ranking.
    Formally accepted children are deduplicated by canonical authoritative state,
    keeping the child with the higher cumulative normalized model log-probability.
    Only then is the beam-width limit applied.
    """

    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if not expansions:
        return BeamStepResult((), (), (), ())

    executor_rejected: list[RejectedBranch] = []
    children: list[SearchNode] = []
    for node, proposals in expansions:
        for proposal in proposals:
            child, rejected = _evaluate_event(
                node,
                proposal,
                state_validator=state_validator,
            )
            if rejected is not None:
                executor_rejected.append(rejected)
            elif child is not None:
                children.append(child)

    unique: dict[str, SearchNode] = {}
    duplicate_pruned: list[RejectedBranch] = []
    for child in children:
        key = canonical_state_key(child.state_smiles)
        incumbent = unique.get(key)
        if incumbent is None:
            unique[key] = child
            continue
        if child.normalized_logprob > incumbent.normalized_logprob:
            duplicate_pruned.append(
                RejectedBranch(
                    parent_trace=incumbent.trace_labels[:-1],
                    proposal_label=incumbent.trace_labels[-1],
                    code="DUPLICATE_SUCCESSOR",
                    message="lower-scoring branch produced the same successor state",
                    successor_smiles=incumbent.state_smiles,
                    normalized_logprob=incumbent.normalized_logprob,
                )
            )
            unique[key] = child
        else:
            duplicate_pruned.append(
                RejectedBranch(
                    parent_trace=child.trace_labels[:-1],
                    proposal_label=child.trace_labels[-1],
                    code="DUPLICATE_SUCCESSOR",
                    message="lower-scoring branch produced the same successor state",
                    successor_smiles=child.state_smiles,
                    normalized_logprob=child.normalized_logprob,
                )
            )

    ranked = sorted(
        unique.values(),
        key=lambda node: (node.normalized_logprob, -len(node.trace_labels), node.trace_labels),
        reverse=True,
    )
    selected = tuple(ranked[:beam_width])
    beam_pruned = tuple(
        RejectedBranch(
            parent_trace=child.trace_labels[:-1],
            proposal_label=child.trace_labels[-1],
            code="BEAM_CAPACITY",
            message=f"valid successor fell outside beam_width={beam_width}",
            successor_smiles=child.state_smiles,
            normalized_logprob=child.normalized_logprob,
        )
        for child in ranked[beam_width:]
    )
    return BeamStepResult(
        selected=selected,
        executor_rejected=tuple(executor_rejected),
        duplicate_pruned=tuple(duplicate_pruned),
        beam_pruned=beam_pruned,
    )
