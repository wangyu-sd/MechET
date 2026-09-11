"""Search runtime for the frozen in-place grounded FLOW policy.

The model-facing action is exactly one transaction: optional first-use imports
plus one coupled electron-flow event.  The executor owns atom maps, state
transitions, cycle checks, proof compilation, and the terminal precursor.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from .electron_flow_trace import ElectronFlowTrace, ElectronFlowTransition, compile_trace_to_proof
from .in_place_grounded_flow import (
    deterministic_unmapped_state,
    execute_grounded_event_transactionally,
    mapped_atom_numbers,
)
from .proof_program import sides_equal


def _copy_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Copy the JSON-only transcript without copying model/cache objects."""

    return tuple(json.loads(json.dumps(item, ensure_ascii=False)) for item in messages)


def initial_private_map(mapped_state: str) -> int:
    maps = mapped_atom_numbers(mapped_state)
    return max(maps, default=0) + 1


@dataclass(frozen=True)
class GroundedProposal:
    """One generated tool action and its actual generated-token likelihood."""

    name: str
    arguments: Mapping[str, Any]
    raw_response: str
    logprob_sum: float
    token_count: int
    seed: int
    call_id: str = "call_000"

    def __post_init__(self) -> None:
        if self.token_count < 1:
            raise ValueError("proposal token_count must be positive")

    @property
    def normalized_logprob(self) -> float:
        return float(self.logprob_sum) / int(self.token_count)


@dataclass(frozen=True)
class GroundedSearchNode:
    """Serializable deterministic episode state; safe to fork between branches."""

    target_mapped_state: str
    current_mapped_state: str
    next_private_map: int
    messages: tuple[dict[str, Any], ...]
    transitions: tuple[dict[str, Any], ...] = ()
    visited_visible_states: tuple[str, ...] = ()
    trace_labels: tuple[str, ...] = ()
    logprob_sum: float = 0.0
    token_count: int = 0

    def __post_init__(self) -> None:
        visible = deterministic_unmapped_state(self.current_mapped_state).text
        if not self.visited_visible_states:
            object.__setattr__(self, "visited_visible_states", (visible,))
        elif self.visited_visible_states[-1] != visible:
            raise ValueError("current state must be the last visited visible state")

    @property
    def normalized_logprob(self) -> float:
        return float(self.logprob_sum) / max(int(self.token_count), 1)

    def transcript(self) -> list[dict[str, Any]]:
        return [json.loads(json.dumps(item, ensure_ascii=False)) for item in self.messages]


@dataclass(frozen=True)
class GroundedRejected:
    parent_trace: tuple[str, ...]
    proposal_name: str
    code: str
    message: str
    depth: int
    normalized_logprob: float
    generated_tokens: int


@dataclass(frozen=True)
class GroundedTerminal:
    node: GroundedSearchNode
    result: Mapping[str, Any]


@dataclass(frozen=True)
class GroundedBeamResult:
    selected: tuple[GroundedSearchNode, ...]
    terminals: tuple[GroundedTerminal, ...]
    rejected: tuple[GroundedRejected, ...]
    duplicate_pruned: tuple[GroundedRejected, ...]
    beam_pruned: tuple[GroundedRejected, ...]


def make_root(row: Mapping[str, Any]) -> GroundedSearchNode:
    messages = [
        dict(item)
        for item in row.get("messages") or []
        if item.get("role") in {"system", "user"}
    ]
    if len(messages) != 2:
        raise ValueError("grounded search row requires one frozen system/user preamble")
    target = str(row.get("target_smiles") or "")
    if not target:
        raise ValueError("grounded search row has no mapped target")
    return GroundedSearchNode(
        target_mapped_state=target,
        current_mapped_state=target,
        next_private_map=initial_private_map(target),
        messages=_copy_messages(messages),
    )


def _assistant_and_tool(
    node: GroundedSearchNode,
    proposal: GroundedProposal,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    transcript = node.transcript()
    transcript.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": proposal.call_id,
                    "type": "function",
                    "function": {
                        "name": proposal.name,
                        "arguments": dict(proposal.arguments),
                    },
                }
            ],
        }
    )
    transcript.append(
        {
            "role": "tool",
            "tool_call_id": proposal.call_id,
            "name": proposal.name,
            "content": json.dumps(dict(result), ensure_ascii=False, separators=(",", ":")),
        }
    )
    return _copy_messages(transcript)


def _reject(
    node: GroundedSearchNode,
    proposal: GroundedProposal,
    code: str,
    message: str,
) -> GroundedRejected:
    return GroundedRejected(
        parent_trace=node.trace_labels,
        proposal_name=proposal.name,
        code=str(code or "EVENT_REJECTED"),
        message=str(message or code or "event rejected"),
        depth=len(node.trace_labels),
        normalized_logprob=proposal.normalized_logprob,
        generated_tokens=proposal.token_count,
    )


def _finish(
    node: GroundedSearchNode,
    proposal: GroundedProposal,
    *,
    expected_precursor: str,
) -> tuple[GroundedTerminal | None, GroundedRejected | None]:
    if proposal.arguments:
        return None, _reject(node, proposal, "FINISH_ARGUMENTS_NOT_EMPTY", "finish_trace takes no arguments")
    try:
        trace = ElectronFlowTrace(
            target_smiles=node.target_mapped_state,
            transitions=[ElectronFlowTransition.parse(item) for item in node.transitions],
        )
        compilation = compile_trace_to_proof(trace)
    except Exception as exc:
        return None, _reject(node, proposal, "TRACE_COMPILATION_FAILED", str(exc))
    exact = bool(
        expected_precursor
        and sides_equal(compilation.precursor_smiles, expected_precursor, ignore_maps=True)
    )
    result = {
        "ok": True,
        "formal_execute": True,
        "trace_bound": True,
        "derived_precursor": compilation.precursor_smiles,
        "endpoint_exact": exact,
        "endpoint_source": "environment_owned_trace",
        "trace_digest": compilation.trace_digest,
        "move_sequence_digest": compilation.move_sequence_digest,
        "n_trace_transitions": compilation.n_transitions,
    }
    terminal_node = GroundedSearchNode(
        target_mapped_state=node.target_mapped_state,
        current_mapped_state=node.current_mapped_state,
        next_private_map=node.next_private_map,
        messages=_assistant_and_tool(node, proposal, result),
        transitions=node.transitions,
        visited_visible_states=node.visited_visible_states,
        trace_labels=node.trace_labels + ("finish_trace",),
        logprob_sum=node.logprob_sum + proposal.logprob_sum,
        token_count=node.token_count + proposal.token_count,
    )
    return GroundedTerminal(terminal_node, result), None


def _apply_event(
    node: GroundedSearchNode,
    proposal: GroundedProposal,
    *,
    max_import_fragments: int = 4,
    max_import_atoms: int = 64,
    max_fragment_heavy_atoms: int = 24,
) -> tuple[GroundedSearchNode | None, GroundedRejected | None]:
    args = dict(proposal.arguments)
    required = {"imports", "marked_state", "flow"}
    if set(args) != required:
        return None, _reject(
            node,
            proposal,
            "EVENT_SCHEMA_INVALID",
            f"apply_grounded_event requires exactly {sorted(required)}",
        )
    outcome = execute_grounded_event_transactionally(
        current_mapped_state=node.current_mapped_state,
        imports=args["imports"],
        marked_state=str(args["marked_state"]),
        flow=str(args["flow"]),
        next_private_map=node.next_private_map,
        seen_visible_states=set(node.visited_visible_states),
        max_import_fragments=max_import_fragments,
        max_import_atoms=max_import_atoms,
        max_fragment_heavy_atoms=max_fragment_heavy_atoms,
    )
    if not outcome.get("ok"):
        message = str(outcome.get("message") or "executor rejected event")
        code = str(outcome.get("code") or "EVENT_REJECTED")
        if "repeats an existing state" in message:
            code = "STATE_CYCLE"
        return None, _reject(node, proposal, code, message)

    successor = str(outcome["current_mapped_state"])
    visible = deterministic_unmapped_state(successor).text
    transition = {
        "step_index": len(node.transitions),
        "state_before": node.current_mapped_state,
        "state_after": successor,
        "moves": list(outcome["compiled_moves"]),
        "imports": list(outcome["mapped_imports"]),
    }
    observation = dict(outcome["observation"])
    observation.pop("imports_committed", None)
    child = GroundedSearchNode(
        target_mapped_state=node.target_mapped_state,
        current_mapped_state=successor,
        next_private_map=int(outcome["next_private_map"]),
        messages=_assistant_and_tool(node, proposal, observation),
        transitions=node.transitions + (transition,),
        visited_visible_states=node.visited_visible_states + (visible,),
        trace_labels=node.trace_labels + (f"event_{len(node.transitions):03d}",),
        logprob_sum=node.logprob_sum + proposal.logprob_sum,
        token_count=node.token_count + proposal.token_count,
    )
    return child, None


def advance_grounded_beam(
    expansions: Sequence[tuple[GroundedSearchNode, Sequence[GroundedProposal]]],
    *,
    beam_width: int,
    expected_precursor: str = "",
    max_import_fragments: int = 4,
    max_import_atoms: int = 64,
    max_fragment_heavy_atoms: int = 24,
) -> GroundedBeamResult:
    """Execute one layer and return terminal, rejected, and surviving branches."""

    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    children: list[GroundedSearchNode] = []
    terminals: list[GroundedTerminal] = []
    rejected: list[GroundedRejected] = []
    for node, proposals in expansions:
        for proposal in proposals:
            if proposal.name == "finish_trace":
                terminal, failure = _finish(
                    node, proposal, expected_precursor=expected_precursor
                )
                if terminal is not None:
                    terminals.append(terminal)
                if failure is not None:
                    rejected.append(failure)
                continue
            if proposal.name != "apply_grounded_event":
                rejected.append(
                    _reject(node, proposal, "TOOL_NOT_AVAILABLE", proposal.name)
                )
                continue
            child, failure = _apply_event(
                node,
                proposal,
                max_import_fragments=max_import_fragments,
                max_import_atoms=max_import_atoms,
                max_fragment_heavy_atoms=max_fragment_heavy_atoms,
            )
            if child is not None:
                children.append(child)
            if failure is not None:
                rejected.append(failure)

    unique: dict[tuple[str, frozenset[str]], GroundedSearchNode] = {}
    duplicate_pruned: list[GroundedRejected] = []
    for child in children:
        key = (
            child.visited_visible_states[-1],
            frozenset(child.visited_visible_states),
        )
        incumbent = unique.get(key)
        if incumbent is None or child.normalized_logprob > incumbent.normalized_logprob:
            loser = incumbent
            unique[key] = child
        else:
            loser = child
        if loser is not None:
            duplicate_pruned.append(
                GroundedRejected(
                    parent_trace=loser.trace_labels[:-1],
                    proposal_name="apply_grounded_event",
                    code="DUPLICATE_SUCCESSOR",
                    message="lower-likelihood branch reached the same state and ancestor set",
                    depth=len(loser.trace_labels) - 1,
                    normalized_logprob=loser.normalized_logprob,
                    generated_tokens=0,
                )
            )

    ranked = sorted(
        unique.values(),
        key=lambda item: (item.normalized_logprob, -len(item.trace_labels), item.trace_labels),
        reverse=True,
    )
    selected = tuple(ranked[:beam_width])
    beam_pruned = tuple(
        GroundedRejected(
            parent_trace=item.trace_labels[:-1],
            proposal_name="apply_grounded_event",
            code="BEAM_CAPACITY",
            message=f"valid successor fell outside beam_width={beam_width}",
            depth=len(item.trace_labels) - 1,
            normalized_logprob=item.normalized_logprob,
            generated_tokens=0,
        )
        for item in ranked[beam_width:]
    )
    return GroundedBeamResult(
        selected=selected,
        terminals=tuple(terminals),
        rejected=tuple(rejected),
        duplicate_pruned=tuple(duplicate_pruned),
        beam_pruned=beam_pruned,
    )


def append_rejection_feedback(
    node: GroundedSearchNode,
    proposal: GroundedProposal,
    rejection: GroundedRejected,
) -> GroundedSearchNode:
    """Continue an independent rollout after an executor-visible failed call.

    Constrained search prunes such a branch.  The independent-rollout baseline
    instead receives the same public error and may spend another model call on
    repair, matching the ordinary interactive inference contract.
    """

    result = {
        "ok": False,
        "code": rejection.code,
        "message": rejection.message,
        "current_state": deterministic_unmapped_state(node.current_mapped_state).text,
    }
    return GroundedSearchNode(
        target_mapped_state=node.target_mapped_state,
        current_mapped_state=node.current_mapped_state,
        next_private_map=node.next_private_map,
        messages=_assistant_and_tool(node, proposal, result),
        transitions=node.transitions,
        visited_visible_states=node.visited_visible_states,
        trace_labels=node.trace_labels,
        logprob_sum=node.logprob_sum + proposal.logprob_sum,
        token_count=node.token_count + proposal.token_count,
    )
