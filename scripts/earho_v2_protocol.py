"""Replay and first-divergence contracts for v2 trajectory-policy EARHO.

Reference actions and endpoints are private training data.  Only the target,
executor state, and accepted-action history reach the policy prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from mechet.in_place_grounded_flow import mapped_atom_numbers
from scripts.run_natural_language_value_search import (
    Action,
    Node,
    execute,
    policy_prompt,
    visible,
)


@dataclass(frozen=True)
class ReferenceTrajectory:
    reaction_id: str
    target: str
    expected_precursor: str
    decisions: tuple[Mapping[str, Any], ...]
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class Divergence:
    decision_index: int | None
    reason: str
    accepted_actions: int
    exact_endpoint: bool


@dataclass(frozen=True)
class V2AnchorTask:
    reaction_id: str
    target: str
    anchor_state: str
    expected_precursor: str
    horizon: int
    total_events: int
    prefix_events: int
    state_hash: str
    anchor_actions: tuple[Mapping[str, Any], ...]
    prefix_states: tuple[str, ...]
    reference_name: str
    reference_arguments: Mapping[str, Any]
    reference_next_state: str
    reference_terminal: bool
    divergence_reason: str

    @property
    def is_full_episode(self) -> bool:
        return self.divergence_reason in {
            "FULL_EPISODE_REHEARSAL", "PRODUCT_ONLY_EVALUATION"
        }


def decision_action(row: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    messages = list(row.get("messages") or ())
    assistants = [message for message in messages if message.get("role") == "assistant"]
    results = [message for message in messages if message.get("role") == "tool"]
    if len(assistants) != 1 or len(results) != 1:
        raise ValueError("reference decision requires one assistant/tool exchange")
    calls = list(assistants[0].get("tool_calls") or ())
    if len(calls) != 1:
        raise ValueError("reference decision requires exactly one tool call")
    function = dict(calls[0].get("function") or {})
    name = str(function.get("name") or "")
    if name not in {"import_fragments", "apply_electron_flow", "finish_trace"}:
        raise ValueError(f"unsupported reference decision: {name}")
    if results[0].get("name") != name:
        raise ValueError("reference tool response name mismatch")
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return name, dict(arguments), json.loads(str(results[0].get("content") or "{}"))


def replay_reference(
    source: Mapping[str, Any],
    decisions: Sequence[Mapping[str, Any]],
    *,
    max_imports: int = 64,
) -> ReferenceTrajectory:
    """Execute every Stage-II decision and audit its exact public observation."""

    reaction_id = str(source.get("source_id") or source.get("id") or "")
    mapped_target = str(source.get("target_smiles") or "")
    expected = visible(str(source.get("full_precursor_state") or source["expected_precursor"]))
    if not reaction_id or not mapped_target or not decisions:
        raise ValueError("reference reaction has no ID, target, or decisions")
    target = visible(mapped_target)
    node = Node(
        target=target,
        state=mapped_target,
        next_map=max(mapped_atom_numbers(mapped_target), default=0) + 1,
        visited={target},
    )
    nodes = [node]
    for index, row in enumerate(decisions):
        if str(row.get("source_id") or "") != reaction_id:
            raise ValueError(f"{reaction_id}: reference reaction ID mismatch")
        if int((row.get("metadata") or {}).get("decision_index", -1)) != index:
            raise ValueError(f"{reaction_id}: reference decision index mismatch")
        users = [message for message in row.get("messages") or () if message.get("role") == "user"]
        if len(users) != 1:
            raise ValueError(f"{reaction_id}: reference prompt is missing")
        actual_prompt = str(users[0].get("content") or "")
        expected_prompt = policy_prompt(
            target,
            node.state,
            include_inventory=True,
            actions=node.actions,
            compact_history=True,
        )
        if actual_prompt != expected_prompt:
            raise ValueError(f"{reaction_id}: Stage-II prompt/replay drift at decision {index}")
        name, arguments, result = decision_action(row)
        if result.get("ok") is not True:
            raise ValueError(f"{reaction_id}: reference tool result failed at {index}")
        child, error = execute(
            node,
            Action(name=name, arguments=arguments, raw="reference", logprob=0.0, tokens=1),
            max_imports=max_imports,
        )
        if child is None:
            raise ValueError(f"{reaction_id}: replay failed at {index}: {error}")
        declared = str(result.get("derived_precursor") or result.get("current_state") or "")
        if visible(child.state) != declared:
            raise ValueError(f"{reaction_id}: successor mismatch at decision {index}")
        if child.terminal != (name == "finish_trace"):
            raise ValueError(f"{reaction_id}: terminal mismatch at decision {index}")
        nodes.append(child)
        node = child
    if not node.terminal or visible(node.state) != expected:
        raise ValueError(f"{reaction_id}: reference endpoint replay mismatch")
    return ReferenceTrajectory(
        reaction_id=reaction_id,
        target=target,
        expected_precursor=expected,
        decisions=tuple(decisions),
        nodes=tuple(nodes),
    )


def locate_first_divergence(
    reference: ReferenceTrajectory,
    policy_step: Callable[[Node], tuple[Node | None, str]],
) -> Divergence:
    """Run the actual policy from product; compare executed, not textual, states."""

    node = reference.nodes[0]
    for index, expected in enumerate(reference.nodes[1:]):
        child, error = policy_step(node)
        if child is None:
            return Divergence(index, error or "INVALID_ACTION", index, False)
        if len(child.actions) != len(node.actions) + 1:
            raise ValueError("policy step must commit exactly one action")
        public = visible(child.state)
        if child.terminal and public == reference.expected_precursor:
            return Divergence(None, "EXACT_ENDPOINT", index + 1, True)
        if public != visible(expected.state) or child.terminal != expected.terminal:
            return Divergence(index, "EXECUTED_SUCCESSOR_DIVERGED", index + 1, False)
        node = child
    return Divergence(None, "REFERENCE_ALIGNED", len(reference.decisions), True)


def anchor_task(
    reference: ReferenceTrajectory,
    decision_index: int,
    *,
    divergence_reason: str,
) -> V2AnchorTask:
    """Reset to the independently replayed reference state and causal history."""

    if not 0 <= decision_index < len(reference.decisions):
        raise ValueError("anchor decision index is outside reference trajectory")
    node = reference.nodes[decision_index]
    successor = reference.nodes[decision_index + 1]
    name, arguments, _ = decision_action(reference.decisions[decision_index])
    return V2AnchorTask(
        reaction_id=reference.reaction_id,
        target=reference.target,
        anchor_state=node.state,
        expected_precursor=reference.expected_precursor,
        horizon=len(reference.decisions) - decision_index,
        total_events=len(reference.decisions),
        prefix_events=decision_index,
        state_hash=hashlib.sha256(node.state.encode()).hexdigest(),
        anchor_actions=tuple(node.actions),
        prefix_states=tuple(visible(previous.state) for previous in reference.nodes[: decision_index + 1]),
        reference_name=name,
        reference_arguments=arguments,
        reference_next_state=successor.state,
        reference_terminal=successor.terminal,
        divergence_reason=divergence_reason,
    )


def promote_productive_horizon(
    horizon: int,
    summaries: Sequence[Mapping[str, Any]],
    *,
    minimum_effective_groups: int,
    productive_pass_rate: float,
    maximum_horizon: int,
) -> tuple[int, dict[str, Any]]:
    """Expand H only when sampled executed successors have verified support."""

    relevant = [item for item in summaries if not item.get("is_full_episode")]
    effective = [item for item in relevant if item.get("effective")]
    productive = sum(
        bool(item.get("successor_success") or item.get("endpoint_success"))
        for item in relevant
    )
    rate = productive / len(relevant) if relevant else 0.0
    promoted = (
        len(effective) >= minimum_effective_groups
        and rate >= productive_pass_rate
    )
    next_horizon = min(maximum_horizon, horizon + int(promoted))
    return next_horizon, {
        "frontier_before": horizon,
        "frontier_after": next_horizon,
        "groups": len(relevant),
        "effective_groups": len(effective),
        "verified_productive_at_k": rate,
        "promoted": promoted,
    }
