"""Reusable multi-turn inference utilities for MechET tool agents."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ParsedToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]

    def to_message_call(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": dict(self.arguments),
            },
        }


_JSON_TOOL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_QWEN_FUNCTION_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def _parse_scalar(text: str) -> Any:
    value = str(text or "").strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalise_call(value: Mapping[str, Any], index: int) -> ParsedToolCall:
    function = dict(value.get("function") or value)
    name = str(function.get("name") or value.get("name") or "").strip()
    arguments = function.get("arguments", value.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not JSON for {name}") from exc
    if not name or not isinstance(arguments, dict):
        raise ValueError(f"invalid tool call: {value}")
    return ParsedToolCall(
        call_id=str(value.get("id") or f"call_{index:03d}"),
        name=name,
        arguments=dict(arguments),
    )


def _calls_from_parsed_response(value: Any) -> list[ParsedToolCall]:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(value, dict):
        if value.get("tool_calls"):
            candidates.extend(value.get("tool_calls") or [])
        elif value.get("function") or value.get("name"):
            candidates.append(value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("tool_calls"):
                candidates.extend(item.get("tool_calls") or [])
            elif isinstance(item, dict) and (
                item.get("function") or item.get("name")
            ):
                candidates.append(item)
    return [
        _normalise_call(item, index) for index, item in enumerate(candidates)
    ]


def parse_tool_calls(
    text: str,
    *,
    tokenizer: Any = None,
    prefix: Any = None,
) -> list[ParsedToolCall]:
    """Parse Transformers/Qwen and generic JSON tool-call responses."""

    raw = str(text or "")
    if tokenizer is not None and hasattr(tokenizer, "parse_response"):
        try:
            parsed = tokenizer.parse_response(raw, prefix=prefix)
            calls = _calls_from_parsed_response(parsed)
            if calls:
                return calls
        except Exception:
            pass

    calls: list[ParsedToolCall] = []
    for match in _JSON_TOOL_RE.finditer(raw):
        value = json.loads(match.group(1))
        calls.append(_normalise_call(value, len(calls)))
    if calls:
        return calls

    for match in _QWEN_FUNCTION_RE.finditer(raw):
        arguments = {
            parameter.group(1).strip(): _parse_scalar(parameter.group(2))
            for parameter in _PARAMETER_RE.finditer(match.group(2))
        }
        calls.append(
            ParsedToolCall(
                call_id=f"call_{len(calls):03d}",
                name=match.group(1).strip(),
                arguments=arguments,
            )
        )
    if calls:
        return calls

    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        value = json.loads(stripped)
        return [_normalise_call(value, 0)]
    return []


def _rejected_call(
    environment: Any, tool_name: str, code: str, message: str = ""
) -> str:
    reject = getattr(environment, "_reject", None)
    if callable(reject):
        try:
            return str(reject(tool_name, code, message))
        except TypeError:
            return str(reject(tool_name, code))
    return json.dumps(
        {
            "ok": False,
            "code": code,
            "tool": tool_name,
            "message": message or code,
        },
        ensure_ascii=False,
    )


def execute_tool_call(environment: Any, call: ParsedToolCall) -> str:
    """Execute one allow-listed facade method and account failures."""

    allowed = {
        "inspect_state",
        "retrieve_textbook_guidance",
        "retrieve_primitives",
        "import_fragment",
        "apply_electron_move",
        "apply_coupled_electron_moves",
        "finish_trace",
        "submit_proof",
        "abstain",
    }
    snapshot = getattr(environment, "_snapshot", lambda: {})()
    if snapshot.get("finalized"):
        return json.dumps(
            {
                "ok": False,
                "code": "EPISODE_ALREADY_FINALIZED",
                "tool": call.name,
            },
            ensure_ascii=False,
        )
    if call.name not in allowed or not hasattr(environment, call.name):
        return _rejected_call(
            environment,
            call.name,
            "TOOL_NOT_AVAILABLE",
            f"tool {call.name!r} is not available in this condition",
        )
    method = getattr(environment, call.name)
    try:
        return str(method(**call.arguments))
    except TypeError as exc:
        return _rejected_call(
            environment, call.name, "TOOL_ARGUMENT_ERROR", str(exc)
        )
    except Exception as exc:
        return _rejected_call(
            environment, call.name, "TOOL_RUNTIME_ERROR", str(exc)
        )


def _skipped_after_terminal(call: ParsedToolCall) -> str:
    return json.dumps(
        {
            "ok": False,
            "code": "SKIPPED_AFTER_TERMINAL",
            "tool": call.name,
            "message": "A previous tool call in the same assistant turn finalized the episode.",
        },
        ensure_ascii=False,
    )


def append_tool_exchange(
    messages: list[dict[str, Any]],
    raw_response: str,
    calls: Iterable[ParsedToolCall],
    environment: Any,
) -> list[dict[str, Any]]:
    """Append one assistant response and exactly one result per declared call.

    A model may emit several calls in one assistant turn. If an earlier call
    finalizes the episode, later calls are not executed, but they still receive
    an explicit ``SKIPPED_AFTER_TERMINAL`` result so the transcript remains a
    valid one-to-one tool-call/tool-result conversation.
    """

    calls = list(calls)
    messages.append(
        {
            "role": "assistant",
            "content": raw_response if not calls else "",
            "tool_calls": [item.to_message_call() for item in calls],
        }
    )
    results: list[dict[str, Any]] = []
    terminal_seen = bool(
        getattr(environment, "_snapshot", lambda: {})().get("finalized")
    )
    for call in calls:
        executed = not terminal_seen
        raw = execute_tool_call(environment, call) if executed else _skipped_after_terminal(call)
        result_message = {
            "role": "tool",
            "tool_call_id": call.call_id,
            "name": call.name,
            "content": raw,
        }
        messages.append(result_message)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = {
                "ok": False,
                "code": "NON_JSON_TOOL_RESULT",
                "raw": raw,
            }
        results.append(
            {
                "call": call.to_message_call(),
                "result": decoded,
                "executed": executed,
            }
        )
        if executed:
            terminal_seen = bool(
                getattr(environment, "_snapshot", lambda: {})().get("finalized")
            )
    return results


def scripted_rollout(
    environment: Any,
    actions: Iterable[Mapping[str, Any]],
    *,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run deterministic scripted actions for CI and intervention validation."""

    transcript = list(messages or [])
    exchanges: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        call = _normalise_call(dict(action), index)
        exchanges.extend(
            append_tool_exchange(
                transcript,
                json.dumps(
                    {"name": call.name, "arguments": call.arguments},
                    ensure_ascii=False,
                ),
                [call],
                environment,
            )
        )
        snapshot = environment._snapshot()
        if snapshot.get("finalized"):
            break
    return {
        "messages": transcript,
        "exchanges": exchanges,
        "rollout_state": environment._snapshot(),
    }


def _row_identifier(row: Mapping[str, Any]) -> str:
    value = row.get("id") or row.get("sample_id") or row.get("source_id")
    if value in {None, ""}:
        raise ValueError("prediction row is missing a stable ID")
    return str(value)


def _row_target(row: Mapping[str, Any]) -> str:
    value = str(row.get("target_smiles") or "").strip()
    if not value:
        raise ValueError(f"{_row_identifier(row)}: prediction row is missing target_smiles")
    return value


def _observations_by_tool(row: Mapping[str, Any]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for message in row.get("messages") or []:
        if message.get("role") != "tool":
            continue
        name = str(message.get("name") or "").strip()
        content = str(message.get("content") or "")
        if name and content:
            output.setdefault(name, []).append(content)
    return output


def tool_result_pool(
    prediction_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic cross-target donor assignments for H1 shuffle.

    Assignments are keyed by target SMILES because the environment receives the
    target during ``reset``. A donor always comes from a different target, which
    prevents self-observation leakage even when several stable IDs share the
    same product. Tools with fewer than two distinct target groups are marked
    unavailable instead of silently reusing the sample's own observation.
    """

    rows = [dict(row) for row in prediction_rows]
    if not rows:
        raise ValueError("shuffle intervention source is empty")

    by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = _row_identifier(row)
        target = _row_target(row)
        entry = by_target.setdefault(
            target,
            {"source_ids": [], "observations": {}},
        )
        entry["source_ids"].append(identifier)
        for tool_name, values in _observations_by_tool(row).items():
            entry["observations"].setdefault(tool_name, []).extend(values)

    tool_targets: dict[str, list[str]] = {}
    for target, entry in by_target.items():
        for tool_name, values in entry["observations"].items():
            if values:
                tool_targets.setdefault(tool_name, []).append(target)
    for values in tool_targets.values():
        values.sort()

    assignments: dict[str, dict[str, Any]] = {}
    unavailable: dict[str, list[str]] = {}
    donor_records: list[dict[str, Any]] = []
    for target in sorted(by_target):
        assignments[target] = {}
        unavailable[target] = []
        for tool_name, eligible_targets in sorted(tool_targets.items()):
            donor_candidates = [item for item in eligible_targets if item != target]
            if not donor_candidates:
                unavailable[target].append(tool_name)
                continue
            selector = hashlib.sha256(f"{target}\0{tool_name}".encode()).digest()
            donor_target = donor_candidates[
                int.from_bytes(selector[:8], "big") % len(donor_candidates)
            ]
            donor_entry = by_target[donor_target]
            values = list(donor_entry["observations"][tool_name])
            donor_id = sorted(donor_entry["source_ids"])[0]
            assignments[target][tool_name] = {
                "donor_id": donor_id,
                "donor_target_smiles": donor_target,
                "values": values,
            }
            donor_records.append(
                {
                    "target_smiles": target,
                    "tool": tool_name,
                    "donor_id": donor_id,
                    "donor_target_smiles": donor_target,
                    "n_observations": len(values),
                }
            )

    contract_valid = not any(unavailable.values())
    donor_digest = hashlib.sha256(
        json.dumps(donor_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "format": "cross_target_tool_observation_donors_v2",
        "assignments": assignments,
        "unavailable": unavailable,
        "contract": {
            "self_donors_forbidden": True,
            "same_tool_type_required": True,
            "different_target_required": True,
            "all_observed_tools_have_donors": contract_valid,
            "donor_manifest_sha256": donor_digest,
        },
        "donor_records": donor_records,
    }
