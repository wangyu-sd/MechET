"""Reusable multi-turn inference utilities for MechET tool agents."""
from __future__ import annotations

from dataclasses import dataclass
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


def append_tool_exchange(
    messages: list[dict[str, Any]],
    raw_response: str,
    calls: Iterable[ParsedToolCall],
    environment: Any,
) -> list[dict[str, Any]]:
    """Append one assistant response and matched results until termination."""

    calls = list(calls)
    messages.append(
        {
            "role": "assistant",
            "content": raw_response if not calls else "",
            "tool_calls": [item.to_message_call() for item in calls],
        }
    )
    results: list[dict[str, Any]] = []
    for call in calls:
        snapshot = getattr(environment, "_snapshot", lambda: {})()
        if snapshot.get("finalized"):
            break
        raw = execute_tool_call(environment, call)
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
        results.append({"call": call.to_message_call(), "result": decoded})
        snapshot = getattr(environment, "_snapshot", lambda: {})()
        if snapshot.get("finalized"):
            break
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


def tool_result_pool(
    prediction_rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[str]]:
    """Collect normal-run tool observations for shuffled-observation controls."""

    pool: dict[str, list[str]] = {}
    for row in prediction_rows:
        for message in row.get("messages") or []:
            if message.get("role") != "tool":
                continue
            name = str(message.get("name") or "")
            content = str(message.get("content") or "")
            if name and content:
                pool.setdefault(name, []).append(content)
    return pool
