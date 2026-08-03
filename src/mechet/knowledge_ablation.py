"""Matched-data and control utilities for evidence-layer experiments."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

KNOWLEDGE_TOOLS = {"retrieve_textbook_guidance", "retrieve_primitives"}
CHEMISTRY_TOOLS = {
    "inspect_state",
    "import_fragment",
    "apply_electron_move",
    "apply_coupled_electron_moves",
    "finish_trace",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(dict(json.loads(line)))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("id") or row.get("source_id") or "").strip()
    if not value:
        raise ValueError("ablation row lacks a stable id")
    return value


def index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row_id(row)
        if identifier in output:
            raise ValueError(f"duplicate ablation row id: {identifier}")
        output[identifier] = dict(row)
    return output


def matched_intersection(
    conditions: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    indexed = {name: index_rows(rows) for name, rows in conditions.items()}
    if not indexed:
        raise ValueError("no ablation conditions")
    common = set.intersection(*(set(rows) for rows in indexed.values()))
    identifiers = sorted(common)
    if not identifiers:
        raise ValueError("ablation conditions have no shared stable IDs")
    matched = {
        name: [rows[identifier] for identifier in identifiers]
        for name, rows in indexed.items()
    }
    validate_alignment(matched)
    return identifiers, matched


def validate_alignment(conditions: Mapping[str, list[Mapping[str, Any]]]) -> None:
    names = list(conditions)
    if not names:
        raise ValueError("no conditions to validate")
    reference = conditions[names[0]]
    for name in names[1:]:
        rows = conditions[name]
        if len(rows) != len(reference):
            raise ValueError(f"condition size mismatch: {name}")
        for left, right in zip(reference, rows):
            if row_id(left) != row_id(right):
                raise ValueError(f"stable-ID order mismatch: {name}")
            for field in ("target_smiles", "expected_precursor"):
                if str(left.get(field) or "") != str(right.get(field) or ""):
                    raise ValueError(
                        f"matched endpoint mismatch for {row_id(left)} in {name}: {field}"
                    )


def _tool_name(message: Mapping[str, Any]) -> str:
    if message.get("role") == "tool":
        return str(message.get("name") or "")
    calls = message.get("tool_calls") or []
    if calls:
        return str((calls[0].get("function") or {}).get("name") or "")
    return ""


def _without_tools(row: Mapping[str, Any], blocked: set[str]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    value["messages"] = [
        message
        for message in value.get("messages") or []
        if _tool_name(message) not in blocked
    ]
    return value


def strip_knowledge_messages(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove all evidence retrieval while retaining the chemistry trace."""

    value = _without_tools(row, KNOWLEDGE_TOOLS)
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "knowledge_condition": "none",
            "textbook_passage_ids": [],
            "textbook_context_sha256": None,
            "textbook_context_characters": 0,
            "structured_primitives_enabled": False,
            "structured_anchors_enabled": False,
        }
    )
    value["metadata"] = metadata
    return value


def strip_textbook_keep_anchors(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the structured-anchor-only trace condition from a combined row."""

    value = _without_tools(row, {"retrieve_textbook_guidance"})
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "knowledge_condition": "structured_anchors",
            "textbook_passage_ids": [],
            "textbook_context_sha256": None,
            "textbook_context_characters": 0,
            "structured_primitives_enabled": True,
            "structured_anchors_enabled": True,
        }
    )
    value["metadata"] = metadata
    return value


def _textbook_tool_result(row: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") == "tool" and message.get("name") == "retrieve_textbook_guidance":
            return index, dict(json.loads(str(message.get("content") or "{}")))
    raise ValueError(f"row has no textbook tool result: {row_id(row)}")


def _fit_length(text: str, length: int) -> str:
    if length <= 0:
        return ""
    if len(text) >= length:
        return text[:length]
    return text + " " * (length - len(text))


def make_irrelevant_context_control(
    rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rotate only evidence text across targets while matching character count."""

    if len(rows) < 2:
        raise ValueError("irrelevant-context control requires at least two rows")
    prepared = []
    for row in rows:
        index, result = _textbook_tool_result(row)
        prepared.append((dict(row), index, result))

    output: list[dict[str, Any]] = []
    for row_index, (row, message_index, original) in enumerate(prepared):
        target = str(row.get("target_smiles") or "")
        donor = None
        for offset in range(1, len(prepared) + 1):
            candidate = prepared[(row_index + offset) % len(prepared)]
            if str(candidate[0].get("target_smiles") or "") != target:
                donor = candidate
                break
        if donor is None:
            raise ValueError("could not find a different-target context donor")

        donor_id = row_id(donor[0])
        donor_context = dict((donor[2].get("context") or {}))
        original_context = dict(original.get("context") or {})
        target_length = int(
            original_context.get("n_characters")
            or len(str(original_context.get("text") or ""))
        )
        donor_text = _fit_length(str(donor_context.get("text") or ""), target_length)

        controlled_context = copy.deepcopy(original_context)
        controlled_context.update(
            {
                "text": donor_text,
                "passage_ids": list(donor_context.get("passage_ids") or []),
                "context_sha256": hashlib.sha256(donor_text.encode()).hexdigest(),
                "n_characters": len(donor_text),
                "truncated": bool(donor_context.get("truncated", False)),
                "control_type": "length_matched_irrelevant",
                "donor_row_id": donor_id,
            }
        )

        controlled_result = copy.deepcopy(original)
        controlled_result["context"] = controlled_context
        controlled_result["matches"] = []
        controlled_result["control_type"] = "length_matched_irrelevant"
        controlled_result["direct_reward"] = False

        value = copy.deepcopy(row)
        value["messages"][message_index]["content"] = json.dumps(
            controlled_result,
            ensure_ascii=False,
        )
        metadata = dict(value.get("metadata") or {})
        metadata.update(
            {
                "knowledge_condition": "length_matched_irrelevant",
                "textbook_control_donor_id": donor_id,
                "textbook_passage_ids": controlled_context["passage_ids"],
                "textbook_context_sha256": controlled_context["context_sha256"],
                "textbook_context_characters": controlled_context["n_characters"],
            }
        )
        value["metadata"] = metadata
        output.append(value)
    return output


def make_direct_textbook_condition(row: Mapping[str, Any]) -> dict[str, Any]:
    """Create a fair open-book direct-answer baseline from a textbook trace row.

    The direct model receives the exact bounded evidence card but no chemistry
    tools or environment observations. The stable ID, target and expected
    structural precursor remain unchanged.
    """

    _, result = _textbook_tool_result(row)
    context = dict(result.get("context") or {})
    evidence = str(context.get("text") or "")
    target = str(row.get("target_smiles") or "")
    precursor = str(row.get("expected_precursor") or "")
    value = copy.deepcopy(dict(row))
    value["messages"] = [
        {
            "role": "system",
            "content": (
                "Predict the structural precursors for the mapped target. "
                "The textbook passage is external evidence, not an answer or validity oracle."
            ),
        },
        {
            "role": "user",
            "content": f"TARGET: {target}\n\nTEXTBOOK EVIDENCE:\n{evidence}",
        },
        {"role": "assistant", "content": f"PRECURSOR: {precursor}"},
    ]
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "reasoning_condition": "direct_answer",
            "knowledge_condition": "textbook_rag",
            "endpoint_source": "direct_answer",
            "trace_digest": None,
            "compiled_proof": None,
            "executor_replayed": False,
            "structured_primitives_enabled": False,
            "structured_anchors_enabled": False,
            "textbook_context_sha256": context.get("context_sha256"),
            "textbook_context_characters": context.get("n_characters", len(evidence)),
        }
    )
    value["metadata"] = metadata
    return value


def extract_terminal_result(row: Mapping[str, Any]) -> dict[str, Any]:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "tool" and message.get("name") == "finish_trace":
            try:
                return dict(json.loads(str(message.get("content") or "{}")))
            except json.JSONDecodeError:
                return {}
    return dict((row.get("metadata") or {}).get("terminal_result") or {})


def condition_contract_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    assistant_characters = 0
    user_characters = 0
    tool_calls = 0
    chemistry_tool_calls = 0
    knowledge_tool_calls = 0
    context_characters = 0
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        context_characters += int(metadata.get("textbook_context_characters") or 0)
        for message in row.get("messages") or []:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "assistant":
                assistant_characters += len(content)
            elif role == "user":
                user_characters += len(content)
            for call in message.get("tool_calls") or []:
                name = str((call.get("function") or {}).get("name") or "")
                tool_calls += 1
                chemistry_tool_calls += int(name in CHEMISTRY_TOOLS)
                knowledge_tool_calls += int(name in KNOWLEDGE_TOOLS)
    ids = sorted(row_id(row) for row in rows)
    return {
        "n_rows": len(rows),
        "stable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "assistant_characters": assistant_characters,
        "user_characters": user_characters,
        "tool_calls": tool_calls,
        "chemistry_tool_calls": chemistry_tool_calls,
        "knowledge_tool_calls": knowledge_tool_calls,
        "textbook_context_characters": context_characters,
    }


def condition_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    terminal = [extract_terminal_result(row) for row in rows]
    textbook_calls = sum(
        any(_tool_name(message) == "retrieve_textbook_guidance" for message in row.get("messages") or [])
        for row in rows
    )
    anchor_calls = sum(
        any(_tool_name(message) == "retrieve_primitives" for message in row.get("messages") or [])
        for row in rows
    )
    direct_reward_violations = 0
    for row in rows:
        for message in row.get("messages") or []:
            if message.get("role") == "tool" and _tool_name(message) in KNOWLEDGE_TOOLS:
                try:
                    result = json.loads(str(message.get("content") or "{}"))
                except json.JSONDecodeError:
                    continue
                if result.get("direct_reward") not in (False, None):
                    direct_reward_violations += 1
    denominator = max(len(rows), 1)
    return {
        **condition_contract_summary(rows),
        "textbook_call_rate": textbook_calls / denominator,
        "structured_anchor_call_rate": anchor_calls / denominator,
        "trace_bound_rate": sum(bool(item.get("trace_bound")) for item in terminal) / denominator,
        "execute_rate": sum(bool(item.get("formal_execute") or item.get("ok")) for item in terminal) / denominator,
        "endpoint_exact_rate": sum(bool(item.get("endpoint_exact")) for item in terminal) / denominator,
        "knowledge_direct_reward_violations": direct_reward_violations,
    }
