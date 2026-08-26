"""Matched-data, control, prediction-alignment, and evaluation utilities."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from .endpoints import (
    mapped_exact,
    reference_structural_precursor,
    split_precursor_endpoints,
    structural_exact,
)
from .proof_program import execute_proof
from .tool_schemas import trace_tool_schemas

KNOWLEDGE_TOOLS = {"retrieve_textbook_guidance", "retrieve_primitives"}
CHEMISTRY_TOOLS = {
    "inspect_state",
    "import_fragment",
    "apply_electron_move",
    "apply_coupled_electron_moves",
    "finish_trace",
}
_DIRECT_ENDPOINT_RE = re.compile(
    r"(?:PRECURSOR|ANSWER)\s*:\s*([^\n]+)", re.IGNORECASE
)
_ANSWER_BLOCK_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(dict(value))
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
        raise ValueError("row lacks a stable id")
    return value


def index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row_id(row)
        if identifier in output:
            raise ValueError(f"duplicate row id: {identifier}")
        output[identifier] = dict(row)
    return output


def matched_intersection(
    conditions: Mapping[str, Iterable[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    """Intersect supervision sources before training, never predictions."""

    indexed = {name: index_rows(rows) for name, rows in conditions.items()}
    if not indexed:
        raise ValueError("no conditions")
    common = set.intersection(*(set(rows) for rows in indexed.values()))
    identifiers = sorted(common)
    if not identifiers:
        raise ValueError("conditions have no shared stable IDs")
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
            for field in (
                "target_smiles",
                "expected_precursor",
                "structural_precursor",
            ):
                left_value = str(left.get(field) or "")
                right_value = str(right.get(field) or "")
                if left_value and right_value and left_value != right_value:
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


_TRACE_ONLY_SYSTEM_PROMPT = """You are MechET, a trace-owned inverse electron-flow agent.
Reconstruct the precursor only through explicit environment tool calls. The
final proof and precursor must be produced by finish_trace."""


def _strip_knowledge_prompt(messages: list[dict[str, Any]]) -> None:
    marker = "INITIAL ENVIRONMENT OBSERVATION:\n"
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        if role == "system" and any(
            term in content.lower() for term in ("textbook", "retrieved evidence")
        ):
            message["content"] = _TRACE_ONLY_SYSTEM_PROMPT
            continue
        if role != "user":
            continue
        content = content.replace(
            "Retrieve relevant textbook guidance, reproduce the executable "
            "inverse trace, and finish the environment-owned program.",
            "Reproduce the executable inverse trace and finish the "
            "environment-owned program.",
        )
        if marker not in content:
            message["content"] = content
            continue
        prefix, raw_observation = content.split(marker, 1)
        try:
            observation = json.loads(raw_observation)
        except json.JSONDecodeError:
            message["content"] = content
            continue
        observation["instructions"] = [
            instruction
            for instruction in observation.get("instructions") or []
            if not any(
                term in str(instruction).lower()
                for term in ("retrieved evidence", "textbook", "retrieve_primitives")
            )
        ]
        observation["knowledge"] = {
            "textbook_enabled": False,
            "structured_primitives_enabled": False,
            "structured_anchors_enabled": False,
            "knowledge_reward": False,
        }
        message["content"] = (
            prefix
            + marker
            + json.dumps(observation, ensure_ascii=False)
        )


def strip_knowledge_messages(row: Mapping[str, Any]) -> dict[str, Any]:
    value = _without_tools(row, KNOWLEDGE_TOOLS)
    _strip_knowledge_prompt(value["messages"])
    value["tools"] = trace_tool_schemas()
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
    value = _without_tools(row, {"retrieve_textbook_guidance"})
    value["tools"] = trace_tool_schemas(anchors=True)
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


def _textbook_tool_result(
    row: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    for index, message in enumerate(row.get("messages") or []):
        if (
            message.get("role") == "tool"
            and message.get("name") == "retrieve_textbook_guidance"
        ):
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
        donor_context = dict(donor[2].get("context") or {})
        original_context = dict(original.get("context") or {})
        target_length = int(
            original_context.get("n_characters")
            or len(str(original_context.get("text") or ""))
        )
        donor_text = _fit_length(
            str(donor_context.get("text") or ""), target_length
        )
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
            controlled_result, ensure_ascii=False
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
    _, result = _textbook_tool_result(row)
    context = dict(result.get("context") or {})
    evidence = str(context.get("text") or "")
    target = str(row.get("target_smiles") or "")
    precursor = reference_structural_precursor(dict(row))
    value = copy.deepcopy(dict(row))
    value["messages"] = [
        {
            "role": "system",
            "content": (
                "Predict the atom-contributing structural precursors for the mapped target. "
                "The textbook passage is external evidence, not an answer or validity oracle."
            ),
        },
        {
            "role": "user",
            "content": f"TARGET: {target}\n\nTEXTBOOK EVIDENCE:\n{evidence}",
        },
        {"role": "assistant", "content": f"PRECURSOR: {precursor}"},
    ]
    value["tools"] = []
    value["artifact_type"] = "supervision"
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "reasoning_condition": "direct_answer",
            "knowledge_condition": "textbook_rag",
            "endpoint_source": "direct_answer",
            "trace_digest": None,
            "move_sequence_digest": None,
            "compiled_proof": None,
            "executor_replayed": False,
            "structured_primitives_enabled": False,
            "structured_anchors_enabled": False,
            "textbook_context_sha256": context.get("context_sha256"),
            "textbook_context_characters": context.get(
                "n_characters", len(evidence)
            ),
        }
    )
    value["metadata"] = metadata
    return value


def condition_contract_summary(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    assistant_characters = user_characters = schema_characters = 0
    tool_calls = chemistry_tool_calls = knowledge_tool_calls = 0
    context_characters = 0
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        context_characters += int(metadata.get("textbook_context_characters") or 0)
        schema_characters += len(
            json.dumps(row.get("tools") or [], sort_keys=True, ensure_ascii=False)
        )
        for message in row.get("messages") or []:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "assistant":
                assistant_characters += len(content)
                assistant_characters += len(
                    json.dumps(
                        message.get("tool_calls") or [],
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
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
        "tool_schema_characters": schema_characters,
        "tool_calls": tool_calls,
        "chemistry_tool_calls": chemistry_tool_calls,
        "knowledge_tool_calls": knowledge_tool_calls,
        "textbook_context_characters": context_characters,
    }


def align_prediction_artifact(
    reference_rows: Iterable[Mapping[str, Any]],
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    condition_name: str,
) -> list[dict[str, Any]]:
    """Align predictions to a frozen reference; missing predictions remain rows."""

    references = index_rows(reference_rows)
    predictions = index_rows(prediction_rows)
    extras = sorted(set(predictions) - set(references))
    if extras:
        raise ValueError(
            f"prediction artifact contains unknown IDs for {condition_name}: {extras[:10]}"
        )
    output: list[dict[str, Any]] = []
    for identifier in references:
        reference = references[identifier]
        prediction = predictions.get(identifier)
        if prediction is None:
            missing = copy.deepcopy(reference)
            missing.update(
                {
                    "artifact_type": "prediction",
                    "prediction_status": "missing",
                    "messages": [],
                    "tools": [],
                }
            )
            metadata = dict(missing.get("metadata") or {})
            metadata["condition_name"] = condition_name
            missing["metadata"] = metadata
            output.append(missing)
            continue
        if str(prediction.get("artifact_type") or "") != "prediction":
            raise ValueError(
                f"row {identifier} is not a prediction artifact; training rows are forbidden"
            )
        merged = copy.deepcopy(dict(prediction))
        for field in (
            "target_smiles",
            "expected_precursor",
            "full_precursor_state",
            "structural_precursor",
            "auxiliary_fragments",
        ):
            reference_value = reference.get(field)
            prediction_value = merged.get(field)
            if prediction_value not in (None, "", []) and reference_value not in (
                None,
                "",
                [],
            ) and prediction_value != reference_value:
                raise ValueError(
                    f"prediction/reference mismatch for {identifier}: {field}"
                )
            if reference_value not in (None, "", []):
                merged[field] = copy.deepcopy(reference_value)
        merged["id"] = identifier
        merged.setdefault("prediction_status", "completed")
        output.append(merged)
    return output


def extract_direct_prediction(row: Mapping[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    for value in (
        row.get("predicted_precursor"),
        row.get("prediction"),
        row.get("completion"),
        metadata.get("predicted_precursor"),
        metadata.get("prediction"),
    ):
        text = str(value or "").strip()
        if text:
            answer = _ANSWER_BLOCK_RE.search(text)
            if answer:
                return answer.group(1).strip()
            match = _DIRECT_ENDPOINT_RE.search(text)
            return (match.group(1) if match else text).strip()
    for message in reversed(row.get("messages") or []):
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content") or "").strip()
        answer = _ANSWER_BLOCK_RE.search(text)
        if answer:
            return answer.group(1).strip()
        match = _DIRECT_ENDPOINT_RE.search(text)
        if match:
            return match.group(1).strip()
    return ""


def _rollout_state(row: Mapping[str, Any]) -> dict[str, Any]:
    for value in (
        row.get("rollout_state"),
        row.get("state"),
        (row.get("metadata") or {}).get("rollout_state"),
    ):
        if isinstance(value, dict):
            return dict(value)
    return {}


def _terminal_result(row: Mapping[str, Any]) -> dict[str, Any]:
    state = _rollout_state(row)
    for value in (
        row.get("terminal_result"),
        row.get("final_result"),
        state.get("final_result"),
        (row.get("metadata") or {}).get("terminal_result"),
    ):
        if isinstance(value, dict):
            return dict(value)
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "tool" and message.get("name") == "finish_trace":
            try:
                return dict(json.loads(str(message.get("content") or "{}")))
            except json.JSONDecodeError:
                return {}
    return {}


def _trace_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    state = _rollout_state(row)
    flow_value = state.get("flow_trace") or row.get("flow_trace")
    terminal = _terminal_result(row)
    trace_bound = False
    formal_execute = False
    full_prediction = ""
    compiled_proof = ""
    recompute_error = ""
    try:
        if isinstance(flow_value, dict) and flow_value.get("transitions"):
            flow = ElectronFlowTrace.parse(flow_value)
            compilation = compile_trace_to_proof(flow)
            full_prediction = compilation.precursor_smiles
            compiled_proof = compilation.proof
            declared_trace = str(terminal.get("trace_digest") or "")
            declared_moves = str(terminal.get("move_sequence_digest") or "")
            trace_bound = (
                (not declared_trace or declared_trace == compilation.trace_digest)
                and (
                    not declared_moves
                    or declared_moves == compilation.move_sequence_digest
                )
            )
            formal_execute = True
        else:
            compiled_proof = str(
                terminal.get("compiled_proof")
                or row.get("compiled_proof")
                or (row.get("metadata") or {}).get("compiled_proof")
                or ""
            )
            if compiled_proof:
                execution = execute_proof(compiled_proof)
                formal_execute = bool(execution.ok)
                full_prediction = execution.precursor_smiles if execution.ok else ""
    except Exception as exc:
        recompute_error = str(exc)
    expected_structural = reference_structural_precursor(dict(row))
    predicted_structural = ""
    if full_prediction and str(row.get("target_smiles") or ""):
        predicted_structural = split_precursor_endpoints(
            full_prediction, str(row.get("target_smiles") or "")
        ).structural
    return {
        "prediction_source": "finish_trace" if full_prediction else "missing",
        "prediction_present": bool(full_prediction),
        "trace_bound": bool(trace_bound),
        "formal_execute": bool(formal_execute),
        "full_precursor_prediction": full_prediction,
        "structural_precursor_prediction": predicted_structural,
        "structural_exact": structural_exact(
            predicted_structural, expected_structural
        ),
        "mapped_exact": mapped_exact(predicted_structural, expected_structural),
        "compiled_proof_recomputed": compiled_proof,
        "evaluation_error": recompute_error,
    }


def _direct_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    prediction = extract_direct_prediction(row)
    expected = reference_structural_precursor(dict(row))
    return {
        "prediction_source": "direct_answer" if prediction else "missing",
        "prediction_present": bool(prediction),
        "trace_bound": False,
        "formal_execute": False,
        "full_precursor_prediction": "",
        "structural_precursor_prediction": prediction,
        "structural_exact": structural_exact(prediction, expected),
        "mapped_exact": mapped_exact(prediction, expected),
        "compiled_proof_recomputed": "",
        "evaluation_error": "",
    }


def endpoint_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    if str(row.get("prediction_status") or "") == "missing":
        return {
            "prediction_source": "missing",
            "prediction_present": False,
            "trace_bound": False,
            "formal_execute": False,
            "structural_exact": False,
            "mapped_exact": False,
            "evaluation_error": "MISSING_PREDICTION",
        }
    metadata = dict(row.get("metadata") or {})
    expected_source = str(
        row.get("prediction_mode")
        or metadata.get("prediction_mode")
        or metadata.get("endpoint_source")
        or ""
    )
    if expected_source in {"direct", "direct_answer"}:
        return _direct_evaluation(row)
    if _rollout_state(row).get("flow_trace") or _terminal_result(row).get(
        "compiled_proof"
    ):
        return _trace_evaluation(row)
    return _direct_evaluation(row)


def _retrieval_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    recall_values: list[float] = []
    precision_values: list[float] = []
    latencies: list[float] = []
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        gold = set(
            str(item)
            for item in (
                row.get("gold_passage_ids")
                or metadata.get("gold_passage_ids")
                or []
            )
        )
        retrieved: set[str] = set()
        for message in row.get("messages") or []:
            if (
                message.get("role") == "tool"
                and message.get("name") == "retrieve_textbook_guidance"
            ):
                try:
                    result = json.loads(str(message.get("content") or "{}"))
                except json.JSONDecodeError:
                    continue
                context = dict(result.get("context") or {})
                retrieved.update(str(item) for item in context.get("passage_ids") or [])
                if result.get("latency_ms") is not None:
                    latencies.append(float(result["latency_ms"]))
        if gold:
            overlap = len(gold & retrieved)
            recall_values.append(overlap / len(gold))
            precision_values.append(overlap / max(len(retrieved), 1))
    return {
        "retrieval_labeled_rows": len(recall_values),
        "retrieval_recall_at_k": (
            sum(recall_values) / len(recall_values) if recall_values else None
        ),
        "retrieval_precision_at_k": (
            sum(precision_values) / len(precision_values)
            if precision_values
            else None
        ),
        "mean_retrieval_latency_ms": (
            sum(latencies) / len(latencies) if latencies else None
        ),
    }


def condition_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    evaluations = [endpoint_evaluation(row) for row in rows]
    textbook_calls = sum(
        any(
            _tool_name(message) == "retrieve_textbook_guidance"
            for message in row.get("messages") or []
        )
        for row in rows
    )
    anchor_calls = sum(
        any(
            _tool_name(message) == "retrieve_primitives"
            for message in row.get("messages") or []
        )
        for row in rows
    )
    direct_reward_violations = 0
    for row in rows:
        for message in row.get("messages") or []:
            if (
                message.get("role") == "tool"
                and _tool_name(message) in KNOWLEDGE_TOOLS
            ):
                try:
                    result = json.loads(str(message.get("content") or "{}"))
                except json.JSONDecodeError:
                    continue
                if result.get("direct_reward") not in (False, None):
                    direct_reward_violations += 1
    denominator = max(len(rows), 1)
    direct_count = sum(
        item["prediction_source"] == "direct_answer" for item in evaluations
    )
    trace_count = sum(
        item["prediction_source"] == "finish_trace" for item in evaluations
    )
    missing_count = sum(
        item["prediction_source"] == "missing" for item in evaluations
    )
    return {
        **condition_contract_summary(rows),
        **_retrieval_metrics(rows),
        "textbook_call_rate": textbook_calls / denominator,
        "structured_anchor_call_rate": anchor_calls / denominator,
        "prediction_present_rate": sum(
            bool(item["prediction_present"]) for item in evaluations
        )
        / denominator,
        "missing_prediction_rate": missing_count / denominator,
        "trace_prediction_rate": trace_count / denominator,
        "direct_prediction_rate": direct_count / denominator,
        "trace_bound_rate": sum(
            bool(item["trace_bound"]) for item in evaluations
        )
        / denominator,
        "execute_rate": sum(
            bool(item["formal_execute"]) for item in evaluations
        )
        / denominator,
        "structural_exact_rate": sum(
            bool(item["structural_exact"]) for item in evaluations
        )
        / denominator,
        "mapped_exact_rate": sum(
            bool(item["mapped_exact"]) for item in evaluations
        )
        / denominator,
        "endpoint_exact_rate": sum(
            bool(item["structural_exact"]) for item in evaluations
        )
        / denominator,
        "evaluation_error_rate": sum(
            bool(item.get("evaluation_error")) for item in evaluations
        )
        / denominator,
        "knowledge_direct_reward_violations": direct_reward_violations,
    }
