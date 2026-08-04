"""Strict prediction evaluation for trace-owned and direct experiment artifacts.

Trace-owned conditions receive endpoint credit only after an explicit, successful
``finish_trace`` tool result.  The evaluator never completes an unfinished trace
on behalf of the model and never falls back to free-form direct answers.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from .endpoints import (
    mapped_exact,
    reference_structural_precursor,
    split_precursor_endpoints,
    structural_exact,
)
from .knowledge_ablation import (
    KNOWLEDGE_TOOLS,
    _retrieval_metrics,
    _tool_name,
    condition_contract_summary,
    extract_direct_prediction,
)
from .proof_program import execute_proof

TRACE_MODES = {"trace", "textbook", "irrelevant", "anchors", "combined"}
DIRECT_MODES = {"direct", "direct_answer"}


def rollout_state(row: Mapping[str, Any]) -> dict[str, Any]:
    for value in (
        row.get("rollout_state"),
        row.get("state"),
        (row.get("metadata") or {}).get("rollout_state"),
    ):
        if isinstance(value, dict):
            return dict(value)
    return {}


def terminal_result(row: Mapping[str, Any]) -> dict[str, Any]:
    state = rollout_state(row)
    for value in (
        row.get("terminal_result"),
        row.get("final_result"),
        state.get("final_result"),
        (row.get("metadata") or {}).get("terminal_result"),
    ):
        if isinstance(value, dict):
            return dict(value)
    return {}


def prediction_mode(row: Mapping[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    return str(
        row.get("prediction_mode")
        or metadata.get("prediction_mode")
        or metadata.get("reasoning_condition")
        or metadata.get("endpoint_source")
        or ""
    ).strip().lower()


def _finish_trace_results(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        if message.get("role") != "tool" or message.get("name") != "finish_trace":
            continue
        try:
            value = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(dict(value))
    return output


def _missing(code: str, *, source: str = "missing") -> dict[str, Any]:
    return {
        "prediction_source": source,
        "prediction_present": False,
        "trace_bound": False,
        "formal_execute": False,
        "full_precursor_prediction": "",
        "structural_precursor_prediction": "",
        "structural_exact": False,
        "mapped_exact": False,
        "compiled_proof_recomputed": "",
        "evaluation_error": "",
        "completion_failure": code,
    }


def strict_trace_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    state = rollout_state(row)
    terminal = terminal_result(row)
    finish_results = _finish_trace_results(row)
    if not bool(state.get("finalized")):
        return _missing("TRACE_NOT_FINALIZED", source="unfinished_trace")
    if bool(state.get("abstained")):
        return _missing("TRACE_ABSTAINED", source="abstention")
    if len(finish_results) != 1:
        return _missing("FINISH_TRACE_RESULT_REQUIRED", source="unfinished_trace")

    finish = finish_results[0]
    if terminal and finish != terminal:
        return _missing("TERMINAL_RESULT_MISMATCH", source="invalid_trace")
    if finish.get("endpoint_source") != "environment_owned_trace":
        return _missing("ENDPOINT_SOURCE_NOT_TRACE_OWNED", source="invalid_trace")
    if finish.get("trace_bound") is not True:
        return _missing("TERMINAL_TRACE_NOT_BOUND", source="invalid_trace")
    if not bool(finish.get("formal_execute") or finish.get("ok")):
        return _missing("FINISH_TRACE_DID_NOT_EXECUTE", source="failed_trace")

    flow_value = state.get("flow_trace") or row.get("flow_trace")
    if not isinstance(flow_value, dict) or not flow_value.get("transitions"):
        return _missing("FLOW_TRACE_MISSING", source="invalid_trace")

    try:
        flow = ElectronFlowTrace.parse(flow_value)
        compilation = compile_trace_to_proof(flow)
        execution = execute_proof(compilation.proof)
        if not execution.ok:
            raise ValueError("RECOMPILED_PROOF_DID_NOT_EXECUTE")
        declared_trace = str(finish.get("trace_digest") or "")
        declared_moves = str(finish.get("move_sequence_digest") or "")
        if not declared_trace or declared_trace != compilation.trace_digest:
            raise ValueError("TRACE_DIGEST_MISMATCH")
        if not declared_moves or declared_moves != compilation.move_sequence_digest:
            raise ValueError("MOVE_SEQUENCE_DIGEST_MISMATCH")
        declared_proof = str(finish.get("compiled_proof") or "")
        if not declared_proof or declared_proof != compilation.proof:
            raise ValueError("COMPILED_PROOF_MISMATCH")
        full_prediction = execution.precursor_smiles
        target = str(row.get("target_smiles") or "")
        predicted_structural = (
            split_precursor_endpoints(full_prediction, target).structural
            if full_prediction and target
            else ""
        )
        expected_structural = reference_structural_precursor(dict(row))
        return {
            "prediction_source": "finish_trace",
            "prediction_present": bool(full_prediction),
            "trace_bound": True,
            "formal_execute": True,
            "full_precursor_prediction": full_prediction,
            "structural_precursor_prediction": predicted_structural,
            "structural_exact": structural_exact(
                predicted_structural, expected_structural
            ),
            "mapped_exact": mapped_exact(
                predicted_structural, expected_structural
            ),
            "compiled_proof_recomputed": compilation.proof,
            "evaluation_error": "",
            "completion_failure": "",
        }
    except Exception as exc:
        value = _missing("TRACE_REEXECUTION_FAILED", source="invalid_trace")
        value["evaluation_error"] = str(exc)
        return value


def strict_direct_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
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
        "completion_failure": "" if prediction else "DIRECT_PREDICTION_MISSING",
    }


def endpoint_evaluation(row: Mapping[str, Any]) -> dict[str, Any]:
    if str(row.get("prediction_status") or "") == "missing":
        return _missing("MISSING_PREDICTION")
    mode = prediction_mode(row)
    if mode in TRACE_MODES:
        return strict_trace_evaluation(row)
    if mode in DIRECT_MODES:
        return strict_direct_evaluation(row)
    if mode == "legacy":
        # Legacy is an explicit complete-proof baseline, not a trace condition.
        proof = str(
            terminal_result(row).get("compiled_proof")
            or row.get("compiled_proof")
            or ""
        )
        if not proof:
            return _missing("LEGACY_PROOF_MISSING")
        try:
            execution = execute_proof(proof)
            if not execution.ok:
                return _missing("LEGACY_PROOF_DID_NOT_EXECUTE", source="failed_proof")
            target = str(row.get("target_smiles") or "")
            structural = split_precursor_endpoints(
                execution.precursor_smiles, target
            ).structural
            expected = reference_structural_precursor(dict(row))
            return {
                "prediction_source": "legacy_complete_proof",
                "prediction_present": True,
                "trace_bound": False,
                "formal_execute": True,
                "full_precursor_prediction": execution.precursor_smiles,
                "structural_precursor_prediction": structural,
                "structural_exact": structural_exact(structural, expected),
                "mapped_exact": mapped_exact(structural, expected),
                "compiled_proof_recomputed": proof,
                "evaluation_error": "",
                "completion_failure": "",
            }
        except Exception as exc:
            value = _missing("LEGACY_PROOF_REEXECUTION_FAILED", source="invalid_proof")
            value["evaluation_error"] = str(exc)
            return value
    # Unknown modes never receive a direct-answer fallback.
    return _missing("PREDICTION_MODE_MISSING_OR_UNKNOWN")


def condition_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    evaluations = [endpoint_evaluation(row) for row in rows]
    denominator = max(len(rows), 1)
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
            if message.get("role") != "tool" or _tool_name(message) not in KNOWLEDGE_TOOLS:
                continue
            try:
                result = json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                continue
            if result.get("direct_reward") not in (False, None):
                direct_reward_violations += 1

    direct_count = sum(
        item["prediction_source"] == "direct_answer" for item in evaluations
    )
    trace_count = sum(
        item["prediction_source"] == "finish_trace" for item in evaluations
    )
    missing_count = sum(not bool(item["prediction_present"]) for item in evaluations)
    unfinished_count = sum(bool(item.get("completion_failure")) for item in evaluations)
    return {
        **condition_contract_summary(rows),
        **_retrieval_metrics(rows),
        "textbook_call_rate": textbook_calls / denominator,
        "structured_anchor_call_rate": anchor_calls / denominator,
        "prediction_present_rate": sum(
            bool(item["prediction_present"]) for item in evaluations
        ) / denominator,
        "missing_prediction_rate": missing_count / denominator,
        "unfinished_or_invalid_prediction_rate": unfinished_count / denominator,
        "trace_prediction_rate": trace_count / denominator,
        "direct_prediction_rate": direct_count / denominator,
        "trace_bound_rate": sum(bool(item["trace_bound"]) for item in evaluations) / denominator,
        "execute_rate": sum(bool(item["formal_execute"]) for item in evaluations) / denominator,
        "structural_exact_rate": sum(bool(item["structural_exact"]) for item in evaluations) / denominator,
        "mapped_exact_rate": sum(bool(item["mapped_exact"]) for item in evaluations) / denominator,
        "endpoint_exact_rate": sum(bool(item["structural_exact"]) for item in evaluations) / denominator,
        "evaluation_error_rate": sum(bool(item.get("evaluation_error")) for item in evaluations) / denominator,
        "knowledge_direct_reward_violations": direct_reward_violations,
    }
