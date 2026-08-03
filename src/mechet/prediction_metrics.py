"""Prediction-set, abstention, and recovery metrics for frozen MechET artifacts."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

from .knowledge_ablation import endpoint_evaluation


def _candidate_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = list(row.get("candidates") or [])
    if not candidates:
        return [dict(row)]
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        value = deepcopy(dict(row))
        value["messages"] = candidate.get("messages") or []
        value["rollout_state"] = candidate.get("rollout_state") or {}
        value["terminal_result"] = (
            candidate.get("terminal_result")
            or (candidate.get("rollout_state") or {}).get("final_result")
            or {}
        )
        value["prediction"] = candidate.get("prediction") or ""
        value["prediction_status"] = "completed"
        value.pop("candidates", None)
        output.append(value)
    return output


def _failure_recovery(row: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(had_failed_tool, recovered_to_formal_terminal)``."""

    state = dict(row.get("rollout_state") or {})
    trace = list(state.get("trace") or [])
    had_failure = False
    for event in trace:
        result = dict(event.get("result") or {})
        if result.get("ok") is False:
            had_failure = True
    final = dict(state.get("final_result") or {})
    recovered = had_failure and bool(final.get("formal_execute") or final.get("ok"))
    return had_failure, recovered


def prediction_set_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    """Compute pass@K, selective risk, abstention, and recovery metrics."""

    rows = list(rows)
    n = max(len(rows), 1)
    structural_at = {k: 0 for k in ks}
    mapped_at = {k: 0 for k in ks}
    execute_at = {k: 0 for k in ks}
    trace_at = {k: 0 for k in ks}
    covered = correct_covered = abstained = 0
    failed_tool_rows = recovered_rows = 0

    for row in rows:
        candidates = _candidate_rows(row)
        evaluations = [endpoint_evaluation(item) for item in candidates]
        for k in ks:
            prefix = evaluations[: min(k, len(evaluations))]
            structural_at[k] += int(any(item.get("structural_exact") for item in prefix))
            mapped_at[k] += int(any(item.get("mapped_exact") for item in prefix))
            execute_at[k] += int(any(item.get("formal_execute") for item in prefix))
            trace_at[k] += int(any(item.get("trace_bound") for item in prefix))

        selected = endpoint_evaluation(row)
        state = dict(row.get("rollout_state") or {})
        is_abstained = bool(state.get("abstained"))
        abstained += int(is_abstained)
        if bool(selected.get("prediction_present")) and not is_abstained:
            covered += 1
            correct_covered += int(bool(selected.get("structural_exact")))
        had_failure, recovered = _failure_recovery(row)
        failed_tool_rows += int(had_failure)
        recovered_rows += int(recovered)

    result: dict[str, Any] = {
        "n_rows": len(rows),
        "coverage": covered / n,
        "selective_risk": (
            1.0 - correct_covered / covered if covered else None
        ),
        "abstention_rate": abstained / n,
        "tool_failure_rows": failed_tool_rows,
        "tool_failure_recovery_rate": (
            recovered_rows / failed_tool_rows if failed_tool_rows else None
        ),
        "reaction_center_accuracy": None,
        "reaction_center_metric_status": "unavailable_without_frozen_reference_center_labels",
        "synthon_exact_match": None,
        "synthon_metric_status": "unavailable_without_frozen_reference_synthon_labels",
    }
    for k in ks:
        result[f"structural_precursor_top{k}"] = structural_at[k] / n
        result[f"mapped_structural_precursor_top{k}"] = mapped_at[k] / n
        result[f"execute_pass_at_{k}"] = execute_at[k] / n
        result[f"trace_bound_pass_at_{k}"] = trace_at[k] / n
    return result
