"""Prediction-set, abstention, recovery, and runtime-contract metrics."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any, Iterable, Mapping

from .knowledge_ablation import endpoint_evaluation


def prediction_runtime_contract(
    rows: Iterable[Mapping[str, Any]],
    *,
    include_adapter: bool,
) -> dict[str, Any]:
    """Summarize and validate model/generation metadata in one artifact."""

    rows = list(rows)
    contracts: set[str] = set()
    adapters: set[str] = set()
    for row in rows:
        model = dict(row.get("model") or {})
        contract = {
            "base_model": model.get("base_model"),
            "model_revision": model.get("model_revision"),
            "temperature": model.get("temperature"),
            "top_p": model.get("top_p"),
            "max_new_tokens": model.get("max_new_tokens"),
            "max_iterations": model.get("max_iterations"),
            "samples_per_target": model.get("samples_per_target"),
        }
        if include_adapter:
            contract["adapter_sha256"] = model.get("adapter_sha256")
            contract["adapter"] = model.get("adapter")
        contracts.add(json.dumps(contract, sort_keys=True, separators=(",", ":")))
        adapters.add(str(model.get("adapter_sha256") or model.get("adapter") or ""))
    decoded = [json.loads(item) for item in sorted(contracts)]
    digest = hashlib.sha256(
        "\n".join(sorted(contracts)).encode()
    ).hexdigest()
    return {
        "n_rows": len(rows),
        "n_unique_runtime_contracts": len(contracts),
        "runtime_contract_consistent": len(contracts) == 1,
        "runtime_contracts": decoded,
        "runtime_contract_sha256": digest,
        "n_unique_adapters": len(adapters),
        "adapter_ids": sorted(adapters),
    }


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
    state = dict(row.get("rollout_state") or {})
    trace = list(state.get("trace") or [])
    had_failure = any(
        dict(event.get("result") or {}).get("ok") is False for event in trace
    )
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
        "selective_risk": 1.0 - correct_covered / covered if covered else None,
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
