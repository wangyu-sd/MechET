#!/usr/bin/env python3
"""Freeze a held-out H1 trace-faithfulness benchmark from replay-verified rows."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import file_sha256, read_jsonl, row_id, write_jsonl


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row.get("metadata") or {})


def _tool_calls(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        output.extend(dict(item) for item in message.get("tool_calls") or [])
    return output


def _finish_trace_result(row: Mapping[str, Any]) -> dict[str, Any] | None:
    for message in row.get("messages") or []:
        if message.get("role") != "tool" or message.get("name") != "finish_trace":
            continue
        try:
            value = json.loads(str(message.get("content") or "{}"))
        except json.JSONDecodeError:
            return None
        return dict(value) if isinstance(value, dict) else None
    return None


def validate_h1_row(row: Mapping[str, Any], *, max_tool_calls: int) -> dict[str, Any]:
    identifier = row_id(row)
    metadata = _metadata(row)
    calls = _tool_calls(row)
    names = [str((item.get("function") or {}).get("name") or "") for item in calls]
    finish_calls = sum(name == "finish_trace" for name in names)
    if finish_calls != 1:
        raise ValueError(f"H1_FINISH_TRACE_COUNT:{identifier}:{finish_calls}")
    if len(calls) > max_tool_calls:
        raise ValueError(
            f"H1_TOOL_BUDGET_EXCEEDED:{identifier}:{len(calls)}>{max_tool_calls}"
        )
    if metadata.get("endpoint_source") != "environment_owned_trace":
        raise ValueError(f"H1_ENDPOINT_NOT_TRACE_OWNED:{identifier}")
    if metadata.get("executor_replayed") is not True:
        raise ValueError(f"H1_EXECUTOR_REPLAY_MISSING:{identifier}")
    if not metadata.get("trace_digest"):
        raise ValueError(f"H1_TRACE_DIGEST_MISSING:{identifier}")
    terminal = _finish_trace_result(row)
    if terminal is None:
        raise ValueError(f"H1_FINISH_TRACE_RESULT_MISSING:{identifier}")
    if terminal.get("ok") is not True:
        raise ValueError(f"H1_FINISH_TRACE_NOT_OK:{identifier}")
    if not str(row.get("target_smiles") or ""):
        raise ValueError(f"H1_TARGET_MISSING:{identifier}")
    if not (
        row.get("structural_precursor")
        or row.get("expected_structural_precursor")
        or row.get("expected_precursor")
    ):
        raise ValueError(f"H1_REFERENCE_ENDPOINT_MISSING:{identifier}")
    plan = metadata.get("trace_plan") or row.get("trace_plan") or {}
    steps = list(plan.get("steps") or []) if isinstance(plan, Mapping) else []
    family = str(
        metadata.get("reaction_family")
        or metadata.get("mechanism_family")
        or metadata.get("mechanism_class")
        or row.get("reaction_family")
        or row.get("mechanism_class")
        or ""
    )
    return {
        "id": identifier,
        "tool_calls": len(calls),
        "trace_steps": len(steps),
        "family": family,
    }


def _digest_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-reference", type=Path)
    parser.add_argument("--max-tool-calls", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("H1 source is empty")

    train_ids: set[str] = set()
    train_sha = None
    if args.train_reference:
        train_rows = read_jsonl(args.train_reference)
        train_ids = {row_id(row) for row in train_rows}
        train_sha = file_sha256(args.train_reference)

    accepted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        try:
            diagnostics.append(
                validate_h1_row(row, max_tool_calls=args.max_tool_calls)
            )
            accepted.append(dict(row))
        except Exception as exc:
            code = str(exc).split(":", 1)[0] or type(exc).__name__
            reasons[code] += 1
            quarantine.append(
                {
                    "id": row.get("id"),
                    "error_code": code,
                    "error": str(exc),
                }
            )

    if not accepted:
        raise ValueError("no H1-eligible rows remain after validation")
    accepted_ids = [row_id(row) for row in accepted]
    overlap = sorted(set(accepted_ids) & train_ids)
    if overlap:
        raise ValueError(
            "H1_TRAIN_TEST_ID_OVERLAP: "
            f"count={len(overlap)} examples={overlap[:10]}"
        )

    families = Counter(item["family"] for item in diagnostics if item["family"])
    tool_counts = [int(item["tool_calls"]) for item in diagnostics]
    step_counts = [int(item["trace_steps"]) for item in diagnostics]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_path = args.output_dir / "test.jsonl"
    quarantine_path = args.output_dir / "quarantine.jsonl"
    write_jsonl(test_path, accepted)
    write_jsonl(quarantine_path, quarantine)
    manifest = {
        "artifact_type": "h1_frozen_benchmark_manifest",
        "input": str(args.input),
        "input_sha256": file_sha256(args.input),
        "train_reference": str(args.train_reference) if args.train_reference else None,
        "train_reference_sha256": train_sha,
        "output": str(test_path),
        "output_sha256": file_sha256(test_path),
        "n_input": len(rows),
        "n_accepted": len(accepted),
        "n_quarantined": len(quarantine),
        "quarantine": str(quarantine_path),
        "quarantine_reasons": dict(reasons),
        "stable_ids_sha256": _digest_ids(accepted_ids),
        "max_tool_calls": args.max_tool_calls,
        "tool_call_summary": {
            "minimum": min(tool_counts),
            "maximum": max(tool_counts),
            "mean": sum(tool_counts) / len(tool_counts),
        },
        "trace_step_summary": {
            "minimum": min(step_counts),
            "maximum": max(step_counts),
            "mean": sum(step_counts) / len(step_counts),
        },
        "family_counts": dict(families),
        "claim_gate": {
            "nonempty": bool(accepted),
            "zero_train_id_overlap": not overlap,
            "all_rows_trace_owned": True,
            "all_rows_executor_replayed": True,
            "all_rows_explicit_finish_trace": True,
            "all_rows_within_tool_budget": True,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
