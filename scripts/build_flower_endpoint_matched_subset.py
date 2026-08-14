#!/usr/bin/env python3
"""Freeze the full-endpoint rows matching the executable FlowER trace test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import sha256_file
from mechet.endpoints import mapped_exact, structural_exact


_TRACE_SOURCE_RE = re.compile(r"(?:^|_)test_(\d+)$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trajectory_id(row: dict[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    for value in (
        metadata.get("trajectory_id"),
        row.get("source_id"),
        row.get("id"),
    ):
        text = str(value or "").strip()
        if not text:
            continue
        match = _TRACE_SOURCE_RE.search(text)
        if match:
            return match.group(1)
        if text.isdigit():
            return text
    raise ValueError(f"cannot resolve FlowER test trajectory ID for {row.get('id')}")


def build_subset(
    full_rows: Iterable[dict[str, Any]], trace_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    full_by_source: dict[str, dict[str, Any]] = {}
    for row in full_rows:
        source_id = str(
            row.get("source_id") or (row.get("metadata") or {}).get("trajectory_id") or ""
        )
        if not source_id or source_id in full_by_source:
            raise ValueError(f"missing or duplicate full-endpoint source ID: {source_id!r}")
        full_by_source[source_id] = row

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in trace_rows:
        trajectory_id = _trajectory_id(trace)
        if trajectory_id in seen:
            raise ValueError(f"duplicate trace trajectory ID: {trajectory_id}")
        seen.add(trajectory_id)
        try:
            full = full_by_source[trajectory_id]
        except KeyError as exc:
            raise ValueError(f"trace trajectory absent from full endpoint test: {trajectory_id}") from exc
        if not structural_exact(
            str(full.get("target_smiles") or ""), str(trace.get("target_smiles") or "")
        ):
            raise ValueError(f"target mismatch for trajectory {trajectory_id}")
        output.append(full)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--trace-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=3080)
    args = parser.parse_args()

    full_rows = _read_jsonl(args.full_reference)
    trace_rows = _read_jsonl(args.trace_reference)
    rows = build_subset(full_rows, trace_rows)
    if args.expected_rows and len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} matched rows, got {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids = [str(row["id"]) for row in rows]
    structural_mismatch_ids = [
        _trajectory_id(trace)
        for full, trace in zip(rows, trace_rows, strict=True)
        if not structural_exact(
            str(full.get("structural_precursor") or ""),
            str(trace.get("structural_precursor") or ""),
        )
    ]
    mapped_target_mismatch_ids = [
        _trajectory_id(trace)
        for full, trace in zip(rows, trace_rows, strict=True)
        if not mapped_exact(
            str(full.get("target_smiles") or ""), str(trace.get("target_smiles") or "")
        )
    ]
    manifest = {
        "artifact_type": "flower_full_endpoint_executable_trace_matched_subset",
        "full_reference": str(args.full_reference.resolve()),
        "full_reference_sha256": sha256_file(args.full_reference),
        "trace_reference": str(args.trace_reference.resolve()),
        "trace_reference_sha256": sha256_file(args.trace_reference),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "rows": len(rows),
        "stable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "matching_key": "FlowER test trajectory_id",
        "endpoint_reference_policy": "full_endpoint_structural_precursor",
        "trace_reference_structural_mismatch_count": len(structural_mismatch_ids),
        "trace_reference_structural_mismatch_trajectory_ids": structural_mismatch_ids,
        "trace_reference_mapped_target_mismatch_count": len(mapped_target_mismatch_ids),
        "trace_reference_mapped_target_mismatch_trajectory_ids": mapped_target_mismatch_ids,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
