#!/usr/bin/env python3
"""Replay every frozen H2 Open-Flow reference without inspecting model outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.open_flow_program import execute_open_flow


def audit_split(manifest_path: Path, split: str, *, progress_every: int = 1000) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "nmi_h2_open_flow_all_step_imports_v2":
        raise ValueError("reference replay requires the frozen repaired Open-Flow v2 artifact")
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"unsupported split: {split}")
    source = manifest_path.parent / f"{split}.jsonl"
    expected_rows = int(manifest["rows"][split])
    expected_sha = str(manifest["output_sha256"][split])
    digest = hashlib.sha256()
    seen: set[str] = set()
    failure_counts: dict[str, int] = {}
    examples: list[dict[str, str]] = []
    n_rows = 0
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            n_rows += 1
            identifier = str(row.get("source_id") or "")
            if not identifier or identifier in seen:
                raise ValueError(f"duplicate or absent source_id at row {n_rows}: {identifier}")
            seen.add(identifier)
            try:
                messages = row["messages"]
                if not messages or messages[-1].get("role") != "assistant":
                    raise ValueError("last message is not the frozen assistant program")
                result = execute_open_flow(str(messages[-1]["content"]), str(row["target_smiles"]))
                if not result.get("execute_ok"):
                    reason = str(result.get("failure_code") or result.get("error") or "EXECUTE_FAILED")
                else:
                    derived = split_precursor_endpoints(
                        str(result.get("derived_precursor") or ""), str(row["target_smiles"])
                    ).structural
                    reason = "" if structural_exact(derived, str(row["structural_precursor"])) else "ENDPOINT_MISMATCH"
            except Exception as exc:
                reason = f"EXCEPTION:{type(exc).__name__}"
            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
                if len(examples) < 12:
                    examples.append({"source_id": identifier, "reason": reason})
            if progress_every and n_rows % progress_every == 0:
                print(json.dumps({"split": split, "processed": n_rows,
                                  "expected": expected_rows, "failures": sum(failure_counts.values())}), flush=True)
    actual_sha = digest.hexdigest()
    if n_rows != expected_rows or actual_sha != expected_sha:
        raise ValueError(f"{split} frozen row count/SHA mismatch: {n_rows}/{actual_sha}")
    return {
        "split": split,
        "rows": n_rows,
        "source_sha256": actual_sha,
        "strict_execution_and_structural_endpoint_exact": n_rows - sum(failure_counts.values()),
        "failure_counts": failure_counts,
        "failure_examples": examples,
        "passed": not failure_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing audit result: {args.output}")
    result = audit_split(args.manifest, args.split, progress_every=args.progress_every)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
