#!/usr/bin/env python3
"""Export frozen R3 corruptions as answer-free *repair* queries.

The current source explicitly shows the corrupted action after a known-good
prefix. Its failure index is therefore recoverable by counting prefix actions;
it must not be used to claim first-failure localization accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


VISIBLE_KEYS = {"target_smiles", "prefix_actions", "corrupted_action", "executor_result"}
FORBIDDEN_KEYS = {"private_reference", "correct_action", "expected_precursor",
                  "expected_successor", "first_failure_index", "suffix_actions"}


def export(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 query artifact already exists: {output}")
    source_hash = verify_evaluation_source(source, name="r3_corruptions")
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    trivial_location_rows = 0
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"R3 source has blank row {number}")
            row = json.loads(line)
            visible = row.get("model_visible")
            if (row.get("source_split") != "test" or not isinstance(visible, dict)
                    or set(visible) != VISIBLE_KEYS
                    or visible["target_smiles"] != row.get("target_smiles")
                    or not isinstance(visible["prefix_actions"], list)
                    or not isinstance(visible["corrupted_action"], dict)
                    or not isinstance(visible["executor_result"], dict)
                    or not isinstance(row.get("private_reference"), dict)):
                raise ValueError(f"R3 source has malformed public/private boundary at row {number}")
            def reject_nested_keys(value: Any) -> None:
                if isinstance(value, dict):
                    if FORBIDDEN_KEYS & set(value):
                        raise ValueError(f"R3 model input leaks reference fields at row {number}")
                    for child in value.values():
                        reject_nested_keys(child)
                elif isinstance(value, list):
                    for child in value:
                        reject_nested_keys(child)
            reject_nested_keys(visible)
            failure_index = row["private_reference"].get("first_failure_index")
            if not isinstance(failure_index, int) or isinstance(failure_index, bool):
                raise ValueError(f"R3 source lacks a valid failure index at row {number}")
            trivial_location_rows += failure_index == len(visible["prefix_actions"])
            case_id = hashlib.sha256(line.rstrip("\r\n").encode("utf-8")).hexdigest()
            if case_id in seen:
                raise ValueError("R3 source has duplicate corruption rows")
            seen.add(case_id)
            queries.append({
                "artifact_type": "r3_answer_free_query_v1",
                "case_id": case_id,
                "model_input": visible,
            })
    if len(queries) != 288:
        raise ValueError(f"R3 query denominator changed: {len(queries)}")
    output.mkdir(parents=True)
    query_path = output / "r3_queries.jsonl"
    with query_path.open("w", encoding="utf-8") as stream:
        for row in queries:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r3_answer_free_query_manifest_v1",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "source_manifest_sha256": digest(source.parent / "manifest.json"),
        "query": str(query_path.resolve()),
        "query_sha256": digest(query_path),
        "cases": len(queries),
        "localization_trivial_from_prefix_count_rows": trivial_location_rows,
        "model_input_fields": sorted(VISIBLE_KEYS),
        "claim_boundary": "Repair-at-exposed-failure queries only. Failure-location Top-1 is degenerate because the corrupted action is singled out after a known-good prefix; no model inference or repair has been run.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r3_query_status_v1",
        "query_sha256": report["query_sha256"],
        "inference_allowed": True,
        "repair_inference_allowed": True,
        "localization_evaluation_allowed": False,
        "training_allowed": False,
        "evaluation_allowed": False,
        "reason": "Repair input only; current failure index is exposed by prefix length, so localization scoring is forbidden",
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
