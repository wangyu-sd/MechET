#!/usr/bin/env python3
"""Score first-reference-divergence localization on frozen *unmarked* R3 queries.

The target is the deliberately mutated action's zero-based position. It is a
synthetic reference-divergence label, not proof of a chemically impossible step.
No repair, oracle action or autonomous endpoint recovery is scored here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.export_r3_unmarked_localization import _flat_action
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


def _paired_source(source: Path, queries: Path,
                   *, expected_cases: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    source_hash = verify_evaluation_source(source, name="r3_corruptions")
    manifest_path = queries.parent / "manifest.json"
    status_path = queries.parent / "ARTIFACT_STATUS.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("R3 unmarked queries require a manifest and status")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    query_hash = digest(queries)
    if (manifest.get("source_sha256") != source_hash
            or manifest.get("query_sha256") != query_hash
            or manifest.get("cases") != expected_cases
            or manifest.get("model_input_fields") != ["target_smiles", "candidate_actions"]
            or status.get("query_sha256") != query_hash
            or status.get("localization_inference_allowed") is not True
            or status.get("evaluation_allowed") is not False):
        raise ValueError("R3 unmarked query/source frozen contract mismatch")
    source_lines = source.read_text().splitlines()
    query_rows = [json.loads(line) for line in queries.read_text().splitlines()]
    if len(source_lines) != expected_cases or len(query_rows) != expected_cases:
        raise ValueError("R3 unmarked query denominator changed")
    paired = []
    seen: set[str] = set()
    for number, (source_line, query) in enumerate(zip(source_lines, query_rows), 1):
        source_row = json.loads(source_line)
        public = source_row["model_visible"]
        private = source_row["private_reference"]
        case_id = hashlib.sha256(source_line.encode("utf-8")).hexdigest()
        actions = (public["prefix_actions"] + [public["corrupted_action"]]
                   + private["suffix_actions"])
        flat = [_flat_action(action, number=number) for action in actions]
        if (case_id in seen or query.get("case_id") != case_id
                or query.get("model_input") != {
                    "target_smiles": public["target_smiles"],
                    "candidate_actions": flat,
                } or source_row.get("source_split") != "test"):
            raise ValueError("R3 unmarked query differs from its frozen private source")
        if (not isinstance(private.get("first_failure_index"), int)
                or isinstance(private["first_failure_index"], bool)
                or not 0 <= private["first_failure_index"] < len(flat)):
            raise ValueError("R3 private localization label is invalid")
        seen.add(case_id)
        paired.append((source_row, query))
    return paired


def _predictions(path: Path, queries: Path,
                 case_ids: set[str], action_counts: dict[str, int]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("R3 localization predictions and sidecar are required")
    meta = json.loads(sidecar.read_text())
    checkpoint_hash = meta.get("checkpoint_sha256")
    if (meta.get("predictions_sha256") != digest(path)
            or meta.get("query_sha256") != digest(queries)
            or meta.get("input_fields") != ["model_input"]
            or meta.get("prediction_semantics") != "zero_based_first_reference_divergence_index_v1"
            or not isinstance(meta.get("checkpoint_identifier"), str)
            or not meta["checkpoint_identifier"]
            or not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint_hash)):
        raise ValueError("R3 localization prediction provenance/hash mismatch")
    predictions: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        case_id = row.get("case_id")
        status = row.get("generation_status")
        index = row.get("predicted_failure_index")
        if (not isinstance(case_id, str) or case_id in predictions
                or case_id not in case_ids
                or set(row) != {"case_id", "generation_status", "predicted_failure_index"}
                or status not in {"completed", "failed"}
                or (status == "failed" and index is not None)
                or (status == "completed" and (
                    not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < action_counts[case_id]))):
            raise ValueError(f"R3 invalid localization prediction: {case_id}")
        predictions[case_id] = row
    if set(predictions) != case_ids:
        raise ValueError("R3 localization predictions must cover all frozen cases")
    return predictions, {"checkpoint_identifier": meta["checkpoint_identifier"],
                         "checkpoint_sha256": checkpoint_hash,
                         "predictions_sha256": digest(path),
                         "predictions_manifest_sha256": digest(sidecar)}


def _first_electron_event(actions: list[dict[str, Any]]) -> int:
    for index, action in enumerate(actions):
        if action["name"] == "apply_electron_flow":
            return index
    raise ValueError("R3 candidate trajectory has no electron-flow event")


def score(source: Path, queries: Path, predictions: Path, output: Path,
          *, expected_cases: int = 288) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 localization score already exists: {output}")
    paired = _paired_source(source, queries, expected_cases=expected_cases)
    ids = {query["case_id"] for _, query in paired}
    counts_by_id = {query["case_id"]: len(query["model_input"]["candidate_actions"])
                    for _, query in paired}
    predicted, provenance = _predictions(predictions, queries, ids, counts_by_id)
    counts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = {}
    by_feedback: dict[str, Counter[str]] = {}
    rows = []
    distance_sum_covered = 0
    distance_sum_all = 0
    shortcut_distance_sum = 0
    for source_row, query in paired:
        case_id = query["case_id"]
        actual = source_row["private_reference"]["first_failure_index"]
        actions = query["model_input"]["candidate_actions"]
        guess = predicted[case_id]["predicted_failure_index"]
        shortcut = _first_electron_event(actions)
        shortcut_distance = abs(shortcut - actual)
        shortcut_distance_sum += shortcut_distance
        distance = abs(guess - actual) if guess is not None else None
        if distance is not None:
            distance_sum_covered += distance
            distance_sum_all += distance
        else:
            # Missing generations pay a penalty larger than any valid index distance.
            distance_sum_all += len(actions)
        stratum = (source_row["strata"]["failure_depth"] + "/"
                   + source_row["strata"]["event_coordination"])
        feedback = "executor_accepted" if source_row["model_visible"][
            "executor_result"].get("ok") is True else "executor_rejected"
        counts["cases"] += 1
        counts["covered"] += guess is not None
        counts["top1_exact"] += guess == actual
        counts["shortcut_first_electron_exact"] += shortcut == actual
        for bucket in (by_stratum.setdefault(stratum, Counter()),
                       by_feedback.setdefault(feedback, Counter())):
            bucket["cases"] += 1
            bucket["covered"] += guess is not None
            bucket["top1_exact"] += guess == actual
        rows.append({"case_id": case_id, "stratum": stratum,
                     "feedback_stratum_private": feedback,
                     "candidate_actions": len(actions),
                     "predicted_failure_index": guess,
                     "private_reference_index": actual,
                     "absolute_distance": distance,
                     "shortcut_first_electron_index": shortcut})
    if counts["cases"] != expected_cases:
        raise AssertionError("R3 localization denominator changed")
    output.mkdir(parents=True)
    details = output / "localization_details.jsonl"
    with details.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"artifact_type": "r3_unmarked_first_reference_divergence_result_v1",
              "cases": expected_cases, "source_sha256": digest(source),
              "query_sha256": digest(queries), "prediction_provenance": provenance,
              "covered": counts["covered"], "coverage": counts["covered"] / expected_cases,
              "top1_exact": counts["top1_exact"],
              "top1_accuracy_full_denominator": counts["top1_exact"] / expected_cases,
              "mean_absolute_distance_covered": (
                  distance_sum_covered / counts["covered"] if counts["covered"] else None),
              "mean_absolute_distance_all_missing_penalized": distance_sum_all / expected_cases,
              "shortcut_baseline": {
                  "rule": "first_apply_electron_flow_action_in_unmarked_input",
                  "predeclared_label_free": True,
                  "top1_exact": counts["shortcut_first_electron_exact"],
                  "top1_accuracy": counts["shortcut_first_electron_exact"] / expected_cases,
                  "mean_absolute_distance": shortcut_distance_sum / expected_cases},
              "by_stratum": {key: dict(sorted(value.items()))
                             for key, value in sorted(by_stratum.items())},
              "by_private_feedback_status": {key: dict(sorted(value.items()))
                                             for key, value in sorted(by_feedback.items())},
              "details": str(details), "details_sha256": digest(details),
              "claim_boundary": "Synthetic first divergence from a recorded trajectory; executor-accepted nonreference successors are not proven chemically impossible. No repair or endpoint recovery is measured."}
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.source, args.queries, args.predictions,
                           args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
