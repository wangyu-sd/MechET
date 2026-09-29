#!/usr/bin/env python3
"""Score one predicted replacement action per frozen R3 exposed-failure query.

The evaluator alone reads private reference actions and endpoints. Predictions
must be generated from the answer-free query artifact, never from this source.
This measures repair at a *given* failure position, not error localization.
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
from scripts.autoresearch.audit_r3_oracle_repair import (
    replay_with_action, require_frozen_rdkit,
)
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


def _source_and_queries(source: Path, queries: Path,
                        *, expected_cases: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    source_sha = verify_evaluation_source(source, name="r3_corruptions")
    query_manifest_path = queries.parent / "manifest.json"
    query_status_path = queries.parent / "ARTIFACT_STATUS.json"
    if not query_manifest_path.is_file() or not query_status_path.is_file():
        raise ValueError("R3 repair queries need their frozen manifest and status")
    manifest = json.loads(query_manifest_path.read_text())
    status = json.loads(query_status_path.read_text())
    query_sha = digest(queries)
    if (manifest.get("source_sha256") != source_sha
            or manifest.get("query_sha256") != query_sha
            or manifest.get("cases") != expected_cases
            or manifest.get("model_input_fields") != sorted(
                ("target_smiles", "prefix_actions", "corrupted_action", "executor_result"))
            or status.get("query_sha256") != query_sha
            or status.get("repair_inference_allowed") is not True
            or status.get("localization_evaluation_allowed") is not False):
        raise ValueError("R3 query/source frozen contract mismatch")
    source_lines = source.read_text().splitlines()
    query_rows = [json.loads(line) for line in queries.read_text().splitlines()]
    if len(source_lines) != expected_cases or len(query_rows) != expected_cases:
        raise ValueError("R3 repair denominator changed")
    pairs = []
    seen: set[str] = set()
    for source_line, query in zip(source_lines, query_rows):
        row = json.loads(source_line)
        case_id = hashlib.sha256(source_line.encode("utf-8")).hexdigest()
        if (case_id in seen or query.get("case_id") != case_id
                or query.get("model_input") != row.get("model_visible")
                or row.get("source_split") != "test"):
            raise ValueError("R3 source/query row alignment or public boundary changed")
        seen.add(case_id)
        pairs.append((row, query))
    return pairs


def _predictions(path: Path, queries: Path,
                 case_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("R3 predictions and sidecar manifest are required")
    meta = json.loads(manifest_path.read_text())
    checkpoint_sha = meta.get("checkpoint_sha256")
    if (meta.get("predictions_sha256") != digest(path)
            or meta.get("query_sha256") != digest(queries)
            or meta.get("input_fields") != ["model_input"]
            or meta.get("prediction_semantics") != "one_replacement_action_at_exposed_failure_v1"
            or not isinstance(meta.get("checkpoint_identifier"), str)
            or not meta["checkpoint_identifier"]
            or not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint_sha)):
        raise ValueError("R3 prediction provenance/hash mismatch")
    predictions: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        prediction = json.loads(line)
        case_id = prediction.get("case_id")
        if (not isinstance(case_id, str) or case_id in predictions
                or set(prediction) != {"case_id", "generation_status", "repair_action"}
                or prediction.get("generation_status") not in {"completed", "failed"}
                or (prediction["generation_status"] == "failed"
                    and prediction.get("repair_action") is not None)
                or (prediction["generation_status"] == "completed"
                    and not isinstance(prediction.get("repair_action"), dict))):
            raise ValueError(f"R3 invalid or duplicate prediction: {case_id}")
        predictions[case_id] = prediction
    if set(predictions) != case_ids:
        raise ValueError("R3 predictions must cover all 288 frozen cases")
    return predictions, {"checkpoint_identifier": meta["checkpoint_identifier"],
                         "checkpoint_sha256": checkpoint_sha,
                         "predictions_sha256": digest(path),
                         "predictions_manifest_sha256": digest(manifest_path)}


def score(source: Path, queries: Path, trace_source: Path, predictions: Path,
          output: Path, *, expected_cases: int = 288) -> dict[str, Any]:
    import rdkit

    require_frozen_rdkit(rdkit.__version__)
    if output.exists():
        raise FileExistsError(f"R3 score output already exists: {output}")
    pairs = _source_and_queries(source, queries, expected_cases=expected_cases)
    source_meta = json.loads((source.parent / "manifest.json").read_text())
    if (str(trace_source.resolve()) != source_meta.get("trace_source")
            or digest(trace_source) != source_meta.get("trace_sha256")):
        raise ValueError("R3 mapped trace source differs from frozen source")
    by_reaction = {row["reaction_id"] for row, _ in pairs}
    mapped_targets: dict[str, str] = {}
    with trace_source.open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            reaction_id = str(trace.get("source_id") or trace.get("id") or "")
            if reaction_id in by_reaction:
                mapped_targets[reaction_id] = str(trace["target_smiles"])
    if set(mapped_targets) != by_reaction:
        raise ValueError("R3 mapped target source is incomplete")
    prediction_rows, provenance = _predictions(
        predictions, queries, {query["case_id"] for _, query in pairs})
    details = []
    counts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = {}
    for row, query in pairs:
        case_id = query["case_id"]
        prediction = prediction_rows[case_id]
        replacement = prediction["repair_action"]
        stratum = row["strata"]["failure_depth"] + "/" + row["strata"]["event_coordination"]
        bucket = by_stratum.setdefault(stratum, Counter())
        counts["cases"] += 1
        bucket["cases"] += 1
        altered = isinstance(replacement, dict) and replacement != row["model_visible"]["corrupted_action"]
        if altered:
            counts["altered_action"] += 1
            bucket["altered_action"] += 1
        if (not isinstance(replacement, dict)
                or replacement.get("name") != "apply_electron_flow"
                or not isinstance(replacement.get("arguments"), dict)):
            result: dict[str, Any] = {"exact": False, "repair_action_accepted": False,
                                      "error": "missing_or_non_electron_replacement_action"}
        else:
            try:
                replay = replay_with_action(row, mapped_targets[row["reaction_id"]], replacement)
                result = {"exact": replay["exact"],
                          "repair_action_accepted": replay.get("failure_index", 10**9)
                              > row["private_reference"]["first_failure_index"],
                          "failure_index": replay.get("failure_index"),
                          "error": replay.get("error"),
                          "terminal": replay.get("terminal"),
                          "first_reference_divergence": replay["first_reference_divergence"]}
            except (TypeError, ValueError, KeyError) as exc:
                result = {"exact": False, "repair_action_accepted": False,
                          "error": f"malformed_replacement_action:{type(exc).__name__}"}
        counts["endpoint_exact"] += bool(result["exact"])
        counts["repair_action_accepted"] += bool(result["repair_action_accepted"])
        bucket["endpoint_exact"] += bool(result["exact"])
        bucket["repair_action_accepted"] += bool(result["repair_action_accepted"])
        details.append({"case_id": case_id, "stratum": stratum,
                        "generation_status": prediction["generation_status"],
                        "altered_action": altered, **result})
    if counts["cases"] != expected_cases:
        raise AssertionError("R3 score denominator changed")
    output.mkdir(parents=True)
    details_path = output / "repair_details.jsonl"
    with details_path.open("w", encoding="utf-8") as stream:
        for detail in details:
            stream.write(json.dumps(detail, sort_keys=True) + "\n")
    report = {"artifact_type": "r3_exposed_failure_one_action_repair_result_v1",
              "cases": expected_cases, "rdkit_version": rdkit.__version__,
              "source_sha256": digest(source), "query_sha256": digest(queries),
              "mapped_trace_sha256": digest(trace_source), "prediction_provenance": provenance,
              "repair_action_accepted": counts["repair_action_accepted"],
              "repair_action_accepted_rate": counts["repair_action_accepted"] / expected_cases,
              "endpoint_exact": counts["endpoint_exact"],
              "oracle_suffix_assisted_endpoint_recovery_rate":
                  counts["endpoint_exact"] / expected_cases,
              "altered_action": counts["altered_action"],
              "by_stratum": {name: dict(sorted(bucket.items()))
                             for name, bucket in sorted(by_stratum.items())},
              "details": str(details_path), "details_sha256": digest(details_path),
              "first_failure_localization_top1": None,
              "claim_boundary": "One predicted action at an exposed failure, then private reference suffix replay; not autonomous full-trajectory recovery or first-failure localization."}
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--trace-source", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(score(args.source, args.queries, args.trace_source,
                           args.predictions, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
