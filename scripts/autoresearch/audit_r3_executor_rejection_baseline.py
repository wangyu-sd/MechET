#!/usr/bin/env python3
"""Audit how often first executor rejection localizes frozen R3 mutations.

This is a deterministic tool-only shortcut, not a model result. Accepted
nonreference successors may fail only in a later reference suffix, or never
be rejected. The private mutation index is used only after the replay.
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


def classify(first_rejection: int | None, mutation_index: int) -> str:
    if first_rejection is None:
        return "no_rejection"
    if first_rejection == mutation_index:
        return "first_rejection_at_mutation"
    if first_rejection > mutation_index:
        return "first_rejection_after_mutation"
    return "first_rejection_before_mutation"


def audit(source: Path, trace_source: Path, output: Path,
          *, expected_cases: int = 288) -> dict[str, Any]:
    import rdkit

    require_frozen_rdkit(rdkit.__version__)
    if output.exists():
        raise FileExistsError(f"R3 rejection audit already exists: {output}")
    source_hash = verify_evaluation_source(source, name="r3_corruptions")
    source_meta = json.loads((source.parent / "manifest.json").read_text())
    if (str(trace_source.resolve()) != source_meta.get("trace_source")
            or digest(trace_source) != source_meta.get("trace_sha256")):
        raise ValueError("R3 mapped trace source differs from frozen source provenance")
    source_lines = source.read_text().splitlines()
    if len(source_lines) != expected_cases:
        raise ValueError(f"R3 rejection denominator changed: {len(source_lines)}")
    rows = [json.loads(line) for line in source_lines]
    wanted = {row["reaction_id"] for row in rows}
    mapped_targets: dict[str, str] = {}
    with trace_source.open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            reaction_id = str(trace.get("source_id") or trace.get("id") or "")
            if reaction_id in wanted:
                mapped_targets[reaction_id] = str(trace["target_smiles"])
    if set(mapped_targets) != wanted:
        raise ValueError("R3 mapped trace source is missing selected reactions")
    counts: Counter[str] = Counter()
    by_feedback: dict[str, Counter[str]] = {}
    by_stratum: dict[str, Counter[str]] = {}
    details = []
    for source_line, row in zip(source_lines, rows):
        public = row["model_visible"]
        private = row["private_reference"]
        mutation_index = private["first_failure_index"]
        if (not isinstance(mutation_index, int) or isinstance(mutation_index, bool)
                or mutation_index != len(public["prefix_actions"])):
            raise ValueError("R3 mutation index differs from frozen prefix length")
        replay = replay_with_action(row, mapped_targets[row["reaction_id"]],
                                    public["corrupted_action"])
        if replay["exact"]:
            raise ValueError("R3 corrupted trajectory unexpectedly reaches its reference endpoint")
        first_rejection = replay.get("failure_index")
        outcome = classify(first_rejection, mutation_index)
        feedback = "executor_accepted" if public["executor_result"].get("ok") is True \
                   else "executor_rejected"
        if ((feedback == "executor_rejected") != (first_rejection == mutation_index)):
            raise ValueError("R3 frozen corrupted-action feedback disagrees with pinned replay")
        stratum = row["strata"]["failure_depth"] + "/" + row["strata"]["event_coordination"]
        for bucket in (counts, by_feedback.setdefault(feedback, Counter()),
                       by_stratum.setdefault(stratum, Counter())):
            bucket["cases"] += 1
            bucket[outcome] += 1
        details.append({"case_id": hashlib.sha256(source_line.encode()).hexdigest(),
                        "feedback_stratum": feedback, "stratum": stratum,
                        "mutation_index_private": mutation_index,
                        "first_executor_rejection_index": first_rejection,
                        "outcome": outcome})
    if counts["cases"] != expected_cases:
        raise AssertionError("R3 rejection denominator changed")
    output.mkdir(parents=True)
    details_path = output / "rejection_details.jsonl"
    with details_path.open("w", encoding="utf-8") as stream:
        for detail in details:
            stream.write(json.dumps(detail, sort_keys=True) + "\n")
    report = {"artifact_type": "r3_executor_first_rejection_baseline_v1",
              "source_sha256": source_hash, "mapped_trace_sha256": digest(trace_source),
              "rdkit_version": rdkit.__version__, "cases": expected_cases,
              "first_rejection_at_mutation": counts["first_rejection_at_mutation"],
              "first_rejection_after_mutation": counts["first_rejection_after_mutation"],
              "first_rejection_before_mutation": counts["first_rejection_before_mutation"],
              "no_rejection": counts["no_rejection"],
              "first_rejection_top1_full_denominator":
                  counts["first_rejection_at_mutation"] / expected_cases,
              "first_rejection_coverage":
                  (expected_cases - counts["no_rejection"]) / expected_cases,
              "by_feedback": {name: dict(sorted(bucket.items()))
                              for name, bucket in sorted(by_feedback.items())},
              "by_stratum": {name: dict(sorted(bucket.items()))
                             for name, bucket in sorted(by_stratum.items())},
              "details": str(details_path), "details_sha256": digest(details_path),
              "claim_boundary": "Deterministic first executor rejection on an unmarked corrupted candidate trajectory. A later/no rejection does not prove the earlier mutation chemically valid or invalid; no model inference or repair is measured."}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--trace-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.trace_source, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
