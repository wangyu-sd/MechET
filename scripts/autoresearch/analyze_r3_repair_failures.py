#!/usr/bin/env python3
"""Summarize frozen R3 model-repair failures without changing its scorer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest


def _category(row: dict[str, Any]) -> str:
    if row["generation_status"] != "completed":
        return "generation_failed"
    if row["exact"]:
        return "oracle_suffix_endpoint_recovered"
    if not row["repair_action_accepted"]:
        return "replacement_action_rejected"
    return "accepted_reference_relative_wrong_endpoint"


def analyze(result: Path, details: Path, output: Path,
            *, expected_cases: int = 288) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 failure analysis already exists: {output}")
    report = json.loads(result.read_text())
    if (report.get("artifact_type") != "r3_exposed_failure_one_action_repair_result_v1"
            or report.get("cases") != expected_cases
            or report.get("details_sha256") != digest(details)):
        raise ValueError("R3 scored result/details provenance mismatch")
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    seen: set[str] = set()
    for line in details.read_text().splitlines():
        row = json.loads(line)
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("R3 detail has missing/duplicate case ID")
        seen.add(case_id)
        category = _category(row)
        counts[category] += 1
        stratum = str(row["stratum"])
        strata[stratum][category] += 1
        error = row.get("error")
        if error:
            errors[str(error).split(":", 1)[0]] += 1
    if (len(seen) != expected_cases
            or counts["oracle_suffix_endpoint_recovered"] != report["endpoint_exact"]
            or (counts["oracle_suffix_endpoint_recovered"]
                + counts["accepted_reference_relative_wrong_endpoint"])
            != report["repair_action_accepted"]):
        raise ValueError("R3 failure categories do not reconstruct frozen totals")
    output.parent.mkdir(parents=True, exist_ok=True)
    analysis = {
        "artifact_type": "r3_exposed_failure_repair_error_taxonomy_v1",
        "score_result_sha256": digest(result),
        "score_details_sha256": digest(details),
        "cases": expected_cases,
        "categories": dict(sorted(counts.items())),
        "error_families": dict(sorted(errors.items())),
        "by_stratum": {name: dict(sorted(bucket.items()))
                       for name, bucket in sorted(strata.items())},
        "claim_boundary": (
            "Existing Stage-II model at an exposed failure, with a private oracle "
            "suffix after one proposed replacement. An accepted nonmatching endpoint "
            "is reference-relative and is not proof of chemical impossibility."
        ),
    }
    output.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.result, args.details, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
