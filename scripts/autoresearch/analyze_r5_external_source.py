#!/usr/bin/env python3
"""Audit a frozen R5 external proposal source before MechET verification.

Recorded-reference absence is not a chemical-invalidity label. This report is
diagnostic and never substitutes for paired Base/Mech trace-support scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts["products"] += 1
        counts["completed_products"] += row["inference_status"] == "completed"
        counts["failed_products"] += row["inference_status"] == "failed"
        ranks = row["candidates"]
        counts["valid_candidate_slots"] += sum(
            candidate["smiles_status"] == "valid_smiles" for candidate in ranks)
        counts["missing_candidate_slots"] += sum(
            candidate["smiles_status"] == "missing" for candidate in ranks)
        counts["invalid_candidate_slots"] += sum(
            candidate["smiles_status"] == "invalid_smiles" for candidate in ranks)
        counts["valid_at_1"] += ranks[0]["smiles_status"] == "valid_smiles"
        counts["recorded_at_1"] += ranks[0]["recorded_reference_status"] == "recorded_reference"
        counts["recorded_at_5"] += any(
            candidate["recorded_reference_status"] == "recorded_reference"
            for candidate in ranks)
    n = counts["products"]
    return {
        "products": n,
        "candidate_slots": n * 5,
        "completed_products": counts["completed_products"],
        "failed_products": counts["failed_products"],
        "valid_candidate_slots": counts["valid_candidate_slots"],
        "missing_candidate_slots": counts["missing_candidate_slots"],
        "invalid_candidate_slots": counts["invalid_candidate_slots"],
        "valid_at_1": counts["valid_at_1"],
        "recorded_at_1": counts["recorded_at_1"],
        "recorded_at_5": counts["recorded_at_5"],
        "valid_candidate_slot_rate": _rate(counts["valid_candidate_slots"], n * 5),
        "recorded_at_1_rate": _rate(counts["recorded_at_1"], n),
        "recorded_at_5_rate": _rate(counts["recorded_at_5"], n),
    }


def analyze(cohort: Path, output: Path, *, expected_products: int = 200) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 source diagnostic already exists: {output}")
    manifest_path = cohort.parent / "manifest.json"
    status_path = cohort.parent / "ARTIFACT_STATUS.json"
    if not cohort.is_file() or not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("R5 source, manifest and artifact status are required")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    cohort_sha = digest(cohort)
    if (manifest.get("cohort_sha256") != cohort_sha
            or manifest.get("products") != expected_products
            or manifest.get("ranks_per_product") != 5
            or manifest.get("target_semantics") != "retrosynthetic_precursor_set"
            or status.get("cohort_sha256") != cohort_sha
            or status.get("evaluation_allowed") is not True
            or status.get("training_allowed") is not False
            or status.get("headline_allowed") is not False):
        raise ValueError("R5 source is not a hash-checked frozen precursor-set cohort")
    rows = [json.loads(line) for line in cohort.read_text().splitlines() if line.strip()]
    if len(rows) != expected_products or len({row["product_smiles"] for row in rows}) != len(rows):
        raise ValueError("R5 product denominator or uniqueness changed")
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if (row.get("model_input") != {"product_smiles": row["product_smiles"]}
                or row.get("inference_status") not in {"completed", "failed"}
                or [item.get("rank") for item in row.get("candidates", [])] != [1, 2, 3, 4, 5]):
            raise ValueError("R5 product input, inference status or Top-5 slots changed")
        for candidate in row["candidates"]:
            if (candidate.get("smiles_status") not in {"valid_smiles", "missing", "invalid_smiles"}
                    or candidate.get("recorded_reference_status") not in {
                        "recorded_reference", "not_recorded_not_proven_invalid", "unassessable"}):
                raise ValueError("R5 candidate validity/reference status is malformed")
            if (candidate["recorded_reference_status"] == "recorded_reference"
                    and candidate["smiles_status"] != "valid_smiles"):
                raise ValueError("R5 missing/invalid proposal cannot match a recorded reference")
        if (row["inference_status"] == "failed"
                and any(item["smiles_status"] == "valid_smiles" for item in row["candidates"])):
            raise ValueError("R5 failed product cannot carry a valid proposal")
        overlap = row.get("external_training_exact_product_overlap")
        overlap_key = ("overlap" if overlap is True else "disjoint" if overlap is False
                       else "unknown")
        strata.setdefault("training_exact_product_" + overlap_key, []).append(row)
        for field in ("reference_support_cohort", "heavy_atom_quartile"):
            value = row.get("strata", {}).get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"R5 source lacks {field} stratum")
            strata.setdefault(field + "/" + value, []).append(row)
    recorded_overlap = manifest.get("training_exact_product_overlap_count")
    if (recorded_overlap is not None
            and recorded_overlap != len(strata.get("training_exact_product_overlap", []))):
        raise ValueError("R5 external training-overlap count differs from frozen rows")
    report = {
        "artifact_type": "r5_external_source_diagnostic_v1",
        "source_cohort_sha256": cohort_sha,
        "source_manifest_sha256": digest(manifest_path),
        "external_model": manifest["model_name"],
        "checkpoint_identifier": manifest["checkpoint_identifier"],
        "ranking_semantics": manifest.get("ranking_semantics"),
        "overall": _summarize(rows),
        "strata": {name: _summarize(group) for name, group in sorted(strata.items())},
        "mechet_verification_performed": False,
        "headline_allowed": False,
        "claim_boundary": (
            "External-source diagnostic only. A recorded-reference miss is not chemical "
            "invalidity; exact-product disjointness is not broader train-set decontamination. "
            "No MechET candidate-conditioned trace or reranking was evaluated."),
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cohort, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
