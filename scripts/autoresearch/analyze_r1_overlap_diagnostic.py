#!/usr/bin/env python3
"""Split an existing-model R1 diagnostic by exact official-train product overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest


def analyze(audit_path: Path, diagnostic_path: Path, rows_path: Path,
            output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R1 overlap diagnostic already exists: {output}")
    audit = json.loads(audit_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())
    if (audit.get("artifact_type") != "r1_existing_model_train_product_overlap_audit_v1"
            or diagnostic.get("artifact_type") != "r1_existing_direct_diagnostic_v1"
            or audit.get("r1_cohort_sha256") != diagnostic.get("r1_cohort_sha256")
            or digest(rows_path) != diagnostic.get("rows_sha256")
            or audit.get("r1_products") != diagnostic.get("products")):
        raise ValueError("R1 overlap/diagnostic provenance mismatch")
    overlaps = set(audit["overlapping_train_ids"])
    buckets: dict[str, dict[str, int]] = {
        "exact_product_disjoint": {"products": 0, "recovered_at_1": 0,
                                   "recovered_at_10": 0, "multi_reference_at_1": 0,
                                   "multi_reference_at_10": 0},
        "exact_product_overlap": {"products": 0, "recovered_at_1": 0,
                                  "recovered_at_10": 0, "multi_reference_at_1": 0,
                                  "multi_reference_at_10": 0},
    }
    seen: set[str] = set()
    with rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product in seen:
                raise ValueError("duplicate R1 diagnostic product")
            seen.add(product)
            metrics = row["metrics"]
            bucket = buckets["exact_product_overlap" if product in overlaps
                             else "exact_product_disjoint"]
            bucket["products"] += 1
            bucket["recovered_at_1"] += int(metrics["multi_reference_at_1"]
                                            and not metrics["single_reference_at_1"])
            bucket["recovered_at_10"] += int(metrics["multi_reference_at_k"]
                                             and not metrics["single_reference_at_k"])
            bucket["multi_reference_at_1"] += int(metrics["multi_reference_at_1"])
            bucket["multi_reference_at_10"] += int(metrics["multi_reference_at_k"])
    if (len(seen) != audit["r1_products"]
            or buckets["exact_product_overlap"]["products"] != len(overlaps)
            or sum(bucket["recovered_at_1"] for bucket in buckets.values())
            != diagnostic["single_reference_false_negatives_recovered_at_1"]
            or sum(bucket["recovered_at_10"] for bucket in buckets.values())
            != diagnostic["single_reference_false_negatives_recovered_at_k"]):
        raise ValueError("R1 overlap partition does not reconstruct frozen totals")
    report = {
        "artifact_type": "r1_existing_direct_by_train_product_overlap_v1",
        "r1_cohort_sha256": audit["r1_cohort_sha256"],
        "overlap_audit_sha256": digest(audit_path),
        "diagnostic_result_sha256": digest(diagnostic_path),
        "diagnostic_rows_sha256": digest(rows_path),
        "by_overlap": buckets,
        "claim_boundary": (
            "Existing full-FlowER-trained Direct model, not a held-out scientific-smoke "
            "comparison. Exact-product-disjoint does not establish reaction-disjointness, "
            "chemical novelty, or absence of backbone pretraining exposure."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--diagnostic-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.overlap_audit, args.diagnostic_result,
                             args.diagnostic_rows, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
