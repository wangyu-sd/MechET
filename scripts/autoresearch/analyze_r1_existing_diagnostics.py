#!/usr/bin/env python3
"""Stratify frozen R1 *existing-model diagnostics*, not the paired smoke result."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


def _rows(path: Path, product_field: str = "product_smiles") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank R1 diagnostic row at {path}:{number}")
        row = json.loads(line)
        product = row.get(product_field)
        if not isinstance(product, str) or product in rows:
            raise ValueError(f"duplicate/invalid R1 product at {path}:{number}")
        rows[product] = row
    return rows


def _diagnostic(directory: Path, *, kind: str, cohort_sha256: str,
                expected_products: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = directory / "result.json"
    rows_path = directory / f"r1_existing_{kind}_rows.jsonl"
    if not report_path.is_file() or not rows_path.is_file():
        raise FileNotFoundError(f"R1 {kind} diagnostic files are required")
    report = json.loads(report_path.read_text())
    if (report.get("artifact_type") != f"r1_existing_{kind}_diagnostic_v1"
            or report.get("status") != "diagnostic_only_not_scientific_smoke"
            or report.get("r1_cohort_sha256") != cohort_sha256
            or report.get("rows_sha256") != digest(rows_path)
            or report.get("products") != expected_products):
        raise ValueError(f"R1 {kind} diagnostic provenance/hash mismatch")
    rows = _rows(rows_path)
    if len(rows) != expected_products:
        raise ValueError(f"R1 {kind} diagnostic denominator changed")
    return rows, {"result_sha256": digest(report_path), "rows_sha256": digest(rows_path),
                  "prediction_sha256": report["existing_predictions_sha256"]}


def _rate(count: int, denominator: int) -> float:
    return count / denominator if denominator else 0.0


def _aggregate(products: list[str], direct: dict[str, dict[str, Any]],
               trace: dict[str, dict[str, Any]]) -> dict[str, Any]:
    count: Counter[str] = Counter()
    for product in products:
        d = direct[product]["metrics"]
        t = trace[product]
        for label in ("single_reference_at_1", "multi_reference_at_1",
                      "single_reference_at_k", "multi_reference_at_k"):
            count[label] += d[label] is True
        count["alternative_recovered_at_1"] += (d["multi_reference_at_1"] is True
                                                 and d["single_reference_at_1"] is False)
        count["alternative_recovered_at_k"] += (d["multi_reference_at_k"] is True
                                                 and d["single_reference_at_k"] is False)
        count["trace_formal_at_1"] += t["formal_at_1"] is True
        count["trace_formal_at_k"] += t["formal_at_k"] is True
        count["trace_recorded_at_1"] += t["recorded_hit_at_1"] is True
        count["trace_recorded_at_k"] += t["recorded_hit_at_k"] is True
    n = len(products)
    return {"products": n, "counts": dict(sorted(count.items())),
            "rates": {key: _rate(value, n) for key, value in sorted(count.items())},
            "alternative_fraction_among_multi_reference_hits_at_1":
                _rate(count["alternative_recovered_at_1"], count["multi_reference_at_1"]),
            "alternative_fraction_among_multi_reference_hits_at_k":
                _rate(count["alternative_recovered_at_k"], count["multi_reference_at_k"])}


def _bootstrap(products: list[str], direct: dict[str, dict[str, Any]],
               *, replicates: int, seed: int) -> dict[str, list[float]]:
    if replicates < 100:
        raise ValueError("R1 bootstrap needs at least 100 replicates")
    rng = random.Random(seed)
    values: dict[str, list[float]] = {"at_1": [], "at_k": []}
    n = len(products)
    for _ in range(replicates):
        sample = [rng.choice(products) for _ in products]
        for suffix, metric in (("at_1", "1"), ("at_k", "k")):
            values[suffix].append(sum(
                direct[p]["metrics"][f"multi_reference_at_{metric}"] is True
                and direct[p]["metrics"][f"single_reference_at_{metric}"] is False
                for p in sample) / n)
    return {key: [sorted(scores)[int(0.025 * (replicates - 1))],
                  sorted(scores)[int(0.975 * (replicates - 1))]]
            for key, scores in values.items()}


def analyze(cohort: Path, direct_dir: Path, trace_dir: Path, output: Path,
            *, expected_products: int = 220, bootstrap_replicates: int = 2000,
            seed: int = 17) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R1 stratified diagnostic already exists: {output}")
    cohort_hash = verify_evaluation_source(cohort, name="r1_multi_reference")
    references = _rows(cohort)
    if len(references) != expected_products:
        raise ValueError("R1 multi-reference product denominator changed")
    direct, direct_prov = _diagnostic(direct_dir, kind="direct",
                                      cohort_sha256=cohort_hash,
                                      expected_products=expected_products)
    trace, trace_prov = _diagnostic(trace_dir, kind="trace",
                                    cohort_sha256=cohort_hash,
                                    expected_products=expected_products)
    if set(references) != set(direct) or set(references) != set(trace):
        raise ValueError("R1 diagnostic products do not match the frozen cohort")
    for product, row in references.items():
        if direct[product]["recorded_reference_count"] != row["reference_count"]:
            raise ValueError("R1 diagnostic reference count differs from cohort")
        if direct[product]["metrics"]["candidate_count"] != 10:
            raise ValueError("R1 Direct diagnostic candidate budget changed")
        if trace[product]["candidate_count"] != 10:
            raise ValueError("R1 trace diagnostic candidate budget changed")
    products = sorted(references)
    strata: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for product, row in references.items():
        for dimension in ("reference_count", "disconnection", "structural_overlap"):
            strata[dimension][str(row["strata"][dimension])].append(product)
    overall = _aggregate(products, direct, trace)
    result = {
        "artifact_type": "r1_existing_models_stratified_diagnostic_v1",
        "status": "diagnostic_only_not_scientific_smoke",
        "r1_cohort_sha256": cohort_hash,
        "direct_provenance": direct_prov, "trace_provenance": trace_prov,
        "products": expected_products, "candidate_budget": 10,
        "overall": overall,
        "by_stratum": {dimension: {label: _aggregate(group, direct, trace)
                                   for label, group in sorted(groups.items())}
                       for dimension, groups in sorted(strata.items())},
        "product_bootstrap_95pct_interval_alternative_recovered_fraction":
            _bootstrap(products, direct, replicates=bootstrap_replicates, seed=seed),
        "bootstrap_replicates": bootstrap_replicates, "bootstrap_seed": seed,
        "claim_boundary": "Existing unmatched FlowER-trained Direct and legacy MechET models, not PR69 paired Base/Mech. The 220 products were selected for multiple independently recorded references; alternative recorded routes are not independently validated chemistry. The confidence interval is conditional on this selected cohort, not a population-wide test estimate.",
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cohort, args.direct_dir, args.trace_dir,
                             args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
