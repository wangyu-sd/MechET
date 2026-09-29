#!/usr/bin/env python3
"""Audit exact product overlap between frozen R1 queries and FlowER train.

This is a provenance diagnostic for existing-model R1 results, not the
scientific-smoke train/evaluation exclusion audit (which uses selected rows).
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def audit(cohort: Path, cohort_manifest: Path, train: Path,
          official_manifest: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R1 train-overlap audit already exists: {output}")
    cohort_meta = json.loads(cohort_manifest.read_text())
    official_meta = json.loads(official_manifest.read_text())
    if digest(cohort) != cohort_meta.get("cohort_sha256"):
        raise ValueError("R1 cohort hash drifted")
    train_meta = official_meta["splits"]["train"]
    if digest(train) != train_meta.get("output_sha256"):
        raise ValueError("FlowER official train hash drifted")

    queries: dict[str, dict[str, Any]] = {}
    with cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product_key(product) != product or product in queries:
                raise ValueError("R1 cohort has noncanonical or duplicate products")
            queries[product] = row
    if len(queries) != int(cohort_meta["products"]):
        raise ValueError("R1 cohort denominator drifted")

    overlap: dict[str, list[str]] = {}
    train_rows = 0
    with train.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            train_rows += 1
            if product in queries:
                overlap.setdefault(product, []).append(str(row["id"]))
    if train_rows != int(train_meta["rows"]):
        raise ValueError("FlowER official train row count drifted")

    by_stratum: dict[str, Counter[str]] = {}
    for product, row in queries.items():
        for name, label in row["strata"].items():
            bucket = by_stratum.setdefault(name, Counter())
            bucket[f"{label}:total"] += 1
            bucket[f"{label}:overlap"] += int(product in overlap)
    report = {
        "artifact_type": "r1_existing_model_train_product_overlap_audit_v1",
        "r1_cohort_sha256": digest(cohort),
        "r1_manifest_sha256": digest(cohort_manifest),
        "official_train_sha256": digest(train),
        "official_manifest_sha256": digest(official_manifest),
        "official_train_rows": train_rows,
        "r1_products": len(queries),
        "overlapping_products": len(overlap),
        "nonoverlapping_products": len(queries) - len(overlap),
        "by_stratum": {name: dict(sorted(counts.items()))
                       for name, counts in sorted(by_stratum.items())},
        "overlapping_train_ids": dict(sorted(overlap.items())),
        "claim_boundary": (
            "Exact canonical product overlap with the full official FlowER train split; "
            "not a selected-scientific-smoke overlap audit, not a near-duplicate audit, "
            "and not a model-pretraining exposure audit."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.cohort, args.cohort_manifest, args.train,
                   args.official_manifest, args.output)
    print(json.dumps({key: report[key] for key in (
        "r1_products", "overlapping_products", "nonoverlapping_products")},
        indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
