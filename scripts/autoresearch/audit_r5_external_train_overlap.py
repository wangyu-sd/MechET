#!/usr/bin/env python3
"""Audit exact product overlap of R5 queries with an external model's train file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def audit(query: Path, query_manifest: Path, external_train: Path,
          output: Path, *, expected_train_sha256: str,
          expected_train_rows: int = 257171) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 external training-overlap audit already frozen: {output}")
    query_meta = json.loads(query_manifest.read_text())
    if digest(query) != query_meta["cohort_sha256"]:
        raise ValueError("R5 query hash differs from its frozen manifest")
    if digest(external_train) != expected_train_sha256:
        raise ValueError("external model training source SHA-256 changed")
    wanted = set()
    with query.open(encoding="utf-8") as stream:
        for line in stream:
            product = str(json.loads(line)["product_smiles"])
            if product_key(product) != product or product in wanted:
                raise ValueError("R5 query products must be unique and canonical")
            wanted.add(product)
    if len(wanted) != query_meta["products"]:
        raise ValueError("R5 query denominator changed")

    overlap: dict[str, list[int]] = {}
    train_rows = 0
    with external_train.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.count(">>") != 1:
                raise ValueError(f"external train reaction line {line_no} has no unique >>")
            product = product_key(line.split(">>", 1)[0])
            train_rows += 1
            if product in wanted:
                overlap.setdefault(product, []).append(line_no)
    if train_rows != expected_train_rows:
        raise ValueError(f"external train has {train_rows}, expected {expected_train_rows} rows")
    report = {
        "artifact_type": "r5_external_model_train_product_overlap_audit_v1",
        "query": str(query), "query_sha256": digest(query),
        "query_manifest_sha256": digest(query_manifest),
        "external_train": str(external_train),
        "external_train_sha256": expected_train_sha256,
        "external_train_rows": train_rows,
        "query_products": len(wanted),
        "overlapping_query_products": len(overlap),
        "overlapping_products_and_train_line_numbers": dict(sorted(overlap.items())),
        "leakage_clean_for_exact_product_overlap": not overlap,
        "claim_boundary": "Exact canonical product overlap only; does not prove absence of near-duplicate reactions, pretraining exposure, or chemistry memorization.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--external-train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.query, args.query_manifest, args.external_train,
                   args.output, expected_train_sha256=args.expected_train_sha256)
    print(json.dumps({key: report[key] for key in (
        "artifact_type", "external_train_rows", "query_products",
        "overlapping_query_products", "leakage_clean_for_exact_product_overlap")},
        indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
