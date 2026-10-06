#!/usr/bin/env python3
"""Audit MechET product-only inference on the unfiltered FlowER endpoint test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def validate_source(
    *, source: Path, manifest: Path, expected_rows: int = 28971,
    check_product_only_mapping: bool = False,
) -> tuple[dict[str, str], str]:
    """Bind the unfiltered endpoint test bytes and all source reaction IDs."""
    if check_product_only_mapping:
        from scripts.run_natural_language_value_search import (
            product_only_private_state, visible,
        )
    source_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    test_manifest = source_manifest["splits"]["test"]
    if (int(test_manifest["rows"]) != expected_rows or
            int(test_manifest["expected_rows"]) != expected_rows):
        raise ValueError("full endpoint test denominator does not match expectation")
    source_sha = _sha256(source)
    if source_sha != test_manifest["output_sha256"]:
        raise ValueError("full endpoint test source SHA-256 mismatch")

    source_ids: dict[str, str] = {}
    for row in _jsonl(source):
        identifier = str(row["id"])
        if identifier in source_ids:
            raise ValueError(f"duplicate full endpoint test ID: {identifier}")
        if (row.get("metadata") or {}).get("coverage_track") != "full_endpoint":
            raise ValueError(f"not a full endpoint test row: {identifier}")
        if not row.get("structural_precursor"):
            raise ValueError(f"missing structural precursor reference: {identifier}")
        if check_product_only_mapping:
            try:
                visible(product_only_private_state(str(row["target_smiles"])))
            except Exception as error:
                raise ValueError(
                    f"product-only private mapping failed for {identifier}"
                ) from error
        source_ids[identifier] = str(row["source_id"])
    if len(source_ids) != expected_rows:
        raise ValueError("source rows do not cover the full endpoint test")
    return source_ids, source_sha


def audit(
    *, source: Path, manifest: Path, results_dir: Path,
    expected_rows: int = 28971,
) -> dict[str, Any]:
    source_ids, source_sha = validate_source(
        source=source, manifest=manifest, expected_rows=expected_rows,
    )

    shard_paths = sorted(results_dir.glob("results.shard-*.jsonl"))
    if not shard_paths:
        raise ValueError("no inference result shards")
    observed: set[str] = set()
    exact = terminal = full_exact = 0
    for shard in shard_paths:
        for row in _jsonl(shard):
            identifier = str(row["id"])
            if identifier not in source_ids:
                raise ValueError(f"prediction outside frozen test: {identifier}")
            if identifier in observed:
                raise ValueError(f"duplicate prediction: {identifier}")
            if row.get("endpoint_metric") != "structural":
                raise ValueError(f"wrong endpoint metric for {identifier}")
            if str(row.get("source_id")) != source_ids[identifier]:
                raise ValueError(f"prediction source ID mismatch: {identifier}")
            observed.add(identifier)
            exact += bool(row["top1_exact"])
            terminal += bool(row["top_terminal"])
            full_exact += bool(row["top1_full_exact"])

    return {
        "artifact_type": "reliable_mechet_full_endpoint_test_audit_v1",
        "source": str(source),
        "source_sha256": source_sha,
        "manifest": str(manifest),
        "test_denominator": expected_rows,
        "observed_predictions": len(observed),
        "missing_predictions": expected_rows - len(observed),
        "top1_structural_exact": exact,
        "top1_structural_accuracy": exact / expected_rows,
        "top1_full_state_exact_secondary": full_exact,
        "terminal": terminal,
        "shards": [{"path": str(path), "sha256": _sha256(path)} for path in shard_paths],
        "note": (
            "The unfiltered 28,971-reaction endpoint denominator includes four "
            "upstream-corrupt reactions. Missing predictions count as failures. "
            "Mechanism/process metrics require the separate 28,967-reaction strict view."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=28971)
    args = parser.parse_args()
    report = audit(
        source=args.source, manifest=args.manifest,
        results_dir=args.results_dir, expected_rows=args.expected_rows,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
