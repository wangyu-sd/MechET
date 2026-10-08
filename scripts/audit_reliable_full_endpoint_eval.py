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


def validate_adapter_identity(
    *, adapter: Path | None, expected_sha256: str | None, stage: str | None,
) -> dict[str, str] | None:
    """Bind a formal result to the same frozen weights checked before inference."""

    if adapter is None and expected_sha256 is None and stage is None:
        return None
    if adapter is None or expected_sha256 is None or stage not in {"state", "trajectory"}:
        raise ValueError("formal evaluation requires adapter, expected SHA-256 and stage")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected adapter SHA-256 is malformed")
    from scripts.run_natural_language_value_search import validate_v2_adapter_manifest

    metadata = validate_v2_adapter_manifest(
        adapter, compact_history=stage == "trajectory",
        expected_model="Qwen/Qwen3-0.6B",
        expected_revision="c1899de289a04d12100db370d81485cdf75e47ca",
    )
    weights = adapter / "adapter_model.safetensors"
    actual_sha256 = _sha256(weights)
    if actual_sha256 != expected_sha256:
        raise ValueError("adapter weights changed since inference began")
    return {
        "stage": stage,
        "path": str(adapter.resolve()),
        "adapter_model_sha256": actual_sha256,
        "adapter_manifest_sha256": _sha256(adapter / "adapter_manifest.json"),
        "base_model": str(metadata["base_model"]),
        "base_model_revision": str(metadata["base_model_revision"]),
        "environment_revision": str(metadata["environment_revision"]),
    }


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
    adapter: Path | None = None, expected_adapter_sha256: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    source_ids, source_sha = validate_source(
        source=source, manifest=manifest, expected_rows=expected_rows,
    )
    adapter_identity = validate_adapter_identity(
        adapter=adapter, expected_sha256=expected_adapter_sha256, stage=stage,
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
        "artifact_type": (
            "reliable_mechet_full_endpoint_test_audit_v1"
            if adapter_identity is not None
            else "reliable_mechet_full_endpoint_test_unbound_diagnostic_v1"
        ),
        "source": str(source),
        "source_sha256": source_sha,
        "adapter_identity": adapter_identity,
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
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--stage", choices=("state", "trajectory"))
    args = parser.parse_args()
    if args.adapter is None or args.expected_adapter_sha256 is None or args.stage is None:
        parser.error("formal full-endpoint audit requires --adapter, --expected-adapter-sha256 and --stage")
    report = audit(
        source=args.source, manifest=args.manifest,
        results_dir=args.results_dir, expected_rows=args.expected_rows,
        adapter=args.adapter, expected_adapter_sha256=args.expected_adapter_sha256,
        stage=args.stage,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
