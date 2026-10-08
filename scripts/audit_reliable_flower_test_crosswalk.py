#!/usr/bin/env python3
"""Compare frozen full-endpoint and strict-proof FlowER test labels by reaction ID."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from mechet.proof_program import _canonical_unmapped, sides_equal
from scripts.run_natural_language_value_search import normal_smiles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _fragments(smiles: str) -> Counter[str]:
    return Counter(
        _canonical_unmapped(part) for part in str(smiles).split(".") if part
    )


def compare(
    *, full_file: Path, full_manifest: Path,
    strict_file: Path, strict_manifest: Path,
    expected_full: int = 28971, expected_strict: int = 28967,
) -> dict[str, Any]:
    full_meta = json.loads(full_manifest.read_text(encoding="utf-8"))["splits"]["test"]
    strict_meta = json.loads(strict_manifest.read_text(encoding="utf-8"))["splits"]["test"]
    if int(full_meta["rows"]) != expected_full or int(strict_meta["rows"]) != expected_strict:
        raise ValueError("test denominators disagree with frozen manifests")
    full_sha, strict_sha = _sha256(full_file), _sha256(strict_file)
    if full_sha != full_meta["output_sha256"] or strict_sha != strict_meta["sha256"]:
        raise ValueError("test bytes disagree with frozen manifests")

    full: dict[str, dict[str, Any]] = {}
    for row in _rows(full_file):
        key = str(row["source_id"])
        if key in full:
            raise ValueError(f"duplicate full-endpoint source ID: {key}")
        full[key] = row
    if len(full) != expected_full:
        raise ValueError("full-endpoint test file is incomplete")

    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for row in _rows(strict_file):
        source = str(row["source_id"])
        prefix = "flower_mech_proof_test_"
        if not source.startswith(prefix):
            raise ValueError(f"unexpected strict-proof ID: {source}")
        key = source.removeprefix(prefix)
        if key in seen:
            raise ValueError(f"duplicate strict-proof source ID: {key}")
        seen.add(key)
        if key not in full:
            raise ValueError(f"strict-proof ID absent from full endpoint test: {key}")
        reference = full[key]
        product_equal = sides_equal(row["target_smiles"], reference["target_smiles"])
        input_equal = (
            normal_smiles(row["target_smiles"])
            == normal_smiles(reference["target_smiles"])
        )
        if input_equal:
            counts["same_product_only_model_input"] += 1
        elif product_equal:
            counts["chemically_equal_but_different_model_input"] += 1
            if len(examples.get("chemically_equal_but_different_model_input", [])) < 8:
                examples.setdefault("chemically_equal_but_different_model_input", []).append({
                    "reaction_id": key,
                    "strict_product": row["target_smiles"],
                    "full_product": reference["target_smiles"],
                })
        if not product_equal:
            counts["product_mismatch"] += 1
            if len(examples.get("product_mismatch", [])) < 8:
                examples.setdefault("product_mismatch", []).append({
                    "reaction_id": key, "strict_product": row["target_smiles"],
                    "full_product": reference["target_smiles"],
                })
        strict_fragments = _fragments(row["structural_precursor"])
        full_fragments = _fragments(reference["structural_precursor"])
        if strict_fragments == full_fragments:
            category = "same_structural_precursor"
        elif not full_fragments - strict_fragments:
            category = "strict_has_extra_structural_fragments"
        elif not strict_fragments - full_fragments:
            category = "full_has_extra_structural_fragments"
        else:
            category = "different_structural_fragments"
        counts[category] += 1
        if category != "same_structural_precursor" and len(examples.get(category, [])) < 8:
            examples.setdefault(category, []).append({
                "reaction_id": key,
                "strict_only": sorted((strict_fragments - full_fragments).elements()),
                "full_only": sorted((full_fragments - strict_fragments).elements()),
            })
    if len(seen) != expected_strict:
        raise ValueError("strict-proof test file is incomplete")
    full_only = sorted(set(full) - seen, key=lambda key: (not key.isdecimal(), int(key) if key.isdecimal() else key))
    if len(full_only) != expected_full - expected_strict:
        raise ValueError("unexpected full/strict reaction-ID difference")
    if sum(counts[key] for key in (
        "same_structural_precursor", "strict_has_extra_structural_fragments",
        "full_has_extra_structural_fragments", "different_structural_fragments",
    )) != expected_strict:
        raise ValueError("structural comparison did not cover every strict-proof row")
    if (counts["same_product_only_model_input"]
            + counts["chemically_equal_but_different_model_input"]
            + counts["product_mismatch"] != expected_strict):
        raise ValueError("model-input comparison did not cover every strict-proof row")
    return {
        "artifact_type": "reliable_mechet_flower_full_strict_test_crosswalk_v1",
        "full_test_rows": expected_full,
        "strict_test_rows": expected_strict,
        "full_test_sha256": full_sha,
        "strict_test_sha256": strict_sha,
        "full_only_reaction_ids": full_only,
        "counts": dict(counts),
        "examples": examples,
        "interpretation": (
            "Matching reaction IDs do not imply matching product-only model input "
            "or structural-precursor label. Reuse a full-endpoint inference trajectory "
            "for the strict view only when normalized model inputs match exactly, "
            "then score against the strict reference label."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-file", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--strict-file", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        full_file=args.full_file, full_manifest=args.full_manifest,
        strict_file=args.strict_file, strict_manifest=args.strict_manifest,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "full_test_rows": report["full_test_rows"],
        "strict_test_rows": report["strict_test_rows"],
        "counts": report["counts"],
        "full_only_reaction_ids": report["full_only_reaction_ids"],
        "report": str(args.output),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
