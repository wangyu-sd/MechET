#!/usr/bin/env python3
"""Freeze label-free H2 reaction covariates before any model comparison.

Test covariates may be audited for coverage, but test endpoint labels/results
must not be used to choose a checkpoint or alter the frozen split.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.materialize_nmi_matched_datasets import _split_ids


def build_covariates(
    source: Path, split_dir: Path, structural_audit: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing covariates: {output}")
    assignments, split_manifest = _split_ids(split_dir)
    audit = json.loads(structural_audit.read_text())
    split_sha = hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest()
    if audit.get("scope") != "full_frozen_split" or audit.get("artifact_type") != "nmi_mechcomp_structural_overlap_v2" or audit.get("split_manifest_sha256") != split_sha:
        raise ValueError("covariates require matching full structural audit")
    if audit.get("reaction_center_undefined_count"):
        raise ValueError("undefined reaction center in structural audit")
    if not all(audit.get("gates", {}).values()):
        raise ValueError("structural exact-reaction audit did not pass")

    primitive_train_frequency: Counter[str] = Counter()
    composition_train_frequency: Counter[str] = Counter()
    heldout: dict[str, dict[str, Any]] = {}
    source_digest = hashlib.sha256()
    scanned = 0
    for line in source.open("rb"):
        source_digest.update(line)
        row = json.loads(line)
        identifier = str(row.get("source_id") or "")
        split = assignments.get(identifier)
        if split is None:
            raise ValueError(f"source ID absent from frozen split: {identifier}")
        metadata = row.get("metadata") or {}
        composition = metadata.get("execution_composition_signature")
        primitives = metadata.get("execution_primitive_signatures")
        if not isinstance(composition, str) or len(composition) != 64:
            raise ValueError(f"invalid composition signature: {identifier}")
        if not isinstance(primitives, list) or not primitives:
            raise ValueError(f"invalid primitive signatures: {identifier}")
        if split == "train":
            composition_train_frequency[composition] += 1
            primitive_train_frequency.update(set(primitives))
        else:
            if identifier in heldout:
                raise ValueError(f"duplicate held-out ID: {identifier}")
            heldout[identifier] = {
                "split": split,
                "program": composition,
                "primitives": tuple(sorted(set(primitives))),
                "trajectory_steps": int(metadata.get("n_trace_steps") or 0),
                "electron_moves": int(metadata.get("n_trace_moves") or 0),
                "fragment_imports": int(metadata.get("n_trace_imports") or 0),
                "be_delta_steps": int(metadata.get("n_be_delta_steps") or 0),
            }
        scanned += 1
    if scanned != split_manifest["n_rows"] or source_digest.hexdigest() != split_manifest["source_sha256"]:
        raise ValueError("covariate source row-count/SHA mismatch")
    if len(heldout) != split_manifest["rows"]["valid"] + split_manifest["rows"]["test"]:
        raise ValueError("held-out covariate coverage mismatch")
    annotations = {
        split: audit["split"][split]["annotations"] for split in ("valid", "test")
    }
    if any(len(annotations[split]) != split_manifest["rows"][split] for split in annotations):
        raise ValueError("structural annotation coverage mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    with output.open("wb") as handle:
        for identifier in sorted(heldout):
            item = heldout[identifier]
            split = item["split"]
            structural = annotations[split].get(identifier)
            if structural is None:
                raise ValueError(f"missing structural annotation: {identifier}")
            frequencies = [primitive_train_frequency[value] for value in item["primitives"]]
            row = {
                "source_id": identifier,
                "split": split,
                "program_train_frequency": composition_train_frequency[item["program"]],
                "minimum_primitive_train_frequency": min(frequencies),
                "mean_primitive_train_frequency": sum(frequencies) / len(frequencies),
                "fraction_primitives_seen_in_train": sum(value > 0 for value in frequencies) / len(frequencies),
                "unique_primitive_count": len(frequencies),
                "trajectory_steps": item["trajectory_steps"],
                "electron_moves": item["electron_moves"],
                "fragment_imports": item["fragment_imports"],
                "be_delta_steps": item["be_delta_steps"],
                "structural_overlap": structural,
            }
            encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            output_digest.update(encoded)
            counts[split] += 1
    report = {
        "artifact_type": "nmi_h2_reaction_covariates_v1",
        "source_sha256": source_digest.hexdigest(),
        "split_manifest_sha256": split_sha,
        "structural_audit_sha256": hashlib.sha256(structural_audit.read_bytes()).hexdigest(),
        "rows": dict(counts),
        "covariates_sha256": output_digest.hexdigest(),
        "covariates_file": str(output.resolve()),
        "label_policy": "No endpoint labels or model outcomes read; test not for model selection",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--structural-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_covariates(
        args.source, args.split_dir, args.structural_audit, args.output,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
