#!/usr/bin/env python3
"""Audit the frozen HF endpoint target against the complete final mixture.

This does not relabel or rewrite the existing benchmark. The alternative
``rxn_prod_equ`` selection remains a deterministic proxy, not an assertion
that the largest component is always the experimentally desired product.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.build_mech_uspto31k_rxnmapper_baseline import (
    EXPECTED, PARQUET_SPLITS, canonical_unmapped, select_main_product,
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_split(raw_path: Path, endpoint_path: Path, *, split: str) -> dict:
    frame = pd.read_parquet(raw_path, columns=[
        "rxn_idx", "step_idx_forward", "rxn_prod_min", "rxn_prod_equ"
    ])
    endpoints = {str(row["source_id"]): row for row in _rows(endpoint_path)}
    if len(endpoints) != EXPECTED[split]:
        raise ValueError(f"{split}: endpoint count/ID uniqueness mismatch")
    counts: Counter[str] = Counter()
    min_disagreement: Counter[str] = Counter()
    changed_ids: list[str] = []
    examples: list[dict] = []
    seen: set[str] = set()
    for rxn_idx, group in frame.groupby("rxn_idx", sort=True):
        reaction_id = str(int(rxn_idx))
        if reaction_id not in endpoints or reaction_id in seen:
            raise ValueError(f"{split}: unmatched or repeated raw reaction {reaction_id}")
        seen.add(reaction_id)
        group = group.sort_values("step_idx_forward", kind="stable")
        if int(group.iloc[0]["step_idx_forward"]) != 0:
            raise ValueError(f"{split}/{reaction_id}: no forward step zero")
        min_mixtures = {canonical_unmapped(value) for value in group["rxn_prod_min"]}
        equ_mixtures = {canonical_unmapped(value) for value in group["rxn_prod_equ"]}
        if len(min_mixtures) != 1 or len(equ_mixtures) != 1:
            raise ValueError(f"{split}/{reaction_id}: non-invariant endpoint fields")
        min_mixture = next(iter(min_mixtures))
        equ_mixture = next(iter(equ_mixtures))
        min_main = select_main_product(min_mixture)
        equ_main = select_main_product(equ_mixture)
        if min_main != endpoints[reaction_id]["product_unmapped"]:
            raise ValueError(f"{split}/{reaction_id}: existing product is not min-main")
        changed = min_main != equ_main
        counts["reactions"] += 1
        counts["main_product_changed"] += int(changed)
        counts["equ_has_multiple_components"] += int("." in equ_mixture)
        counts["min_has_multiple_components"] += int("." in min_mixture)
        equ_components = equ_mixture.split(".")
        counts["min_main_present_in_equ"] += int(min_main in equ_components)
        if changed:
            counts["changed_min_still_in_equ"] += int(min_main in equ_components)
            min_disagreement[min_main] += 1
            changed_ids.append(reaction_id)
            if len(examples) < 16:
                examples.append({
                    "reaction_id": reaction_id,
                    "existing_min_main": min_main,
                    "equ_main": equ_main,
                    "min_mixture": min_mixture,
                    "equ_mixture": equ_mixture,
                    "min_main_present_in_equ": min_main in equ_components,
                })
    if seen != set(endpoints) or len(seen) != EXPECTED[split]:
        raise ValueError(f"{split}: raw and endpoint reaction coverage differ")
    return {
        "split": split,
        "raw_sha256": sha256(raw_path),
        "endpoint_sha256": sha256(endpoint_path),
        "counts": dict(counts),
        "changed_reaction_ids": changed_ids,
        "top_existing_min_main_when_changed": min_disagreement.most_common(12),
        "examples": examples,
    }


def audit(raw_dir: Path, endpoint_dir: Path) -> dict:
    manifest = json.loads((endpoint_dir / "manifest.json").read_text())
    if (manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or manifest["executor_filtering"] is not False):
        raise ValueError("existing endpoint benchmark contract mismatch")
    splits = {}
    for split, stem in PARQUET_SPLITS.items():
        raw_path = raw_dir / f"{stem}-00000-of-00001.parquet"
        endpoint_path = endpoint_dir / f"{split}.jsonl"
        if (sha256(raw_path) != manifest["source_files"][split]["sha256"]
                or sha256(endpoint_path) != manifest["splits"][split]["endpoint_sha256"]
                or manifest["splits"][split]["rows"] != EXPECTED[split]):
            raise ValueError(f"{split}: frozen source hash/count mismatch")
        splits[split] = audit_split(raw_path, endpoint_path, split=split)
    return {
        "artifact_type": "mech_uspto31k_existing_min_target_vs_equ_final_mixture_audit",
        "selection_policy": "largest_organic_component_in_each_field",
        "interpretation_limit": "equ-main is an alternate proxy, not a gold desired-product label",
        "endpoint_manifest_sha256": sha256(endpoint_dir / "manifest.json"),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.raw_dir, args.endpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({split: row["counts"] for split, row in result["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
