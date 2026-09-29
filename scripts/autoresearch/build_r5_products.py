#!/usr/bin/env python3
"""Freeze an external-model-independent, 200-product PR #69 R5 query cohort."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def heavy_atoms(smiles: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid R5 product: {smiles}")
    return int(molecule.GetNumHeavyAtoms())


def balanced_products(products: list[str], *, count: int, seed: int,
                      cohort: str) -> list[tuple[str, str]]:
    """Select equally from empirical size quartiles before model inference."""
    if count % 4:
        raise ValueError("R5 product count per cohort must divide into four quartiles")
    if len(products) < count:
        raise ValueError(f"R5 {cohort} has only {len(products)} unique products")
    ranked = sorted(products, key=lambda product: (heavy_atoms(product), product))
    selected: list[tuple[str, str]] = []
    for quartile in range(4):
        part = ranked[len(ranked) * quartile // 4:len(ranked) * (quartile + 1) // 4]
        if len(part) < count // 4:
            raise ValueError(f"R5 {cohort} quartile {quartile + 1} is underfilled")
        chosen = sorted(part, key=lambda product: (
            hashlib.sha256(f"{seed}:{cohort}:{product}".encode()).hexdigest(),
            product))[:count // 4]
        selected.extend((product, f"Q{quartile + 1}") for product in chosen)
    return selected


def build(source: Path, official_manifest: Path, r1_cohort: Path,
          r1_manifest: Path, output: Path, *, seed: int = 17,
          per_cohort: int = 100) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 product cohort already frozen: {output}")
    official = json.loads(official_manifest.read_text())
    source_hash = digest(source)
    if source_hash != official["splits"]["test"]["output_sha256"]:
        raise ValueError("R5 source differs from official full-endpoint test manifest")
    r1_metadata = json.loads(r1_manifest.read_text())
    r1_hash = digest(r1_cohort)
    if r1_hash != r1_metadata["cohort_sha256"]:
        raise ValueError("R5 R1 multi-reference source hash drifted")
    r1_products = set()
    with r1_cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product_key(product) != product:
                raise ValueError("R1 product is not canonical")
            if int(row["reference_count"]) < 2:
                raise ValueError("R1 product is not multi-reference")
            if product in r1_products:
                raise ValueError("R1 cohort has duplicate products")
            r1_products.add(product)
    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            precursor = product_key(str(row["structural_precursor"]))
            groups[product][precursor].add(str(row.get("source_id") or row["id"]))
    if not r1_products <= groups.keys():
        raise ValueError("R1 products are missing from official R5 source")
    for product in r1_products:
        if len(groups[product]) < 2:
            raise ValueError("R1 multi-reference product lost distinct endpoints")
    multi = balanced_products(sorted(r1_products), count=per_cohort, seed=seed,
                              cohort="multi_recorded")
    ordinary = balanced_products(sorted(groups.keys() - r1_products),
                                 count=per_cohort, seed=seed,
                                 cohort="other_official_test")
    records = []
    for cohort_name, selected in (("multi_recorded", multi),
                                  ("other_official_test", ordinary)):
        for product, quartile in selected:
            references = [{"precursor_smiles": precursor,
                           "support_record_ids": sorted(ids)}
                          for precursor, ids in sorted(groups[product].items())]
            records.append({
                "artifact_type": "r5_frozen_product_query_v1",
                "product_smiles": product,
                "source_dataset": "FlowER official full endpoint",
                "source_split": "test",
                "model_input": {"product_smiles": product},
                "strata": {"reference_support_cohort": cohort_name,
                           "heavy_atom_quartile": quartile,
                           "reaction_family": "unavailable"},
                "private_reference": {"recorded_precursor_sets": references,
                                      "reference_count": len(references)},
            })
    records.sort(key=lambda item: item["product_smiles"])
    output.mkdir(parents=True)
    cohort = output / "r5_products.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r5_frozen_product_query_manifest_v1",
        "source": str(source), "source_sha256": source_hash,
        "official_manifest": str(official_manifest),
        "official_manifest_sha256": digest(official_manifest),
        "r1_cohort": str(r1_cohort), "r1_cohort_sha256": r1_hash,
        "r1_manifest_sha256": digest(r1_manifest),
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "products": len(records), "seed": seed,
        "multi_recorded": len(multi), "other_official_test": len(ordinary),
        "quartiles": {name: sum(row["strata"]["heavy_atom_quartile"] == name
                                for row in records) for name in ("Q1", "Q2", "Q3", "Q4")},
        "external_predictions_present": False,
        "claim_boundary": "Product query cohort only; no RetroChimera predictions or MechET verifier scores yet.",
        "training_overlap_caveat": "External checkpoint training-universe overlap must be audited before headline comparisons.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--r1-cohort", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-cohort", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.official_manifest,
                           args.r1_cohort, args.r1_manifest, args.output,
                           seed=args.seed, per_cohort=args.per_cohort),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
