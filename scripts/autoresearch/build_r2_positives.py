#!/usr/bin/env python3
"""Freeze recorded R2 positives without inventing chemically invalid negatives."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.build_r5_products import balanced_products
from scripts.autoresearch.stratified_manifest import digest, product_key


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build(source: Path, official_manifest: Path, r1_cohort: Path,
          r1_manifest: Path, output: Path, *, seed: int = 17,
          per_stratum: int = 200) -> dict[str, Any]:
    """Select 200 recorded and 200 independently documented alternative routes.

    These are *positive proposal records*, not a completed R2 evaluation set.
    Recorded reaction pairs are evidence of reported routes, not a proof of
    unique mechanism or of feasibility under every unspecified condition.
    """
    if output.exists():
        raise FileExistsError(f"R2 positive cohort already frozen: {output}")
    if per_stratum <= 0 or per_stratum % 4:
        raise ValueError("R2 per-stratum count must be positive and divisible by four")
    official = json.loads(official_manifest.read_text())
    source_hash = digest(source)
    if source_hash != official["splits"]["test"]["output_sha256"]:
        raise ValueError("R2 source differs from official full-endpoint test manifest")
    r1_metadata = json.loads(r1_manifest.read_text())
    if digest(r1_cohort) != r1_metadata["cohort_sha256"]:
        raise ValueError("R2 R1 source hash drifted")

    groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = product_key(str(row["target_smiles"]))
            precursor = product_key(str(row["structural_precursor"]))
            groups[product][precursor].add(str(row.get("source_id") or row["id"]))

    r1: dict[str, dict[str, Any]] = {}
    with r1_cohort.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product in r1 or product_key(product) != product:
                raise ValueError("duplicate or noncanonical R1 product")
            references = {item["precursor_smiles"]: set(map(str, item["support_record_ids"]))
                          for item in row["references"]}
            if len(references) < 2 or references != groups.get(product):
                raise ValueError("R1 alternatives disagree with official held-out source")
            r1[product] = row

    multi = balanced_products(sorted(r1), count=per_stratum, seed=seed,
                              cohort="documented_alternative")
    single_products = sorted(product for product, references in groups.items()
                             if len(references) == 1 and product not in r1)
    single = balanced_products(single_products, count=per_stratum, seed=seed,
                               cohort="recorded_precursor")
    records: list[dict[str, Any]] = []
    for product, quartile in multi:
        references = sorted(groups[product].items(),
                            key=lambda item: (min(item[1]), item[0]))
        primary_precursor, primary_ids = references[0]
        alternative_precursor, alternative_ids = references[1]
        records.append({
            "artifact_type": "r2_recorded_positive_proposal_v1",
            "proposal_id": f"r2:alternative:{min(alternative_ids)}",
            "source_dataset": "FlowER official full endpoint",
            "source_split": "test",
            "product_smiles": product,
            "model_input": {"product_smiles": product,
                            "proposed_precursors": alternative_precursor},
            "private_label": {"known_recorded_positive": True,
                              "evidence_kind": "independent_heldout_recorded_alternative",
                              "proposal_record_ids": sorted(alternative_ids),
                              "primary_record_ids": sorted(primary_ids),
                              "primary_precursors": primary_precursor},
            "strata": {"positive_type": "documented_alternative",
                       "heavy_atom_quartile": quartile,
                       "reaction_family": "unavailable"},
        })
    for product, quartile in single:
        precursor, ids = next(iter(groups[product].items()))
        records.append({
            "artifact_type": "r2_recorded_positive_proposal_v1",
            "proposal_id": f"r2:recorded:{min(ids)}",
            "source_dataset": "FlowER official full endpoint",
            "source_split": "test",
            "product_smiles": product,
            "model_input": {"product_smiles": product,
                            "proposed_precursors": precursor},
            "private_label": {"known_recorded_positive": True,
                              "evidence_kind": "heldout_recorded_precursor",
                              "proposal_record_ids": sorted(ids)},
            "strata": {"positive_type": "recorded_precursor",
                       "heavy_atom_quartile": quartile,
                       "reaction_family": "unavailable"},
        })
    records.sort(key=lambda row: row["product_smiles"])
    if len(records) != 2 * per_stratum or len({row["product_smiles"] for row in records}) != len(records):
        raise ValueError("R2 positive cohort is not balanced and product-unique")

    output.mkdir(parents=True)
    cohort = output / "r2_positives.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r2_frozen_positive_source_manifest_v1",
        "source": str(source), "source_sha256": source_hash,
        "official_manifest": str(official_manifest),
        "official_manifest_sha256": digest(official_manifest),
        "r1_cohort": str(r1_cohort), "r1_cohort_sha256": digest(r1_cohort),
        "r1_manifest_sha256": digest(r1_manifest),
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "seed": seed, "positive_proposals": len(records),
        "recorded_precursor": len(single),
        "documented_alternative": len(multi),
        "negative_proposals": 0,
        "quartiles": {f"Q{index}": sum(row["strata"]["heavy_atom_quartile"] == f"Q{index}"
                                     for row in records) for index in range(1, 5)},
        "claim_boundary": "Held-out recorded positive proposals only; no hard negatives, verifier scores, AUROC or AUPRC.",
    }
    _write_json(output / "manifest.json", report)
    _write_json(output / "ARTIFACT_STATUS.json", {
        "artifact_type": "r2_positive_source_status_v1",
        "cohort_sha256": report["cohort_sha256"],
        "positive_source_allowed": True,
        "evaluation_allowed": False,
        "training_allowed": False,
        "reason": "R2 requires 400 evidence-audited hard negatives before evaluator registration",
    })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-manifest", type=Path, required=True)
    parser.add_argument("--r1-cohort", type=Path, required=True)
    parser.add_argument("--r1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--per-stratum", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.official_manifest,
                           args.r1_cohort, args.r1_manifest, args.output,
                           seed=args.seed, per_stratum=args.per_stratum),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
