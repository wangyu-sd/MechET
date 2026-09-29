#!/usr/bin/env python3
"""Freeze actual external-model Top-K proposals on the fixed R5 products."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"R5 provenance needs nonempty {key}")
    return value.strip()


def _canonical_or_invalid(value: str) -> tuple[str | None, str]:
    if not value.strip():
        return None, "missing"
    try:
        return product_key(value), "valid_smiles"
    except ValueError:
        return None, "invalid_smiles"


def freeze(query: Path, query_manifest: Path, predictions: Path,
           provenance: Path, output: Path, *, expected_products: int = 200,
           top_k: int = 5) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 external prediction source already frozen: {output}")
    if expected_products <= 0 or top_k <= 0:
        raise ValueError("R5 expected products and Top-K must be positive")
    query_meta = json.loads(query_manifest.read_text())
    query_hash = digest(query)
    if query_hash != query_meta["cohort_sha256"]:
        raise ValueError("R5 frozen query cohort hash drifted")
    if query_meta["products"] != expected_products:
        raise ValueError("R5 query product denominator changed")
    model = json.loads(provenance.read_text())
    for field in ("model_name", "checkpoint_identifier", "checkpoint_source",
                  "training_corpus", "license_or_terms"):
        _required_text(model, field)
    if model.get("input_fields") != ["product_smiles"]:
        raise ValueError("R5 external model provenance must attest product-only input")
    if model.get("target_semantics") != "retrosynthetic_precursor_set":
        raise ValueError("R5 scientific intake requires retrosynthetic precursor-set predictions, not a full reaction world")
    if model.get("inference_status") != "completed":
        raise ValueError("R5 source requires a completed external inference run")
    if not isinstance(model.get("inference_config"), dict) or not model["inference_config"]:
        raise ValueError("R5 source requires a recorded inference configuration")

    products: dict[str, dict[str, Any]] = {}
    canonical_products: set[str] = set()
    with query.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            canonical = product_key(product)
            if product in products or canonical in canonical_products:
                raise ValueError("R5 query has duplicate chemical products")
            if row["model_input"] != {"product_smiles": product}:
                raise ValueError("R5 query exposes more than its frozen product input")
            products[product] = row
            canonical_products.add(canonical)
    if len(products) != expected_products:
        raise ValueError("R5 query row count changed")
    overlap_map = model.get("training_exact_product_overlap")
    if overlap_map is not None:
        if not isinstance(overlap_map, dict) or set(overlap_map) != set(products):
            raise ValueError("R5 external training-overlap map must cover every frozen product")
        if any(type(value) is not bool for value in overlap_map.values()):
            raise ValueError("R5 external training-overlap map must contain booleans")
        if model.get("training_exact_product_overlap_count") != sum(overlap_map.values()):
            raise ValueError("R5 external training-overlap count differs from the map")

    raw: dict[str, dict[str, Any]] = {}
    with predictions.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product not in products or product in raw:
                raise ValueError("R5 external predictions have an unknown or duplicate product")
            raw[product] = row
    if set(raw) != set(products):
        raise ValueError("R5 external predictions must include every frozen product row")

    frozen = []
    counts: Counter[str] = Counter()
    for product, query_row in sorted(products.items()):
        prediction_row = raw[product]
        status = str(prediction_row.get("inference_status") or "")
        if status not in {"completed", "failed"}:
            raise ValueError("R5 per-product inference status must be completed or failed")
        candidates = prediction_row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) > top_k:
            raise ValueError("R5 candidates must be a list of at most Top-K entries")
        if status == "failed" and candidates:
            raise ValueError("failed R5 inference row cannot carry candidates")
        by_rank = {}
        for candidate in candidates:
            rank = candidate.get("rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= top_k:
                raise ValueError("R5 candidate has an invalid rank")
            if rank in by_rank:
                raise ValueError("R5 candidate has a duplicated rank")
            precursor = candidate.get("precursors")
            if not isinstance(precursor, str):
                raise ValueError("R5 precursor proposal must be a string")
            frequency = candidate.get("frequency_confidence")
            if frequency is not None and (not isinstance(frequency, (int, float))
                                          or isinstance(frequency, bool)
                                          or not 0 <= frequency <= 1):
                raise ValueError("R5 source frequency confidence must be in [0,1]")
            by_rank[rank] = precursor
        references = {str(item["precursor_smiles"])
                      for item in query_row["private_reference"]["recorded_precursor_sets"]}
        normalized = []
        for rank in range(1, top_k + 1):
            raw_precursor = by_rank.get(rank, "")
            canonical, validity = _canonical_or_invalid(raw_precursor)
            counts[validity] += 1
            normalized.append({
                "rank": rank,
                "rank_stratum": "1" if rank == 1 else "2-3" if rank <= 3 else "4-5",
                "raw_precursors": raw_precursor or None,
                "canonical_precursors": canonical,
                "smiles_status": validity,
                "source_frequency_confidence": (
                    next((candidate.get("frequency_confidence") for candidate in candidates
                          if candidate["rank"] == rank), None)),
                "recorded_reference_status": (
                    "recorded_reference" if canonical in references
                    else "not_recorded_not_proven_invalid" if canonical
                    else "unassessable"),
            })
        counts["product_" + status] += 1
        frozen.append({
            "artifact_type": "r5_external_ranked_predictions_v1",
            "product_smiles": product,
            "source_dataset": query_row["source_dataset"],
            "source_split": query_row["source_split"],
            "external_model": model["model_name"],
            "model_input": {"product_smiles": product},
            "inference_status": status,
            "external_training_exact_product_overlap": (
                overlap_map[product] if overlap_map is not None else None),
            "candidates": normalized,
            "strata": {**query_row["strata"], "model_agreement": "unavailable_one_model"},
            "private_reference": query_row["private_reference"],
        })
    if counts["product_completed"] == 0 or counts["valid_smiles"] == 0:
        raise ValueError("R5 external source contains no completed valid prediction")

    output.mkdir(parents=True)
    cohort = output / "r5_external_predictions.jsonl"
    with cohort.open("w", encoding="utf-8") as stream:
        for row in frozen:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {
        "artifact_type": "r5_external_prediction_manifest_v1",
        "query": str(query), "query_sha256": query_hash,
        "query_manifest_sha256": digest(query_manifest),
        "source_predictions": str(predictions),
        "source_predictions_sha256": digest(predictions),
        "provenance": str(provenance), "provenance_sha256": digest(provenance),
        "model_name": model["model_name"],
        "checkpoint_identifier": model["checkpoint_identifier"],
        "checkpoint_source": model["checkpoint_source"],
        "training_corpus": model["training_corpus"],
        "target_semantics": model["target_semantics"],
        "cohort": str(cohort), "cohort_sha256": digest(cohort),
        "products": len(frozen), "ranks_per_product": top_k,
        "counts": dict(sorted(counts.items())),
        "training_overlap_audited": bool(model.get("training_overlap_audited")),
        "training_exact_product_overlap_count": model.get("training_exact_product_overlap_count"),
        "ranking_semantics": model.get("inference_config", {}).get("ranking"),
        "claim_boundary": "Frozen external proposals only; recorded-reference absence does not prove chemical invalidity, and no MechET reranking has run.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r5_external_prediction_status_v1",
        "cohort_sha256": report["cohort_sha256"],
        "evaluation_allowed": True,
        "training_allowed": False,
        "headline_allowed": False,
        "reason": "Complete external-prediction intake; MechET verification and overlap audit are separate gates",
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-products", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(freeze(args.query, args.query_manifest,
                            args.predictions, args.provenance, args.output,
                            expected_products=args.expected_products,
                            top_k=args.top_k), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
