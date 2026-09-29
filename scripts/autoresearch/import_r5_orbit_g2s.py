#!/usr/bin/env python3
"""Adapt the archived ORBIT Graph2SMILES predictions to frozen R5 queries.

This is an external-model *diagnostic* source, not a RetroChimera run or a
matched headline baseline. The archived checkpoint is not present locally.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, product_key


def convert(query: Path, query_manifest: Path, source: Path,
            source_provenance: Path, output: Path, *,
            expected_source_sha256: str, expected_provenance_sha256: str,
            expected_products: int = 200, top_k: int = 5) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R5 G2S conversion already frozen: {output}")
    if digest(source) != expected_source_sha256:
        raise ValueError("archived G2S prediction SHA-256 changed")
    if digest(source_provenance) != expected_provenance_sha256:
        raise ValueError("archived G2S provenance SHA-256 changed")
    query_meta = json.loads(query_manifest.read_text())
    if digest(query) != query_meta["cohort_sha256"]:
        raise ValueError("frozen R5 query SHA-256 changed")
    if query_meta["products"] != expected_products or top_k != 5:
        raise ValueError("R5 G2S diagnostic requires frozen 200 products and Top-5")

    query_products = []
    with query.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            product = str(row["product_smiles"])
            if product_key(product) != product:
                raise ValueError("noncanonical R5 product query")
            query_products.append(product)
    if len(query_products) != expected_products or len(set(query_products)) != expected_products:
        raise ValueError("R5 product query denominator or uniqueness changed")

    archived = json.loads(source.read_text())
    if not isinstance(archived, list):
        raise ValueError("archived G2S predictions must be a JSON array")
    wanted = set(query_products)
    by_product: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(archived):
        if not isinstance(row, dict) or not isinstance(row.get("product_smiles"), str):
            raise ValueError(f"malformed archived G2S prediction at index {index}")
        product = product_key(row["product_smiles"])
        if product in wanted:
            by_product[product].append((index, row))
    if set(by_product) != wanted:
        raise ValueError(f"archived G2S predictions miss {len(wanted - set(by_product))} R5 products")

    predictions = []
    duplicates = 0
    conflicting_top5 = 0
    for product in sorted(wanted):
        matches = by_product[product]
        # The source-file order is frozen by its hash. This uses neither the
        # recorded precursors nor any model/evaluator score to select a row.
        source_index, chosen = matches[0]
        candidates = chosen.get("predicted_precursors")
        if not isinstance(candidates, list) or any(not isinstance(x, str) for x in candidates):
            raise ValueError(f"malformed G2S candidate list at source index {source_index}")
        if len(matches) > 1:
            duplicates += 1
            conflicting_top5 += int(any(
                other.get("predicted_precursors", [])[:top_k] != candidates[:top_k]
                for _, other in matches[1:]
            ))
        predictions.append({
            "product_smiles": product,
            "inference_status": "completed",
            "source_row_index": source_index,
            "canonical_product_match_count": len(matches),
            "candidates": [{"rank": rank, "precursors": precursor}
                           for rank, precursor in enumerate(candidates[:top_k], 1)],
        })

    output.mkdir(parents=True)
    prediction_path = output / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "model_name": "Graph2SMILES ORBIT collaborator archive (diagnostic)",
        "checkpoint_identifier": "ORBIT_g2s_series_rel_smiles_smiles.1/model.5000_0.pt (archive record; weights unavailable locally)",
        "checkpoint_source": str(source_provenance),
        "training_corpus": "FlowER flower_completion train; archived record says 257171 rows, 5000 updates",
        "license_or_terms": "Collaborator-produced local archive; redistribution and original model terms not independently verified",
        "input_fields": ["product_smiles"],
        "target_semantics": "full_reaction_world",
        "inference_status": "completed",
        "inference_config": {
            "beam_size": 10, "n_best": 10, "retained_top_k": top_k,
            "canonical_duplicate_policy": "first source-file row; no reference or score used",
        },
        "training_overlap_audited": False,
        "source_predictions": str(source),
        "source_predictions_sha256": expected_source_sha256,
        "source_provenance_sha256": expected_provenance_sha256,
        "source_rows_observed": len(archived),
        "source_rows_claimed_by_archive": 28971,
        "matched_query_products": len(predictions),
        "query_products_with_duplicate_source_rows": duplicates,
        "duplicate_products_with_conflicting_top5": conflicting_top5,
        "claim_boundary": "Archived 5000-step G2S diagnostic with unavailable weights and unaudited training overlap; not RetroChimera or a headline R5 comparison.",
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {
        "predictions": str(prediction_path),
        "predictions_sha256": digest(prediction_path),
        "provenance": str(provenance_path),
        "provenance_sha256": digest(provenance_path),
        "products": len(predictions),
        "source_rows": len(archived),
        "duplicate_query_products": duplicates,
        "conflicting_top5_products": conflicting_top5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.query, args.query_manifest, args.source,
                             args.source_provenance, args.output,
                             expected_source_sha256=args.expected_source_sha256,
                             expected_provenance_sha256=args.expected_provenance_sha256),
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
