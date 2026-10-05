#!/usr/bin/env python3
"""Train-only context proposals for every full-endpoint mech-USPTO product.

Labels come from full-endpoint training final mixtures. Validation/test final
mixtures enter only the offline scorer, never the proposal function.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import component_counter, sha256
from scripts.eval_system_one_context_knn import METHOD as STRICT_METHOD, rank_weighted_batches

METHOD = "full_train_only_morgan_radius2_2048_weighted_knn_k11_p2"


def load_full_rows(full_dir: Path, split: str) -> tuple[list[dict], dict]:
    manifest = json.loads((full_dir / "manifest.json").read_text())
    product_field = manifest.get("product_source_field", "rxn_prod_min")
    expected = 24959 if split == "train" else 3120
    declared = manifest["splits"][split]
    path = full_dir / f"{split}.jsonl"
    if (product_field not in {"rxn_prod_min", "rxn_prod_equ"}
            or manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or manifest["executor_filtering"] is not False
            or declared["rows"] != expected
            or declared["endpoint_sha256"] != sha256(path)):
        raise ValueError(f"{split}: full endpoint source contract mismatch")
    rows, seen = [], set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in seen:
            raise ValueError(f"{split}: duplicate full endpoint ID {reaction_id}")
        seen.add(reaction_id)
        product = str(row["product_unmapped"])
        principal = component_counter(product, isomeric=False)
        final = component_counter(str(row["final_mixture_unmapped"]), isomeric=False)
        if principal - final:
            raise ValueError(f"{split}/{reaction_id}: principal product absent from final mixture")
        rows.append({
            "reaction_id": reaction_id,
            "product_unmapped": product,
            "context_batch": tuple(sorted((final - principal).elements())),
        })
    if len(rows) != expected:
        raise ValueError(f"{split}: full endpoint row denominator mismatch")
    return rows, {"sha256": declared["endpoint_sha256"], "rows": expected,
                  "benchmark_universe": manifest["benchmark_universe"],
                  "product_source_field": product_field}


def evaluate(train: list[dict], heldout: list[dict]) -> tuple[dict, list[dict]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles: str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"invalid principal product: {smiles}")
        return generator.GetFingerprint(molecule)

    train_fps = [fingerprint(row["product_unmapped"]) for row in train]
    train_batches = [row["context_batch"] for row in train]
    train_support = set(train_batches)
    counts: Counter[str] = Counter()
    cases = []
    for row in heldout:
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint(row["product_unmapped"]), train_fps
        )
        ranked = rank_weighted_batches(similarities, train_batches)
        predicted = ranked[0][0]
        reference = row["context_batch"]
        case = {
            "reaction_id": row["reaction_id"],
            "product_unmapped": row["product_unmapped"],
            "predicted_context_batch": list(predicted),
            "reference_context_batch": list(reference),
            "top3_context_batches": [list(batch) for batch, _score in ranked[:3]],
            "top1_exact": predicted == reference,
            "reference_context_seen_in_train": reference in train_support,
        }
        cases.append(case)
        counts["evaluated"] += 1
        counts["top1_exact"] += int(case["top1_exact"])
        counts["reference_context_seen_in_train"] += int(case["reference_context_seen_in_train"])
        counts["top3_contains_reference"] += int(reference in [batch for batch, _ in ranked[:3]])
    return {
        "evaluated": counts["evaluated"],
        "top1_exact": counts["top1_exact"],
        "top1_exact_rate": counts["top1_exact"] / counts["evaluated"],
        "top3_contains_reference": counts["top3_contains_reference"],
        "reference_context_seen_in_train": counts["reference_context_seen_in_train"],
        "train_context_batch_support": len(train_support),
        "strict_view_method_ancestor": STRICT_METHOD,
    }, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train, train_source = load_full_rows(args.full_endpoint_dir, "train")
    heldout, heldout_source = load_full_rows(args.full_endpoint_dir, args.split)
    metrics, cases = evaluate(train, heldout)
    args.output.mkdir(parents=True)
    path = args.output / "cases.jsonl"
    with path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, separators=(",", ":")) + "\n")
    report = {
        "artifact_type": "system_one_full_endpoint_principal_product_context_proposal",
        "split": args.split,
        "train_source": train_source,
        "heldout_source": heldout_source,
        "product_source_field": train_source["product_source_field"],
        "method": METHOD,
        "selection": "k11_p2_frozen_from_strict_validation_then_applied_to_full_train",
        "cases_sha256": sha256(path),
        **metrics,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
