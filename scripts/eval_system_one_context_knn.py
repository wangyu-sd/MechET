#!/usr/bin/env python3
"""Frozen train-only weighted kNN context proposal for principal products.

The k=11, squared-Tanimoto vote was selected on strict validation only. The
test script uses exactly the same rule; reference context is scorer-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.eval_system_one_context_retrieval import ContextRow, load_rows
from scripts.eval_system_one_product_start_pilot import canonical_visible

K = 11
POWER = 2
METHOD = "train_only_morgan_radius2_2048_weighted_knn_k11_p2"


def rank_weighted_batches(similarities: list[float], train_batches: list[tuple[str, ...]]) -> list[tuple[tuple[str, ...], float]]:
    if len(similarities) != len(train_batches) or len(similarities) < K:
        raise ValueError("insufficient aligned training rows for context kNN")
    nearest = sorted(range(len(similarities)), key=lambda i: (-similarities[i], i))[:K]
    votes: dict[tuple[str, ...], float] = defaultdict(float)
    for index in nearest:
        votes[train_batches[index]] += float(similarities[index]) ** POWER
    return sorted(votes.items(), key=lambda item: (-item[1], item[0]))


def evaluate(train: list[ContextRow], heldout: list[ContextRow]) -> tuple[dict, list[dict]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid principal product SMILES: {smiles}")
        return generator.GetFingerprint(mol)

    train_fp = [fp(row.product) for row in train]
    train_labels = [row.batch for row in train]
    train_products = {canonical_visible(row.product) for row in train}
    counts: Counter[str] = Counter()
    cases = []
    for row in heldout:
        similarities = DataStructs.BulkTanimotoSimilarity(fp(row.product), train_fp)
        ranked = rank_weighted_batches(similarities, train_labels)
        proposal = ranked[0][0]
        top3 = [item[0] for item in ranked[:3]]
        case = {
            "reaction_id": row.reaction_id,
            "product_unmapped": row.product,
            "reference_context_batch": list(row.batch),
            "predicted_context_batch": list(proposal),
            "top3_context_batches": [list(batch) for batch in top3],
            "top1_exact": proposal == row.batch,
            "top3_contains_reference": row.batch in top3,
            "product_exact_seen_in_train": canonical_visible(row.product) in train_products,
        }
        cases.append(case)
        counts["evaluated"] += 1
        counts["top1_exact"] += int(case["top1_exact"])
        counts["top3_contains_reference"] += int(case["top3_contains_reference"])
    return {
        "evaluated": counts["evaluated"],
        "top1_exact": counts["top1_exact"],
        "top1_exact_rate": counts["top1_exact"] / counts["evaluated"],
        "top3_contains_reference": counts["top3_contains_reference"],
        "top3_contains_reference_rate": counts["top3_contains_reference"] / counts["evaluated"],
    }, cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train, train_source = load_rows(args.strict_dir, args.full_endpoint_dir, "train")
    heldout, heldout_source = load_rows(args.strict_dir, args.full_endpoint_dir, args.split)
    metrics, cases = evaluate(train, heldout)
    args.output.mkdir(parents=True)
    path = args.output / "cases.jsonl"
    with path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, separators=(",", ":")) + "\n")
    report = {
        "artifact_type": "system_one_principal_product_context_retrieval_diagnostic",
        "scope": "strict_trace_view_only_not_full_endpoint_benchmark",
        "split": args.split,
        "train_source": train_source,
        "heldout_source": heldout_source,
        "method": METHOD,
        "selection": "k_and_power_selected_on_validation_then_frozen_on_test",
        "reference_context_label": "stereo_agnostic_final_mixture_minus_principal_product_components",
        "cases_sha256": sha256(path),
        **metrics,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
