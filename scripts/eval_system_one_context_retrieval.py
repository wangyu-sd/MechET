#!/usr/bin/env python3
"""Train-only principal-product -> final-mixture context retrieval diagnostic.

The label is the stereo-agnostic component difference between a strict
mechanism-final mixture and the full endpoint benchmark's principal product.
This predicts context *without* a held-out mechanism at inference, but it does
not infer missing stereochemistry, execute electrons, or cover reactions with
no strict trace. It is a candidate subsystem, not full product-only accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import component_counter, sha256
from scripts.eval_system_one_product_start_pilot import canonical_visible
from scripts.train_system_one_electron_flow import verify_source


@dataclass(frozen=True)
class ContextRow:
    reaction_id: str
    product: str
    batch: tuple[str, ...]


def load_rows(strict_dir: Path, full_dir: Path, split: str) -> tuple[list[ContextRow], dict]:
    strict_path, full_path = strict_dir / f"{split}.jsonl", full_dir / f"{split}.jsonl"
    strict_source = verify_source(strict_path)
    full_manifest = json.loads((full_dir / "manifest.json").read_text())
    full_declared = full_manifest["splits"][split]
    expected_full = 24959 if split == "train" else 3120
    if (full_manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or full_manifest["executor_filtering"] is not False
            or full_declared["rows"] != expected_full
            or sha256(full_path) != full_declared["endpoint_sha256"]):
        raise ValueError(f"{split}: invalid full endpoint source contract")
    full = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in full:
            raise ValueError(f"{split}: duplicate full endpoint ID {reaction_id}")
        full[reaction_id] = row
    if len(full) != expected_full:
        raise ValueError(f"{split}: full endpoint denominator mismatch")
    rows = []
    seen = set()
    for line in strict_path.read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["decision_type"] != "finish":
            continue
        reaction_id = str(row["metadata"]["reaction_id"])
        if reaction_id in seen or reaction_id not in full:
            raise ValueError(f"{split}: duplicate/missing strict ID {reaction_id}")
        seen.add(reaction_id)
        product = str(full[reaction_id]["product_unmapped"])
        principal = component_counter(product, isomeric=False)
        mixture = component_counter(str(row["target_smiles"]), isomeric=False)
        if principal - mixture:
            raise ValueError(f"{split}/{reaction_id}: principal product absent even without stereo")
        rows.append(ContextRow(
            reaction_id=reaction_id,
            product=product,
            batch=tuple(sorted((mixture - principal).elements())),
        ))
    if len(rows) != strict_source["reaction_denominator"]:
        raise ValueError(f"{split}: strict reaction denominator mismatch")
    rows.sort(key=lambda row: int(row.reaction_id))
    return rows, {
        "strict_source_sha256": strict_source["sha256"],
        "strict_reactions": len(rows),
        "full_endpoint_sha256": full_declared["endpoint_sha256"],
        "full_endpoint_reactions": expected_full,
    }


def rank_batches(similarities: list[float], train_batches: list[tuple[str, ...]]) -> list[tuple[tuple[str, ...], float, int]]:
    if len(similarities) != len(train_batches) or not similarities:
        raise ValueError("similarity vector and training context labels do not align")
    best: dict[tuple[str, ...], tuple[float, int]] = {}
    for index, (similarity, batch) in enumerate(zip(similarities, train_batches, strict=True)):
        previous = best.get(batch)
        if previous is None or similarity > previous[0]:
            best[batch] = (float(similarity), index)
    return sorted(
        ((batch, similarity, index) for batch, (similarity, index) in best.items()),
        key=lambda item: (-item[1], item[2]),
    )


def evaluate(train: list[ContextRow], heldout: list[ContextRow]) -> tuple[dict, list[dict]]:
    if not train or not heldout:
        raise ValueError("context retrieval requires nonempty train and heldout rows")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fingerprint(smiles: str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"unparseable product for fingerprint: {smiles}")
        return generator.GetFingerprint(molecule)

    train_fps = [fingerprint(row.product) for row in train]
    train_batches = [row.batch for row in train]
    train_products: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in train:
        train_products[canonical_visible(row.product)].add(row.batch)
    majority = Counter(train_batches).most_common(1)[0][0]
    support = set(train_batches)
    counts: Counter[str] = Counter()
    cases = []
    for row in heldout:
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint(row.product), train_fps)
        ranked = rank_batches(similarities, train_batches)
        exact_product_labels = train_products.get(canonical_visible(row.product), set())
        top1 = ranked[0][0]
        case = {
            "reaction_id": row.reaction_id,
            "product_unmapped": row.product,
            "reference_context_batch": list(row.batch),
            "predicted_context_batch": list(top1),
            "nearest_similarity": ranked[0][1],
            "top3_context_batches": [list(item[0]) for item in ranked[:3]],
            "product_exact_seen_in_train": bool(exact_product_labels),
            "exact_product_train_context_ambiguous": len(exact_product_labels) > 1,
            "top1_exact": top1 == row.batch,
            "top3_contains_reference": row.batch in [item[0] for item in ranked[:3]],
        }
        cases.append(case)
        counts["evaluated"] += 1
        counts["reference_batch_in_train_support"] += int(row.batch in support)
        counts["top1_exact"] += int(case["top1_exact"])
        counts["top3_contains_reference"] += int(case["top3_contains_reference"])
        counts["train_majority_exact"] += int(majority == row.batch)
        counts["product_exact_seen_in_train"] += int(bool(exact_product_labels))
        counts["exact_product_train_context_ambiguous"] += int(len(exact_product_labels) > 1)
    return {
        **dict(counts),
        "top1_exact_rate": counts["top1_exact"] / counts["evaluated"],
        "top3_contains_reference_rate": counts["top3_contains_reference"] / counts["evaluated"],
        "train_majority_exact_rate": counts["train_majority_exact"] / counts["evaluated"],
        "train_context_batch_support": len(support),
        "train_context_fragment_support": len({fragment for batch in support for fragment in batch}),
        "train_majority_batch": list(majority),
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
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case, separators=(",", ":")) + "\n")
    report = {
        "artifact_type": "system_one_principal_product_context_retrieval_diagnostic",
        "scope": "strict_trace_view_only_not_full_endpoint_benchmark",
        "split": args.split,
        "train_source": train_source,
        "heldout_source": heldout_source,
        "method": "train_only_morgan_radius2_2048_nearest_reaction_distinct_context_batches",
        "reference_context_label": "stereo_agnostic_final_mixture_minus_principal_product_components",
        "cases_sha256": sha256(cases_path),
        **metrics,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
