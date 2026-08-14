#!/usr/bin/env python3
"""Evaluate direct endpoint candidates in generation and frozen-NLL order."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import sha256_file
from mechet.endpoints import mapped_exact, reference_structural_precursor, structural_exact
from mechet.knowledge_ablation import extract_direct_prediction


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in output:
            raise ValueError(f"{label} contains a missing or duplicate ID: {identifier!r}")
        output[identifier] = row
    return output


def _candidate_row(parent: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_mode": "direct",
        "prediction": candidate.get("prediction") or "",
        "messages": candidate.get("messages") or [],
        "metadata": parent.get("metadata") or {},
    }


def _canonical_key(smiles: str, *, neutralize: bool) -> tuple[str, ...] | None:
    if not smiles:
        return None
    uncharger = rdMolStandardize.Uncharger()
    parts: list[str] = []
    for fragment in smiles.split("."):
        mol = Chem.MolFromSmiles(fragment.strip())
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        if neutralize:
            mol = uncharger.uncharge(mol)
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return tuple(sorted(parts))


def _load_rankings(directory: Path | None) -> dict[str, dict[str, Any]]:
    if directory is None:
        return {}
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("nll_scores.shard-*.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise ValueError(f"no NLL ranking shards found in {directory}")
    return _index(rows, "rankings")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-candidates", type=int, default=10)
    parser.add_argument("--allow-prediction-superset", action="store_true")
    args = parser.parse_args()

    references = _index(_read_jsonl(args.reference), "references")
    predictions = _index(_read_jsonl(args.predictions), "predictions")
    rankings = _load_rankings(args.ranking_dir)
    if args.expected_rows and len(references) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} references, got {len(references)}")
    missing_predictions = sorted(set(references) - set(predictions))
    extras = sorted(set(predictions) - set(references))
    if missing_predictions:
        raise ValueError(f"missing predictions: {missing_predictions[:10]}")
    if extras and not args.allow_prediction_superset:
        raise ValueError(f"prediction IDs absent from reference: {extras[:10]}")
    if rankings:
        missing_rankings = sorted(set(references) - set(rankings))
        if missing_rankings:
            raise ValueError(f"missing NLL rankings: {missing_rankings[:10]}")

    ks = (1, 3, 5, 10)
    counts = Counter()
    rows_out: list[dict[str, Any]] = []
    for identifier, reference in references.items():
        prediction = predictions[identifier]
        candidates = list(prediction.get("candidates") or [])
        if len(candidates) != args.expected_candidates:
            raise ValueError(
                f"{identifier} has {len(candidates)} candidates, expected {args.expected_candidates}"
            )
        expected = reference_structural_precursor(reference)
        expected_neutral = _canonical_key(expected, neutralize=True)
        evaluated: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            value = extract_direct_prediction(_candidate_row(prediction, candidate))
            valid_key = _canonical_key(value, neutralize=False)
            neutral_key = _canonical_key(value, neutralize=True)
            item = {
                "candidate_index": index,
                "sample_index": int(candidate.get("sample_index") or index),
                "prediction": value,
                "valid_smiles": valid_key is not None,
                "structural_exact": valid_key is not None
                and structural_exact(value, expected),
                "mapped_exact": valid_key is not None and mapped_exact(value, expected),
                "neutralized_exact": neutral_key is not None
                and neutral_key == expected_neutral,
                "generation_error": str(candidate.get("generation_error") or ""),
            }
            evaluated.append(item)
            counts["valid_candidates"] += int(item["valid_smiles"])

        generation_order = list(range(len(evaluated)))
        ranked_order: list[int] = []
        ranking = rankings.get(identifier)
        if ranking:
            ranked_order = [int(index) for index in ranking["ranked_candidate_indices"]]
            if sorted(ranked_order) != generation_order:
                raise ValueError(f"invalid ranking permutation for {identifier}")
            scores = list(ranking.get("candidate_scores") or [])
            for index, score in enumerate(scores):
                if index < len(evaluated):
                    evaluated[index]["assistant_mean_nll"] = score.get("assistant_mean_nll")
                    evaluated[index]["score_status"] = score.get("score_status")

        for k in ks:
            prefix = generation_order[: min(k, len(generation_order))]
            counts[f"pass_{k}"] += int(any(evaluated[i]["structural_exact"] for i in prefix))
            if ranked_order:
                prefix = ranked_order[: min(k, len(ranked_order))]
                counts[f"top_{k}"] += int(any(evaluated[i]["structural_exact"] for i in prefix))
        first = evaluated[generation_order[0]]
        counts["generation_top1_valid"] += int(first["valid_smiles"])
        counts["generation_top1_neutral"] += int(first["neutralized_exact"])
        if ranked_order:
            selected = evaluated[ranked_order[0]]
            counts["ranked_top1_valid"] += int(selected["valid_smiles"])
            counts["ranked_top1_neutral"] += int(selected["neutralized_exact"])
        rows_out.append(
            {
                "id": identifier,
                "expected": expected,
                "generation_order": generation_order,
                "nll_ranked_order": ranked_order,
                "candidates": evaluated,
            }
        )

    denominator = len(references)
    total_candidates = denominator * args.expected_candidates
    report: dict[str, Any] = {
        "artifact_type": "direct_endpoint_candidate_evaluation",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "n_reference_rows": denominator,
        "n_prediction_rows_used": denominator,
        "n_prediction_rows_ignored_as_superset": len(extras),
        "candidates_per_target": args.expected_candidates,
        "n_candidates": total_candidates,
        "candidate_validity_rate": counts["valid_candidates"] / max(total_candidates, 1),
        "generation_order": {
            "semantics": "independent-sample Pass@K; not ranked Top-K",
            **{f"structural_pass_at_{k}": counts[f"pass_{k}"] / denominator for k in ks},
            "top1_validity": counts["generation_top1_valid"] / denominator,
            "neutralized_pass_at_1": counts["generation_top1_neutral"] / denominator,
        },
        "nll_ranked": None,
    }
    if rankings:
        report["nll_ranked"] = {
            "semantics": "gold-independent teacher-forced assistant mean-NLL rank",
            "ranker": next(iter(rankings.values())).get("ranker"),
            **{f"structural_top_{k}": counts[f"top_{k}"] / denominator for k in ks},
            "top1_validity": counts["ranked_top1_valid"] / denominator,
            "neutralized_top_1": counts["ranked_top1_neutral"] / denominator,
            "selection_uses_ground_truth": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    row_path = args.output.with_suffix(".rows.jsonl")
    with row_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
