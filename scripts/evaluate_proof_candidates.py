#!/usr/bin/env python3
"""Evaluate sampled complete-proof candidates by execution and derived endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import sha256_file
from mechet.endpoints import (
    mapped_exact,
    reference_structural_precursor,
    split_precursor_endpoints,
    structural_exact,
)
from mechet.proof_program import sides_equal, verify_proof


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


def _load_rankings(directory: Path | None) -> dict[str, dict[str, Any]]:
    if directory is None:
        return {}
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("nll_scores.shard-*.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise ValueError(f"no NLL ranking shards found in {directory}")
    return _index(rows, "rankings")


def _target(row: dict[str, Any]) -> str:
    value = str(row.get("target_smiles") or (row.get("metadata") or {}).get("target_smiles") or "")
    if value:
        return value
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith("TARGET:"):
            return content.split("\n", 1)[0].removeprefix("TARGET:").strip()
    return ""


def _full_precursor(row: dict[str, Any]) -> str:
    metadata = dict(row.get("metadata") or {})
    return str(
        row.get("full_precursor_state")
        or row.get("expected_precursor")
        or metadata.get("full_precursor_state")
        or metadata.get("derived_precursor")
        or ""
    )


def _score_candidate(
    candidate: dict[str, Any], reference: dict[str, Any], index: int
) -> dict[str, Any]:
    proof = str(candidate.get("prediction") or "")
    verified = verify_proof(proof, expected_precursor=None)
    execute_ok = bool(verified.get("execute_ok"))
    derived_full = str(verified.get("derived_precursor") or "") if execute_ok else ""
    derived_structural = ""
    endpoint_error = ""
    if execute_ok:
        try:
            derived_structural = split_precursor_endpoints(
                derived_full, _target(reference)
            ).structural
        except Exception as exc:
            endpoint_error = str(exc)
    expected_structural = reference_structural_precursor(reference)
    expected_full = _full_precursor(reference)
    diagnostics = list(verified.get("diagnostics") or [])
    first = dict(diagnostics[0]) if diagnostics and isinstance(diagnostics[0], dict) else {}
    return {
        "candidate_index": index,
        "sample_index": int(candidate.get("sample_index") or index),
        "proof": proof,
        "format_ok": bool(verified.get("format_ok")),
        "execute_ok": execute_ok,
        "derived_full_precursor": derived_full,
        "derived_structural_precursor": derived_structural,
        "structural_exact": bool(
            derived_structural
            and expected_structural
            and structural_exact(derived_structural, expected_structural)
        ),
        "mapped_exact": bool(
            derived_structural
            and expected_structural
            and mapped_exact(derived_structural, expected_structural)
        ),
        "full_precursor_exact": bool(
            derived_full
            and expected_full
            and sides_equal(derived_full, expected_full, ignore_maps=True)
        ),
        "failure_code": str(first.get("code") or ""),
        "endpoint_error": endpoint_error,
        "generation_error": str(candidate.get("generation_error") or ""),
    }


def _formal_nll_order(
    evaluated: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> list[int]:
    if len(scores) != len(evaluated):
        raise ValueError(
            f"candidate-score count {len(scores)} != candidate count {len(evaluated)}"
        )

    def key(index: int) -> tuple[Any, ...]:
        score = scores[index]
        nll = score.get("assistant_mean_nll")
        score_ok = score.get("score_status") == "ok" and nll is not None
        return (
            int(evaluated[index]["execute_ok"]),
            int(score_ok),
            -float(nll) if score_ok else float("-inf"),
            -int(evaluated[index]["sample_index"]),
        )

    for index, score in enumerate(scores):
        evaluated[index]["assistant_mean_nll"] = score.get("assistant_mean_nll")
        evaluated[index]["score_status"] = score.get("score_status")
    return sorted(range(len(evaluated)), key=key, reverse=True)


def _hit(items: list[dict[str, Any]], order: list[int], key: str, k: int) -> bool:
    return any(items[index][key] for index in order[: min(k, len(order))])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ranking-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--expected-candidates", type=int, default=10)
    args = parser.parse_args()

    references = _index(_read_jsonl(args.reference), "references")
    predictions = _index(_read_jsonl(args.predictions), "predictions")
    rankings = _load_rankings(args.ranking_dir)
    if args.expected_rows and len(references) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} references, got {len(references)}")
    if set(predictions) != set(references):
        raise ValueError(
            f"prediction/reference ID mismatch: predictions={len(predictions)} references={len(references)}"
        )
    if rankings and set(rankings) != set(references):
        raise ValueError(
            f"ranking/reference ID mismatch: rankings={len(rankings)} references={len(references)}"
        )

    ks = (1, 5, 10)
    metrics = ("structural_exact", "mapped_exact", "full_precursor_exact", "execute_ok")
    counts: Counter[str] = Counter()
    failure_codes: Counter[str] = Counter()
    rows_out: list[dict[str, Any]] = []
    for identifier, reference in references.items():
        candidates = list(predictions[identifier].get("candidates") or [])
        if len(candidates) != args.expected_candidates:
            raise ValueError(
                f"{identifier} has {len(candidates)} candidates, expected {args.expected_candidates}"
            )
        evaluated = [
            _score_candidate(candidate, reference, index)
            for index, candidate in enumerate(candidates)
        ]
        generation_order = list(range(len(evaluated)))
        ranked_order: list[int] = []
        if rankings:
            ranked_order = _formal_nll_order(
                evaluated, list(rankings[identifier].get("candidate_scores") or [])
            )
        for item in evaluated:
            if not item["execute_ok"]:
                failure_codes[item["failure_code"] or "UNCLASSIFIED"] += 1
        for k in ks:
            for metric in metrics:
                counts[f"generation_{metric}_{k}"] += int(
                    _hit(evaluated, generation_order, metric, k)
                )
                if ranked_order:
                    counts[f"ranked_{metric}_{k}"] += int(
                        _hit(evaluated, ranked_order, metric, k)
                    )
        rows_out.append(
            {
                "id": identifier,
                "expected_structural_precursor": reference_structural_precursor(reference),
                "generation_order": generation_order,
                "formal_nll_ranked_order": ranked_order,
                "candidates": evaluated,
            }
        )

    denominator = len(references)

    def block(prefix: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k in ks:
            for metric in metrics:
                result[f"{metric}_at_{k}"] = counts[f"{prefix}_{metric}_{k}"] / max(denominator, 1)
        return result

    report: dict[str, Any] = {
        "artifact_type": "complete_proof_candidate_evaluation",
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "n_reference_rows": denominator,
        "candidates_per_target": args.expected_candidates,
        "n_candidates": denominator * args.expected_candidates,
        "generation_order": {
            "semantics": "independent-sample Pass@K; not ranked Top-K",
            **block("generation"),
        },
        "formal_nll_ranked": None,
        "execution_failure_codes": dict(failure_codes),
    }
    if rankings:
        report["formal_nll_ranked"] = {
            "semantics": "formal-execution gate, then gold-independent assistant mean-NLL",
            "ranker": "proof_execute_gate__assistant_mean_nll_v1",
            "selection_uses_ground_truth": False,
            **block("ranked"),
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
