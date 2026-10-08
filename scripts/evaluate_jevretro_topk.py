#!/usr/bin/env python3
"""Evaluate ranked JevRetro single-step precursor candidates at Top-1/5/10.

References define the denominator. Missing predictions, invalid SMILES and
empty candidate lists remain failures. Candidates are ordered by the frozen
numeric score emitted by the model/decoder and deduplicated after canonical
map-free normalization.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.jevretro_endpoint import canonical_unmapped


def load_references(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            identifier = str(row.get("id") or "")
            if not identifier or identifier in seen:
                raise ValueError(f"invalid/duplicate reference id at line {line_number}")
            seen.add(identifier)
            gold = str(
                row.get("structural_precursor")
                or row.get("expected_precursor")
                or ""
            ).strip()
            if not gold:
                raise ValueError(f"{identifier}: missing structural precursor")
            rows.append(
                {
                    "id": identifier,
                    "gold": canonical_unmapped(gold),
                }
            )
    return rows


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            identifier = str(row.get("id") or "")
            if not identifier or identifier in output:
                raise ValueError(f"invalid/duplicate prediction id at line {line_number}")
            candidates = list(row.get("candidates") or [])
            normalized = []
            for rank, candidate in enumerate(candidates):
                score = candidate.get("score")
                precursor = str(candidate.get("precursor") or "").strip()
                if score is None or not math.isfinite(float(score)) or not precursor:
                    normalized.append(
                        {
                            "source_rank": rank,
                            "score": float("-inf"),
                            "canonical": None,
                            "valid": False,
                        }
                    )
                    continue
                try:
                    canonical = canonical_unmapped(precursor)
                except Exception:
                    canonical = None
                normalized.append(
                    {
                        "source_rank": rank,
                        "score": float(score),
                        "canonical": canonical,
                        "valid": canonical is not None,
                    }
                )
            output[identifier] = normalized
    return output


def ranked_unique(candidates: list[dict[str, Any]]) -> list[str]:
    ordered = sorted(
        candidates,
        key=lambda item: (-float(item["score"]), int(item["source_rank"])),
    )
    result = []
    seen = set()
    for item in ordered:
        canonical = item.get("canonical")
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        result.append(str(canonical))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    refs = load_references(args.references)
    preds = load_predictions(args.predictions)
    reference_ids = {row["id"] for row in refs}
    extra_ids = sorted(set(preds) - reference_ids)
    if extra_ids:
        raise ValueError(f"prediction file contains {len(extra_ids)} unknown IDs")

    ks = (1, 5, 10)
    hits = {k: 0 for k in ks}
    missing = 0
    invalid_only = 0
    candidate_counts = []
    unique_counts = []
    cases = []

    for row in refs:
        raw = preds.get(row["id"])
        if raw is None:
            missing += 1
            ranked = []
        else:
            candidate_counts.append(len(raw))
            ranked = ranked_unique(raw)
            unique_counts.append(len(ranked))
            invalid_only += int(bool(raw) and not ranked)
        for k in ks:
            hits[k] += int(row["gold"] in ranked[:k])
        cases.append(
            {
                "id": row["id"],
                "gold": row["gold"],
                "ranked_candidates": ranked[:10],
                "hit_at_1": row["gold"] in ranked[:1],
                "hit_at_5": row["gold"] in ranked[:5],
                "hit_at_10": row["gold"] in ranked[:10],
            }
        )

    n = len(refs)
    report = {
        "artifact_type": "jevretro_ranked_single_step_endpoint_evaluation_v1",
        "denominator": n,
        "ranking_contract": "descending_frozen_candidate_score_then_source_rank",
        "canonicalization": "RDKit canonical map-free isomeric SMILES",
        "missing_prediction_rows": missing,
        "invalid_only_prediction_rows": invalid_only,
        "top1": hits[1] / n if n else None,
        "top5": hits[5] / n if n else None,
        "top10": hits[10] / n if n else None,
        "hits": {str(k): hits[k] for k in ks},
        "candidate_count_mean": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "unique_valid_candidate_count_mean": (
            sum(unique_counts) / len(unique_counts) if unique_counts else 0.0
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cases_path = args.output.with_suffix(".cases.jsonl")
    with cases_path.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
