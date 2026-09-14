#!/usr/bin/env python3
"""Export ranked R-SMILES predictions to the shared MechET JSONL contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-predictions", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-seconds", type=Path, required=True)
    parser.add_argument("--candidates-per-target", type=int, default=10)
    args = parser.parse_args()

    references = read_jsonl(args.reference)
    ranked = args.ranked_predictions.read_text(encoding="utf-8").splitlines()
    expected = len(references) * args.candidates_per_target
    if len(ranked) != expected:
        raise ValueError(
            f"ranked prediction count mismatch: expected {expected}, got {len(ranked)}"
        )
    runtime_seconds = float(args.runtime_seconds.read_text(encoding="utf-8").strip())
    runtime_ms = runtime_seconds * 1000.0 / max(len(references), 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as sink:
        for row_index, reference in enumerate(references):
            identifier = str(reference.get("stable_id") or reference.get("id") or "")
            if not identifier:
                raise ValueError(f"reference row {row_index} has no stable ID")
            start = row_index * args.candidates_per_target
            values = ranked[start : start + args.candidates_per_target]
            candidates = [
                {
                    "rank": rank,
                    "precursors": value,
                    "prediction": value,
                    # score.py has already applied native TTA voting. This
                    # monotonic value records the resulting order without
                    # pretending to retain its internal aggregate score.
                    "score": -float(rank),
                }
                for rank, value in enumerate(values, start=1)
            ]
            sink.write(
                json.dumps(
                    {
                        "id": identifier,
                        "stable_id": identifier,
                        "product": reference.get("product_mapped")
                        or reference.get("product_unmapped")
                        or "",
                        "reference_precursors": reference.get("precursor_mapped")
                        or reference.get("precursor_unmapped")
                        or "",
                        "candidates": candidates,
                        "runtime_ms": runtime_ms,
                        "source_method": "R-SMILES",
                        "checkpoint": str(args.checkpoint.resolve()),
                        "ranking": "native_tta_score_alpha_1",
                        "score_semantics": "negative_final_rank_after_native_tta",
                        "evaluation_status": "ok" if values[0] else "empty_top1",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"exported {len(references)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
