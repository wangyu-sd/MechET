#!/usr/bin/env python3
"""Audit reusable execution primitives and complete-program novelty."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_splits import extract_split_features


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def features(rows):
    out = []
    quarantine = []
    for row in rows:
        try:
            out.append((str(row.get("id")), extract_split_features(row)))
        except Exception as exc:
            quarantine.append({"id": row.get("id"), "error": str(exc)})
    return out, quarantine


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows-output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train_rows = read_jsonl(args.train)
    eval_rows = read_jsonl(args.eval)
    train_feat, train_bad = features(train_rows)
    eval_feat, eval_bad = features(eval_rows)
    if not train_feat or not eval_feat:
        raise SystemExit("operator audit requires non-empty replay-verified train/eval rows")

    primitive_counts = Counter()
    composition_counts = Counter()
    for _, f in train_feat:
        primitive_counts.update(f.primitives)
        composition_counts[f.composition] += 1

    row_reports = []
    counts = Counter()
    for rid, f in eval_feat:
        freqs = [primitive_counts[x] for x in f.primitives]
        all_seen = all(v > 0 for v in freqs)
        program_seen = composition_counts[f.composition] > 0
        if program_seen:
            counts["program_seen"] += 1
        elif all_seen:
            counts["program_unseen_all_primitives_seen"] += 1
        else:
            counts["program_unseen_with_unseen_primitive"] += 1
        row_reports.append(
            {
                "id": rid,
                "composition": f.composition,
                "program_train_count": composition_counts[f.composition],
                "n_primitives": len(f.primitives),
                "primitive_seen_fraction": sum(v > 0 for v in freqs) / len(freqs),
                "min_primitive_train_count": min(freqs),
                "mean_log1p_primitive_train_count": sum(math.log1p(v) for v in freqs) / len(freqs),
                "all_primitives_seen": all_seen,
                "program_seen": program_seen,
            }
        )

    rng = random.Random(args.seed)
    shuffled = list(train_feat)
    rng.shuffle(shuffled)
    growth = []
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
        n = max(1, round(len(shuffled) * frac))
        prim = set()
        comp = set()
        for _, f in shuffled[:n]:
            prim.update(f.primitives)
            comp.add(f.composition)
        growth.append({"fraction": frac, "rows": n, "unique_primitives": len(prim), "unique_compositions": len(comp)})

    n_eval = len(eval_feat)
    summary = {
        "artifact_type": "nmi_operator_basis_audit_v1",
        "primitive_basis": "source_to_sink_execution_moves_v1",
        "train_rows_total": len(train_rows),
        "train_rows_eligible": len(train_feat),
        "eval_rows_total": len(eval_rows),
        "eval_rows_eligible": n_eval,
        "train_unique_primitives": len(primitive_counts),
        "train_unique_compositions": len(composition_counts),
        "program_seen_count": counts["program_seen"],
        "program_unseen_all_primitives_seen_count": counts["program_unseen_all_primitives_seen"],
        "program_unseen_with_unseen_primitive_count": counts["program_unseen_with_unseen_primitive"],
        "program_unseen_all_primitives_seen_rate": counts["program_unseen_all_primitives_seen"] / n_eval,
        "any_unseen_primitive_rate": counts["program_unseen_with_unseen_primitive"] / n_eval,
        "vocabulary_growth": growth,
        "train_quarantine": train_bad,
        "eval_quarantine": eval_bad,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows_path = args.rows_output or args.output.with_name(args.output.stem + ".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in row_reports:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
