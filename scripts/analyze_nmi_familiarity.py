#!/usr/bin/env python3
"""Analyze primitive familiarity versus complete-program novelty for matched predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import align_prediction_artifact, read_jsonl, row_id
from mechet.proof_splits import extract_split_features
from mechet.strict_prediction_evaluation import endpoint_evaluation


def parse_named(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prediction", action="append", type=parse_named, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    train = read_jsonl(args.train)
    ref = read_jsonl(args.reference)
    prim = Counter()
    comp = Counter()
    for row in train:
        f = extract_split_features(row)
        prim.update(f.primitives)
        comp[f.composition] += 1

    ref_by_id = {row_id(r): r for r in ref}
    row_meta = {}
    for rid, row in ref_by_id.items():
        f = extract_split_features(row)
        freqs = [prim[x] for x in f.primitives]
        row_meta[rid] = {
            "program_train_count": comp[f.composition],
            "all_primitives_seen": all(v > 0 for v in freqs),
            "min_primitive_train_count": min(freqs),
            "mean_log1p_primitive_train_count": sum(math.log1p(v) for v in freqs) / len(freqs),
        }

    report = {
        "artifact_type": "nmi_familiarity_landscape_v1",
        "n_reference": len(ref),
        "conditions": {},
    }
    for name, path in args.prediction:
        preds = read_jsonl(path)
        aligned = align_prediction_artifact(ref, preds, condition_name=name)
        buckets = defaultdict(list)
        rows = []
        for pred in aligned:
            rid = row_id(pred)
            meta = row_meta[rid]
            exact = bool(endpoint_evaluation(pred)["structural_exact"])
            key = (
                "program_seen" if meta["program_train_count"] > 0 else "program_unseen",
                "all_primitives_seen" if meta["all_primitives_seen"] else "primitive_unseen",
            )
            buckets[key].append(exact)
            rows.append({"id": rid, **meta, "structural_exact": exact})
        report["conditions"][name] = {
            "n": len(rows),
            "strata": {
                "|".join(k): {"n": len(v), "structural_exact": sum(v) / max(len(v), 1)}
                for k, v in sorted(buckets.items())
            },
            "rows": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "conditions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
