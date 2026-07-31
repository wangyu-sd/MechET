#!/usr/bin/env python3
"""Audit train--benchmark overlap across exact, product, scaffold, center and proof levels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import (
    KEY_LEVELS,
    NormalizationConfig,
    ReactionRecord,
    build_key_index,
    iter_mechet_jsonl,
    iter_reaction_table,
    max_tanimoto_against_reference,
    overlap_summary,
    reaction_keys,
    sha256_file,
)


def load_records(path: Path, fmt: str, reaction_field: str) -> Iterable[ReactionRecord]:
    if fmt == "mechet_jsonl":
        return iter_mechet_jsonl(path)
    if fmt == "reaction_table":
        return iter_reaction_table(path, reaction_field=reaction_field)
    raise ValueError(f"unsupported format: {fmt}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--train-format", choices=["mechet_jsonl", "reaction_table"], default="mechet_jsonl")
    parser.add_argument("--benchmark-format", choices=["mechet_jsonl", "reaction_table"], default="reaction_table")
    parser.add_argument("--reaction-field", default="reaction_smiles")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--similarity-sample", type=int, default=0)
    parser.add_argument("--no-stereo", action="store_true")
    args = parser.parse_args()

    config = NormalizationConfig(keep_stereo=not args.no_stereo)
    train_records = list(load_records(args.train, args.train_format, args.reaction_field))
    benchmark_records = list(load_records(args.benchmark, args.benchmark_format, args.reaction_field))
    train_rows = [(record.record_id, reaction_keys(record, config)) for record in train_records]
    benchmark_rows = [(record.record_id, reaction_keys(record, config)) for record in benchmark_records]
    train_index = build_key_index(train_rows)
    counts, conflicts = overlap_summary(train_index, benchmark_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        "config": config.__dict__,
        "normalization_digest": config.digest(),
        "train_path": str(args.train),
        "benchmark_path": str(args.benchmark),
        "train_sha256": sha256_file(args.train),
        "benchmark_sha256": sha256_file(args.benchmark),
    }
    (args.out_dir / "normalization_config.json").write_text(json.dumps(normalization, indent=2) + "\n")
    summary = {
        "n_train": len(train_rows),
        "n_benchmark": len(benchmark_rows),
        "overlap_counts": counts,
        "overlap_rates": {level: counts[level] / max(len(benchmark_rows), 1) for level in KEY_LEVELS},
    }

    if args.similarity_sample:
        reference = [record.product for record in train_records[: args.similarity_sample]]
        similarities = [
            {
                "id": record.record_id,
                "max_tanimoto": max_tanimoto_against_reference(record.product, reference, config),
            }
            for record in benchmark_records
        ]
        bins = {"0.00-0.40": 0, "0.40-0.60": 0, "0.60-0.80": 0, "0.80-0.95": 0, "0.95-1.00": 0}
        for item in similarities:
            value = item["max_tanimoto"]
            key = "0.00-0.40" if value < 0.4 else "0.40-0.60" if value < 0.6 else "0.60-0.80" if value < 0.8 else "0.80-0.95" if value < 0.95 else "0.95-1.00"
            bins[key] += 1
        summary["sampled_product_similarity"] = {
            "reference_sample_size": len(reference),
            "bins": bins,
            "note": "exact Tanimoto against a bounded reference sample; overlap keys are exhaustive",
        }
        with (args.out_dir / "sampled_product_similarity.jsonl").open("w") as handle:
            for item in similarities:
                handle.write(json.dumps(item) + "\n")

    (args.out_dir / "overlap_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out_dir / "benchmark_conflicts.jsonl").open("w") as handle:
        for row in conflicts:
            handle.write(json.dumps(row) + "\n")
    with (args.out_dir / "overlap_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["train", "benchmark", "level", "count", "rate"])
        for level in KEY_LEVELS:
            writer.writerow([args.train.name, args.benchmark.name, level, counts[level], counts[level] / max(len(benchmark_rows), 1)])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
