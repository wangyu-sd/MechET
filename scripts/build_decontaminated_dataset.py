#!/usr/bin/env python3
"""Remove training rows conflicting with a frozen benchmark under an explicit policy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import (
    KEY_LEVELS,
    NormalizationConfig,
    build_key_index,
    iter_mechet_jsonl,
    iter_reaction_table,
    quarantine_reason,
    reaction_keys,
    record_from_mechet_row,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-format", choices=["mechet_jsonl", "reaction_table"], default="reaction_table")
    parser.add_argument("--reaction-field", default="reaction_smiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", nargs="+", choices=list(KEY_LEVELS), default=["exact_structural", "product"])
    args = parser.parse_args()

    config = NormalizationConfig()
    benchmark_records = (
        iter_mechet_jsonl(args.benchmark)
        if args.benchmark_format == "mechet_jsonl"
        else iter_reaction_table(args.benchmark, reaction_field=args.reaction_field)
    )
    benchmark_index = build_key_index(
        (record.record_id, reaction_keys(record, config)) for record in benchmark_records
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    reason_counts: Counter[str] = Counter()
    kept = removed = total = 0
    quarantine_path = args.output.with_suffix(args.output.suffix + ".quarantine.jsonl")
    with args.train.open("r", encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as out, quarantine_path.open("w", encoding="utf-8") as quarantine:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            record = record_from_mechet_row(row)
            if not record.record_id:
                record.record_id = f"{args.train.name}:{line_no}"
            keys = reaction_keys(record, config)
            reasons = quarantine_reason(keys, benchmark_index, args.policy)
            if reasons:
                removed += 1
                reason_counts.update(reasons)
                quarantine.write(json.dumps({"id": record.record_id, "reasons": reasons, "keys": keys.as_dict()}) + "\n")
            else:
                kept += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "train_input": str(args.train),
        "benchmark": str(args.benchmark),
        "output": str(args.output),
        "policy": args.policy,
        "normalization_digest": config.digest(),
        "input_sha256": sha256_file(args.train),
        "benchmark_sha256": sha256_file(args.benchmark),
        "n_total": total,
        "n_kept": kept,
        "n_removed": removed,
        "removal_rate": removed / max(total, 1),
        "reason_counts": dict(reason_counts),
        "quarantine": str(quarantine_path),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
