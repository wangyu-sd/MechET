#!/usr/bin/env python3
"""Merge distributed value-search shards and materialize distillation data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--distill-dir", type=Path)
    parser.add_argument("--minimum-successes", type=int, default=16)
    args = parser.parse_args()
    results: list[dict] = []
    for path in sorted(args.search_dir.glob("results.shard-*.jsonl")):
        results.extend(json.loads(line) for line in path.open() if line.strip())
    if not results:
        raise RuntimeError("no search results found")
    counts = Counter()
    for row in results:
        counts["reactions"] += 1
        counts["top1_exact"] += int(bool(row["top1_exact"]))
        counts["pass_at_beam"] += int(bool(row["pass_at_beam"]))
        counts["terminal"] += int(bool(row["top_terminal"]))
    report = {
        **dict(counts),
        "top1_accuracy": counts["top1_exact"] / counts["reactions"],
        "pass_at_beam_accuracy": counts["pass_at_beam"] / counts["reactions"],
        "terminal_rate": counts["terminal"] / counts["reactions"],
    }
    (args.search_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if args.distill_dir is not None:
        if counts["pass_at_beam"] < args.minimum_successes:
            raise RuntimeError(
                f"distillation gate failed: {counts['pass_at_beam']} successes "
                f"< {args.minimum_successes}"
            )
        args.distill_dir.mkdir(parents=True, exist_ok=True)
        target = args.distill_dir / "train.jsonl"
        rows = 0
        with target.open("w", encoding="utf-8") as sink:
            for path in sorted(args.search_dir.glob("distill.shard-*.jsonl")):
                for line in path.open(encoding="utf-8"):
                    if line.strip():
                        sink.write(line)
                        rows += 1
        if not rows:
            raise RuntimeError("successful searches produced no distillation rows")
        # A validation split is required by the shared trainer; use a stable,
        # disjoint tail of the generated decision rows.
        all_rows = [json.loads(line) for line in target.open(encoding="utf-8") if line.strip()]
        source_ids = sorted({str(row["source_id"]) for row in all_rows})
        validation_sources = {
            source_id
            for source_id in source_ids
            if int(hashlib.sha256(f"17:{source_id}".encode()).hexdigest(), 16) % 10 == 0
        }
        if not validation_sources and source_ids:
            validation_sources = {source_ids[0]}
        train_rows = [row for row in all_rows if str(row["source_id"]) not in validation_sources]
        validation_rows = [row for row in all_rows if str(row["source_id"]) in validation_sources]
        if not train_rows or not validation_rows:
            raise RuntimeError("distillation dataset is too small")
        def encode(rows: list[dict]) -> str:
            return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
        (args.distill_dir / "valid.jsonl").write_text(encode(validation_rows), encoding="utf-8")
        target.write_text(encode(train_rows), encoding="utf-8")
        manifest = {
            "artifact_type": "natural_language_value_search_distillation_v2",
            "status": "validated_pilot",
            "training_allowed": True,
            "selection_uses_endpoint_only_after_product_only_search": True,
            "search_summary": report,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "train_reactions": len(source_ids) - len(validation_sources),
            "validation_reactions": len(validation_sources),
            "train_sha256": digest(target),
            "validation_sha256": digest(args.distill_dir / "valid.jsonl"),
        }
        (args.distill_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (args.distill_dir / "ARTIFACT_STATUS.json").write_text(
            json.dumps(
                {
                    "artifact_id": manifest["artifact_type"],
                    "status": "validated_pilot",
                    "training_allowed": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
