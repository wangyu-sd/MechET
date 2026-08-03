#!/usr/bin/env python3
"""Build source-to-sink composition-disjoint Tool-SFT splits."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_splits import (
    build_compositional_ood_split,
    extract_split_features,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--min-train-primitive-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    accepted: list[dict] = []
    quarantined: list[dict] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        try:
            extract_split_features(row)
            accepted.append(row)
        except Exception as exc:
            code = str(exc).split(":", 1)[0].strip() or type(exc).__name__
            reasons[code] += 1
            quarantined.append(
                {
                    "id": row.get("id"),
                    "error_code": code,
                    "error": str(exc),
                }
            )
    if not accepted:
        raise ValueError("no replay-verified source-to-sink trace rows remain")

    splits, manifest = build_compositional_ood_split(
        accepted,
        test_fraction=args.test_fraction,
        valid_fraction=args.valid_fraction,
        min_train_primitive_count=args.min_train_primitive_count,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        _write_jsonl(args.output_dir / f"{split}.jsonl", split_rows)
    quarantine_path = args.output_dir / "quarantine.jsonl"
    _write_jsonl(quarantine_path, quarantined)
    manifest.update(
        {
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "n_input": len(rows),
            "n_split_eligible": len(accepted),
            "n_quarantined": len(quarantined),
            "quarantine": str(quarantine_path),
            "quarantine_reasons": dict(reasons),
        }
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
