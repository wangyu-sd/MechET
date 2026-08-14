#!/usr/bin/env python3
"""Select mech-USPTO reactions whose every raw elementary step is executable."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.trace_stitching import complete_reaction_ids  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument("--standardized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_ids = [
        str(int(value))
        for value in pd.read_parquet(
            args.raw_parquet, columns=["rxn_idx"]
        )["rxn_idx"]
    ]
    standardized_rows = [
        json.loads(line)
        for line in args.standardized.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    complete = complete_reaction_ids(raw_ids, standardized_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in complete),
        encoding="utf-8",
    )
    report = {
        "raw_parquet": str(args.raw_parquet),
        "raw_parquet_sha256": sha256(args.raw_parquet),
        "raw_step_rows": len(raw_ids),
        "raw_reactions": len(set(raw_ids)),
        "standardized": str(args.standardized),
        "standardized_sha256": sha256(args.standardized),
        "standardized_rows": len(standardized_rows),
        "standardized_steps": sum(
            len(row.get("steps") or []) for row in standardized_rows
        ),
        "complete_reactions": len(complete),
        "incomplete_reactions": len(set(raw_ids)) - len(complete),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
