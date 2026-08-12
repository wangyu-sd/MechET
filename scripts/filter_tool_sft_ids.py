#!/usr/bin/env python3
"""Filter explicitly audited Tool-SFT IDs and record a hash-bound report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--id", action="append", dest="ids", required=True)
    args = parser.parse_args()

    excluded = set(args.ids)
    found: set[str] = set()
    kept = removed = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.quarantine.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.input.open(encoding="utf-8") as source,
        args.output.open("w", encoding="utf-8") as accepted,
        args.quarantine.open("w", encoding="utf-8") as quarantine,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("id") or "")
            if identifier in excluded:
                found.add(identifier)
                quarantine.write(line)
                removed += 1
            else:
                accepted.write(line)
                kept += 1
    missing = sorted(excluded - found)
    if missing:
        raise ValueError(f"requested IDs were absent: {missing}")
    report = {
        "artifact_type": "explicit_tool_sft_id_filter",
        "reason": args.reason,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "quarantine": str(args.quarantine),
        "quarantine_sha256": sha256(args.quarantine),
        "kept": kept,
        "removed": removed,
        "removed_ids": sorted(found),
    }
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
