#!/usr/bin/env python3
"""Recover a resumable JSONL artifact after an interrupted concurrent append."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    if args.backup.exists():
        raise FileExistsError(f"backup already exists: {args.backup}")
    temporary = args.path.with_name(args.path.name + ".repairing")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")

    seen: set[str] = set()
    read = written = duplicates = malformed = 0
    with args.path.open("r", encoding="utf-8") as source, temporary.open(
        "x", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            read += 1
            try:
                row = dict(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
                print(f"discard malformed line {line_number}", flush=True)
                continue
            identifier = str(row.get("source_id") or "")
            if not identifier:
                raise ValueError(f"accepted row has no source_id at line {line_number}")
            if identifier in seen:
                duplicates += 1
                continue
            seen.add(identifier)
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
        target.flush()
        os.fsync(target.fileno())

    args.path.rename(args.backup)
    temporary.rename(args.path)
    print(
        json.dumps(
            {
                "path": str(args.path),
                "backup": str(args.backup),
                "read": read,
                "written_unique": written,
                "duplicates_removed": duplicates,
                "malformed_removed": malformed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
