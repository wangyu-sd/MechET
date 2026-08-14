#!/usr/bin/env python3
"""Import one hash-verified source from an existing evidence snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    args = parser.parse_args()
    source_manifest_path = args.input / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in source_manifest.get("artifacts") or []
        if row.get("source_id") == args.source_id
    ]
    if not rows:
        raise ValueError(f"no artifacts for {args.source_id} in {args.input}")
    copied = 0
    for row in rows:
        relative = Path(str(row["path"]))
        source = args.input / args.source_id / relative
        target = args.output / args.source_id / relative
        if not source.exists() or digest(source) != row.get("sha256"):
            raise ValueError(f"source hash mismatch: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    args.output.mkdir(parents=True, exist_ok=True)
    target_manifest_path = args.output / "manifest.json"
    target_manifest = (
        json.loads(target_manifest_path.read_text(encoding="utf-8"))
        if target_manifest_path.exists()
        else {}
    )
    retained = [
        row
        for row in target_manifest.get("artifacts") or []
        if row.get("source_id") != args.source_id
    ]
    target_manifest.update(
        {
            "schema_version": max(int(target_manifest.get("schema_version") or 1), 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": sorted(
                set([*(target_manifest.get("sources") or []), args.source_id])
            ),
            "artifacts": [*retained, *rows],
            "source_snapshots": {
                **dict(target_manifest.get("source_snapshots") or {}),
                args.source_id: {
                    "imported_from": str(args.input),
                    "source_manifest_sha256": digest(source_manifest_path),
                    "artifacts": copied,
                },
            },
            "candidate_evidence_only": True,
        }
    )
    target_manifest_path.write_text(
        json.dumps(target_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "copied": copied,
                "manifest": str(target_manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
