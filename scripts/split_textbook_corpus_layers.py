#!/usr/bin/env python3
"""Physically separate redistributable and non-commercial textbook layers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.textbook_store import TextbookStore


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    combined = TextbookStore.load(args.corpus)
    redistributable = TextbookStore(
        [row for row in combined.passages if "NC" not in row.license.upper()]
    )
    noncommercial = TextbookStore(
        [row for row in combined.passages if "NC" in row.license.upper()]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "redistributable": args.output_dir / "passages.redistributable.jsonl",
        "noncommercial": args.output_dir / "passages.noncommercial.jsonl",
    }
    redistributable.save(paths["redistributable"])
    noncommercial.save(paths["noncommercial"])
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "combined_input": str(args.corpus),
        "combined": combined.manifest(),
        "layers": {
            "redistributable": {
                **redistributable.manifest(),
                "path": str(paths["redistributable"]),
                "sha256": sha256_file(paths["redistributable"]),
                "policy": "contains no license identifier with NC",
            },
            "noncommercial": {
                **noncommercial.manifest(),
                "path": str(paths["noncommercial"]),
                "sha256": sha256_file(paths["noncommercial"]),
                "policy": "non-commercial research only; preserve attribution/share-alike",
            },
        },
    }
    output = args.output_dir / "layers.manifest.json"
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["layers"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
