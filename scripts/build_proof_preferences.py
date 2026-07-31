#!/usr/bin/env python3
"""Build safe DPO pairs from executor-labelled corruptions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.proof_curriculum import ProofCorruption, preference_pair_from_corruption


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corruptions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.corruptions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            fields = {
                key: row.get(key)
                for key in ProofCorruption.__dataclass_fields__
            }
            fields["changed_lines"] = tuple(fields.get("changed_lines") or [])
            corruption = ProofCorruption(**fields)
            pair = preference_pair_from_corruption(
                corruption,
                prompt_messages=list(row.get("prompt_messages") or []),
            )
            if pair is None:
                skipped += 1
                continue
            pair["metadata"] = {
                **dict(pair.get("metadata") or {}),
                "source_metadata": dict(row.get("source_metadata") or {}),
            }
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1
    manifest = {
        "input": str(args.corruptions),
        "output": str(args.output),
        "n_input": len(rows),
        "pairs_written": written,
        "skipped": skipped,
        "policy": "executable proof preferred only over formally invalid proof",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
