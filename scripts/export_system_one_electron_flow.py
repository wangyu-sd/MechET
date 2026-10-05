#!/usr/bin/env python3
"""Export MechET Stage-II electron-flow decisions for a System-One policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechet.electron_pointer import (
    UnsupportedPointerEvent,
    candidate_keys,
    parse_pointer_example,
)


def label(key):
    kind, a, b = key
    return f"ATOM:A{a + 1:02d}" if kind == "atom" else f"BOND:A{a + 1:02d}-A{b + 1:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    kept = unsupported = nonflow = 0
    with args.input.open() as src, args.output.open("w") as dst:
        for line in src:
            row = json.loads(line)
            try:
                ex = parse_pointer_example(row)
            except UnsupportedPointerEvent:
                unsupported += 1
                continue
            if ex is None:
                nonflow += 1
                continue
            source = candidate_keys(len(ex.atom_names), ex.bonds, source=True)
            sink = candidate_keys(len(ex.atom_names), ex.bonds, source=False)
            record = {
                "id": ex.row_id,
                "state": next(m["content"] for m in reversed(ex.messages) if m["role"] == "user"),
                "source_options": [label(x) for x in source],
                "sink_options": [label(x) for x in sink],
                "gold_source": [source.index(x) for x in ex.source_targets],
                "gold_sink": [sink.index(x) for x in ex.sink_targets],
                "gold_pairs": [
                    [source.index(s), sink.index(t)]
                    for s, t in zip(ex.source_targets, ex.sink_targets, strict=True)
                ],
            }
            dst.write(json.dumps(record, separators=(",", ":")) + "\n")
            kept += 1
    print(json.dumps({"kept": kept, "unsupported": unsupported, "nonflow": nonflow}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
