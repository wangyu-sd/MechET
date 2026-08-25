#!/usr/bin/env python3
"""Build State-CoT source rows for exactly the replayable trace-owned IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mechet.mech_graph import build_mechanism_graph, parse_flower_line
from scripts.build_mechet_sft import _build_row


def load_targets(path: Path, split: str) -> dict[str, str]:
    prefix = "textbook-tool-sft:"
    marker = f"flower_mech_proof_{split}_"
    targets: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row_id = str(json.loads(line).get("id") or "")
            if row_id.startswith(prefix):
                row_id = row_id[len(prefix) :]
            if not row_id.startswith(marker):
                raise ValueError(f"unexpected ID for split={split}: {row_id}")
            targets[row_id[len(marker) :]] = row_id
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-input", type=Path, required=True)
    parser.add_argument("--flower-input", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    targets = load_targets(args.trace_input, args.split)
    groups: dict[str, list[tuple[str, str]]] = {tid: [] for tid in targets}
    with args.flower_input.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_flower_line(line)
            if parsed is None:
                continue
            reactants, products, tid = parsed
            if tid in groups:
                groups[tid].append((reactants, products))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures: Counter[str] = Counter()
    accepted = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for tid, proof_id in sorted(targets.items(), key=lambda item: item[1]):
            steps = groups[tid]
            if not steps:
                failures["missing_trajectory"] += 1
                continue
            graph = build_mechanism_graph(tid, steps, source_path=str(args.flower_input))
            if graph is None:
                failures["invalid_graph"] += 1
                continue
            row = _build_row(graph, source_split=args.split)
            if row is None:
                failures["verify_failed"] += 1
                continue
            row["id"] = proof_id
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted += 1

    report = {
        "trace_input": str(args.trace_input),
        "flower_input": str(args.flower_input),
        "split": args.split,
        "requested": len(targets),
        "accepted": accepted,
        "failures": dict(failures),
        "output": str(args.output),
        "ok": accepted == len(targets),
    }
    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
