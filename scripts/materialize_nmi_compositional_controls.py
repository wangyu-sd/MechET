#!/usr/bin/env python3
"""Materialize matched endpoint/open-flow controls from one frozen MechComp split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import row_id


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--outcome-source", type=Path, required=True)
    p.add_argument("--open-flow-source", type=Path, required=True)
    args = p.parse_args()

    split_rows = {}
    split_ids = {}
    for split in ("train", "valid", "test"):
        path = args.split_dir / f"{split}.jsonl"
        rows = read_jsonl(path)
        split_rows[split] = rows
        ids = [row_id(r) for r in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"duplicate IDs in strict split {split}")
        split_ids[split] = ids

    all_ids = set().union(*(set(v) for v in split_ids.values()))
    if sum(len(v) for v in split_ids.values()) != len(all_ids):
        raise SystemExit("stable-ID overlap across strict train/valid/test")

    sources = {
        "outcome_only": read_jsonl(args.outcome_source),
        "open_flow": read_jsonl(args.open_flow_source),
    }
    manifest = {
        "artifact_type": "nmi_compositional_matched_controls_v1",
        "strict_split_dir": str(args.split_dir),
        "tasks": {},
    }

    # Trace-owned rows are already the authoritative strict split.
    trace_dir = args.out_root / "trace_closed_loop"
    task = {}
    for split, rows in split_rows.items():
        out = trace_dir / f"{split}.jsonl"
        write_jsonl(out, rows)
        task[split] = {"rows": len(rows), "sha256": sha(out)}
    manifest["tasks"]["trace_closed_loop"] = task

    for name, rows in sources.items():
        by_id = {row_id(r): r for r in rows}
        if len(by_id) != len(rows):
            raise SystemExit(f"duplicate IDs in source {name}")
        missing = sorted(all_ids - set(by_id))
        if missing:
            raise SystemExit(f"{name} missing {len(missing)} matched IDs; first={missing[:5]}")
        task = {}
        for split, ids in split_ids.items():
            selected = [by_id[i] for i in ids]
            if [row_id(r) for r in selected] != ids:
                raise SystemExit(f"{name}/{split}: stable-ID order drift")
            out = args.out_root / name / f"{split}.jsonl"
            write_jsonl(out, selected)
            task[split] = {"rows": len(selected), "sha256": sha(out)}
        manifest["tasks"][name] = task

    # Strong equality gate.
    counts = {
        name: tuple(manifest["tasks"][name][s]["rows"] for s in ("train", "valid", "test"))
        for name in manifest["tasks"]
    }
    if len(set(counts.values())) != 1:
        raise SystemExit(f"matched condition row-count mismatch: {counts}")
    manifest["matched_counts"] = dict(zip(("train", "valid", "test"), next(iter(counts.values()))))
    manifest["matched_id_universe"] = True
    manifest_path = args.out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
