#!/usr/bin/env python3
"""Validate that all ICLR baselines use identical frozen row ids and endpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row.get("id")): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", required=True, help="name=path.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    datasets = {}
    for item in args.task:
        name, value = item.split("=", 1)
        datasets[name] = load(Path(value))
    names = sorted(datasets)
    reference = set(datasets[names[0]])
    report = {"reference": names[0], "n_reference": len(reference), "tasks": {}, "ok": True}
    for name in names:
        ids = set(datasets[name])
        missing = sorted(reference - ids)
        extra = sorted(ids - reference)
        endpoint_mismatch = []
        for row_id in sorted(reference & ids):
            ref_meta = datasets[names[0]][row_id].get("metadata") or {}
            meta = datasets[name][row_id].get("metadata") or {}
            if str(ref_meta.get("core_precursor") or "") != str(meta.get("core_precursor") or ""):
                endpoint_mismatch.append(row_id)
        task_ok = not missing and not extra and not endpoint_mismatch
        report["tasks"][name] = {
            "n": len(ids),
            "missing_ids": missing[:100],
            "extra_ids": extra[:100],
            "endpoint_mismatch_ids": endpoint_mismatch[:100],
            "ok": task_ok,
        }
        report["ok"] = report["ok"] and task_ok
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
