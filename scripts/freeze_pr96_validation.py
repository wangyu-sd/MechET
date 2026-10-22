#!/usr/bin/env python3
"""Freeze paired and disjoint validation reaction IDs before model training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.pr96_validation import frozen_cohorts
from scripts.build_mech31k_main_product_only_sft import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    manifest = args.data_dir / "manifest.json"
    data = json.loads(manifest.read_text())
    audit = args.data_dir / "audits/valid.jsonl"
    if sha(audit) != data["reports"]["valid"]["outputs"]["audits"]["sha256"]:
        raise ValueError("VALID_AUDIT_HASH_MISMATCH")
    rows = [json.loads(l) for l in audit.open()]
    cohort = frozen_cohorts(rows, seed=args.seed)
    cohort.update(data_manifest_sha256=sha(manifest), valid_audit_sha256=sha(audit),
                  source_track=data["track"], stage_one_started=False,
                  test_used=False, frozen_before_training=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(cohort, indent=2) + "\n")
    print(json.dumps({"stage": "freeze-validation", "path": str(args.output),
                      "sha256": sha(args.output), **cohort}), flush=True)


if __name__ == "__main__":
    main()
