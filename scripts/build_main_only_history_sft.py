#!/usr/bin/env python3
"""Build Stage-II accepted-prefix Tool-SFT from audited main-product-only Stage-I.

Does not rebuild reactants, alter event labels, or reveal gold future actions.
Unlike the FlowER-only legacy history builder, this uses the verified subset
denominators from each separately audited corpus (including mech-31k).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.build_natural_language_history_sft import transform_rows


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as r:
        for b in iter(lambda: r.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    src = args.state_dir
    status = json.loads((src / "ARTIFACT_STATUS.json").read_text())
    manifest = json.loads((src / "manifest.json").read_text())
    if status.get("training_allowed") is not True:
        raise ValueError("Unapproved Stage-I reference supervision")
    if status.get("input_condition") != "main_product_only_v1":
        raise ValueError("History source is not main-product-only")
    expected = manifest["reaction_denominator"]
    if args.output_dir.exists():
        raise ValueError("Refusing to overwrite previous Stage-II history data")
    args.output_dir.mkdir(parents=True)
    reports: dict[str, Any] = {}
    start = time.monotonic()
    for split in ("train", "valid", "test"):
        origin = src / f"{split}.jsonl"
        outpath = args.output_dir / f"{split}.jsonl"
        tmp = outpath.with_suffix(".jsonl.building")
        n = 0
        ids = set()
        counts: Counter[str] = Counter()
        active = ""
        with origin.open(encoding="utf-8") as reader, tmp.open("w", encoding="utf-8") as writer:
            def yield_source():
                for line in reader:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (row.get("metadata") or {}).get("input_condition") != "main_product_only_v1":
                        raise ValueError(f"{split}: non-main-only supervised decision")
                    yield row
            for row in transform_rows(yield_source()):
                rid = str(row["source_id"])
                if rid != active:
                    if rid in ids:
                        raise ValueError(f"{split}: repeated noncontiguous reaction")
                    ids.add(rid)
                    active = rid
                name = str(row["metadata"]["decision_type"])
                counts[name] += 1
                n += 1
                writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if n % 50000 == 0:
                    print(json.dumps({
                        "phase": "MAIN_ONLY_STAGE_II_HISTORY",
                        "split": split, "decisions": n, "reactions": len(ids),
                        "elapsed_sec": round(time.monotonic()-start,1),
                    }), flush=True)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp, outpath)
        if len(ids) != int(expected[split]) or counts["finish"] != len(ids):
            raise ValueError(
                f"{split}: history converted denominator mismatch: {len(ids)}/{counts['finish']} != {expected[split]}"
            )
        reports[split] = {
            "reactions": len(ids), "decision_rows": n,
            "tool_type_counts": dict(counts),
            "source_sha256": sha(origin), "output_sha256": sha(outpath),
        }
    document = {
        "artifact_type": "main_product_only_compressed_history_sft_v2",
        "input_condition": "main_product_only_v1",
        "observation_contract": "target_current_state_inventory_compact_history_v2",
        "decision_contract": "unified_inventory_compressed_history_tool_decision_v2",
        "history_contract": "executor_compact_accepted_actions_v1",
        "source_stage1_manifest_sha256": sha(src / "manifest.json"),
        "source_status_sha256": sha(src / "ARTIFACT_STATUS.json"),
        "reports": reports,
        "reaction_denominator": {k:r["reactions"] for k,r in reports.items()},
        "original_reaction_denominator": manifest["original_reaction_denominator"],
        "decision_rows": {k:r["decision_rows"] for k,r in reports.items()},
        "training_allowed": True,
        "gold_future_visible": False,
        "zero_failed_actions": True,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "status": "validated_explicitly_quarantined_subset",
        "training_allowed": True,
        "input_condition": "main_product_only_v1",
        "source_stage1_manifest_sha256": document["source_stage1_manifest_sha256"],
    }, indent=2) + "\n")
    return document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(build(args), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
