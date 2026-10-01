#!/usr/bin/env python3
"""Compile exact-endpoint search paths into Stage-II v2 distillation decisions.

Search may consult the private train reference *only to select a teacher path*.
Actor prompts receive the product, executor state and accepted-action capsule,
never the reference endpoint or a gold future horizon. Validation/test source
reactions are forbidden as teacher examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.build_natural_language_history_sft import transform_rows
from scripts.run_natural_language_value_search import distill_rows


def read_jsonl(path: Path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_rows(
    source_rows: list[dict[str, Any]],
    search_rows: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    source_by_id = {str(row["source_id"]): row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise ValueError("train source must have unique reaction IDs")
    output: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    seen: set[str] = set()
    skipped = 0
    for result in search_rows:
        rid = str(result["source_id"])
        if rid not in source_by_id:
            raise ValueError(f"search result {rid} is not in the declared train-only source")
        if rid in seen:
            raise ValueError(f"duplicate search result for {rid}")
        seen.add(rid)
        if not bool(result.get("successful_full_exact")):
            skipped += 1
            continue
        event_rows = distill_rows(result, source_by_id[rid])
        if not event_rows:
            raise ValueError(f"{rid}: exact endpoint result lacks executable action path")
        history_rows = list(transform_rows(event_rows))
        if history_rows[-1]["messages"][-2]["tool_calls"][0]["function"]["name"] != "finish_trace":
            raise ValueError(f"{rid}: teacher path does not finish")
        split = (
            "valid"
            if int.from_bytes(hashlib.sha256(f"17:{rid}".encode()).digest()[:4], "big") % 10 == 0
            else "train"
        )
        for row in history_rows:
            metadata = dict(row.get("metadata") or {})
            metadata.update({
                "teacher_provenance": "search_generated_executor_verified_full_endpoint",
                "search_private_reference_selected": True,
                "reference_endpoint_model_visible": False,
                "gold_standard_only": False,
            })
            row["metadata"] = metadata
            row["task_type"] = "vnext_search_distilled_stage2_v2"
            output[split].append(row)
    train_ids = {row["source_id"] for row in output["train"]}
    valid_ids = {row["source_id"] for row in output["valid"]}
    if train_ids & valid_ids:
        raise ValueError("distillation reaction leakage")
    if not output["train"] or not output["valid"]:
        raise ValueError("distillation requires nonempty train and internal valid")
    return output, {
        "source_reactions": len(source_by_id),
        "search_reactions": len(seen),
        "exact_teacher_reactions": len(train_ids | valid_ids),
        "nonexact_search_reactions_skipped": skipped,
        "train_decisions": len(output["train"]),
        "valid_decisions": len(output["valid"]),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-source", type=Path, required=True)
    p.add_argument("--search-result", type=Path, action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    staging = args.output_dir.with_name(args.output_dir.name + ".building")
    if staging.exists():
        raise FileExistsError(staging)
    source = list(read_jsonl(args.train_source))
    results = [r for path in args.search_result for r in read_jsonl(path)]
    splits, counts = build_rows(source, results)
    staging.mkdir(parents=True)
    manifest = {
        "artifact_type": "vnext_search_distilled_stage2_v2",
        "source_train_sha256": hashlib.sha256(args.train_source.read_bytes()).hexdigest(),
        "search_result_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.search_result
        },
        "selection_contract": "train_only_full_endpoint_exact_executor_verified",
        "reference_endpoint_model_visible": False,
        "search_private_reference_selected": True,
        "reaction_overlap": 0,
        "counts": counts,
        "splits": {},
    }
    for split, rows in splits.items():
        path = staging / f"{split}.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "decisions": len(rows),
            "reactions": len({r["source_id"] for r in rows}),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (staging / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "status": "validated_search_teacher_subset",
        "training_allowed": True,
        "not_full_reaction_universe": True,
        "search_private_reference_selected": True,
    }, indent=2))
    staging.rename(args.output_dir)
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
