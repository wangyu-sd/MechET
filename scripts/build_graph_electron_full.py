#!/usr/bin/env python3
"""Compile the complete strict FlowER universe into graph-policy decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

from mechet.electron_policy_protocol import (
    CompressedTrajectory,
    PROTOCOL_VERSION,
    STAGE_ONLINE_RL,
    STAGE_STATE_BC,
    STAGE_TRAJECTORY_BC,
)
from mechet.graph_fragment_actions import classify_imports, decompose_reactive_fragment


EXPECTED = {"train": 257167, "valid": 2890, "test": 28967}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def append_fragment(state: str, fragment: str) -> str:
    return f"{state}.{fragment}" if state else fragment


def compile_row(payload: tuple[int, str]) -> tuple[int, list[dict[str, Any]]]:
    row_index, line = payload
    row = json.loads(line)
    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    target = str(plan.get("target_smiles") or "")
    if not target:
        raise ValueError(f"row {row_index} lacks target_smiles")
    current = target
    decisions: list[dict[str, Any]] = []
    supervision = iter(classify_imports(plan))
    history = CompressedTrajectory()

    def add_import(fragment: str) -> None:
        nonlocal current, history
        item = next(supervision)
        if item.fragment != fragment:
            raise ValueError(f"row {row_index}: chronological import supervision drift")
        decision: dict[str, Any] = {
            "reaction_id": row.get("id"),
            "kind": item.kind,
            "current": current,
            "target": target,
            "fragment": fragment,
        }
        # Both reactive and spectator imports are generated as molecular graph
        # programs.  IMPORT_ENV therefore no longer retrieves or ranks a
        # train-only fragment catalog.  Empty active atoms explicitly identify
        # an environment fragment that creates no next-event obligation.
        decision["program"] = decompose_reactive_fragment(
            fragment,
            participating_maps=item.participating_maps,
            role=str(item.role) if item.role else "ENVIRONMENT",
        ).to_dict()
        decision["history"] = history.to_dict()
        decisions.append(decision)
        history = history.append(decision)
        current = append_fragment(current, fragment)

    for fragment in plan.get("initial_imports") or ():
        add_import(str(fragment))
    for step in plan.get("steps") or ():
        for fragment in step.get("imports") or ():
            add_import(str(fragment))
        moves = list(step.get("moves") or ())
        kind = (
            "BE_DELTA"
            if len(moves) == 1 and moves[0].get("mode") == "BE_DELTA"
            else "FLOW"
        )
        decision = {
            "reaction_id": row.get("id"),
            "kind": kind,
            "current": current,
            "target": target,
            "moves": moves,
            "history": history.to_dict(),
        }
        decisions.append(decision)
        history = history.append(decision)
        current = str(step["state_after"])
    decisions.append(
        {
            "reaction_id": row.get("id"),
            "kind": "FINISH",
            "current": current,
            "target": target,
            "history": history.to_dict(),
        }
    )
    try:
        next(supervision)
    except StopIteration:
        pass
    else:
        raise ValueError(f"row {row_index}: unconsumed import supervision")
    return row_index, decisions


def indexed_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open() as handle:
        for index, line in enumerate(handle):
            yield index, line


def compile_split(
    source: Path,
    output: Path,
    *,
    split: str,
    workers: int,
    shards: int,
) -> dict[str, Any]:
    partials = [output / f"{split}.rank{index:02d}.jsonl.partial" for index in range(shards)]
    finals = [output / f"{split}.rank{index:02d}.jsonl" for index in range(shards)]
    handles = [path.open("w") for path in partials]
    family_counts: Counter[str] = Counter()
    shard_rows = [0] * shards
    reaction_rows = 0
    decision_index = 0
    started = time.time()
    context = mp.get_context("spawn")
    try:
        with context.Pool(processes=workers, maxtasksperchild=1000) as pool:
            for row_index, decisions in pool.imap(
                compile_row, indexed_lines(source), chunksize=4
            ):
                if row_index != reaction_rows:
                    raise ValueError(f"{split}: ordered worker stream drift at {row_index}")
                reaction_rows += 1
                for decision in decisions:
                    shard = decision_index % shards
                    decision["decision_index"] = decision_index
                    handles[shard].write(json.dumps(decision, separators=(",", ":")) + "\n")
                    family_counts[decision["kind"]] += 1
                    shard_rows[shard] += 1
                    decision_index += 1
                if reaction_rows % 1000 == 0:
                    rate = reaction_rows / max(1e-6, time.time() - started)
                    print(
                        f"[graph-build] split={split} reactions={reaction_rows}/{EXPECTED[split]} "
                        f"decisions={decision_index} rate={rate:.2f}rxn/s",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.close()
    if reaction_rows != EXPECTED[split]:
        raise ValueError(f"{split}: expected {EXPECTED[split]} reactions, got {reaction_rows}")
    for partial, final in zip(partials, finals):
        os.replace(partial, final)
    return {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "reactions": reaction_rows,
        "decisions": decision_index,
        "family_counts": dict(family_counts),
        "shards": [
            {"file": path.name, "rows": rows, "sha256": sha256_file(path)}
            for path, rows in zip(finals, shard_rows)
        ],
        "wall_seconds": time.time() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shards", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.shards < 1:
        raise SystemExit("workers and shards must be positive")
    source_manifest = json.loads(args.source_manifest.read_text())
    if not source_manifest.get("strict_trace_universe_complete"):
        raise SystemExit("source is not the frozen strict trace universe")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "artifact_type": "graph_electron_direct_pointer_decisions",
        "policy_protocol": {
            "version": PROTOCOL_VERSION,
            "tracks": ["llm", "graph"],
            "stages": [STAGE_STATE_BC, STAGE_TRAJECTORY_BC, STAGE_ONLINE_RL],
            "history": "accepted_action_ledger_without_state_snapshots_or_gold_horizon",
        },
        "action_contract": {
            "flow": "direct_conditional_node_pointers_no_candidate_inventory",
            "imports": "open_graph_program_for_environment_and_reactive_fragments",
            "executor": "post_generation_validation_only",
        },
        "source_manifest": str(args.source_manifest),
        "reaction_denominator": EXPECTED,
        "shard_count": args.shards,
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        source = args.source_root / f"{split}.jsonl"
        frozen = source_manifest["splits"][split]
        actual_sha = sha256_file(source)
        if int(frozen["rows"]) != EXPECTED[split] or actual_sha != frozen["sha256"]:
            raise SystemExit(f"{split}: frozen source contract mismatch")
        print(f"[graph-build] verified split={split} rows={EXPECTED[split]} sha256={actual_sha}", flush=True)
        result = compile_split(
            source,
            args.output,
            split=split,
            workers=args.workers,
            shards=args.shards,
        )
        manifest["splits"][split] = result
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[graph-build] completed manifest={manifest_path} train_decisions="
        f"{manifest['splits']['train']['decisions']} action_contract=direct_pointer",
        flush=True,
    )


if __name__ == "__main__":
    main()
