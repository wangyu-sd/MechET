#!/usr/bin/env python3
"""Add compact gold-trajectory history to natural-language decision SFT.

This is a lossless protocol transform over the validated standard trajectories.
It does not sample the actor, inject errors, expose gold horizon information, or
change any supervised tool call.  The history can be reconstructed identically
by the inference runtime from accepted tool results.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.trajectory_history import TrajectoryHistory


EXPECTED_REACTIONS = {"train": 257167, "valid": 2890, "test": 28967}
VERSION = "natural_language_electron_event_history_v2"
DECISION_CONTRACT = "unified_inventory_compressed_history_tool_decision_v2"
PROMPT_SUFFIX = "\nChoose the single next retrosynthetic action."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_exchange(row: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    messages = list(row.get("messages") or [])
    assistants = [m for m in messages if m.get("role") == "assistant"]
    tools = [m for m in messages if m.get("role") == "tool"]
    if len(assistants) != 1 or len(tools) != 1:
        raise ValueError(f"expected one paired tool exchange: {row.get('id')}")
    calls = list(assistants[0].get("tool_calls") or [])
    if len(calls) != 1:
        raise ValueError(f"expected one supervised tool call: {row.get('id')}")
    function = dict(calls[0].get("function") or {})
    name = str(function.get("name") or "")
    if tools[0].get("name") != name:
        raise ValueError(f"tool result mismatch: {row.get('id')}")
    arguments = dict(function.get("arguments") or {})
    result = dict(json.loads(str(tools[0].get("content") or "{}")))
    return name, arguments, result


def add_history(row: dict[str, Any], history: TrajectoryHistory) -> dict[str, Any]:
    output = deepcopy(row)
    users = [m for m in output.get("messages") or [] if m.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"expected one user prompt: {row.get('id')}")
    prompt = str(users[0].get("content") or "")
    if not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError(f"unexpected prompt contract: {row.get('id')}")
    users[0]["content"] = (
        prompt[: -len(PROMPT_SUFFIX)]
        + "\n\n"
        + history.render()
        + PROMPT_SUFFIX
    )
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        {
            "representation": VERSION,
            "decision_contract": DECISION_CONTRACT,
            "history_contract": "executor_compact_accepted_actions_v1",
            "history_accepted_actions": len(history.accepted_action_types),
            "history_model_visible_gold_horizon": False,
            "history_contains_failed_actions": False,
        }
    )
    output["metadata"] = metadata
    output["task_type"] = VERSION
    output["id"] = f"{row['id']}::history_v2"
    encoded_messages = json.dumps(output["messages"], ensure_ascii=False)
    forbidden = ("remaining_events", "expected_precursor", "reference_successor")
    if any(value in encoded_messages for value in forbidden):
        raise ValueError(f"private trajectory field leaked into prompt: {row.get('id')}")
    return output


def transform_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    current_reaction = ""
    completed: set[str] = set()
    history = TrajectoryHistory()
    expected_index = 0
    for row in rows:
        reaction = str(row.get("source_id") or "")
        if not reaction:
            raise ValueError("missing source reaction ID")
        if reaction != current_reaction:
            if reaction in completed:
                raise ValueError(f"non-contiguous reaction decisions: {reaction}")
            if current_reaction:
                completed.add(current_reaction)
            current_reaction = reaction
            history = TrajectoryHistory()
            expected_index = 0
        metadata = dict(row.get("metadata") or {})
        if int(metadata.get("decision_index", -1)) != expected_index:
            raise ValueError(f"decision index discontinuity: {row.get('id')}")
        name, arguments, result = _tool_exchange(row)
        yield add_history(row, history)
        history = history.accept(name, arguments, result)
        expected_index += 1


def build_split(source: Path, target: Path, limit_reactions: int = 0) -> dict[str, Any]:
    temporary = target.with_suffix(".jsonl.building")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    reactions: set[str] = set()
    counts: Counter[str] = Counter()

    def source_rows() -> Iterable[dict[str, Any]]:
        selected_reactions = 0
        active = ""
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = dict(json.loads(line))
                reaction = str(row.get("source_id") or "")
                if reaction != active:
                    if limit_reactions and selected_reactions >= limit_reactions:
                        break
                    active = reaction
                    selected_reactions += 1
                yield row

    with temporary.open("w", encoding="utf-8") as out:
        for row in transform_rows(source_rows()):
            reactions.add(str(row["source_id"]))
            dtype = str((row.get("metadata") or {}).get("decision_type") or "unknown")
            counts[f"{dtype}_decisions"] += 1
            counts["decision_rows"] += 1
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if counts["decision_rows"] % 100000 == 0:
                print(
                    f"[history-sft] split={source.stem} reactions={len(reactions)} "
                    f"decisions={counts['decision_rows']}",
                    flush=True,
                )
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, target)
    report = {
        "source": str(source),
        "target": str(target),
        "reactions": len(reactions),
        **dict(counts),
        "source_sha256": sha256(source),
        "output_sha256": sha256(target),
        "gold_standard_only": True,
        "failed_actions": 0,
        "model_visible_gold_horizon": False,
    }
    target.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-reactions", type=int, default=0)
    args = parser.parse_args()
    reports = {}
    for split in ("valid", "test", "train"):
        reports[split] = build_split(
            args.source_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            args.limit_reactions,
        )
        if not args.limit_reactions and reports[split]["reactions"] != EXPECTED_REACTIONS[split]:
            raise ValueError(f"{split} reaction denominator mismatch")
    manifest = {
        "artifact_type": VERSION,
        "source_artifact": str(args.source_dir),
        "reaction_denominator": {k: v["reactions"] for k, v in reports.items()},
        "decision_rows": {k: v["decision_rows"] for k, v in reports.items()},
        "splits": reports,
        "observation_contract": "target_current_state_inventory_compact_history_v2",
        "decision_contract": DECISION_CONTRACT,
        "history_contract": "executor_compact_accepted_actions_v1",
        "gold_standard_only": True,
        "failed_actions": 0,
        "model_visible_gold_horizon": False,
        "model_visible_atom_maps": False,
        "training_allowed": not bool(args.limit_reactions),
        "status": "validated_complete" if not args.limit_reactions else "diagnostic_smoke",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(
            {
                "artifact_id": VERSION,
                "training_allowed": not bool(args.limit_reactions),
                "status": manifest["status"],
                "reason": "full audited gold-history build" if not args.limit_reactions else "limited diagnostic build",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
