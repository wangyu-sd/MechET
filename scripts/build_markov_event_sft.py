#!/usr/bin/env python3
"""Expand replayed FlowER traces into runtime-matched one-step decisions.

Each output row conditions on TARGET and the executor's current SMILES state,
then supervises exactly one tool call.  Gold horizon fields are deliberately
removed.  The paired tool result is retained for chat-template validity but is
causally after the supervised assistant span.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_REACTIONS = {"train": 257167, "valid": 2890, "test": 28967}
SYSTEM = (
    "Infer the precursor one executable electron-flow decision at a time. "
    "Given TARGET and the executor-owned CURRENT STATE, call exactly one tool: "
    "apply_grounded_event for the next chemical event, or finish_trace only "
    "when the current state is the complete precursor. For an event, import "
    "only first-use unmapped fragments; copy the exact CURRENT STATE and insert "
    "event-local <A>..<Z> markers immediately before atom occurrences; then "
    "emit compact FLOW clauses. A>AB means LP(A)->BOND(A,B), AB>B means "
    "BOND(A,B)->ATOM(B), and AB>BC means a bond shift. The environment owns "
    "state transitions and endpoint derivation."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_state(row: dict[str, Any]) -> str:
    users = [m for m in row["messages"] if m.get("role") == "user"]
    if len(users) != 1:
        raise ValueError(f"expected one user message: {row.get('id')}")
    text = str(users[0].get("content") or "")
    marker = "\nCURRENT STATE: "
    if marker not in text:
        raise ValueError(f"missing current state: {row.get('id')}")
    return text.split(marker, 1)[1]


def public_result(name: str, content: str) -> dict[str, Any]:
    value = dict(json.loads(content))
    if name == "apply_grounded_event":
        return {
            "ok": bool(value.get("ok")),
            "code": str(value.get("code") or ""),
            "current_state": str(value.get("current_state") or ""),
        }
    return value


def decision_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = list(row["messages"])
    tools = list(row["tools"])
    target = str(row["target_smiles"])
    state = initial_state(row)
    output: list[dict[str, Any]] = []
    decision_index = 0
    for index, message in enumerate(messages):
        calls = list(message.get("tool_calls") or [])
        if message.get("role") != "assistant" or not calls:
            continue
        if len(calls) != 1 or index + 1 >= len(messages):
            raise ValueError(f"invalid assistant call: {row.get('id')}:{index}")
        result_message = messages[index + 1]
        call = calls[0]
        name = str((call.get("function") or {}).get("name") or "")
        if result_message.get("role") != "tool" or result_message.get("name") != name:
            raise ValueError(f"unpaired tool call: {row.get('id')}:{index}")
        result = public_result(name, str(result_message.get("content") or "{}"))
        if result.get("ok") is not True:
            raise ValueError(f"gold call did not pass: {row.get('id')}:{index}")
        call_id = f"decision_{decision_index:03d}"
        clean_call = {
            "id": call_id,
            "type": "function",
            "function": dict(call["function"]),
        }
        decision_type = "finish" if name == "finish_trace" else "event"
        identifier = f"{row['source_id']}::decision_{decision_index:03d}_{decision_type}"
        metadata = {
            "representation": "markov_grounded_event_v2",
            "decision_contract": "markov_tool_decision_v1",
            "reaction_id": row["source_id"],
            "decision_index": decision_index,
            "decision_type": decision_type,
            "executor_replayed": True,
            "mapping_model_visible": False,
            "gold_horizon_model_visible": False,
            "source_trace_digest": (row.get("metadata") or {}).get("trace_digest"),
        }
        output.append(
            {
                "id": identifier,
                "source_id": row["source_id"],
                "artifact_type": "supervision",
                "task_type": "markov_grounded_event_v2",
                "target_smiles": target,
                "expected_precursor": row["expected_precursor"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"TARGET: {target}\nCURRENT STATE: {state}",
                    },
                    {"role": "assistant", "content": "", "tool_calls": [clean_call]},
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, separators=(",", ":")),
                    },
                ],
                "tools": tools,
                "metadata": metadata,
            }
        )
        if name == "apply_grounded_event":
            state = str(result["current_state"])
        elif name != "finish_trace":
            raise ValueError(f"unexpected tool {name}: {row.get('id')}")
        decision_index += 1
    expected = int((row.get("metadata") or {}).get("n_events", -1)) + 1
    if len(output) != expected or output[-1]["metadata"]["decision_type"] != "finish":
        raise ValueError(f"decision coverage mismatch: {row.get('id')}")
    return output


def build_split(source: Path, target: Path, limit: int = 0) -> tuple[dict[str, Any], set[str]]:
    temporary = target.with_suffix(".jsonl.building")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    reactions: set[str] = set()
    decisions: set[str] = set()
    with source.open(encoding="utf-8") as inp, temporary.open("w", encoding="utf-8") as out:
        for reaction_index, line in enumerate(inp):
            if limit and reaction_index >= limit:
                break
            if not line.strip():
                continue
            row = dict(json.loads(line))
            reaction_id = str(row.get("source_id") or "")
            if not reaction_id or reaction_id in reactions:
                raise ValueError(f"duplicate reaction ID: {reaction_id}")
            reactions.add(reaction_id)
            for decision in decision_rows(row):
                identifier = str(decision["id"])
                if identifier in decisions:
                    raise ValueError(f"duplicate decision ID: {identifier}")
                decisions.add(identifier)
                dtype = decision["metadata"]["decision_type"]
                counts[f"{dtype}_decisions"] += 1
                encoded = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
                if "remaining_events" in encoded:
                    raise ValueError(f"gold horizon leakage: {identifier}")
                out.write(encoded + "\n")
            if len(reactions) % 5000 == 0:
                print(
                    f"[markov-build] split={source.stem} reactions={len(reactions)} "
                    f"decisions={len(decisions)}",
                    flush=True,
                )
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, target)
    report = {
        "source": str(source),
        "target": str(target),
        "reactions": len(reactions),
        "decision_rows": len(decisions),
        **dict(counts),
        "source_sha256": sha256(source),
        "output_sha256": sha256(target),
        "remaining_events_fields": 0,
        "one_paired_tool_call_per_row": True,
    }
    target.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    return report, reactions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    reports: dict[str, Any] = {}
    ids: dict[str, set[str]] = {}
    for split in ("valid", "test", "train"):
        reports[split], ids[split] = build_split(
            args.source_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            args.limit,
        )
        if not args.limit and reports[split]["reactions"] != EXPECTED_REACTIONS[split]:
            raise ValueError(f"{split} reaction denominator mismatch")
    overlaps = {
        "train_valid": len(ids["train"] & ids["valid"]),
        "train_test": len(ids["train"] & ids["test"]),
        "valid_test": len(ids["valid"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"reaction ID overlap: {overlaps}")
    manifest = {
        "artifact_type": "markov_grounded_event_sft_v2",
        "observation_contract": "target_plus_executor_current_state_v1",
        "decision_contract": "markov_tool_decision_v1",
        "source_artifact": str(args.source_dir),
        "reaction_denominator": {k: v["reactions"] for k, v in reports.items()},
        "decision_rows": {k: v["decision_rows"] for k, v in reports.items()},
        "splits": reports,
        "split_reaction_id_overlap": overlaps,
        "gold_horizon_model_visible": False,
        "model_visible_atom_maps": False,
        "training_allowed": not bool(args.limit),
        "gate_passed": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    status = {
        "artifact_id": "flower_markov_event_sft_v2",
        "training_allowed": not bool(args.limit),
        "reason": "full audited build" if not args.limit else "limited test build",
    }
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
