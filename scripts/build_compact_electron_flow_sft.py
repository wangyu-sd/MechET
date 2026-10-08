#!/usr/bin/env python3
"""Convert frozen v2 decision SFT to opt-in compact electron-flow v3.

Only the system/tool schema and assistant event serialization change.
The product, current state, fragment actions, terminal actions, tool results,
data split and executor-derived successors are preserved verbatim.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from mechet.compact_electron_flow import (
    compact_from_natural_arguments, natural_from_compact_arguments,
)
from scripts.build_natural_language_event_sft import TOOLS as V2_TOOLS
from scripts.train_tool_sft import validate_conversation

VERSION = "compact_electron_flow_v3"
STATE = "unified_inventory_compact_flow_tool_decision_v3"
HISTORY = "unified_inventory_compact_flow_history_tool_decision_v3"
PARENTS = {
    "unified_inventory_tool_decision_v2": STATE,
    "unified_inventory_compressed_history_tool_decision_v2": HISTORY,
}
REACTIONS = {"train": 257167, "valid": 2890, "test": 28967}
DECISIONS = {"train": 2007421, "valid": 22341, "test": 225613}
SYSTEM = (
    "You are MechET, performing RETROSYNTHETIC electron-flow reasoning. "
    "Read TARGET PRODUCT SMILES, CURRENT STATE SMILES and ANNOTATED CURRENT STATE. "
    "Axx labels refer only to the current annotated molecular state. "
    "Call exactly one tool: import_fragments for missing participants, "
    "apply_electron_flow for one coupled electron-pair-transfer event, "
    "or finish_trace only at a complete precursor. "
    "Compact flow grammar: B(A01,A02)>A02;LP(A03)>NB(A01,A03). "
    "B is a bond electron source/sink, LP is a lone-pair source, "
    "A is an atom destination, and RP is a radical-pair container. "
    "Use DELTA|B(A01,A02):+1;Q(A03):0>+1 only for existing graph-delta events. "
    "Do not generate explanatory prose or next-state/precursor SMILES. "
    "The frozen executor applies and checks each event and owns the next state."
)
TOOLS = deepcopy(V2_TOOLS)
TOOLS[1] = {
    "type": "function",
    "function": {
        "name": "apply_electron_flow",
        "description": "Execute one coupled inverse electron-transfer event.",
        "parameters": {
            "type": "object",
            "properties": {
                "flow": {
                    "type": "string",
                    "description": (
                        "B(A01,A02)>A02;LP(A03)>NB(A01,A03); "
                        "or DELTA|B(A01,A02):+1;Q(A03):0>+1"
                    ),
                }
            },
            "required": ["flow"],
            "additionalProperties": False,
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _core_signature(arguments: Mapping[str, Any]) -> tuple[Any, ...]:
    """Ignore redundant natural-language instructions but retain all chemistry."""
    return (
        arguments["direction"],
        tuple((x["source"], x["destination"]) for x in arguments["electron_flow"]),
        tuple((tuple(x["atoms"]), int(x["delta"])) for x in arguments["bond_order_changes"]),
        tuple((x["atom"], int(x["from"]), int(x["to"])) for x in arguments["charge_changes"]),
    )


def convert_decision(row: Mapping[str, Any]) -> dict[str, Any]:
    """Turn exactly one v2 teacher decision into one v3 teacher decision."""
    parent_contract = str((row.get("metadata") or {}).get("decision_contract") or "")
    if parent_contract not in PARENTS:
        raise ValueError(f"COMPACT_UNKNOWN_PARENT_CONTRACT:{parent_contract}")
    converted = deepcopy(dict(row))
    messages = converted["messages"]
    if len(messages) != 4 or [x.get("role") for x in messages] != [
        "system", "user", "assistant", "tool"
    ]:
        raise ValueError("COMPACT_UNEXPECTED_CONVERSATION_SHAPE")
    calls = messages[2].get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError("COMPACT_EXPECTED_ONE_ASSISTANT_TOOL_CALL")
    func = calls[0]["function"]
    name = str(func["name"])
    if name == "apply_electron_flow":
        original = dict(func["arguments"])
        compact = compact_from_natural_arguments(original)
        if _core_signature(natural_from_compact_arguments(compact)) != _core_signature(original):
            raise ValueError("COMPACT_EVENT_INFORMATION_LOSS")
        func["arguments"] = compact
    elif name not in ("import_fragments", "finish_trace"):
        raise ValueError(f"COMPACT_UNSUPPORTED_TOOL:{name}")
    messages[0]["content"] = SYSTEM
    converted["tools"] = deepcopy(TOOLS)
    metadata = converted["metadata"]
    metadata["decision_contract"] = PARENTS[parent_contract]
    metadata["parent_decision_contract"] = parent_contract
    metadata["representation"] = VERSION
    metadata["compact_event_codec"] = "arrow_or_BE_DELTA_v3"
    converted["task_type"] = VERSION + (
        "_history" if PARENTS[parent_contract] == HISTORY else "_state"
    )
    converted["id"] = f"{row['id']}::compact_v3"
    validate_conversation(
        converted, require_trace_owned=False, require_tool_decision=True
    )
    return converted


def convert_split(source: Path, destination: Path, limit: int) -> dict[str, Any]:
    if source.resolve() == destination.resolve():
        raise ValueError("v3 cannot overwrite v2")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jsonl.building")
    seen: set[str] = set()
    closed: set[str] = set()
    current = ""
    decisions = events = 0
    with source.open(encoding="utf-8") as inp, temporary.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            if not line.strip():
                continue
            row = json.loads(line)
            reaction = str(row["source_id"])
            if reaction != current:
                if current:
                    closed.add(current)
                if reaction in closed:
                    raise ValueError("non-contiguous source reaction")
                if limit and len(seen) == limit:
                    break
                seen.add(reaction)
                current = reaction
            converted = convert_decision(row)
            events += int(
                converted["messages"][2]["tool_calls"][0]["function"]["name"]
                == "apply_electron_flow"
            )
            out.write(json.dumps(converted, ensure_ascii=False, separators=(",", ":")) + "\n")
            decisions += 1
    os.replace(temporary, destination)
    return {
        "reactions": len(seen), "decision_rows": decisions,
        "electron_flow_decisions": events,
        "source_sha256": sha256(source),
        "output_sha256": sha256(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=["valid", "test", "train"],
                        default=["valid", "test", "train"])
    parser.add_argument("--limit-reactions", type=int, default=0)
    args = parser.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        raise ValueError("source and output directories must differ")
    if args.limit_reactions < 0:
        raise ValueError("limit-reactions cannot be negative")
    status_path = args.source_dir / "ARTIFACT_STATUS.json"
    manifest_path = args.source_dir / "manifest.json"
    if not status_path.is_file() or not manifest_path.is_file():
        raise ValueError("frozen source v2 status and manifest required")
    if json.loads(status_path.read_text())["training_allowed"] is not True:
        raise ValueError("source v2 dataset is not authorized for training")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("reaction_denominator") != REACTIONS:
        raise ValueError("v2 strict universe differs from the expected split")
    reports = {}
    for split in args.splits:
        source = args.source_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        report = convert_split(source, args.output_dir / f"{split}.jsonl",
                               args.limit_reactions)
        frozen = (manifest.get("splits") or {}).get(split) or {}
        if report["source_sha256"] != frozen.get("output_sha256"):
            raise ValueError(f"v2 {split} file SHA disagrees with frozen manifest")
        if not args.limit_reactions and (
            report["reactions"] != REACTIONS[split]
            or report["decision_rows"] != DECISIONS[split]
        ):
            raise ValueError(f"{split}: incomplete strict source conversion")
        reports[split] = report
        print(json.dumps({"split": split, **report}), flush=True)
    complete = not args.limit_reactions and set(args.splits) == set(REACTIONS)
    output_manifest = {
        "artifact_type": VERSION,
        "source_manifest_sha256": sha256(manifest_path),
        "source_dir": str(args.source_dir),
        "reaction_denominator": {
            split: row["reactions"] for split, row in reports.items()
        },
        "decision_rows": {
            split: row["decision_rows"] for split, row in reports.items()
        },
        "splits": reports,
        "decision_contract": STATE,
        "conversion": "lossless_v2_surface_only",
        "reference_replay_verified": manifest.get("reference_replay_verified"),
        "training_allowed": bool(complete and manifest.get("reference_replay_verified")),
    }
    allowed = output_manifest["training_allowed"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps({
            "artifact_id": VERSION, "training_allowed": allowed,
            "status": "validated_complete" if allowed else "diagnostic_only",
            "reason": "full frozen v2 serialization conversion" if allowed
                      else "subset/incomplete source; training prohibited",
        }, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if allowed or args.limit_reactions else 1


if __name__ == "__main__":
    raise SystemExit(main())
