#!/usr/bin/env python3
"""Build one-decision natural-language inverse electron-flow supervision.

The source is the frozen, first-electron-use program artifact.  Each reaction
becomes import decisions (only when needed), executor-grounded electron-event
decisions, and one finish decision.  No private atom map is model-visible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.a7_rescue import canonical_event
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)
from mechet.natural_language_electron_flow import (
    build_inventory,
    compile_event_arguments,
    render_event_arguments,
)
from mechet.forward_expert import verify_electron_step


VERSION = "natural_language_electron_event_v1"
EXPECTED_REACTIONS = {"train": 257167, "valid": 2890, "test": 28967}
SYSTEM = (
    "You are MechET, performing RETROSYNTHETIC electron-flow reasoning from the "
    "product side toward a precursor mixture. Read the CURRENT STATE SMILES and "
    "its temporary natural-language atom/bond inventory. The Axx/Bxx names are "
    "regenerated at every decision and are not atom maps. Call exactly one tool. "
    "If a chemically required external species is absent, import it first. "
    "Otherwise generate one complete coupled electron-flow event in natural "
    "language, explicitly moving electrons from a lone pair, bond, or radical "
    "pair to an atom or bond. The executor resolves the temporary names, checks "
    "electron/valence consistency, applies the event atomically, and owns the "
    "next state. Never copy or predict the next-state SMILES. Call finish_trace "
    "only when the current state is the complete precursor."
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "import_fragments",
            "description": (
                "Import absent external species immediately before first electron "
                "use, or endpoint-only context immediately before completion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fragments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "smiles": {"type": "string"},
                                "count": {"type": "integer", "minimum": 1},
                                "purpose": {
                                    "type": "string",
                                    "enum": ["electron_participant", "endpoint_context"],
                                },
                            },
                            "required": ["smiles", "count", "purpose"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["fragments"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_electron_flow",
            "description": (
                "Apply one retrosynthetic electron event. Each instruction is a "
                "natural-language explanation; source/destination phrases must use "
                "the temporary atoms and bonds listed in the current inventory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["retrosynthetic"]},
                    "electron_flow": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "destination": {"type": "string"},
                                "instruction": {"type": "string"},
                            },
                            "required": ["source", "destination", "instruction"],
                            "additionalProperties": False,
                        },
                    },
                    "bond_order_changes": {"type": "array", "items": {"type": "object"}},
                    "charge_changes": {"type": "array", "items": {"type": "object"}},
                },
                "required": [
                    "direction",
                    "electron_flow",
                    "bond_order_changes",
                    "charge_changes",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_trace",
            "description": "Finish and let the executor derive the precursor endpoint.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _call(name: str, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _visible(mapped_state: str) -> str:
    return deterministic_unmapped_state(mapped_state).text


def _prompt(target: str, mapped_state: str, *, include_inventory: bool) -> str:
    inventory = build_inventory(mapped_state)
    prefix = (
        f"TARGET PRODUCT SMILES: {target}\n"
        f"CURRENT STATE SMILES: {inventory.visible_smiles}\n"
    )
    if include_inventory:
        prefix += f"\nMOLECULAR INVENTORY\n{inventory.prompt}\n"
    return prefix + "\nChoose the single next retrosynthetic action."


def _decision_row(
    *,
    row: Mapping[str, Any],
    sequence_index: int,
    decision_type: str,
    mapped_state: str,
    name: str,
    arguments: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    call_id = f"decision_{sequence_index:03d}"
    reaction_id = str(row["source_id"])
    return {
        "id": f"{reaction_id}::nl_decision_{sequence_index:03d}_{decision_type}",
        "source_id": reaction_id,
        "artifact_type": "supervision",
        "task_type": VERSION,
        "target_smiles": str(row["target_smiles"]),
        "expected_precursor": str(row["expected_precursor"]),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": _prompt(
                    str(row["target_smiles"]),
                    mapped_state,
                    include_inventory=name == "apply_electron_flow",
                ),
            },
            {"role": "assistant", "content": "", "tool_calls": [_call(name, arguments, call_id)]},
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(dict(result), ensure_ascii=False, separators=(",", ":")),
            },
        ],
        "tools": TOOLS,
        "metadata": {
            "representation": VERSION,
            "decision_contract": "markov_tool_decision_v1",
            "reaction_id": reaction_id,
            "decision_index": sequence_index,
            "decision_type": decision_type,
            "executor_replayed": True,
            "endpoint_source": "environment_owned_trace",
            "mapping_model_visible": False,
            "state_prediction_supervised": False,
            "direction": "retrosynthetic",
            "source_program_version": (row.get("metadata") or {}).get("program_version"),
        },
    }


def _event_maps(moves: Sequence[Mapping[str, Any]]) -> set[int]:
    output: set[int] = set()
    for move in moves:
        if move.get("mode") == "BE_DELTA":
            for item in move.get("bond_deltas") or []:
                output.update(int(value) for value in item.get("atoms") or [])
            for item in move.get("charge_actions") or []:
                output.add(int(item["atom_map"]))
        else:
            output.update(int(value) for value in move["source"]["atoms"])
            output.update(int(value) for value in move["sink"]["atoms"])
    return output


def _import_arguments(
    mapped_fragments: Sequence[str], visible_fragments: Sequence[str], moves: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    touched = _event_maps(moves)
    grouped: dict[tuple[str, str], int] = defaultdict(int)
    for mapped_fragment, visible_fragment in zip(mapped_fragments, visible_fragments):
        purpose = "electron_participant" if mapped_atom_numbers(mapped_fragment) & touched else "endpoint_context"
        grouped[(visible_fragment, purpose)] += 1
    return {
        "fragments": [
            {"smiles": smiles, "count": count, "purpose": purpose}
            for (smiles, purpose), count in sorted(grouped.items())
        ]
    }


def convert_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert and independently replay one complete source reaction."""

    metadata = dict(row.get("metadata") or {})
    plan = dict(metadata.get("trace_plan") or {})
    steps = [dict(value) for value in plan.get("steps") or []]
    target = str(row.get("target_smiles") or plan.get("target_smiles") or "")
    expected = str(row.get("full_precursor_state") or row.get("expected_precursor") or "")
    if not steps or not target or not expected:
        raise ValueError("source row lacks a complete strict trace")
    fragments = extract_import_fragments(row)
    scheduled = schedule_imports(fragments, steps)
    target_maps = mapped_atom_numbers(target)
    present_maps = set(target_maps)
    current = retain_mapped_components(str(steps[0]["state_before"]), present_maps)
    target_visible = _visible(current)
    public_row = dict(
        row,
        target_smiles=target_visible,
        expected_precursor=_visible(expected),
    )
    output: list[dict[str, Any]] = []
    sequence = 0
    events = 0
    for event_index, (step, mapped_imports) in enumerate(zip(steps, scheduled)):
        authoritative_prefix = retain_mapped_components(
            str(step.get("state_before") or ""), present_maps
        )
        if mapped_state_signature(current) != mapped_state_signature(authoritative_prefix):
            raise ValueError(f"authoritative prefix changed at event {event_index}")
        moves = [dict(value) for value in step.get("moves") or []]
        visible_imports = [_visible(fragment) for fragment in mapped_imports]
        event_state = append_mapped_fragments_verbatim(current, mapped_imports)
        if mapped_imports:
            for fragment in mapped_imports:
                present_maps.update(mapped_atom_numbers(fragment))
            import_arguments = _import_arguments(mapped_imports, visible_imports, moves)
            output.append(
                _decision_row(
                    row=public_row,
                    sequence_index=sequence,
                    decision_type="import",
                    mapped_state=current,
                    name="import_fragments",
                    arguments=import_arguments,
                    result={
                        "ok": True,
                        "code": "PASS",
                        "current_state": _visible(event_state),
                        "imported_fragments": len(mapped_imports),
                    },
                )
            )
            sequence += 1

        arguments = render_event_arguments(event_state, moves)
        compiled = compile_event_arguments(event_state, arguments)
        if canonical_event(compiled) != canonical_event(moves):
            raise ValueError(f"natural-language move round-trip failed at {events}")
        replay = verify_electron_step(event_state, compiled)
        if not replay.get("ok"):
            raise ValueError(
                f"natural-language replay failed at {events}: {replay.get('code')}"
            )
        successor = retain_mapped_components(
            str(step.get("state_after") or replay["state_smiles"]), present_maps
        )
        if mapped_state_signature(str(replay["state_smiles"])) != mapped_state_signature(successor):
            raise ValueError(f"successor state changed at event {events}")
        output.append(
            _decision_row(
                row=public_row,
                sequence_index=sequence,
                decision_type="event",
                mapped_state=event_state,
                name="apply_electron_flow",
                arguments=arguments,
                result={
                    "ok": True,
                    "code": "PASS",
                    "current_state": _visible(successor),
                },
            )
        )
        sequence += 1
        events += 1
        current = successor

    if events != len(steps):
        raise ValueError("event-decision coverage mismatch")
    if mapped_state_signature(current) != mapped_state_signature(expected):
        raise ValueError("natural-language trace endpoint changed")
    output.append(
        _decision_row(
            row=public_row,
            sequence_index=sequence,
            decision_type="finish",
            mapped_state=current,
            name="finish_trace",
            arguments={},
            result={
                "ok": True,
                "code": "PASS",
                "derived_precursor": _visible(current),
                "endpoint_exact": True,
            },
        )
    )
    return output


def _worker_init() -> None:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    def timeout(*_: Any) -> None:
        raise TimeoutError("NATURAL_LANGUAGE_CONVERSION_TIMEOUT")

    signal.signal(signal.SIGALRM, timeout)


def _worker(line: str) -> tuple[bool, Any]:
    row = json.loads(line)
    try:
        signal.setitimer(signal.ITIMER_REAL, 180)
        return True, convert_row(row)
    except Exception as exc:
        return False, {
            "id": row.get("id"),
            "source_id": row.get("source_id"),
            "exception": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def build_split(
    source: Path,
    target: Path,
    *,
    workers: int,
    limit_reactions: int,
) -> tuple[dict[str, Any], set[str]]:
    building = target.with_suffix(".jsonl.building")
    unresolved = target.with_suffix(".unresolved.jsonl")
    unresolved_building = target.with_suffix(".unresolved.jsonl.building")
    for path in (building, unresolved_building):
        path.unlink(missing_ok=True)
    counts: Counter[str] = Counter()
    reactions: set[str] = set()
    decisions: set[str] = set()
    started = time.monotonic()
    context = mp.get_context("fork")
    with (
        source.open(encoding="utf-8") as inp,
        building.open("w", encoding="utf-8") as out,
        unresolved_building.open("w", encoding="utf-8") as bad,
        context.Pool(max(1, workers), initializer=_worker_init) as pool,
    ):
        lines = inp if not limit_reactions else (line for _, line in zip(range(limit_reactions), inp))
        for ok, value in pool.imap(_worker, lines, chunksize=2):
            if not ok:
                bad.write(json.dumps(value, ensure_ascii=False) + "\n")
                counts["unresolved_reactions"] += 1
                reaction_id = str(value.get("source_id") or value.get("id") or "")
            else:
                rows = list(value)
                if not rows:
                    raise ValueError("converted reaction produced no decisions")
                reaction_id = str(rows[0]["source_id"])
                for decision in rows:
                    identifier = str(decision["id"])
                    if identifier in decisions:
                        raise ValueError(f"duplicate decision ID: {identifier}")
                    decisions.add(identifier)
                    dtype = str(decision["metadata"]["decision_type"])
                    counts[f"{dtype}_decisions"] += 1
                    encoded = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
                    if '"atom_map"' in encoded or '"state_after"' in encoded:
                        raise ValueError(f"private-state leakage in {identifier}")
                    out.write(encoded + "\n")
            if not reaction_id or reaction_id in reactions:
                raise ValueError(f"invalid or duplicate reaction ID: {reaction_id}")
            reactions.add(reaction_id)
            if len(reactions) % 100 == 0:
                out.flush()
                bad.flush()
                print(
                    json.dumps(
                        {
                            "stage": "natural-language-build",
                            "split": source.stem,
                            "reactions": len(reactions),
                            "decisions": len(decisions),
                            "unresolved": counts["unresolved_reactions"],
                            "seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
        out.flush()
        bad.flush()
        os.fsync(out.fileno())
        os.fsync(bad.fileno())
    os.replace(building, target)
    os.replace(unresolved_building, unresolved)
    report = {
        "source": str(source),
        "target": str(target),
        "reactions": len(reactions),
        "decision_rows": len(decisions),
        **dict(counts),
        "source_sha256": sha256(source),
        "output_sha256": sha256(target),
        "unresolved_sha256": sha256(unresolved),
        "mapping_model_visible": False,
        "state_prediction_supervised": False,
        "reference_replay_verified": counts["unresolved_reactions"] == 0,
    }
    target.with_suffix(".audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    return report, reactions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_action_delta_v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 32))
    parser.add_argument("--limit-reactions", type=int, default=0)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "valid", "test"],
        default=["valid", "test", "train"],
    )
    args = parser.parse_args()
    manifest_path = args.source_dir / "training_manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    source_status = json.loads((args.source_dir / "ARTIFACT_STATUS.json").read_text())
    if (
        source_status.get("training_allowed") is not True
        or source_manifest.get("strict_trace_universe_complete") is not True
    ):
        raise ValueError("source artifact is not approved for training")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    split_ids: dict[str, set[str]] = {}
    for split in args.splits:
        source = args.source_dir / f"{split}.jsonl"
        expected_hash = source_manifest["splits"][split]["sha256"]
        if sha256(source) != expected_hash:
            raise ValueError(f"source hash mismatch: {split}")
        reports[split], split_ids[split] = build_split(
            source,
            args.output_dir / f"{split}.jsonl",
            workers=args.workers,
            limit_reactions=args.limit_reactions,
        )
        if not args.limit_reactions and reports[split]["reactions"] != EXPECTED_REACTIONS[split]:
            raise ValueError(f"reaction denominator mismatch: {split}")

    overlap = {}
    names = sorted(split_ids)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap[f"{left}_{right}"] = len(split_ids[left] & split_ids[right])
    unresolved = sum(int(value.get("unresolved_reactions", 0)) for value in reports.values())
    complete = (
        not args.limit_reactions
        and set(args.splits) == set(EXPECTED_REACTIONS)
        and unresolved == 0
        and not any(overlap.values())
    )
    manifest = {
        "artifact_type": VERSION,
        "source_artifact": str(args.source_dir),
        "source_manifest_sha256": sha256(manifest_path),
        "reaction_denominator": {key: value["reactions"] for key, value in reports.items()},
        "decision_rows": {key: value["decision_rows"] for key, value in reports.items()},
        "splits": reports,
        "split_reaction_id_overlap": overlap,
        "observation_contract": "target_current_state_plus_natural_inventory_v1",
        "decision_contract": "markov_tool_decision_v1",
        "import_policy": "first_electron_use_with_explicit_final_spectators",
        "model_visible_atom_maps": False,
        "model_predicts_state": False,
        "electron_flow_model_generated": True,
        "reference_replay_verified": unresolved == 0,
        "training_allowed": complete,
        "status": "validated_complete" if complete else "diagnostic_only",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(
            {
                "artifact_id": VERSION,
                "training_allowed": complete,
                "status": manifest["status"],
                "reason": (
                    "all frozen strict-executable reaction IDs converted and replayed"
                    if complete
                    else "limited or unresolved conversion; forbidden for training"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": complete, "unresolved": unresolved, "reports": reports}), flush=True)
    return 0 if complete or args.limit_reactions else 1


if __name__ == "__main__":
    raise SystemExit(main())
