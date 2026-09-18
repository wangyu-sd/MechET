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
import re
import signal
import sys
import time
from typing import Any, Mapping, Sequence

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.a7_rescue import canonical_event
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    map_unmapped_fragment,
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


VERSION = "natural_language_electron_event_v2"
DECISION_CONTRACT = "unified_inventory_tool_decision_v2"
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


_ATOM_MAP_RE = re.compile(r":(\d+)\]")


def _translate_mapped_smiles(smiles: str, translation: Mapping[int, int]) -> str:
    """Relabel maps without changing the authoritative bond serialization."""

    def replace(match: re.Match[str]) -> str:
        old = int(match.group(1))
        if old not in translation:
            raise ValueError(f"missing runtime map translation for {old}")
        return f":{translation[old]}]"

    return _ATOM_MAP_RE.sub(replace, str(smiles))


def _translate_moves(
    moves: Sequence[Mapping[str, Any]], translation: Mapping[int, int]
) -> list[dict[str, Any]]:
    output = json.loads(json.dumps(list(moves)))
    for move in output:
        if move.get("mode") == "BE_DELTA":
            for item in move.get("bond_deltas") or []:
                item["atoms"] = [translation[int(value)] for value in item["atoms"]]
            for item in move.get("charge_actions") or []:
                item["atom_map"] = translation[int(item["atom_map"])]
        else:
            for side in ("source", "sink"):
                move[side]["atoms"] = [
                    translation[int(value)] for value in move[side]["atoms"]
                ]
    return output


def _raw_graph_delta(source: str, destination: str) -> dict[str, Any]:
    """Encode a representation repair only when the normal arrow replay drifts."""

    def graph(smiles: str) -> tuple[dict[tuple[int, int], int], dict[int, int]]:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(smiles, params)
        if mol is None:
            raise ValueError("cannot derive graph delta from invalid state")
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        bonds = {
            tuple(
                sorted(
                    (
                        int(bond.GetBeginAtom().GetAtomMapNum()),
                        int(bond.GetEndAtom().GetAtomMapNum()),
                    )
                )
            ): int(round(bond.GetBondTypeAsDouble()))
            for bond in mol.GetBonds()
        }
        charges = {
            int(atom.GetAtomMapNum()): int(atom.GetFormalCharge())
            for atom in mol.GetAtoms()
        }
        return bonds, charges

    source_bonds, source_charges = graph(source)
    destination_bonds, destination_charges = graph(destination)
    if set(source_charges) != set(destination_charges):
        raise ValueError("graph delta atom sets differ")
    bond_deltas = [
        {"atoms": list(pair), "delta": destination_bonds.get(pair, 0) - source_bonds.get(pair, 0)}
        for pair in sorted(set(source_bonds) | set(destination_bonds))
        if destination_bonds.get(pair, 0) != source_bonds.get(pair, 0)
    ]
    charge_actions = [
        {
            "atom_map": atom_map,
            "q0": source_charges[atom_map],
            "q1": destination_charges[atom_map],
        }
        for atom_map in sorted(source_charges)
        if source_charges[atom_map] != destination_charges[atom_map]
    ]
    if not bond_deltas and not charge_actions:
        raise ValueError("graph delta is empty")
    return {
        "mode": "BE_DELTA",
        "bond_deltas": bond_deltas,
        "charge_actions": charge_actions,
    }


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
                    # Every action is predicted from one identical observation
                    # contract.  Conditioning inventory visibility on the gold
                    # action type leaks the teacher-forced tool class and makes
                    # import/event/finish likelihoods incomparable at inference.
                    include_inventory=True,
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
            "decision_contract": DECISION_CONTRACT,
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
    translation = {value: value for value in target_maps}
    next_private_map = max(target_maps, default=0) + 1
    # Start from the actual benchmark product representation.  Earlier builds
    # substituted the hidden authoritative step-0 state, whose Kekule form can
    # differ from the product available to a real product-only runtime.
    current = target
    target_visible = _visible(target)
    public_row = dict(
        row,
        target_smiles=target_visible,
        expected_precursor=_visible(expected),
    )
    output: list[dict[str, Any]] = []
    sequence = 0
    events = 0
    for step, source_imports in zip(steps, scheduled):
        # The v2 trajectory is executor-owned from the public product onward.
        # Do not replace it with the hidden authoritative prefix between
        # decisions; doing so makes teacher-forced states unreachable online.
        source_moves = [dict(value) for value in step.get("moves") or []]
        visible_imports = [_visible(fragment) for fragment in source_imports]
        runtime_imports: list[str] = []
        for source_fragment, visible_fragment in zip(
            source_imports, visible_imports, strict=True
        ):
            runtime_fragment, next_private_map = map_unmapped_fragment(
                visible_fragment, first_map=next_private_map
            )
            source_serialization = deterministic_unmapped_state(source_fragment)
            runtime_serialization = deterministic_unmapped_state(runtime_fragment)
            if source_serialization.text != runtime_serialization.text:
                raise ValueError("runtime import changed visible fragment")
            for source_map, runtime_map in zip(
                source_serialization.atom_maps,
                runtime_serialization.atom_maps,
                strict=True,
            ):
                if source_map in translation and translation[source_map] != runtime_map:
                    raise ValueError("inconsistent source-to-runtime atom map")
                translation[source_map] = runtime_map
            runtime_imports.append(runtime_fragment)
            present_maps.update(mapped_atom_numbers(source_fragment))
        moves = _translate_moves(source_moves, translation)
        event_state = append_mapped_fragments_verbatim(current, runtime_imports)
        if runtime_imports:
            import_arguments = _import_arguments(
                runtime_imports, visible_imports, moves
            )
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
                        "imported_fragments": len(runtime_imports),
                    },
                )
            )
            sequence += 1

        authoritative_successor = _translate_mapped_smiles(
            retain_mapped_components(str(step.get("state_after") or ""), present_maps),
            translation,
        )
        replay = verify_electron_step(event_state, moves)
        if not replay.get("ok") or mapped_state_signature(
            str(replay.get("state_smiles") or "")
        ) != mapped_state_signature(authoritative_successor):
            try:
                repair = _raw_graph_delta(event_state, authoritative_successor)
                repaired = verify_electron_step(event_state, [repair])
                if repaired.get("ok") and mapped_state_signature(
                    str(repaired["state_smiles"])
                ) == mapped_state_signature(authoritative_successor):
                    moves = [repair]
                    replay = repaired
            except Exception:
                # Some aromatic deltas are representation-equivalent but not
                # directly executable from this Kekule choice.  Keep the
                # ordinary electron replay and let the next event continue.
                pass
        arguments = render_event_arguments(event_state, moves)
        compiled = compile_event_arguments(event_state, arguments)
        if canonical_event(compiled) != canonical_event(moves):
            raise ValueError(f"natural-language move round-trip failed at {events}")
        replay = verify_electron_step(event_state, compiled)
        if not replay.get("ok"):
            raise ValueError(
                f"natural-language replay failed at {events}: {replay.get('code')}"
            )
        successor = str(replay["state_smiles"])
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
    runtime_expected = _translate_mapped_smiles(expected, translation)
    if mapped_state_signature(current) != mapped_state_signature(runtime_expected):
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
        "unresolved_reactions": int(counts["unresolved_reactions"]),
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
        "observation_contract": "target_current_state_plus_natural_inventory_v2_unified",
        "decision_contract": DECISION_CONTRACT,
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
