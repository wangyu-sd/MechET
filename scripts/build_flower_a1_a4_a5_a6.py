#!/usr/bin/env python3
"""Build the missing A1/A4/A5/A6 controls on the frozen FlowER program IDs.

This is a representation-only transform.  It does not select, filter,
deduplicate, decontaminate, or intersect reaction IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


SPLITS = ("train", "valid", "test")
EXPECTED = {"train": 257_167, "valid": 2_890, "test": 28_967}
TASKS = ("free_cot", "open_flow", "loose_trace_answer", "mechsmiles")


SYSTEM = {
    "free_cot": (
        "Given only the mapped product SMILES, reason in ordinary chemical "
        "language about the reverse electron flow, then predict the structural "
        "precursors. Output one <reasoning> block and one <answer> block."
    ),
    "open_flow": (
        "Given only the mapped product SMILES, emit the complete inverse "
        "electron-flow program in one <flow> block. The program is executed "
        "only after the complete block is generated; do not emit an answer."
    ),
    "mechsmiles": (
        "Given only the mapped product SMILES, emit the complete inverse "
        "mechanism as a sequence of MechSMILES elementary steps. Each line is "
        "SMILES|arrows, with semicolon-separated arrows. Do not emit an answer."
    ),
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def trace_plan(row: dict[str, Any]) -> dict[str, Any]:
    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    if not plan.get("steps"):
        raise ValueError(f"row {row.get('id')} has no frozen trace plan")
    return plan


def product(row: dict[str, Any]) -> str:
    value = str(row.get("target_smiles") or "").strip()
    if not value:
        raise ValueError(f"row {row.get('id')} has no product")
    return value


def precursor(row: dict[str, Any]) -> str:
    value = str(row.get("structural_precursor") or "").strip()
    if not value:
        raise ValueError(f"row {row.get('id')} has no structural precursor")
    return value


def base_row(row: dict[str, Any], task: str, assistant: str) -> dict[str, Any]:
    out = {key: value for key, value in row.items() if key not in {"messages", "tools"}}
    out["task_type"] = task
    out["messages"] = [
        {"role": "system", "content": SYSTEM[task]},
        {"role": "user", "content": f"TARGET: {product(row)}"},
        {"role": "assistant", "content": assistant},
    ]
    metadata = dict(out.get("metadata") or {})
    metadata["paper_control"] = task
    metadata["source_observation_mode"] = metadata.pop("observation_mode", None)
    metadata["endpoint_source"] = "independent_answer" if task == "free_cot" else "program_execution"
    out["metadata"] = metadata
    return out


def move_phrase(move: dict[str, Any]) -> str:
    if move.get("mode") == "BE_DELTA":
        bonds = ", ".join(
            f"bond({','.join(str(x) for x in item['atoms'])}) {int(item['delta']):+d}"
            for item in move.get("bond_deltas") or []
        )
        charges = ", ".join(
            f"charge({item['atom_map']}) {item['q0']}->{item['q1']}"
            for item in move.get("charge_actions") or []
        )
        return "apply the coupled bond-electron changes " + "; ".join(
            value for value in (bonds, charges) if value
        )
    source = move["source"]
    sink = move["sink"]
    source_atoms = ",".join(str(x) for x in source["atoms"])
    sink_atoms = ",".join(str(x) for x in sink["atoms"])
    return f"move an electron pair from {source['kind']}({source_atoms}) to {sink['kind']}({sink_atoms})"


def build_free_cot(row: dict[str, Any]) -> dict[str, Any]:
    plan = trace_plan(row)
    statements: list[str] = []
    imports = [str(value) for value in plan.get("initial_imports") or []]
    if imports:
        statements.append("First introduce the required precursor fragments: " + "; ".join(imports) + ".")
    for index, step in enumerate(plan["steps"], start=1):
        phrases = [move_phrase(move) for move in step.get("moves") or []]
        statements.append(f"Reverse step {index}: " + "; then ".join(phrases) + ".")
    statements.append("These changes give the atom-contributing structural precursor shown below.")
    assistant = "<reasoning>\n" + "\n".join(statements) + "\n</reasoning>\n<answer>\n" + precursor(row) + "\n</answer>"
    return base_row(row, "free_cot", assistant)


def build_open_flow(row: dict[str, Any]) -> dict[str, Any]:
    plan = trace_plan(row)
    lines = ["OPEN_FLOW v1"]
    lines.extend(f"IMPORT {value}" for value in plan.get("initial_imports") or [])
    for index, step in enumerate(plan["steps"]):
        lines.append(f"STEP {index} {compact(step.get('moves') or [])}")
    lines.append("EXECUTE")
    return base_row(row, "open_flow", "<flow>\n" + "\n".join(lines) + "\n</flow>")


def arrow_atom(source_atoms: list[int], sink_atoms: list[int]) -> int:
    for atom in sink_atoms:
        if atom not in source_atoms:
            return int(atom)
    return int(sink_atoms[-1])


def mechsmiles_arrow(move: dict[str, Any]) -> str:
    source = move["source"]
    sink = move["sink"]
    source_atoms = [int(x) for x in source["atoms"]]
    sink_atoms = [int(x) for x in sink["atoms"]]
    if source["kind"] in {"BOND", "RADICAL_PAIR"}:
        if len(source_atoms) != 2:
            raise ValueError(f"invalid bond source: {move}")
        target = arrow_atom(source_atoms, sink_atoms)
        return f"(({source_atoms[0]},{source_atoms[1]}),{target})"
    if len(source_atoms) != 1:
        raise ValueError(f"invalid atom source: {move}")
    target = arrow_atom(source_atoms, sink_atoms)
    return f"({source_atoms[0]},{target})"


def mechsmiles_arrows(move: dict[str, Any]) -> list[str]:
    if move.get("mode") != "BE_DELTA":
        return [mechsmiles_arrow(move)]
    arrows: list[str] = []
    # MechSMILES has no separate net-edit opcode.  Preserve each coupled bond
    # redistribution in its native arrow tuple syntax and keep the complete
    # elementary event on one line.  Negative deltas are bond ionizations;
    # positive deltas are attacks forming the addressed bond.
    for item in move.get("bond_deltas") or []:
        a, b = (int(value) for value in item["atoms"])
        delta = int(item["delta"])
        for _ in range(abs(delta)):
            arrows.append(f"(({a},{b}),{b})" if delta < 0 else f"({a},{b})")
    return arrows


def build_mechsmiles(row: dict[str, Any]) -> dict[str, Any]:
    lines = []
    for step in trace_plan(row)["steps"]:
        arrows = ";".join(
            arrow
            for move in step.get("moves") or []
            for arrow in mechsmiles_arrows(move)
        )
        lines.append(f"{step['state_before']}|{arrows}")
    assistant = "<mechsmiles>\n" + "\n".join(lines) + "\n</mechsmiles>"
    return base_row(row, "mechsmiles", assistant)


def build_loose_trace_answer(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["messages"] = [dict(message) for message in row.get("messages") or []]
    answer = f"<answer>\n{precursor(row)}\n</answer>"
    if out["messages"] and out["messages"][-1].get("role") == "assistant":
        out["messages"][-1] = {"role": "assistant", "content": answer}
    else:
        out["messages"].append({"role": "assistant", "content": answer})
    out["task_type"] = "loose_trace_answer"
    metadata = dict(out.get("metadata") or {})
    metadata["paper_control"] = "loose_trace_answer"
    # Keep the successfully replayed trace provenance and explicitly declare
    # the extra bypass channel that distinguishes A5 from A7.
    metadata["independent_answer_channel"] = True
    out["metadata"] = metadata
    return out


BUILDERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "free_cot": build_free_cot,
    "open_flow": build_open_flow,
    "loose_trace_answer": build_loose_trace_answer,
    "mechsmiles": build_mechsmiles,
}


def build_file(source: Path, target: Path, builder: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    rows = 0
    with source.open(encoding="utf-8") as reader, temporary.open("wb") as writer:
        for line in reader:
            if not line.strip():
                continue
            encoded = (json.dumps(builder(json.loads(line)), ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            writer.write(encoded)
            digest.update(encoded)
            rows += 1
            if rows % 20_000 == 0:
                print(f"{target.parent.name}/{target.stem}: {rows}", flush=True)
    temporary.replace(target)
    return {"path": str(target), "rows": rows, "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/flower_inverse_tool_sft_action_delta_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/iclr_program_controls_v1"))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    args = parser.parse_args()

    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "artifact_type": "flower_fixed_program_id_internal_controls_v1",
        "source": str(args.source_dir),
        "protocol": "representation_only_no_selection_filtering_or_id_intersection",
        "expected_rows": EXPECTED,
        "tasks": {},
    }
    for task in args.tasks:
        report: dict[str, Any] = {}
        for split in SPLITS:
            result = build_file(args.source_dir / f"{split}.jsonl", args.output_dir / task / f"{split}.jsonl", BUILDERS[task])
            if result["rows"] != EXPECTED[split]:
                raise RuntimeError(f"{task}/{split}: {result['rows']} != {EXPECTED[split]}")
            report[split] = result
        manifest["tasks"][task] = report
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
