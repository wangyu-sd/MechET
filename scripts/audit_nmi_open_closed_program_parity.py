#!/usr/bin/env python3
"""Audit that the matched H2 open/closed controls supervise identical programs.

This reads only frozen dataset rows, never model predictions or test outcomes.
Import order and ordered electron-move steps are compared by source ID; the
closed-loop import schedule is reported separately because Open-Flow grammar
always places imports before its first STEP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mechet.open_flow_program import parse_open_flow


def _digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _rows(path: Path, expected_sha: str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                yield json.loads(line)
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"frozen H2 data SHA mismatch: {path}")


def _open_program(row: dict[str, Any]) -> tuple[list[str], list[list[dict[str, Any]]]]:
    assistants = [message for message in row.get("messages") or []
                  if message.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError(f"Open-Flow needs one assistant program: {row.get('source_id')}")
    return parse_open_flow(str(assistants[0].get("content") or ""))


def _closed_program(row: dict[str, Any]) -> tuple[list[str], list[list[dict[str, Any]]], int]:
    imports: list[str] = []
    steps: list[list[dict[str, Any]]] = []
    names: list[str] = []
    late_imports = 0
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = dict(call.get("function") or {})
            name = str(function.get("name") or "")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"invalid closed-loop tool arguments: {row.get('source_id')}")
            names.append(name)
            if name == "import_fragment":
                fragment = str(arguments.get("fragment_smiles") or "")
                if not fragment:
                    raise ValueError(f"empty closed-loop import: {row.get('source_id')}")
                imports.append(fragment)
                late_imports += bool(steps)
            elif name == "apply_coupled_electron_moves":
                moves = arguments.get("moves")
                if not isinstance(moves, list) or not moves or any(
                    not isinstance(move, dict) for move in moves
                ):
                    raise ValueError(f"invalid closed-loop moves: {row.get('source_id')}")
                steps.append(moves)
            elif name != "finish_trace":
                raise ValueError(f"unexpected closed-loop action {name}: {row.get('source_id')}")
    if not steps or names.count("finish_trace") != 1 or names[-1] != "finish_trace":
        raise ValueError(f"incomplete closed-loop program: {row.get('source_id')}")
    return imports, steps, late_imports


def audit(matched_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing audit: {output}")
    manifests = {
        condition: json.loads((matched_dir / condition / "manifest.json").read_text())
        for condition in ("open_flow", "closed_loop")
    }
    if (manifests["open_flow"]["parent_split_manifest_sha256"]
            != manifests["closed_loop"]["parent_split_manifest_sha256"]):
        raise ValueError("Open-Flow and Closed-Loop use different H2 splits")
    reports = {}
    for split in ("train", "valid", "test"):
        open_manifest = manifests["open_flow"]
        closed_manifest = manifests["closed_loop"]
        open_path = matched_dir / "open_flow" / f"{split}.jsonl"
        closed_path = matched_dir / "closed_loop" / f"{split}.jsonl"
        expected = int(open_manifest["rows"][split])
        if expected != int(closed_manifest["rows"][split]):
            raise ValueError(f"{split} representation row counts differ")
        open_signatures: dict[str, tuple[str, str, str, int]] = {}
        for row in _rows(open_path, open_manifest["output_sha256"][split]):
            identifier = str(row.get("source_id") or "")
            if not identifier or identifier in open_signatures:
                raise ValueError(f"missing/duplicate Open-Flow ID: {identifier}")
            imports, steps = _open_program(row)
            open_signatures[identifier] = (
                _digest(imports), _digest(sorted(imports)), _digest(steps), len(imports),
            )
        if len(open_signatures) != expected:
            raise ValueError(f"Open-Flow {split} row count mismatch")
        seen: set[str] = set()
        import_mismatches: list[str] = []
        import_multiset_mismatches: list[str] = []
        move_mismatches: list[str] = []
        late_import_rows = late_import_calls = 0
        closed_minus_open_import_calls = 0
        for row in _rows(closed_path, closed_manifest["output_sha256"][split]):
            identifier = str(row.get("source_id") or "")
            if identifier in seen or identifier not in open_signatures:
                raise ValueError(f"missing/duplicate Closed-Loop ID: {identifier}")
            seen.add(identifier)
            imports, steps, late_imports = _closed_program(row)
            expected_imports, expected_import_multiset, expected_steps, open_import_count = open_signatures[identifier]
            if _digest(imports) != expected_imports:
                import_mismatches.append(identifier)
            if _digest(sorted(imports)) != expected_import_multiset:
                import_multiset_mismatches.append(identifier)
            if _digest(steps) != expected_steps:
                move_mismatches.append(identifier)
            late_import_rows += bool(late_imports)
            late_import_calls += late_imports
            closed_minus_open_import_calls += len(imports) - open_import_count
        if len(seen) != expected or seen != set(open_signatures):
            raise ValueError(f"Closed-Loop {split} ID coverage mismatch")
        reports[split] = {
            "rows": expected,
            "import_sequence_mismatch_count": len(import_mismatches),
            "import_multiset_mismatch_count": len(import_multiset_mismatches),
            "electron_step_sequence_mismatch_count": len(move_mismatches),
            "closed_import_after_first_step_rows": late_import_rows,
            "closed_import_after_first_step_calls": late_import_calls,
            "closed_minus_open_import_calls": closed_minus_open_import_calls,
            "import_mismatch_example_ids": import_mismatches[:10],
            "import_multiset_mismatch_example_ids": import_multiset_mismatches[:10],
            "electron_step_mismatch_example_ids": move_mismatches[:10],
            "open_flow_sha256": open_manifest["output_sha256"][split],
            "closed_loop_sha256": closed_manifest["output_sha256"][split],
        }
        print(json.dumps({"split": split, "rows": expected,
                          "import_sequence_mismatches": len(import_mismatches),
                          "import_multiset_mismatches": len(import_multiset_mismatches),
                          "electron_step_mismatches": len(move_mismatches)},
                         sort_keys=True), flush=True)
        if move_mismatches:
            raise ValueError(f"{split} Open-Flow/Closed-Loop electron steps differ: {reports[split]}")
    report = {
        "artifact_type": "nmi_h2_open_closed_program_parity_v1",
        "parent_split_manifest_sha256": manifests["open_flow"]["parent_split_manifest_sha256"],
        "splits": reports,
        "passed": True,
        "interpretation": "Electron steps must match; import differences are measured and reported, not silently filtered or reclassified as step errors.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.matched_dir, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
