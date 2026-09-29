#!/usr/bin/env python3
"""Freeze answer-free Stage-II prompts for R3 exposed-failure repair.

The input is the public R3 query file only. Its accepted prefix is summarized
using the existing Stage-II trajectory capsule; the proposed action and actual
executor feedback are included so this is a repair prompt, not a teacher-forced
next-action prompt. No mapped trace, reference action, or endpoint is read.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from rdkit import Chem

from scripts.autoresearch.stratified_manifest import digest
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS
from scripts.run_natural_language_value_search import (
    PROMPT_SUFFIX, policy_prompt, visible,
)


PROMPT_VERSION = "r3_exposed_failure_stageii_revision_v1"
QUERY_FIELDS = {"target_smiles", "prefix_actions", "corrupted_action", "executor_result"}


def mapped_for_inventory(unmapped: str) -> str:
    """Assign disposable private maps; the model sees only positional aliases."""

    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    molecule = Chem.MolFromSmiles(unmapped, parameters)
    if molecule is None:
        raise ValueError("R3 public current state is not parseable")
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    mapped = Chem.MolToSmiles(molecule, canonical=False, isomericSmiles=True)
    if visible(mapped) != unmapped:
        raise ValueError("R3 public current-state serialization drifted")
    return mapped


def prompt_from_query(model_input: dict[str, Any]) -> str:
    if set(model_input) != QUERY_FIELDS:
        raise ValueError("R3 repair query has extra or missing model-visible fields")
    target = model_input["target_smiles"]
    prefix = model_input["prefix_actions"]
    bad_action = model_input["corrupted_action"]
    feedback = model_input["executor_result"]
    if (not isinstance(target, str) or not isinstance(prefix, list)
            or not isinstance(bad_action, dict) or not isinstance(feedback, dict)
            or bad_action.get("name") != "apply_electron_flow"
            or not isinstance(bad_action.get("arguments"), dict)
            or feedback.get("ok") not in (True, False)):
        raise ValueError("R3 repair query is malformed")
    for item in prefix:
        if (not isinstance(item, dict) or item.get("name") not in
                {"import_fragments", "apply_electron_flow"}
                or not isinstance(item.get("arguments"), dict)
                or not isinstance(item.get("result"), dict)
                or item["result"].get("ok") is not True):
            raise ValueError("R3 repair prefix contains a non-accepted action")
    current = (prefix[-1]["result"].get("current_state") if prefix else target)
    if not isinstance(current, str) or not current:
        raise ValueError("R3 repair prefix has no current state")
    base = policy_prompt(target, mapped_for_inventory(current),
                         include_inventory=True, actions=prefix,
                         compact_history=True)
    if not base.endswith(PROMPT_SUFFIX):
        raise ValueError("Stage-II policy prompt suffix drifted")
    submitted = json.dumps(bad_action, sort_keys=True, ensure_ascii=False)
    observed = json.dumps(feedback, sort_keys=True, ensure_ascii=False)
    repair = (
        "\n\nEXPOSED ACTION FOR REVISION\n"
        f"submitted_action: {submitted}\n"
        f"executor_result: {observed}\n"
        "This action is flagged for revision at the current, pre-action state. "
        "An executor PASS establishes formal execution, not that the action "
        "is the intended chemistry. Return exactly one replacement "
        "apply_electron_flow tool call using the temporary atom names in "
        "MOLECULAR INVENTORY. Do not import or finish here."
    )
    return base[:-len(PROMPT_SUFFIX)] + repair + PROMPT_SUFFIX


def build(queries: Path, output: Path, *, expected_cases: int = 288) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 repair prompt artifact already exists: {output}")
    manifest_path = queries.parent / "manifest.json"
    status_path = queries.parent / "ARTIFACT_STATUS.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("R3 repair query needs frozen manifest and status")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    query_sha = digest(queries)
    if (manifest.get("query_sha256") != query_sha
            or status.get("query_sha256") != query_sha
            or manifest.get("cases") != expected_cases
            or manifest.get("model_input_fields") != sorted(QUERY_FIELDS)
            or status.get("repair_inference_allowed") is not True
            or status.get("localization_evaluation_allowed") is not False):
        raise ValueError("R3 repair query provenance or permission drifted")
    rows = []
    seen: set[str] = set()
    prefix_counts: Counter[int] = Counter()
    with queries.open(encoding="utf-8") as stream:
        for line in stream:
            query = json.loads(line)
            case_id = query.get("case_id")
            if (set(query) != {"artifact_type", "case_id", "model_input"}
                    or not isinstance(case_id, str) or len(case_id) != 64
                    or case_id in seen or not isinstance(query.get("model_input"), dict)):
                raise ValueError("R3 repair query row is malformed or duplicated")
            seen.add(case_id)
            prefix_counts[len(query["model_input"]["prefix_actions"])] += 1
            rows.append({"case_id": case_id,
                         "prompt_version": PROMPT_VERSION,
                         "user_prompt": prompt_from_query(query["model_input"])})
    if len(rows) != expected_cases:
        raise ValueError("R3 repair query denominator changed")
    output.mkdir(parents=True)
    prompts = output / "r3_repair_prompts.jsonl"
    with prompts.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    report = {
        "artifact_type": "r3_exposed_failure_stageii_prompts_manifest_v1",
        "prompt_version": PROMPT_VERSION,
        "queries_sha256": query_sha,
        "queries_manifest_sha256": digest(manifest_path),
        "prompts": str(prompts.resolve()),
        "prompts_sha256": digest(prompts),
        "cases": len(rows),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(row["case_id"] for row in rows).encode()).hexdigest(),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "tools_sha256": hashlib.sha256(json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
        "prefix_action_counts": dict(sorted(prefix_counts.items())),
        "max_prompt_chars": max(len(row["user_prompt"]) for row in rows),
        "input_fields": ["model_input"],
        "training_allowed": False,
        "evaluation_allowed": False,
        "claim_boundary": "Answer-free, frozen prompts for one-action repair at an exposed failure; no model prediction or localization result.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r3_exposed_failure_stageii_prompts_status_v1",
        "prompts_sha256": report["prompts_sha256"],
        "inference_allowed": True,
        "training_allowed": False,
        "evaluation_allowed": False,
        "localization_evaluation_allowed": False,
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.queries, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
