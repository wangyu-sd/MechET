#!/usr/bin/env python3
"""Freeze answer-free prompts for R3 unmarked-trajectory localization.

Only the public unmarked query is read. The private corruption index and
executor feedback are neither consulted nor reconstructed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.export_r3_unmarked_localization import _flat_action
from scripts.autoresearch.stratified_manifest import digest


PROMPT_VERSION = "r3_unmarked_stageii_localization_v1"
SYSTEM = (
    "You audit a proposed RETROSYNTHETIC electron-flow trajectory. "
    "The atom labels in each action are temporary local names, not persistent "
    "atom maps. The trajectory may contain one altered action. Identify the "
    "earliest action you judge inconsistent with the proposed chemistry. "
    "Return only a JSON object with the zero-based integer key first_error_index. "
    "Do not claim that an unrecorded but executable alternative is impossible."
)


def prompt_from_query(model_input: dict[str, Any]) -> str:
    if set(model_input) != {"target_smiles", "candidate_actions"}:
        raise ValueError("R3 localization query has extra or missing model-visible fields")
    target = model_input["target_smiles"]
    actions = model_input["candidate_actions"]
    if (not isinstance(target, str) or not target
            or not isinstance(actions, list) or len(actions) < 2):
        raise ValueError("R3 localization query is malformed")
    lines = [f"PRODUCT SMILES: {target}", "PROPOSED INVERSE TRAJECTORY:"]
    for index, action in enumerate(actions):
        if (not isinstance(action, dict) or set(action) != {"name", "arguments"}
                or not isinstance(action["name"], str)
                or not isinstance(action["arguments"], dict)):
            raise ValueError(f"R3 localization action {index} is malformed")
        action = _flat_action(action, number=index + 1)
        lines.append(f"{index}: {json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(',', ':'))}")
    lines.append(
        f"Select one index from 0 to {len(actions) - 1}. "
        'Output only {"first_error_index": INTEGER}.')
    return "\n".join(lines)


def build(queries: Path, output: Path, *, expected_cases: int = 288) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 localization prompt artifact already exists: {output}")
    manifest_path = queries.parent / "manifest.json"
    status_path = queries.parent / "ARTIFACT_STATUS.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise ValueError("R3 unmarked queries require frozen manifest and status")
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    query_sha = digest(queries)
    if (manifest.get("query_sha256") != query_sha
            or status.get("query_sha256") != query_sha
            or manifest.get("cases") != expected_cases
            or manifest.get("model_input_fields") != ["target_smiles", "candidate_actions"]
            or status.get("localization_inference_allowed") is not True
            or status.get("training_allowed") is not False
            or status.get("evaluation_allowed") is not False):
        raise ValueError("R3 unmarked query provenance or permissions drifted")
    rows = []
    seen: set[str] = set()
    for line in queries.read_text(encoding="utf-8").splitlines():
        query = json.loads(line)
        case_id = query.get("case_id")
        if (set(query) != {"artifact_type", "case_id", "model_input"}
                or not isinstance(case_id, str) or len(case_id) != 64
                or any(char not in "0123456789abcdef" for char in case_id)
                or case_id in seen or not isinstance(query.get("model_input"), dict)):
            raise ValueError("R3 unmarked query row is malformed or duplicated")
        seen.add(case_id)
        rows.append({"case_id": case_id, "prompt_version": PROMPT_VERSION,
                     "user_prompt": prompt_from_query(query["model_input"]),
                     "candidate_action_count": len(query["model_input"]["candidate_actions"])})
    if len(rows) != expected_cases:
        raise ValueError("R3 unmarked query denominator changed")
    output.mkdir(parents=True)
    prompts = output / "r3_localization_prompts.jsonl"
    with prompts.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    report = {
        "artifact_type": "r3_unmarked_stageii_localization_prompts_manifest_v1",
        "prompt_version": PROMPT_VERSION,
        "queries_sha256": query_sha,
        "queries_manifest_sha256": digest(manifest_path),
        "prompts": str(prompts.resolve()), "prompts_sha256": digest(prompts),
        "cases": len(rows),
        "case_ids_sha256": hashlib.sha256(
            "\n".join(row["case_id"] for row in rows).encode()).hexdigest(),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "max_prompt_chars": max(len(row["user_prompt"]) for row in rows),
        "input_fields": ["model_input"],
        "training_allowed": False, "evaluation_allowed": False,
        "claim_boundary": "Unmarked action-index diagnostic only; no private label, reference successor, or executor feedback is input.",
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r3_unmarked_stageii_localization_prompts_status_v1",
        "prompts_sha256": report["prompts_sha256"],
        "inference_allowed": True, "training_allowed": False,
        "evaluation_allowed": False,
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
