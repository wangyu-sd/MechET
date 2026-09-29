#!/usr/bin/env python3
"""Export R3 flat candidate trajectories without marking the corrupted step.

The frozen source remains the evaluation authority. Its reference suffix is
used only to create the candidate action sequence; no reference successor,
correct action, execution feedback or failure index enters model_input.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


FORBIDDEN = {"private_reference", "correct_action", "expected_precursor",
             "expected_successor", "first_failure_index", "corruption_kind",
             "prefix_actions", "corrupted_action", "suffix_actions", "result",
             "executor_result", "failure_depth", "corruption"}


def _flat_action(action: Any, *, number: int) -> dict[str, Any]:
    if (not isinstance(action, dict) or not isinstance(action.get("name"), str)
            or not isinstance(action.get("arguments"), dict)):
        raise ValueError(f"R3 candidate has malformed action in source row {number}")
    flattened = {"name": action["name"], "arguments": action["arguments"]}

    def check(value: Any) -> None:
        if isinstance(value, dict):
            if FORBIDDEN & set(value):
                raise ValueError(f"R3 candidate leaks reference or feedback in source row {number}")
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(flattened)
    return flattened


def export(source: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R3 unmarked query artifact already exists: {output}")
    source_hash = verify_evaluation_source(source, name="r3_corruptions")
    queries = []
    positions: Counter[int] = Counter()
    lengths: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    seen: set[str] = set()
    with source.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"R3 source has blank row {number}")
            row = json.loads(line)
            visible = row.get("model_visible")
            private = row.get("private_reference")
            if (row.get("source_split") != "test" or not isinstance(visible, dict)
                    or not isinstance(private, dict)
                    or not isinstance(visible.get("target_smiles"), str)
                    or visible["target_smiles"] != row.get("target_smiles")
                    or not isinstance(visible.get("prefix_actions"), list)
                    or not isinstance(visible.get("corrupted_action"), dict)
                    or not isinstance(private.get("suffix_actions"), list)):
                raise ValueError(f"R3 source has malformed trajectory boundary at row {number}")
            index = private.get("first_failure_index")
            if (not isinstance(index, int) or isinstance(index, bool)
                    or index != len(visible["prefix_actions"])):
                raise ValueError(f"R3 source has inconsistent private failure index at row {number}")
            candidate = (visible["prefix_actions"] + [visible["corrupted_action"]]
                         + private["suffix_actions"])
            actions = [_flat_action(action, number=number) for action in candidate]
            if len(actions) < 2 or index >= len(actions):
                raise ValueError(f"R3 source has degenerate candidate trajectory at row {number}")
            case_id = hashlib.sha256(line.rstrip("\r\n").encode("utf-8")).hexdigest()
            if case_id in seen:
                raise ValueError("R3 source has duplicate corruption rows")
            seen.add(case_id)
            queries.append({"artifact_type": "r3_unmarked_localization_query_v1",
                            "case_id": case_id,
                            "model_input": {"target_smiles": visible["target_smiles"],
                                            "candidate_actions": actions}})
            positions[index] += 1
            lengths[len(actions)] += 1
            kinds[str(private.get("corruption_kind"))] += 1
    if len(queries) != 288:
        raise ValueError(f"R3 unmarked query denominator changed: {len(queries)}")
    if len(positions) < 3 or max(positions.values()) == len(queries):
        raise ValueError("R3 unmarked queries have a degenerate failure-position distribution")
    output.mkdir(parents=True)
    query = output / "r3_unmarked_queries.jsonl"
    with query.open("w", encoding="utf-8") as stream:
        for row in queries:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"artifact_type": "r3_unmarked_localization_manifest_v1",
              "source": str(source.resolve()), "source_sha256": source_hash,
              "source_manifest_sha256": digest(source.parent / "manifest.json"),
              "query": str(query.resolve()), "query_sha256": digest(query),
              "cases": len(queries), "model_input_fields": ["target_smiles", "candidate_actions"],
              "private_first_divergence_index_histogram": dict(sorted(positions.items())),
              "candidate_action_count_histogram": dict(sorted(lengths.items())),
              "private_corruption_kind_counts": dict(sorted(kinds.items())),
              "claim_boundary": "Flat, unmarked candidate actions for synthetic first-reference-divergence localization. Accepted wrong successors are not proven chemically impossible; suffix actions were reference actions and may not execute after the changed step. No model localization result exists."}
    (output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "ARTIFACT_STATUS.json").write_text(json.dumps({
        "artifact_type": "r3_unmarked_localization_query_status_v1",
        "query_sha256": report["query_sha256"],
        "inference_allowed": True, "localization_inference_allowed": True,
        "training_allowed": False, "evaluation_allowed": False,
        "reason": "Model-facing unmarked query only; original frozen R3 source holds private labels and remains evaluation authority",
    }, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
