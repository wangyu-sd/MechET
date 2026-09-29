#!/usr/bin/env python3
"""Check that each frozen R3 corruption has an executable oracle one-action repair.

The private correct action and suffix are used only in this data-quality audit;
they must never be supplied to an inference policy. This is an oracle upper
bound, not a model repair or localization result.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mechet.in_place_grounded_flow import mapped_atom_numbers
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source
from scripts.run_natural_language_value_search import Action, Node, execute, visible


FROZEN_RDKIT_VERSION = "2026.03.4"


def require_frozen_rdkit(version: str) -> None:
    if version != FROZEN_RDKIT_VERSION:
        raise RuntimeError(
            f"R3 replay requires RDKit {FROZEN_RDKIT_VERSION}; got {version}. "
            "A different RDKit serialization changes atom aliases and can create false failures.")


def replay_with_action(row: dict[str, Any], mapped_target: str,
                       replacement_action: dict[str, Any],
                       *, max_imports: int = 64) -> dict[str, Any]:
    public = row["model_visible"]
    private = row["private_reference"]
    target = str(public["target_smiles"])
    if visible(mapped_target) != target:
        raise ValueError("R3 mapped trace target differs from frozen visible target")
    actions = public["prefix_actions"] + [replacement_action] + private["suffix_actions"]
    if len(public["prefix_actions"]) != private["first_failure_index"]:
        raise ValueError("R3 private failure index differs from prefix length")
    node = Node(target=target, state=mapped_target,
                next_map=max(mapped_atom_numbers(mapped_target), default=0) + 1,
                visited={target})
    first_reference_divergence = None
    first_reference_divergence_detail = None
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or not isinstance(action.get("name"), str) \
                or not isinstance(action.get("arguments"), dict):
            raise ValueError(f"R3 oracle action {index} is malformed")
        child, error = execute(node, Action(name=action["name"],
                                            arguments=action["arguments"], raw="r3_oracle_audit",
                                            logprob=0.0, tokens=1), max_imports=max_imports)
        if child is None:
            return {"exact": False, "failure_index": index, "error": error,
                    "executed_actions": index,
                    "first_reference_divergence": first_reference_divergence,
                    "first_reference_divergence_detail": first_reference_divergence_detail}
        node = child
        reference_successor = (
            public["prefix_actions"][index].get("result", {}).get("current_state")
            if index < len(public["prefix_actions"])
            else private["expected_successor"]
            if index == len(public["prefix_actions"]) else None)
        if (reference_successor is not None and visible(node.state) != reference_successor
                and first_reference_divergence is None):
            first_reference_divergence = index
            first_reference_divergence_detail = {
                "action": action["name"], "expected": reference_successor,
                "observed": visible(node.state)}
    expected = str(private["expected_precursor"])
    if not expected:
        raise ValueError("R3 oracle row lacks the frozen visible precursor")
    return {"exact": bool(node.terminal and visible(node.state) == expected),
            "terminal": bool(node.terminal), "observed_precursor": visible(node.state),
            "expected_precursor": expected, "executed_actions": len(actions),
            "first_reference_divergence": first_reference_divergence,
            "first_reference_divergence_detail": first_reference_divergence_detail}


def replay_repaired(row: dict[str, Any], mapped_target: str,
                    *, max_imports: int = 64) -> dict[str, Any]:
    """Replay the private oracle action; never use this for model predictions."""
    return replay_with_action(row, mapped_target,
                              row["private_reference"]["correct_action"],
                              max_imports=max_imports)


def audit(source: Path, trace_source: Path, output: Path,
          *, expected_cases: int = 288) -> dict[str, Any]:
    import rdkit

    require_frozen_rdkit(rdkit.__version__)
    if output.exists():
        raise FileExistsError(f"R3 oracle audit output already exists: {output}")
    source_hash = verify_evaluation_source(source, name="r3_corruptions")
    source_meta = json.loads((source.parent / "manifest.json").read_text())
    if (str(trace_source.resolve()) != source_meta.get("trace_source")
            or digest(trace_source) != source_meta.get("trace_sha256")):
        raise ValueError("R3 mapped trace source differs from frozen source provenance")
    source_rows = [json.loads(line) for line in source.read_text().splitlines()]
    wanted = {row["reaction_id"] for row in source_rows}
    mapped_targets: dict[str, str] = {}
    with trace_source.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            reaction_id = str(row.get("source_id") or row.get("id") or "")
            if reaction_id in wanted:
                mapped_targets[reaction_id] = str(row["target_smiles"])
    if set(mapped_targets) != wanted:
        raise ValueError("R3 oracle audit is missing mapped source reactions")
    results = []
    counts: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = {}
    for number, row in enumerate(source_rows, 1):
        if row.get("source_split") != "test":
            raise ValueError(f"R3 row {number} is not held-out test")
        result = replay_repaired(row, mapped_targets[row["reaction_id"]])
        stratum = (row["strata"]["failure_depth"] + "/"
                   + row["strata"]["event_coordination"])
        bucket = by_stratum.setdefault(stratum, Counter())
        counts["cases"] += 1
        bucket["cases"] += 1
        if result["exact"]:
            counts["oracle_exact"] += 1
            bucket["oracle_exact"] += 1
        else:
            counts["oracle_failed"] += 1
            bucket["oracle_failed"] += 1
        if result["first_reference_divergence"] is not None:
            counts["reference_divergence"] += 1
        results.append({"row_number": number, "stratum": stratum,
                        "corruption_kind": row["private_reference"]["corruption_kind"],
                        **result})
    if counts["cases"] != expected_cases:
        raise ValueError(f"R3 oracle audit denominator changed: {counts['cases']}")
    output.mkdir(parents=True)
    details = output / "oracle_replay.jsonl"
    with details.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    report = {"artifact_type": "r3_private_oracle_repair_audit_v1",
              "rdkit_version": rdkit.__version__,
              "source": str(source), "source_sha256": source_hash,
              "mapped_trace_source": str(trace_source),
              "mapped_trace_source_sha256": source_meta["trace_sha256"],
              "details": str(details), "details_sha256": digest(details),
              "cases": counts["cases"], "oracle_exact": counts["oracle_exact"],
              "oracle_failed": counts["oracle_failed"],
              "reference_divergence": counts["reference_divergence"],
              "replay_gate_passed": (counts["oracle_exact"] == expected_cases
                                     and counts["reference_divergence"] == 0),
              "by_stratum": {name: dict(sorted(bucket.items()))
                             for name, bucket in sorted(by_stratum.items())},
              "claim_boundary": "Private one-action oracle upper bound only; no model prediction, localization or repair success is measured."}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not report["replay_gate_passed"]:
        raise RuntimeError("R3 oracle replay gate failed; inspect the preserved report and details")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trace-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=288)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.trace_source, args.output,
                           expected_cases=args.expected_cases), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
