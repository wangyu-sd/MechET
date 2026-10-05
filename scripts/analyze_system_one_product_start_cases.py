#!/usr/bin/env python3
"""Audit product-start cases and replay ranked alternatives at failed events.

The legality probe uses only a frozen predicted ranking and the predicted
current state. It does not continue the episode and cannot claim an endpoint
improvement from a recoverable local electron event.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.electron_pointer import parse_pointer_observation
from mechet.trajectory_history import TrajectoryHistory
from scripts.audit_system_one_observation_parity import mapped_from_visible, runtime_prompt
from scripts.eval_system_one_import_retrieval import chemical_batch
from scripts.eval_system_one_product_start_pilot import canonical_visible
from mechet.system_one_replay import execute_pair_indices


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_legal_single_rank(case: dict) -> int | None:
    failed = [action for action in case["actions"] if action.get("execute_ok") is False]
    if len(failed) != 1:
        raise ValueError(f"{case['id']}: expected one failed electron event")
    action = failed[0]
    ranked = action.get("ranked_top8") or []
    if not ranked or len(ranked) > 8 or len(set(ranked)) != len(ranked):
        raise ValueError(f"{case['id']}: invalid frozen Top-8 ranking")
    current = action["state_before"]
    prompt = runtime_prompt(case["target"], current, TrajectoryHistory())
    observation = parse_pointer_observation(prompt, row_id=f"{case['id']}::legality_probe")
    mapped = mapped_from_visible(current)
    for rank, candidate in enumerate(ranked, 1):
        if execute_pair_indices(mapped, observation, [candidate]).get("ok"):
            return rank
    return None


def reference_import_batches(report: dict) -> dict[str, list[tuple[tuple[str, int], ...]]]:
    """Read held-out actions only for retrospective error analysis, never policy input."""
    path = Path(report["source"]["path"])
    if sha256(path) != report["source"]["sha256"]:
        raise ValueError("held-out reference source hash mismatch")
    batches: dict[str, list[tuple[tuple[str, int], ...]]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["decision_type"] != "import":
            continue
        calls = [call for message in row["messages"] if message.get("role") == "assistant"
                 for call in message.get("tool_calls", ())]
        if len(calls) != 1 or calls[0]["function"]["name"] != "import_fragments":
            raise ValueError(f"{row['id']}: malformed reference IMPORT")
        reaction_id = str(row["metadata"]["reaction_id"])
        batches.setdefault(reaction_id, []).append(
            chemical_batch(calls[0]["function"]["arguments"])
        )
    return batches


def analyze(report: dict, cases: list[dict]) -> dict:
    if report["evaluated_reactions"] != len(cases):
        raise ValueError("case count differs from frozen report")
    ids = [str(case["id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate reaction ID in cases")
    exact = 0
    terminals: Counter[str] = Counter()
    event_failures = []
    repeated_imports = []
    wrong_finished_without_import = 0
    reference_imports = reference_import_batches(report)
    import_counts: Counter[str] = Counter()
    if not set(reference_imports).issubset(set(ids)) and len(cases) == report["reaction_denominator"]:
        raise ValueError("reference IMPORT reaction is absent from full evaluation")
    for case in cases:
        completed = case["terminal"] == "FINISHED"
        recomputed = bool(
            completed and canonical_visible(case["predicted_precursor"])
            == canonical_visible(case["expected_precursor"])
        )
        if recomputed != case["endpoint_exact"] or completed != case["completed"]:
            raise ValueError(f"{case['id']}: terminal or endpoint scoring mismatch")
        exact += int(recomputed)
        terminals[case["terminal"]] += 1
        if completed and not recomputed and case["import_batches"] == 0:
            wrong_finished_without_import += 1
        imports = [action.get("batch") for action in case["actions"]
                   if action["action"] == "import_fragments"]
        expected_imports = reference_imports.get(str(case["id"]), [])
        if expected_imports:
            import_counts["gt_import_reactions"] += 1
            import_counts["gt_import_decisions"] += len(expected_imports)
            import_counts["endpoint_exact_given_gt_import"] += int(recomputed)
            if imports:
                import_counts["pred_import_given_gt"] += 1
                first_exact = tuple(tuple(item) for item in imports[0]) == expected_imports[0]
                import_counts["first_batch_exact_given_gt"] += int(first_exact)
                import_counts["first_batch_wrong_given_gt"] += int(not first_exact)
                import_counts["endpoint_exact_given_first_batch_exact"] += int(
                    first_exact and recomputed
                )
        else:
            import_counts["gt_no_import_reactions"] += 1
            import_counts["endpoint_exact_given_gt_no_import"] += int(recomputed)
            import_counts["false_positive_import"] += int(bool(imports))
        if any(left == right for left, right in zip(imports, imports[1:])):
            repeated_imports.append(case["id"])
        if case["terminal"] == "ELECTRON_EXECUTION_FAILED":
            event_failures.append({
                "id": case["id"],
                "accepted_electron_events_before_failure": case["electron_events"],
                "first_legal_single_rank_in_top8": first_legal_single_rank(case),
            })
    if exact != report["endpoint_exact"] or dict(terminals) != report["terminal_counts"]:
        raise ValueError("recomputed totals differ from frozen report")
    return {
        "evaluated_reactions": len(cases),
        "endpoint_exact": exact,
        "terminal_counts": dict(sorted(terminals.items())),
        "wrong_finished_without_import": wrong_finished_without_import,
        "repeated_import_reaction_ids": repeated_imports,
        "reference_import_breakdown": dict(sorted(import_counts.items())),
        "electron_failures": event_failures,
        "electron_failures_with_legal_single_in_top8": sum(
            item["first_legal_single_rank_in_top8"] is not None for item in event_failures
        ),
        "interpretation": "Local legality rescue only; endpoint after alternative action not evaluated",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report_path, cases_path = args.run / "report.json", args.run / "cases.jsonl"
    report = json.loads(report_path.read_text())
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line]
    summary = {
        "artifact_type": "system_one_product_start_offline_failure_analysis",
        "report_sha256": sha256(report_path),
        "cases_sha256": sha256(cases_path),
        "source_sha256": report["source"]["sha256"],
        "policy": report["policy"],
        **analyze(report, cases),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
