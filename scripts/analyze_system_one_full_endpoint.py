#!/usr/bin/env python3
"""Hash-bound stratification of complete PR81 endpoint rollout coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.endpoints import structural_exact
from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.train_system_one_electron_flow import verify_source


def analyze(run_dir: Path, context_dir: Path, strict_dir: Path, full_dir: Path) -> dict:
    report = json.loads((run_dir / "report.json").read_text())
    path = run_dir / "cases.jsonl"
    if (report["artifact_type"] != "system_one_pr81_complete_hf_principal_product_structural_endpoint_evaluation"
            or report["full_endpoint_reaction_denominator"] != 3120
            or report["evaluated_reactions"] != 3120
            or report["cases_sha256"] != sha256(path)):
        raise ValueError("not a complete hash-verified full endpoint rollout")
    split = report["split"]
    full_path = full_dir / f"{split}.jsonl"
    full_manifest = json.loads((full_dir / "manifest.json").read_text())
    full_declared = full_manifest["splits"][split]
    if (full_manifest["benchmark_universe"] != "complete_hf_reaction_level_split"
            or full_manifest["executor_filtering"] is not False
            or full_declared["rows"] != 3120
            or full_declared["endpoint_sha256"] != sha256(full_path)
            or report["full_endpoint_source"]["sha256"] != full_declared["endpoint_sha256"]):
        raise ValueError("full endpoint reference source mismatch")
    full = {}
    for line in full_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in full:
            raise ValueError(f"duplicate full endpoint ID {reaction_id}")
        full[reaction_id] = row
    if len(full) != 3120:
        raise ValueError("full endpoint reference denominator mismatch")
    context_report = json.loads((context_dir / "report.json").read_text())
    context_path = context_dir / "cases.jsonl"
    if (report["context_report_sha256"] != sha256(context_dir / "report.json")
            or context_report["cases_sha256"] != sha256(context_path)
            or report["context_cases_sha256"] != context_report["cases_sha256"]
            or context_report["split"] != split):
        raise ValueError("context prediction source mismatch")
    contexts = {}
    for line in context_path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["reaction_id"])
        if reaction_id in contexts:
            raise ValueError(f"duplicate context reaction ID {reaction_id}")
        contexts[reaction_id] = row
    strict_path = strict_dir / f"{split}.jsonl"
    strict_source = verify_source(strict_path)
    strict_ids = set()
    for line in strict_path.read_text().splitlines():
        row = json.loads(line)
        if row["metadata"]["decision_type"] == "finish":
            strict_ids.add(str(row["metadata"]["reaction_id"]))
    if len(strict_ids) != strict_source["reaction_denominator"]:
        raise ValueError("strict reference ID denominator mismatch")
    if report["strict_policy_source"]["sha256"] != strict_source["sha256"]:
        raise ValueError("frozen policy lineage strict source mismatch")
    counts: dict[str, Counter[str]] = {
        "all": Counter(), "strict_trace_overlap": Counter(), "outside_strict_trace": Counter()
    }
    seen = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        reaction_id = str(row["id"])
        if reaction_id in seen or reaction_id not in contexts or reaction_id not in full:
            raise ValueError(f"duplicate/unmatched rollout ID {reaction_id}")
        seen.add(reaction_id)
        if (row["principal_product_input"] != contexts[reaction_id]["product_unmapped"]
                or row["principal_product_input"] != full[reaction_id]["product_unmapped"]
                or row["predicted_context_batch"] != contexts[reaction_id]["predicted_context_batch"]
                or row["expected_structural_precursor"] != full[reaction_id]["structural_precursor"]):
            raise ValueError(f"policy input differs from context proposal: {reaction_id}")
        rescored = bool(row["completed"] and structural_exact(
            row["predicted_structural_precursor"], row["expected_structural_precursor"]
        ))
        if rescored != row["structural_exact"]:
            raise ValueError(f"structural endpoint score mismatch: {reaction_id}")
        groups = [counts["all"], counts[
            "strict_trace_overlap" if reaction_id in strict_ids else "outside_strict_trace"
        ]]
        for group in groups:
            group["reactions"] += 1
            group["structural_exact"] += int(row["structural_exact"])
            group["completed"] += int(row["completed"])
            group["context_top1_exact"] += int(contexts[reaction_id]["top1_exact"])
            group["electron_events"] += int(row["electron_events"])
            group["import_batches"] += int(row["import_batches"])
            group[f"terminal_{row['terminal']}"] += 1
    if len(seen) != 3120 or len(contexts) != 3120:
        raise ValueError("full endpoint reaction denominator/coverage mismatch")
    if (counts["all"]["structural_exact"] != report["structural_exact"]
            or counts["strict_trace_overlap"]["reactions"] != len(strict_ids)):
        raise ValueError("report/strict-overlap count mismatch")
    return {
        "artifact_type": "system_one_pr81_full_endpoint_stratified_audit",
        "split": split,
        "rollout_report_sha256": sha256(run_dir / "report.json"),
        "rollout_cases_sha256": sha256(path),
        "context_report_sha256": sha256(context_dir / "report.json"),
        "full_endpoint_source_sha256": full_declared["endpoint_sha256"],
        "strict_source_sha256": strict_source["sha256"],
        "groups": {name: dict(group) for name, group in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run_dir, args.context_dir, args.strict_dir,
                     args.full_endpoint_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
