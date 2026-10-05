#!/usr/bin/env python3
"""Localize complete System-One endpoint errors by compiler coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.endpoints import structural_exact
from scripts.analyze_system_one_full_endpoint import analyze
from scripts.audit_system_one_full_endpoint_input_gap import sha256


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _group_name(reaction_id: str, strict_ids: set[str], complete_ids: set[str]) -> str:
    if reaction_id in strict_ids:
        return "stitched_strict_trace"
    if reaction_id in complete_ids:
        return "all_steps_executable_but_unstitched"
    return "incomplete_elementary_steps"


def summarize_cases(
    cases: list[dict], contexts: dict[str, dict], strict_ids: set[str],
    complete_ids: set[str], product_changed_ids: set[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Summarize mutually exclusive source strata without re-ranking cases."""
    if not strict_ids <= complete_ids:
        raise ValueError("stitched IDs must be a subset of all-step-executable IDs")
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in (
            "all", "stitched_strict_trace", "all_steps_executable_but_unstitched",
            "incomplete_elementary_steps",
        )
    }
    seen: set[str] = set()
    product_changed_ids = product_changed_ids or set()
    for row in cases:
        reaction_id = str(row["id"])
        if reaction_id in seen or reaction_id not in contexts:
            raise ValueError(f"duplicate/unmatched reaction ID: {reaction_id}")
        seen.add(reaction_id)
        category = _group_name(reaction_id, strict_ids, complete_ids)
        complete = bool(row["completed"])
        exact = bool(row["structural_exact"])
        context_exact = bool(contexts[reaction_id]["top1_exact"])
        no_transform = bool(complete and structural_exact(
            row["predicted_structural_precursor"], row["principal_product_input"]
        ))
        for group in (counts["all"], counts[category]):
            group["reactions"] += 1
            group["structural_exact"] += int(exact)
            group["completed"] += int(complete)
            group["context_top1_exact"] += int(context_exact)
            group["context_exact_and_endpoint_exact"] += int(context_exact and exact)
            group["context_exact_but_endpoint_wrong"] += int(context_exact and not exact)
            group["min_equ_target_changed"] += int(reaction_id in product_changed_ids)
            group["min_equ_changed_and_endpoint_exact"] += int(
                reaction_id in product_changed_ids and exact
            )
            group["finished_no_transform"] += int(no_transform)
            group["finished_wrong_no_transform"] += int(no_transform and not exact)
            group["finished_wrong_with_electron_event"] += int(
                complete and not exact and int(row["electron_events"]) > 0
            )
            group["finished_wrong_with_import"] += int(
                complete and not exact and int(row["import_batches"]) > 0
            )
            group["zero_electron_events"] += int(int(row["electron_events"]) == 0)
            group[f"terminal_{row['terminal']}"] += 1
    if seen != set(contexts):
        raise ValueError("case/context reaction ID coverage mismatch")
    return {name: dict(group) for name, group in counts.items()}


def stratify(
    run_dir: Path, context_dir: Path, strict_dir: Path, full_dir: Path,
    compiler_dir: Path, product_audit_path: Path | None = None,
) -> dict:
    baseline = analyze(run_dir, context_dir, strict_dir, full_dir)
    split = baseline["split"]
    complete_path = compiler_dir / "complete_reaction_ids" / f"{split}.jsonl"
    complete_report_path = complete_path.with_suffix(".report.json")
    complete_report = json.loads(complete_report_path.read_text())
    compiler_lineage = json.loads((compiler_dir / "COMPILER_LINEAGE.json").read_text())
    full_manifest = json.loads((full_dir / "manifest.json").read_text())
    if (complete_report["output_sha256"] != sha256(complete_path)
            or complete_report["raw_parquet_sha256"]
            != full_manifest["source_files"][split]["sha256"]
            or complete_report["raw_reactions"] != 3120
            or complete_report["complete_reactions"]
            != compiler_lineage["complete_reactions"][split]):
        raise ValueError("all-step-executable source lineage mismatch")
    complete_ids = {str(row["reaction_id"]) for row in _jsonl(complete_path)}
    strict_rows = _jsonl(strict_dir / f"{split}.jsonl")
    strict_ids = {
        str(row["metadata"]["reaction_id"]) for row in strict_rows
        if row["metadata"]["decision_type"] == "finish"
    }
    if (len(complete_ids) != complete_report["complete_reactions"]
            or len(strict_ids) != baseline["groups"]["strict_trace_overlap"]["reactions"]):
        raise ValueError("compiler/strict reaction count mismatch")
    contexts = {str(row["reaction_id"]): row for row in _jsonl(context_dir / "cases.jsonl")}
    cases = _jsonl(run_dir / "cases.jsonl")
    product_changed_ids: set[str] = set()
    if product_audit_path is not None:
        product_audit = json.loads(product_audit_path.read_text())
        split_audit = product_audit["splits"][split]
        product_changed_ids = set(split_audit["changed_reaction_ids"])
        if (product_audit["artifact_type"]
                != "mech_uspto31k_existing_min_target_vs_equ_final_mixture_audit"
                or product_audit["endpoint_manifest_sha256"] != sha256(full_dir / "manifest.json")
                or split_audit["raw_sha256"] != full_manifest["source_files"][split]["sha256"]
                or split_audit["endpoint_sha256"] != baseline["full_endpoint_source_sha256"]
                or len(product_changed_ids) != split_audit["counts"]["main_product_changed"]):
            raise ValueError("product field audit lineage mismatch")
    groups = summarize_cases(cases, contexts, strict_ids, complete_ids,
                             product_changed_ids)
    if (groups["all"]["reactions"] != 3120
            or groups["all"]["structural_exact"]
            != baseline["groups"]["all"]["structural_exact"]
            or groups["all"]["min_equ_target_changed"] != len(product_changed_ids)
            or groups["stitched_strict_trace"]["reactions"] != len(strict_ids)
            or groups["all_steps_executable_but_unstitched"]["reactions"]
            != len(complete_ids) - len(strict_ids)
            or groups["incomplete_elementary_steps"]["reactions"]
            != 3120 - len(complete_ids)):
        raise ValueError("compiler stratification does not reconcile")
    return {
        "artifact_type": "system_one_pr81_full_endpoint_compiler_coverage_stratification",
        "split": split,
        "rollout_report_sha256": baseline["rollout_report_sha256"],
        "rollout_cases_sha256": baseline["rollout_cases_sha256"],
        "context_report_sha256": baseline["context_report_sha256"],
        "full_endpoint_source_sha256": baseline["full_endpoint_source_sha256"],
        "strict_source_sha256": baseline["strict_source_sha256"],
        "complete_ids_sha256": complete_report["output_sha256"],
        "compiler_lineage_sha256": sha256(compiler_dir / "COMPILER_LINEAGE.json"),
        "product_field_audit_sha256": sha256(product_audit_path) if product_audit_path else None,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--compiler-dir", type=Path, required=True)
    parser.add_argument("--product-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = stratify(args.run_dir, args.context_dir, args.strict_dir,
                      args.full_endpoint_dir, args.compiler_dir,
                      args.product_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
