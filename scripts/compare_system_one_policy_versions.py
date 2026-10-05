#!/usr/bin/env python3
"""Independently replay and pair two System-One policies on one frozen endpoint slice."""
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
from scripts.compare_system_one_pr71_successor import cluster_bootstrap
from scripts.eval_system_one_full_endpoint import load_policy_contexts, select_ids
from scripts.score_system_one_structural_bridge import replay_provenance


def read_cases(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank rollout row: {path}")
            case = json.loads(line)
            reaction_id = str(case["id"])
            if reaction_id in rows:
                raise ValueError(f"duplicate rollout reaction: {reaction_id}")
            rows[reaction_id] = case
    return rows


def executed_signature(case: dict) -> tuple:
    """Compare executed paths, excluding token lengths, logits and metadata."""
    accepted = []
    for action in case["actions"]:
        # An exhausted legality-backoff attempt records execute_ok=False but
        # has no accepted field; it never changed the executor state.
        if "accepted" not in action:
            if action.get("execute_ok") is False:
                continue
            raise ValueError("rollout action lacks execution/acceptance status")
        if action["accepted"]:
            accepted.append((action["action"],
                             tuple(action.get("selected_pairs", ())),
                             tuple(tuple(item) for item in action.get("batch", ())),
                             action.get("state_after")))
    return tuple(accepted)


def summarize_paired(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    if set(baseline) != set(candidate):
        raise ValueError("paired policies have different reaction IDs")
    count: Counter[str] = Counter()
    gained = []
    lost = []
    paired = []
    for reaction_id in sorted(baseline):
        old, new = baseline[reaction_id], candidate[reaction_id]
        for field in ("principal_product_input", "predicted_context_batch",
                      "inferred_final_mixture", "expected_structural_precursor"):
            if old[field] != new[field]:
                raise ValueError(f"{reaction_id}: paired input/reference differs in {field}")
        count["reactions"] += 1
        count["baseline_exact"] += int(old["structural_exact"])
        count["candidate_exact"] += int(new["structural_exact"])
        count["baseline_formal_finish"] += int(old["completed"])
        count["candidate_formal_finish"] += int(new["completed"])
        count["executed_path_changed"] += int(executed_signature(old) != executed_signature(new))
        paired.append((1, int(new["structural_exact"]), int(old["structural_exact"])))
        if new["structural_exact"] and not old["structural_exact"]:
            gained.append(reaction_id)
        if old["structural_exact"] and not new["structural_exact"]:
            lost.append(reaction_id)
    count["gained_exact"] = len(gained)
    count["lost_exact"] = len(lost)
    ci = cluster_bootstrap(paired)
    return {
        "counts": dict(count), "gained_ids": gained, "lost_ids": lost,
        "candidate_minus_baseline_exact_rate": (
            count["candidate_exact"] - count["baseline_exact"]
        ) / count["reactions"],
        "reaction_bootstrap_95pct_ci": list(ci),
        "bootstrap_seed": 17,
        "bootstrap_repetitions": 5000,
    }


def audit_run(run_dir: Path, references: dict[str, dict], contexts: dict,
              *, split: str, full_sha: str, expected: int) -> tuple[dict, dict]:
    report_path = run_dir / "report.json"
    cases_path = run_dir / "cases.jsonl"
    report = json.loads(report_path.read_text())
    if (report.get("artifact_type")
        != "system_one_pr81_complete_hf_principal_product_structural_endpoint_evaluation"
        or report.get("split") != split
        or report.get("product_source_field") != "rxn_prod_equ"
        or report.get("full_endpoint_reaction_denominator") != 3120
        or report.get("full_endpoint_source", {}).get("sha256") != full_sha
        or report.get("evaluated_reactions") != expected
        or report.get("cases_sha256") != sha256(cases_path)):
        raise ValueError(f"rollout report/source/hash mismatch: {run_dir}")
    selection = report.get("selection", {})
    selection_limit = 0 if expected == 3120 else expected
    if (selection.get("method") != "sha256_seed_reaction_id"
            or selection.get("seed") != 17 or selection.get("limit") != selection_limit):
        raise ValueError(f"rollout selection differs from frozen {split} slice: {run_dir}")
    rows = read_cases(cases_path)
    selected = set(select_ids(list(references), seed=17, limit=selection_limit))
    if set(rows) != selected:
        raise ValueError(f"rollout IDs differ from deterministic selection: {run_dir}")
    exact = completed = 0
    for reaction_id, case in rows.items():
        reference = references[reaction_id]
        product, batch = contexts[reaction_id]
        if (case["principal_product_input"] != product
                or case["predicted_context_batch"] != list(batch)
                or case["expected_structural_precursor"] != reference["structural_precursor"]):
            raise ValueError(f"{reaction_id}: input, context or reference mismatch")
        full, structural, failure = replay_provenance(case)
        if (case.get("predicted_full_precursor_replayed")
            != (full if case["completed"] else None)
            or case.get("predicted_structural_precursor")
            != (structural if case["completed"] else None)
            or case.get("replay_failure") != failure):
            raise ValueError(f"{reaction_id}: reported prediction differs from independent replay")
        hit = bool(case["completed"] and structural_exact(
            structural, reference["structural_precursor"]
        ))
        if hit != case["structural_exact"]:
            raise ValueError(f"{reaction_id}: reported exactness differs from independent score")
        exact += int(hit)
        completed += int(case["completed"])
    if exact != report["structural_exact"] or completed != report["completed"]:
        raise ValueError(f"rollout aggregate differs from independent case audit: {run_dir}")
    return report, rows


def compare(baseline_dir: Path, candidate_dir: Path, context_dir: Path,
            full_endpoint_dir: Path, *, split: str = "valid", expected: int = 64) -> dict:
    if (split, expected) not in (("valid", 64), ("valid", 3120), ("test", 3120)):
        raise ValueError("this paired gate requires valid64, valid3120 or test3120")
    manifest = json.loads((full_endpoint_dir / "manifest.json").read_text())
    full_path = full_endpoint_dir / f"{split}.jsonl"
    train_path = full_endpoint_dir / "train.jsonl"
    if (manifest.get("product_source_field") != "rxn_prod_equ"
            or manifest["splits"][split]["rows"] != 3120
            or manifest["splits"][split]["endpoint_sha256"] != sha256(full_path)
            or manifest["splits"]["train"]["endpoint_sha256"] != sha256(train_path)):
        raise ValueError("frozen equ-proxy endpoint source mismatch")
    full_sha = manifest["splits"][split]["endpoint_sha256"]
    contexts, context_report = load_policy_contexts(
        context_dir, split=split,
        train_sha=manifest["splits"]["train"]["endpoint_sha256"],
        heldout_sha=full_sha, product_field="rxn_prod_equ",
    )
    references = {str(row["source_id"]): row for row in map(json.loads, full_path.open())}
    if len(references) != 3120 or set(contexts) != set(references):
        raise ValueError("endpoint/context universe is not the complete held-out split")
    baseline_report, baseline = audit_run(
        baseline_dir, references, contexts, split=split, full_sha=full_sha, expected=expected,
    )
    candidate_report, candidate = audit_run(
        candidate_dir, references, contexts, split=split, full_sha=full_sha, expected=expected,
    )
    if (baseline_report["context_cases_sha256"] != candidate_report["context_cases_sha256"]
            or baseline_report["context_cases_sha256"] != context_report["cases_sha256"]
            or baseline_report.get("principal_target_prompt", False)
            or not candidate_report.get("principal_target_prompt", False)):
        raise ValueError("baseline/candidate prompt or context protocol mismatch")
    return {
        "artifact_type": f"system_one_pr81_paired_principal_training_{split}{expected}_audit",
        "split": split,
        "frozen_reactions": expected,
        "endpoint_source_sha256": full_sha,
        "context_cases_sha256": context_report["cases_sha256"],
        "baseline_report_sha256": sha256(baseline_dir / "report.json"),
        "candidate_report_sha256": sha256(candidate_dir / "report.json"),
        "comparison": summarize_paired(baseline, candidate),
        "limitations": ["old and new policies differ in training target/weights and import target mode",
                        "equ-field principal product is a deterministic proxy, not an authenticated desired-product label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--expected", type=int, choices=(64, 3120), default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.baseline_dir, args.candidate_dir,
                     args.context_dir, args.full_endpoint_dir,
                     split=args.split, expected=args.expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
