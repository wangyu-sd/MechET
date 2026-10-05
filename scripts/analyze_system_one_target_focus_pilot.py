#!/usr/bin/env python3
"""Independently rescore paired 64-row System-One decoder/prompt diagnostics."""
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
from scripts.eval_system_one_full_endpoint import select_ids
from scripts.score_system_one_structural_bridge import replay_provenance


def _rows(path: Path, field: str) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row[field])
        if key in rows:
            raise ValueError(f"duplicate reaction {key}: {path}")
        rows[key] = row
    return rows


def trajectory_signature(case: dict) -> tuple:
    """Compare executed chemistry, ignoring logits, token lengths and audit tags."""
    return tuple((action["action"], bool(action.get("accepted")),
                  tuple(action.get("selected_pairs") or ()),
                  tuple(tuple(item) for item in (action.get("batch") or ())),
                  action.get("state_after")) for action in case["actions"])


def validate_run(run: Path, full: dict[str, dict], context: dict[str, dict],
                 *, source_sha: str, context_report_sha: str,
                 focus: bool, principal_prompt: bool = False) -> tuple[dict, dict[str, dict]]:
    report = json.loads((run / "report.json").read_text())
    cases_path = run / "cases.jsonl"
    cases = _rows(cases_path, "id")
    if (report["artifact_type"]
            != "system_one_pr81_complete_hf_principal_product_structural_endpoint_evaluation"
            or report["split"] != "valid"
            or report["product_source_field"] != "rxn_prod_equ"
            or report["full_endpoint_source"]["sha256"] != source_sha
            or report["full_endpoint_reaction_denominator"] != 3120
            or report["context_report_sha256"] != context_report_sha
            or report.get("first_event_target_focus", False) is not focus
            or report.get("principal_target_prompt", False) is not principal_prompt
            or report["selection"] != {"method": "sha256_seed_reaction_id",
                                       "seed": 17, "limit": 64}
            or report["evaluated_reactions"] != len(cases)
            or len(cases) != 64
            or report["cases_sha256"] != sha256(cases_path)):
        raise ValueError("paired pilot report/source contract mismatch")
    expected_ids = set(select_ids(list(full), seed=17, limit=64))
    if set(cases) != expected_ids:
        raise ValueError("pilot reaction selection differs from frozen hash sample")
    exact = completed = 0
    for reaction_id, case in cases.items():
        reference = full[reaction_id]
        proposal = context[reaction_id]
        if (case["principal_product_input"] != reference["product_unmapped"]
                or case["principal_product_input"] != proposal["product_unmapped"]
                or case["predicted_context_batch"] != proposal["predicted_context_batch"]
                or case["expected_structural_precursor"]
                != reference["structural_precursor"]):
            raise ValueError(f"{reaction_id}: pilot input/reference join mismatch")
        _, structural, failure = replay_provenance(case)
        rescored = bool(case["completed"] and structural_exact(
            structural, reference["structural_precursor"]
        ))
        if (rescored != case["structural_exact"]
                or structural != (case["predicted_structural_precursor"] or "")
                or failure != case["replay_failure"]):
            raise ValueError(f"{reaction_id}: independent replay/score mismatch")
        exact += int(rescored)
        completed += int(case["completed"])
    if exact != report["structural_exact"] or completed != report["completed"]:
        raise ValueError("pilot report aggregate differs from independent scoring")
    return report, cases


def compare(baseline_dir: Path, focused_dir: Path, full_dir: Path,
            context_dir: Path, *, intervention: str = "first_event_target_focus") -> dict:
    if intervention not in {"first_event_target_focus", "principal_target_prompt"}:
        raise ValueError("unsupported paired pilot intervention")
    manifest = json.loads((full_dir / "manifest.json").read_text())
    full_path = full_dir / "valid.jsonl"
    source_sha = sha256(full_path)
    if (manifest["product_source_field"] != "rxn_prod_equ"
            or manifest["splits"]["valid"]["endpoint_sha256"] != source_sha
            or manifest["splits"]["valid"]["rows"] != 3120):
        raise ValueError("equ-proxy full endpoint source mismatch")
    full = _rows(full_path, "source_id")
    if len(full) != 3120:
        raise ValueError("full endpoint source denominator mismatch")
    context_report_path = context_dir / "report.json"
    context_report = json.loads(context_report_path.read_text())
    context_path = context_dir / "cases.jsonl"
    if (context_report["product_source_field"] != "rxn_prod_equ"
            or context_report["heldout_source"]["sha256"] != source_sha
            or context_report["cases_sha256"] != sha256(context_path)):
        raise ValueError("equ-proxy context source mismatch")
    context = _rows(context_path, "reaction_id")
    if set(context) != set(full):
        raise ValueError("context/full source ID coverage mismatch")
    baseline_report, baseline = validate_run(
        baseline_dir, full, context, source_sha=source_sha,
        context_report_sha=sha256(context_report_path), focus=False,
    )
    focused_report, focused = validate_run(
        focused_dir, full, context, source_sha=source_sha,
        context_report_sha=sha256(context_report_path),
        focus=intervention == "first_event_target_focus",
        principal_prompt=intervention == "principal_target_prompt",
    )
    for key in ("weights", "strict_policy_source", "train_import_source", "max_actions",
                "legality_backoff", "context_cases_sha256"):
        if baseline_report[key] != focused_report[key]:
            raise ValueError(f"pilot intervention changed frozen policy field {key}")
    counts: Counter[str] = Counter()
    for reaction_id, before in baseline.items():
        after = focused[reaction_id]
        overrides = [action["first_event_target_focus"] for action in after["actions"]
                     if action.get("first_event_target_focus", {}).get("overrode_baseline")]
        counts["reactions"] += 1
        counts["trajectory_changed"] += int(
            trajectory_signature(before) != trajectory_signature(after)
        )
        counts["first_event_override_reactions"] += int(bool(overrides))
        counts["first_event_override_steps"] += len(overrides)
        counts["baseline_exact"] += int(before["structural_exact"])
        counts["focused_exact"] += int(after["structural_exact"])
        counts["both_exact"] += int(before["structural_exact"] and after["structural_exact"])
        counts["baseline_only_exact"] += int(before["structural_exact"] and not after["structural_exact"])
        counts["focused_only_exact"] += int(after["structural_exact"] and not before["structural_exact"])
        counts["baseline_completed"] += int(before["completed"])
        counts["focused_completed"] += int(after["completed"])
    return {
        "artifact_type": f"system_one_pr81_equ_proxy_{intervention}_valid64_paired_audit",
        "intervention": intervention,
        "scope": "validation_only_diagnostic_not_full_3120_or_test_result",
        "full_endpoint_sha256": source_sha,
        "context_report_sha256": sha256(context_report_path),
        "baseline_report_sha256": sha256(baseline_dir / "report.json"),
        "focused_report_sha256": sha256(focused_dir / "report.json"),
        "counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--focused-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--intervention", choices=("first_event_target_focus",
                                                  "principal_target_prompt"),
                        default="first_event_target_focus")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.baseline_dir, args.focused_dir,
                     args.full_endpoint_dir, args.context_dir,
                     intervention=args.intervention)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
