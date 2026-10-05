#!/usr/bin/env python3
"""Paired diagnostics for historical min-field and new equ-field proxies.

This does not estimate a gain in desired-product accuracy: the evaluated target
changes for some reactions, and both versions remain heuristic product labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.analyze_system_one_full_endpoint import analyze
from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.stratify_system_one_full_endpoint import verified_changed_ids


def _rows(path: Path, id_key: str) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row[id_key])
        if key in rows:
            raise ValueError(f"duplicate reaction ID {key}: {path}")
        rows[key] = row
    return rows


def summarize_paired(old: dict[str, dict], new: dict[str, dict],
                     old_context: dict[str, dict], new_context: dict[str, dict],
                     changed_ids: set[str]) -> dict[str, dict[str, int]]:
    if not (set(old) == set(new) == set(old_context) == set(new_context)):
        raise ValueError("proxy version reaction coverage differs")
    if not changed_ids <= set(old):
        raise ValueError("raw-audit changed IDs not covered by both runs")
    counters: dict[str, Counter[str]] = {
        "all": Counter(), "target_changed": Counter(), "target_unchanged": Counter(),
    }
    for reaction_id in old:
        before, after = old[reaction_id], new[reaction_id]
        changed = reaction_id in changed_ids
        if (before["principal_product_input"] != after["principal_product_input"]) != changed:
            raise ValueError(f"observed target difference disagrees with audit: {reaction_id}")
        if not changed and (
            before["expected_structural_precursor"]
            != after["expected_structural_precursor"]
        ):
            raise ValueError(f"unchanged target has changed reference: {reaction_id}")
        for group in (counters["all"], counters[
            "target_changed" if changed else "target_unchanged"
        ]):
            group["reactions"] += 1
            group["old_exact"] += int(before["structural_exact"])
            group["new_exact"] += int(after["structural_exact"])
            group["both_exact"] += int(before["structural_exact"] and after["structural_exact"])
            group["old_only_exact"] += int(before["structural_exact"] and not after["structural_exact"])
            group["new_only_exact"] += int(after["structural_exact"] and not before["structural_exact"])
            group["old_completed"] += int(before["completed"])
            group["new_completed"] += int(after["completed"])
            group["old_context_top1"] += int(old_context[reaction_id]["top1_exact"])
            group["new_context_top1"] += int(new_context[reaction_id]["top1_exact"])
            group["context_proposal_changed"] += int(
                before["predicted_context_batch"] != after["predicted_context_batch"]
            )
    return {name: dict(count) for name, count in counters.items()}


def compare(old_run: Path, new_run: Path, old_context: Path, new_context: Path,
            strict_dir: Path, old_endpoint: Path, new_endpoint: Path,
            product_audit_path: Path, handoff_path: Path) -> dict:
    old_audit = analyze(old_run, old_context, strict_dir, old_endpoint)
    new_audit = analyze(new_run, new_context, strict_dir, new_endpoint)
    if (old_audit["split"] != new_audit["split"]
            or old_audit["product_source_field"] != "rxn_prod_min"
            or new_audit["product_source_field"] != "rxn_prod_equ"
            or old_audit["strict_source_sha256"] != new_audit["strict_source_sha256"]):
        raise ValueError("proxy comparison split/source contract mismatch")
    split = old_audit["split"]
    audit = json.loads(product_audit_path.read_text())
    handoff = json.loads(handoff_path.read_text())
    old_manifest_path = old_endpoint / "manifest.json"
    new_manifest_path = new_endpoint / "manifest.json"
    new_manifest = json.loads(new_manifest_path.read_text())
    if (handoff["old_manifest_sha256"] != sha256(old_manifest_path)
            or old_audit["full_endpoint_source_sha256"]
            != handoff["splits"][split]["old_endpoint_sha256"]):
        raise ValueError("historical proxy does not match handoff")
    changed_ids = verified_changed_ids(
        audit, handoff, split=split, product_field="rxn_prod_equ",
        full_manifest_sha=sha256(new_manifest_path),
        full_source_sha=new_audit["full_endpoint_source_sha256"],
        raw_sha=new_manifest["source_files"][split]["sha256"],
        product_audit_sha=sha256(product_audit_path),
    )
    groups = summarize_paired(
        _rows(old_run / "cases.jsonl", "id"),
        _rows(new_run / "cases.jsonl", "id"),
        _rows(old_context / "cases.jsonl", "reaction_id"),
        _rows(new_context / "cases.jsonl", "reaction_id"),
        changed_ids,
    )
    if (groups["all"]["reactions"] != 3120
            or groups["target_changed"]["reactions"] != len(changed_ids)
            or groups["all"]["old_exact"] != old_audit["groups"]["all"]["structural_exact"]
            or groups["all"]["new_exact"] != new_audit["groups"]["all"]["structural_exact"]):
        raise ValueError("paired summary does not reconcile with independent audits")
    return {
        "artifact_type": "system_one_pr81_paired_proxy_version_diagnostic",
        "split": split,
        "interpretation": "Different proxy targets; paired counts are diagnostics, not desired-product accuracy change",
        "old_rollout_report_sha256": old_audit["rollout_report_sha256"],
        "new_rollout_report_sha256": new_audit["rollout_report_sha256"],
        "raw_field_audit_sha256": sha256(product_audit_path),
        "handoff_sha256": sha256(handoff_path),
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("old-run", "new-run", "old-context", "new-context", "strict-dir",
                 "old-endpoint", "new-endpoint", "product-audit", "handoff", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.old_run, args.new_run, args.old_context, args.new_context,
                     args.strict_dir, args.old_endpoint, args.new_endpoint,
                     args.product_audit, args.handoff)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
