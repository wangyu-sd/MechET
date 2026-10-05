#!/usr/bin/env python3
"""Independently gate principal-target history data before training."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256
from scripts.build_system_one_principal_target_history import (
    ARTIFACT, CONTRACT, load_products, principal_component_from_mixture,
)
from scripts.train_system_one_electron_flow import verify_source


def check_row(original: dict, converted: dict, proxy_product: str) -> None:
    """Prove the model-visible target line is the only changed supervision."""
    if original["id"] != converted["id"] or original["source_id"] != converted["source_id"]:
        raise ValueError("reaction/decision identity changed")
    principal, _ = principal_component_from_mixture(
        str(original["target_smiles"]), proxy_product
    )
    if converted.get("principal_product_smiles") != principal:
        raise ValueError("selected principal product differs from proxy-aligned component")
    restored = copy.deepcopy(converted)
    restored.pop("principal_product_smiles")
    metadata = restored["metadata"]
    if metadata.pop("target_prompt_contract", None) != CONTRACT:
        raise ValueError("principal-target prompt contract missing")
    users = [message for message in restored["messages"] if message.get("role") == "user"]
    source_users = [message for message in original["messages"] if message.get("role") == "user"]
    if len(users) != 1 or len(source_users) != 1:
        raise ValueError("observation count changed")
    expected = f"TARGET PRODUCT SMILES: {principal}\n"
    original_prefix = f"TARGET PRODUCT SMILES: {original['target_smiles']}\n"
    if (not users[0]["content"].startswith(expected)
            or not source_users[0]["content"].startswith(original_prefix)
            or users[0]["content"][len(expected):]
            != source_users[0]["content"][len(original_prefix):]):
        raise ValueError("non-target observation text changed")
    users[0]["content"] = source_users[0]["content"]
    if restored != original:
        raise ValueError("action, executor state, metadata or non-target data changed")


def audit(source_dir: Path, full_dir: Path, converted_dir: Path) -> dict:
    manifest_path = converted_dir / "manifest.json"
    status_path = converted_dir / "ARTIFACT_STATUS.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if (manifest["artifact_type"] != ARTIFACT
            or manifest["target_prompt_contract"] != CONTRACT
            or status["artifact_id"] != ARTIFACT
            or status["target_prompt_contract"] != CONTRACT
            or manifest["source_manifest_sha256"] != sha256(source_dir / "manifest.json")
            or manifest["full_endpoint_manifest_sha256"] != sha256(full_dir / "manifest.json")):
        raise ValueError("principal-target artifact lineage mismatch")
    summaries = {}
    for split in ("train", "valid", "test"):
        source_path = source_dir / f"{split}.jsonl"
        output_path = converted_dir / f"{split}.jsonl"
        source = verify_source(source_path)
        products, endpoint_sha = load_products(full_dir, split)
        declared = manifest["splits"][split]
        if (declared["source_sha256"] != source["sha256"]
                or declared["full_endpoint_sha256"] != endpoint_sha
                or declared["output_sha256"] != sha256(output_path)):
            raise ValueError(f"{split}: output/source hash mismatch")
        seen: set[str] = set()
        counts: Counter[str] = Counter()
        with source_path.open() as source_reader, output_path.open() as output_reader:
            for source_line, output_line in zip_longest(source_reader, output_reader):
                if source_line is None or output_line is None:
                    raise ValueError(f"{split}: source/output decision denominator differs")
                original, converted = json.loads(source_line), json.loads(output_line)
                reaction_id = str(original["source_id"])
                if reaction_id not in products:
                    raise ValueError(f"{split}: strict ID absent from full endpoint")
                check_row(original, converted, products[reaction_id])
                seen.add(reaction_id)
                counts["decision_rows"] += 1
                counts[f"{original['metadata']['decision_type']}_decisions"] += 1
        if (len(seen) != source["reaction_denominator"]
                or counts["decision_rows"] != source["decision_rows"]
                or counts["event_decisions"] != source["event_decisions"]
                or counts["decision_rows"] != declared["decision_rows"]
                or len(seen) != declared["reactions"]):
            raise ValueError(f"{split}: independent reaction/decision counts mismatch")
        summaries[split] = {
            "source_sha256": source["sha256"],
            "full_endpoint_sha256": endpoint_sha,
            "output_sha256": declared["output_sha256"],
            "reactions": len(seen),
            **dict(counts),
        }
    return {
        "artifact_type": "system_one_principal_target_history_independent_audit",
        "prompt_contract": CONTRACT,
        "source_manifest_sha256": sha256(source_dir / "manifest.json"),
        "full_endpoint_manifest_sha256": sha256(full_dir / "manifest.json"),
        "splits": summaries,
        "all_actions_and_executor_states_unchanged": True,
    }


def promote(converted_dir: Path, report: dict) -> None:
    """Publish training permission last, only after complete independent audit."""
    manifest_path = converted_dir / "manifest.json"
    status_path = converted_dir / "ARTIFACT_STATUS.json"
    report_path = converted_dir / "INDEPENDENT_AUDIT.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if (manifest.get("training_allowed") is not False
            or status.get("training_allowed") is not False
            or report_path.exists()):
        raise ValueError("principal-target data is not in pending-audit state")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest["audit_report_sha256"] = sha256(report_path)
    manifest["training_allowed"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status["status"] = "validated_principal_target_trace_view"
    status["audit_report_sha256"] = sha256(report_path)
    status["manifest_sha256"] = sha256(manifest_path)
    status["training_allowed"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--full-endpoint-dir", type=Path, required=True)
    parser.add_argument("--converted-dir", type=Path, required=True)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    report = audit(args.source_dir, args.full_endpoint_dir, args.converted_dir)
    if args.promote:
        promote(args.converted_dir, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
