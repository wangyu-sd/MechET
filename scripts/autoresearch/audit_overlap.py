#!/usr/bin/env python3
"""Independently audit PR #69 frozen training versus all R1–R5 cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.stratified_manifest import (
    digest, eval_exclusions, product_key, row_reaction_id,
)


def audit(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    freeze_path = output / "scientific_freeze/manifests/freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("scientific freeze required for overlap audit")
    frozen = json.loads(freeze_path.read_text())
    if frozen.get("engineering_only") is not False or frozen.get("campaign_id") != config["campaign_id"]:
        raise ValueError("overlap audit requires this campaign's scientific freeze")
    excluded_ids, excluded_products, evaluation_hashes = eval_exclusions(config, root)
    if evaluation_hashes != frozen.get("evaluation_hashes"):
        raise ValueError("overlap audit evaluation cohort hashes drifted")
    conditions = {}
    overlap_count = 0
    for condition in ("base", "mech"):
        selection = frozen["files"][condition]
        path = Path(selection["train"])
        if not path.is_file() or digest(path) != selection["train_sha256"]:
            raise ValueError(f"{condition} selected training file drifted")
        reaction_ids: set[str] = set()
        products: set[str] = set()
        decision_ids: set[str] = set()
        rows = 0
        decision_rows_with_overlap = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                decision_id = str(row["id"])
                if decision_id in decision_ids:
                    raise ValueError(f"duplicate {condition} decision ID: {decision_id}")
                decision_ids.add(decision_id)
                reaction = row_reaction_id(row)
                product = product_key(str(row["target_smiles"]))
                rows += 1
                reaction_ids.add(reaction)
                products.add(product)
                decision_rows_with_overlap += int(
                    reaction in excluded_ids or product in excluded_products)
        if rows != int(selection["rows"]):
            raise ValueError(f"{condition} selected training row count drifted")
        reaction_overlap = sorted(reaction_ids & excluded_ids)
        product_overlap = sorted(products & excluded_products)
        overlap_count += len(reaction_overlap) + len(product_overlap)
        conditions[condition] = {
            "selected_rows": rows,
            "selected_train_sha256": selection["train_sha256"],
            "unique_reaction_ids": len(reaction_ids),
            "unique_products": len(products),
            "reaction_overlap_count": len(reaction_overlap),
            "product_overlap_count": len(product_overlap),
            "decision_rows_with_overlap": decision_rows_with_overlap,
            "reaction_overlap_examples": reaction_overlap[:20],
            "product_overlap_examples": product_overlap[:20],
        }
    report = {
        "artifact_type": "pr69_scientific_train_eval_overlap_audit_v1",
        "campaign_id": config["campaign_id"],
        "scientific_freeze_sha256": digest(freeze_path),
        "evaluation_hashes": evaluation_hashes,
        "evaluation_reaction_ids": len(excluded_ids),
        "evaluation_products": len(excluded_products),
        "conditions": conditions,
        "overlap_count": overlap_count,
        "passed": overlap_count == 0,
    }
    path = output / "manifests/train_eval_overlap_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise ValueError("existing train/eval overlap audit differs from frozen inputs")
    else:
        path.write_text(content)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    print(json.dumps(audit(config, args.data_root, args.output), indent=2, sort_keys=True),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
