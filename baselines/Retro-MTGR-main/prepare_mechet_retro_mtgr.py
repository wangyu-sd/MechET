#!/usr/bin/env python3
"""Audit and prepare frozen MechET splits for the Retro-MTGR adapter."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from mechet_retro_mtgr import analyze_row, read_jsonl, sha256_file, stable_id, write_jsonl


SPLITS = ("train", "valid", "test")


def label_catalog(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {spec["key"]: spec for spec in specs}
    return [
        {"id": index, **by_key[key]}
        for index, key in enumerate(sorted(by_key))
    ]


def analyze_split(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    targets: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        analysis = analyze_row(row)
        targets.append(analysis.target)
        if analysis.target["status"] == "supported":
            specs.extend(analysis.label_specs)
    return targets, specs


def audit(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_targets: dict[str, list[dict[str, Any]]] = {}
    all_specs: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        source = args.input_dir / f"{split}.jsonl"
        targets, specs = analyze_split(source)
        all_targets[split] = targets
        all_specs[split] = specs
        write_jsonl(args.output_dir / f"{split}_targets.jsonl", targets)

    train_catalog = label_catalog(all_specs["train"])
    train_keys = {item["key"] for item in train_catalog}
    (args.output_dir / "label_vocabulary.json").write_text(
        json.dumps({
            "source_splits": ["train"],
            "labels": train_catalog,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    split_reports: dict[str, Any] = {}
    for split in SPLITS:
        targets = all_targets[split]
        reasons = Counter(
            "supported" if row["status"] == "supported" else row["reason"]
            for row in targets
        )
        supported = [row for row in targets if row["status"] == "supported"]
        heldout_keys = {
            key
            for row in supported
            for key in (row["left_label_key"], row["right_label_key"])
        }
        source_rows = read_jsonl(args.input_dir / f"{split}.jsonl")
        split_reports[split] = {
            "input_rows": len(source_rows),
            "output_rows": len(targets),
            "no_rows_dropped": len(source_rows) == len(targets),
            "stable_id_order_preserved": [stable_id(row) for row in source_rows]
            == [row["stable_id"] for row in targets],
            "stable_ids_unique": len({stable_id(row) for row in source_rows}) == len(source_rows),
            "product_identity_preserved": all(
                target["product_mapped"] == source.get("product_mapped", "")
                and target["product_unmapped"] == source.get("product_unmapped", "")
                for source, target in zip(source_rows, targets)
            ),
            "precursor_identity_preserved": all(
                target["reference_precursors"] == source.get("precursor_unmapped", "")
                for source, target in zip(source_rows, targets)
            ),
            "status_counts": dict(sorted(reasons.items())),
            "native_supported_rows": len(supported),
            "native_supported_rate": len(supported) / max(len(targets), 1),
            "label_keys": len(heldout_keys),
            "label_keys_unseen_from_train": len(heldout_keys - train_keys),
            "source_sha256": sha256_file(args.input_dir / f"{split}.jsonl"),
            "targets_sha256": sha256_file(args.output_dir / f"{split}_targets.jsonl"),
        }
    report = {
        "artifact_type": "retro_mtgr_mechet_preprocessing_audit",
        "method": "Retro-MTGR",
        "input_dir": str(args.input_dir.resolve()),
        "target_contract": "single product-bond deletion plus two endpoint leaving-group labels",
        "unsupported_rows_retained": True,
        "label_vocabulary_source_splits": ["train"],
        "label_vocabulary_size": len(train_catalog),
        "splits": split_reports,
    }
    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def overfit(args: argparse.Namespace) -> int:
    source_path = args.input_dir / "train.jsonl"
    selected_rows: list[dict[str, Any]] = []
    selected_targets: list[dict[str, Any]] = []
    selected_specs: list[dict[str, Any]] = []
    for row in read_jsonl(source_path):
        analysis = analyze_row(row)
        if analysis.target["status"] != "supported":
            continue
        selected_rows.append(row)
        selected_targets.append(analysis.target)
        selected_specs.extend(analysis.label_specs)
        if len(selected_rows) == args.size:
            break
    if len(selected_rows) != args.size:
        raise RuntimeError(f"found only {len(selected_rows)} supported rows, need {args.size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        write_jsonl(args.output_dir / f"{split}.jsonl", selected_rows)
        write_jsonl(args.output_dir / f"{split}_targets.jsonl", selected_targets)
    catalog = label_catalog(selected_specs)
    (args.output_dir / "label_vocabulary.json").write_text(
        json.dumps({"source_splits": ["train"], "labels": catalog}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact_type": "retro_mtgr_overfit_diagnostic_dataset",
        "diagnostic_duplicate_splits": True,
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "selection": "first N natively supported train rows in frozen source order",
        "size": args.size,
        "stable_ids": [stable_id(row) for row in selected_rows],
        "label_vocabulary_source_splits": ["train"],
        "label_vocabulary_size": len(catalog),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "rows_per_split": args.size,
        "labels": len(catalog),
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--input-dir", type=Path, required=True)
    audit_parser.add_argument("--output-dir", type=Path, required=True)
    audit_parser.set_defaults(func=audit)
    overfit_parser = subparsers.add_parser("overfit")
    overfit_parser.add_argument("--input-dir", type=Path, required=True)
    overfit_parser.add_argument("--output-dir", type=Path, required=True)
    overfit_parser.add_argument("--size", type=int, choices=(32, 128), required=True)
    overfit_parser.set_defaults(func=overfit)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(parsed.func(parsed))
