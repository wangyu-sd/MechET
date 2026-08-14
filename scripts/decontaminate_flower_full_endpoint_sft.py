#!/usr/bin/env python3
"""Remove exact held-out reaction overlap from the full FlowER endpoint SFT.

The official test denominator is preserved.  Validation rows whose normalized
reaction key occurs in test are quarantined; train rows occurring in either
official validation or test are quarantined.  Normalization removes atom maps,
canonicalizes with RDKit, preserves stereochemistry, and is fragment-order
invariant through RDKit's disconnected-molecule canonical SMILES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.structural_overlap import canonical_unmapped_smiles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def reaction_key(row: dict[str, Any]) -> str:
    product = canonical_unmapped_smiles(str(row.get("target_smiles") or ""))
    precursor = canonical_unmapped_smiles(
        str(row.get("structural_precursor") or row.get("expected_precursor") or "")
    )
    return f"{product}>>{precursor}"


def load_keys(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    values: list[dict[str, Any]] = []
    keys: list[str] = []
    for row in rows(path):
        values.append(row)
        keys.append(reaction_key(row))
    return values, keys


def write_partition(
    source_rows: Iterator[dict[str, Any]] | list[dict[str, Any]],
    *,
    blocked_keys: set[str],
    kept_path: Path,
    quarantine_path: Path,
) -> dict[str, int]:
    kept_tmp = kept_path.with_suffix(kept_path.suffix + ".part")
    quarantine_tmp = quarantine_path.with_suffix(quarantine_path.suffix + ".part")
    kept = removed = 0
    with kept_tmp.open("w", encoding="utf-8") as keep, quarantine_tmp.open(
        "w", encoding="utf-8"
    ) as quarantine:
        for row in source_rows:
            target = quarantine if reaction_key(row) in blocked_keys else keep
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            if target is quarantine:
                removed += 1
            else:
                kept += 1
            total = kept + removed
            if total % 25_000 == 0:
                print(
                    f"[{kept_path.stem}] processed={total:,} kept={kept:,} removed={removed:,}",
                    flush=True,
                )
    kept_tmp.replace(kept_path)
    quarantine_tmp.replace(quarantine_path)
    return {"source_rows": kept + removed, "kept": kept, "removed": removed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=REPO / "data/flower_full_endpoint_sft"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/flower_full_endpoint_sft_decontaminated",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_source = args.input_dir / "train.jsonl"
    valid_source = args.input_dir / "valid.jsonl"
    test_source = args.input_dir / "test.jsonl"
    for path in (train_source, valid_source, test_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    test_rows, test_key_list = load_keys(test_source)
    test_keys = set(test_key_list)
    valid_rows, valid_key_list = load_keys(valid_source)
    valid_report = write_partition(
        valid_rows,
        blocked_keys=test_keys,
        kept_path=args.output_dir / "valid.jsonl",
        quarantine_path=args.output_dir / "valid.test_overlap.quarantine.jsonl",
    )
    train_report = write_partition(
        rows(train_source),
        blocked_keys=test_keys | set(valid_key_list),
        kept_path=args.output_dir / "train.jsonl",
        quarantine_path=args.output_dir / "train.heldout_overlap.quarantine.jsonl",
    )
    test_output = args.output_dir / "test.jsonl"
    test_tmp = test_output.with_suffix(test_output.suffix + ".part")
    shutil.copyfile(test_source, test_tmp)
    test_tmp.replace(test_output)

    artifacts = {}
    for name in (
        "train.jsonl",
        "train.heldout_overlap.quarantine.jsonl",
        "valid.jsonl",
        "valid.test_overlap.quarantine.jsonl",
        "test.jsonl",
    ):
        path = args.output_dir / name
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": 1,
        "artifact_type": "flower_full_endpoint_inverse_sft_decontaminated",
        "normalization": (
            "RDKit canonical isomeric SMILES; atom maps removed; disconnected "
            "fragment order invariant"
        ),
        "policy": {
            "test": "preserve complete official test denominator",
            "valid": "remove exact reaction keys occurring in test",
            "train": "remove exact reaction keys occurring in official valid or test",
            "within_split_deduplication": False,
        },
        "source_manifest": str(args.input_dir / "manifest.json"),
        "source_manifest_sha256": sha256_file(args.input_dir / "manifest.json"),
        "train": train_report,
        "valid": valid_report,
        "test": {
            "rows": len(test_rows),
            "unique_reaction_keys": len(test_keys),
            "duplicate_reaction_rows": len(test_key_list) - len(test_keys),
        },
        "artifacts": artifacts,
    }
    output = args.output_dir / "manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
