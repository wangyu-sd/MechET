#!/usr/bin/env python3
"""Verify that the equ-field endpoint remap changes only audited targets."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_system_one_full_endpoint_input_gap import sha256

EXPECTED = {"train": 24959, "valid": 3120, "test": 3120}


def _read_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reaction_id = str(row["source_id"])
        if reaction_id in rows:
            raise ValueError(f"duplicate reaction {reaction_id}: {path}")
        rows[reaction_id] = row
    return rows


def compare_rows(old: dict[str, dict], new: dict[str, dict],
                 changed_ids: set[str]) -> dict[str, int]:
    if set(old) != set(new):
        raise ValueError("reaction ID coverage differs")
    observed_changed = set()
    counts: Counter[str] = Counter()
    for reaction_id in old:
        before, after = old[reaction_id], new[reaction_id]
        if (before["id"] != after["id"]
                or before["reactants_unmapped"] != after["reactants_unmapped"]):
            raise ValueError(f"reaction identity/full reactants changed: {reaction_id}")
        changed = before["product_unmapped"] != after["product_unmapped"]
        structural_changed = before["precursor_unmapped"] != after["precursor_unmapped"]
        if structural_changed and not changed:
            raise ValueError(f"structural precursor changed without product change: {reaction_id}")
        counts["reactions"] += 1
        counts["product_changed"] += int(changed)
        counts["product_unchanged"] += int(not changed)
        counts["structural_precursor_changed"] += int(structural_changed)
        if changed:
            observed_changed.add(reaction_id)
    if observed_changed != changed_ids:
        raise ValueError("changed product IDs differ from raw-field audit")
    return dict(counts)


def verify(old_dir: Path, new_dir: Path, audit_path: Path) -> dict:
    old_manifest = json.loads((old_dir / "manifest.json").read_text())
    new_manifest = json.loads((new_dir / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    if (old_manifest["artifact_type"] != "mech_uspto_31k_full_hf_endpoint_rxnmapper"
            or new_manifest["artifact_type"]
            != "mech_uspto_31k_full_hf_endpoint_rxnmapper_equ_proxy"
            or new_manifest["product_source_field"] != "rxn_prod_equ"
            or new_manifest["product_selection_is_proxy"] is not True
            or old_manifest["executor_filtering"] is not False
            or new_manifest["executor_filtering"] is not False
            or old_manifest["source_files"] != new_manifest["source_files"]
            or audit["endpoint_manifest_sha256"] != sha256(old_dir / "manifest.json")):
        raise ValueError("source/proxy manifest lineage mismatch")
    splits = {}
    for split, expected in EXPECTED.items():
        old_path, new_path = old_dir / f"{split}.jsonl", new_dir / f"{split}.jsonl"
        if (old_manifest["splits"][split]["rows"] != expected
                or new_manifest["splits"][split]["rows"] != expected
                or old_manifest["splits"][split]["endpoint_sha256"] != sha256(old_path)
                or new_manifest["splits"][split]["endpoint_sha256"] != sha256(new_path)
                or audit["splits"][split]["endpoint_sha256"] != sha256(old_path)):
            raise ValueError(f"{split}: endpoint hash/denominator mismatch")
        changed_ids = set(audit["splits"][split]["changed_reaction_ids"])
        if len(changed_ids) != audit["splits"][split]["counts"]["main_product_changed"]:
            raise ValueError(f"{split}: raw-field audit count mismatch")
        counts = compare_rows(_read_rows(old_path), _read_rows(new_path), changed_ids)
        if counts["reactions"] != expected:
            raise ValueError(f"{split}: reaction denominator mismatch")
        splits[split] = {
            "counts": counts,
            "old_endpoint_sha256": sha256(old_path),
            "new_endpoint_sha256": sha256(new_path),
        }
    return {
        "artifact_type": "mech_uspto31k_equ_proxy_endpoint_handoff_verification",
        "old_manifest_sha256": sha256(old_dir / "manifest.json"),
        "new_manifest_sha256": sha256(new_dir / "manifest.json"),
        "raw_field_audit_sha256": sha256(audit_path),
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--raw-field-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(args.old_dir, args.new_dir, args.raw_field_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
