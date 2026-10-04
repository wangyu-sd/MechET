#!/usr/bin/env python3
"""Verify the H2 direct/open/closed datasets have identical product endpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.materialize_nmi_matched_datasets import CONDITIONS, FIXED_TOOL_BUDGET, INITIAL_OBSERVATION_MARKER


def _read_condition(path: Path, manifest: dict[str, Any], split: str,
                    *, check_preamble: bool) -> dict[str, tuple[str, str]]:
    digest = hashlib.sha256()
    signatures: dict[str, tuple[str, str]] = {}
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            identifier = str(row.get("source_id") or "")
            target = str(row.get("target_smiles") or "")
            structural = str(row.get("structural_precursor") or "")
            if not identifier or not target or not structural or identifier in signatures:
                raise ValueError(f"missing/duplicate ID or endpoint in {path}: {identifier}")
            if str((row.get("metadata") or {}).get("nmi_split") or "") != split:
                raise ValueError(f"row assigned to wrong H2 split: {identifier}")
            if check_preamble:
                users = [item for item in row.get("messages") or [] if item.get("role") == "user"]
                if len(users) != 1:
                    raise ValueError(f"closed-loop initial prompt count mismatch: {identifier}")
                prompt = str(users[0].get("content") or "")
                if not prompt.startswith(f"TARGET: {target}\n") or INITIAL_OBSERVATION_MARKER not in prompt:
                    raise ValueError(f"closed-loop product prompt mismatch: {identifier}")
                observation = json.loads(prompt.split(INITIAL_OBSERVATION_MARKER, 1)[1])
                if int(observation.get("max_tool_calls") or 0) != FIXED_TOOL_BUDGET:
                    raise ValueError(f"closed-loop gold-dependent budget survived: {identifier}")
            signatures[identifier] = target, structural
    if len(signatures) != int(manifest["rows"][split]):
        raise ValueError(f"row count mismatch: {path}")
    if digest.hexdigest() != manifest["output_sha256"][split]:
        raise ValueError(f"output SHA mismatch: {path}")
    return signatures


def verify(matched_dir: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing verification: {output}")
    manifests = {
        condition: json.loads((matched_dir / condition / "manifest.json").read_text())
        for condition in CONDITIONS
    }
    parent_hashes = {manifest["parent_split_manifest_sha256"] for manifest in manifests.values()}
    stable_hashes = {json.dumps(manifest["stable_reaction_ids"], sort_keys=True) for manifest in manifests.values()}
    if len(parent_hashes) != 1 or len(stable_hashes) != 1:
        raise ValueError("condition split/ID hashes differ")
    reports = {}
    for split in ("train", "valid", "test"):
        baseline = _read_condition(matched_dir / "direct" / f"{split}.jsonl",
                                   manifests["direct"], split, check_preamble=False)
        for condition in ("open_flow", "closed_loop"):
            other = _read_condition(matched_dir / condition / f"{split}.jsonl",
                                    manifests[condition], split,
                                    check_preamble=condition == "closed_loop")
            if baseline.keys() != other.keys():
                raise ValueError(f"{condition}/{split} stable reaction IDs differ")
            mismatches = Counter()
            for identifier, (target, structural) in baseline.items():
                alternate_target, alternate_structural = other[identifier]
                mismatches["target"] += target != alternate_target
                mismatches["structural_precursor"] += structural != alternate_structural
            if any(mismatches.values()):
                raise ValueError(f"{condition}/{split} product or endpoint mismatch: {dict(mismatches)}")
        reports[split] = {"rows": len(baseline), "mapped_product_exact_parity": True,
                          "structural_endpoint_exact_parity": True}
    report = {
        "artifact_type": "nmi_h2_matched_representation_verification_v1",
        "parent_split_manifest_sha256": next(iter(parent_hashes)),
        "stable_reaction_ids": manifests["direct"]["stable_reaction_ids"],
        "splits": reports,
        "fixed_closed_loop_initial_budget": FIXED_TOOL_BUDGET,
        "passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.matched_dir, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
