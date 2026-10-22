#!/usr/bin/env python3
"""Explicitly promote audited main-product-only SFT subsets for frozen training.

The source audit always retains every original reaction in its denominator.
Only replay-verified rows are supervised; quarantined source IDs/codes remain
in an immutable ledger, with per-split coverage reported. NEVER silently
treat the supervised subset as the complete official benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

EXPECTED = {
    "mech31k": {"train": 10152, "valid": 1319, "test": 1253},
    "flower": {"train": 257167, "valid": 2890, "test": 28967},
}
# Explicitly audited exclusion caps; exceeding any cap FAILS CLOSED.
# 31k reflects the already inspected 15/2/1 stereo/mapping corner cases.
# Flower must first be audited; tolerance does not license unreported drops.
MAX_UNRESOLVED = {
    "mech31k": {"train": 15, "valid": 2, "test": 1},
    "flower": {"train": 257, "valid": 2, "test": 28},
}


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as handle:
        for p in iter(lambda: handle.read(8*1024*1024), b""):
            d.update(p)
    return d.hexdigest()


def audit_metadata(audit: Path, corpus: str) -> dict[str, Any]:
    document = json.loads((audit / "manifest.json").read_text())
    if corpus == "flower":
        if document.get("input_condition") != "main_product_only_v1":
            raise ValueError("FlowER main-product contract absent")
        reports = document["reports"]
        src_dir = audit / "source"
        sft_dir = audit / "state_sft"
        rows = "read"
        passed = "reference_pass"
    else:
        if document.get("input_condition") != "main_product_only_v1":
            raise ValueError("31k main-product contract absent")
        reports = document["reports"]
        src_dir = audit / "inverse"
        sft_dir = audit / "state_sft"
        rows = "reactions"
        passed = "reference_pass"
    return {
        "manifest": document, "reports": reports,
        "source_dir": src_dir, "state_sft_dir": sft_dir,
        "original_rows_key": rows, "verified_rows_key": passed,
    }


def promote(args: argparse.Namespace) -> dict[str, Any]:
    if args.destination.exists():
        raise ValueError("Refusing to overwrite promoted training data")
    paths = audit_metadata(args.audit, args.corpus)
    ledger: dict[str, Any] = {
        "schema": "main_product_only_explicit_quarantine_v1",
        "corpus": args.corpus,
        "input_condition": "main_product_only_v1",
        "source_audit_manifest_sha256": sha(args.audit / "manifest.json"),
        "original_denominators": EXPECTED[args.corpus],
        "audited_quarantine": {},
        "training_allowed": True,
        "no_unlabeled_removed_reactions": True,
        "GT_model_visible": False,
    }
    for split in ("train", "valid", "test"):
        report = paths["reports"][split]
        original = int(report[paths["original_rows_key"]])
        verified = int(report[paths["verified_rows_key"]])
        unresolved = int(report.get("unresolved", 0))
        if original != EXPECTED[args.corpus][split]:
            raise ValueError(f"{split}: original denominator changed ({original})")
        if verified + unresolved != original:
            raise ValueError(f"{split}: coverage not exhaustively accounted for")
        if unresolved > MAX_UNRESOLVED[args.corpus][split]:
            raise ValueError(f"{split}: unresolved references exceed audited cap")
        bad_path = (args.audit / "audits" / f"{split}.unresolved.jsonl")
        if not bad_path.exists():
            raise ValueError(f"{split}: missing unresolved ledger")
        bad_rows = [json.loads(line) for line in bad_path.open() if line.strip()]
        if len(bad_rows) != unresolved:
            raise ValueError(f"{split}: unresolved ledger count mismatch")
        ids = [str(row.get("source_id") or row.get("id") or "") for row in bad_rows]
        if any(not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{split}: unresolved reaction IDs invalid")
        if unresolved and not args.allow_quarantine:
            raise ValueError(
                f"{split}: {unresolved} reference cases unresolved; "
                "explicit --allow-quarantine required"
            )
        supervised = paths["state_sft_dir"] / f"{split}.jsonl"
        source = paths["source_dir"] / f"{split}.jsonl"
        for p in (supervised, source):
            if not p.is_file() or p.stat().st_size < 2:
                raise ValueError(f"{split}: missing supervised/source rows: {p}")
        ledger["audited_quarantine"][split] = {
            "original_reactions": original,
            "reference_replay_pass": verified,
            "unresolved_reactions": unresolved,
            "unresolved_ids": ids,
            "unresolved_sha256": sha(bad_path),
            "source_sha256": sha(source),
            "state_sft_sha256": sha(supervised),
            "supervision_label": "audited_replayable_subset_not_full_corpus",
            "raw_denominator_retained": True,
        }
    (args.destination / "source").mkdir(parents=True)
    (args.destination / "state_sft").mkdir(parents=True)
    # Symlink immutable, SHA-checked source references without copying huge
    # FlowER JSONL. The user owns the original source; checks reject drift.
    for split in ("train", "valid", "test"):
        for folder, origin in (
            ("source", paths["source_dir"]),
            ("state_sft", paths["state_sft_dir"]),
        ):
            link = args.destination / folder / f"{split}.jsonl"
            os.symlink((origin / f"{split}.jsonl").resolve(), link)
    for folder in ("source", "state_sft"):
        src_artifact_status = {
            "status": "validated_explicitly_quarantined_subset",
            "training_allowed": folder == "state_sft",
            "input_condition": "main_product_only_v1",
            "original_corpus_denominator_preserved_in_manifest": True,
            "approved_exclusions": ledger["audited_quarantine"],
        }
        (args.destination / folder / "ARTIFACT_STATUS.json").write_text(
            json.dumps(src_artifact_status, indent=2) + "\n"
        )
    (args.destination / "manifest.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n"
    )
    (args.destination / "state_sft" / "manifest.json").write_text(
        json.dumps({
            "artifact_type": "main_product_only_tool_sft_audited_subset",
            "training_allowed": True,
            "input_condition": "main_product_only_v1",
            "source_ledger": str(args.destination / "manifest.json"),
            "reaction_denominator": {k:v["reference_replay_pass"] for k,v in ledger["audited_quarantine"].items()},
            "original_reaction_denominator": EXPECTED[args.corpus],
        }, indent=2) + "\n"
    )
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--allow-quarantine", action="store_true")
    args = parser.parse_args()
    result = promote(args)
    print(json.dumps({
        "status": "AUDITED_MAIN_PRODUCT_TRAIN_SUBSET_APPROVED",
        "corpus": args.corpus, "coverage": result["audited_quarantine"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
