#!/usr/bin/env python3
"""Freeze and verify the full-reaction FlowER trace-owned Tool-SFT artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {"train": 257_171, "valid": 2_890, "test": 28_971}
EXPECTED_FALLBACK = {"train": 4, "valid": 0, "test": 4}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> tuple[dict[str, Any], set[str]]:
    ids: set[str] = set()
    rows = fallback = max_calls = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row.get("id") or "")
            if not identifier or identifier in ids:
                raise ValueError(f"invalid/duplicate ID at {path}:{line_number}")
            metadata = dict(row.get("metadata") or {})
            is_fallback = metadata.get("upstream_endpoint_fallback") is True
            if is_fallback:
                if metadata.get("executor_replayed") is not False:
                    raise ValueError(f"fallback claims executor replay: {identifier}")
                if metadata.get("endpoint_source") != "upstream_frozen_endpoint_fallback":
                    raise ValueError(f"fallback provenance invalid: {identifier}")
            else:
                if metadata.get("executor_replayed") is not True:
                    raise ValueError(f"row was not executor replayed: {identifier}")
                if metadata.get("endpoint_source") != "environment_owned_trace":
                    raise ValueError(f"row endpoint is not trace-owned: {identifier}")
            if metadata.get("corpus_used") is not False:
                raise ValueError(f"no-knowledge row used a corpus: {identifier}")
            calls = sum(
                len(message.get("tool_calls") or [])
                for message in row.get("messages") or []
            )
            finish = sum(
                str((call.get("function") or {}).get("name") or "")
                == "finish_trace"
                for message in row.get("messages") or []
                for call in message.get("tool_calls") or []
            )
            expected_finish = 0 if is_fallback else 1
            if finish != expected_finish:
                raise ValueError(
                    f"row finish_trace count is {finish}, expected "
                    f"{expected_finish}: {identifier}"
                )
            ids.add(identifier)
            rows += 1
            max_calls = max(max_calls, calls)
            fallback += int(is_fallback)
    return {
        "file": str(path),
        "rows": rows,
        "sha256": sha256(path),
        "unique_ids": len(ids),
        "max_tool_calls": max_calls,
        "upstream_endpoint_fallback_rows": fallback,
    }, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir", type=Path, default=Path("data/flower_inverse_tool_sft_full_v4")
    )
    parser.add_argument(
        "--proof-dir", type=Path, default=Path("data/mechet_proof_sft_flower_full_v4")
    )
    args = parser.parse_args()

    proof_manifest_path = args.proof_dir / "manifest.json"
    proof_manifest = json.loads(proof_manifest_path.read_text(encoding="utf-8"))
    if int(proof_manifest.get("skipped_total") or 0) != sum(EXPECTED_FALLBACK.values()):
        raise ValueError("MECH_PROOF skipped count does not match named corrupt endpoints")

    splits: dict[str, dict[str, Any]] = {}
    ids: dict[str, set[str]] = {}
    for split, expected in EXPECTED.items():
        report_path = args.trace_dir / f"{split}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("quarantined") or 0) != 0:
            raise ValueError(f"trace conversion rejected rows in {split}: {report}")
        splits[split], ids[split] = audit(args.trace_dir / f"{split}.jsonl")
        if splits[split]["rows"] != expected:
            raise ValueError(
                f"full FlowER denominator mismatch in {split}: "
                f"{splits[split]['rows']} != {expected}"
            )
        if splits[split]["upstream_endpoint_fallback_rows"] != EXPECTED_FALLBACK[split]:
            raise ValueError(f"fallback count mismatch in {split}")

    overlap = {
        "train_valid": len(ids["train"] & ids["valid"]),
        "train_test": len(ids["train"] & ids["test"]),
        "valid_test": len(ids["valid"] & ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"split ID overlap: {overlap}")

    manifest = {
        "schema_version": 1,
        "artifact_type": "flower_full_reaction_level_trace_owned_tool_sft",
        "source_dataset": "FlowER flower_new_dataset",
        "condition": "trace_no_knowledge",
        "reaction_coverage_complete": True,
        "proof_replay_complete": False,
        "strict_trace_rows": sum(EXPECTED.values()) - sum(EXPECTED_FALLBACK.values()),
        "upstream_corrupt_endpoint_fallback_rows": sum(EXPECTED_FALLBACK.values()),
        "corpus_used": False,
        "reaction_filtering": False,
        "overlap_filtering": False,
        "token_length_filtering": False,
        "pairing_policy": "canonical_local_charge_exact_with_radical_pair",
        "named_test_endpoint_fallback_policy": "explicitly labelled RC/PC/PM/RS only",
        "splits": splits,
        "split_id_overlap": overlap,
        "proof_manifest": {
            "file": str(proof_manifest_path),
            "sha256": sha256(proof_manifest_path),
        },
    }
    output = args.trace_dir / "training_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
