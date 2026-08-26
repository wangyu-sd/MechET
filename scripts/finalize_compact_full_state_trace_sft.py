#!/usr/bin/env python3
"""Freeze and verify the strict FlowER compact-full-state A7 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STRICT_EXPECTED = {"train": 257_167, "valid": 2_890, "test": 28_967}
OFFICIAL_EXPECTED = {"train": 257_171, "valid": 2_890, "test": 28_971}
UPSTREAM_CORRUPT = {"train": 4, "valid": 0, "test": 4}
FORBIDDEN_VISIBLE_KEYS = {
    "state_smiles",
    "state_before",
    "state_after",
    "local_state_before",
    "local_state_after",
    "trace_step",
    "trace_digest",
    "move_sequence_digest",
    "compiled_proof",
    "full_precursor_state",
    "structural_precursor",
    "pending_imports",
    "imported_fragment",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> tuple[dict[str, Any], set[str]]:
    ids: set[str] = set()
    rows = max_calls = tool_results = 0
    for line_number, line in enumerate(path.open(encoding="utf-8"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        identifier = str(row.get("id") or "")
        if not identifier or identifier in ids:
            raise ValueError(f"invalid/duplicate ID at {path}:{line_number}")
        metadata = dict(row.get("metadata") or {})
        if metadata.get("observation_mode") != "compact_full_state_v1":
            raise ValueError(f"observation contract mismatch: {identifier}")
        if metadata.get("executor_replayed") is not True:
            raise ValueError(f"row was not executor replayed: {identifier}")
        if metadata.get("endpoint_source") != "environment_owned_trace":
            raise ValueError(f"endpoint is not trace-owned: {identifier}")
        if metadata.get("corpus_used") is not False:
            raise ValueError(f"compact A7 used a corpus: {identifier}")

        calls = finish = 0
        for message in row.get("messages") or []:
            for call in message.get("tool_calls") or []:
                calls += 1
                finish += int(
                    str((call.get("function") or {}).get("name") or "")
                    == "finish_trace"
                )
            if message.get("role") != "tool":
                continue
            tool_results += 1
            result = json.loads(str(message.get("content") or "{}"))
            leaked = FORBIDDEN_VISIBLE_KEYS.intersection(result)
            if leaked:
                raise ValueError(
                    f"audit/full-state field leaked: {identifier}:{sorted(leaked)}"
                )
            name = str(message.get("name") or "")
            if result.get("observation_mode") != "compact_full_state_v1":
                raise ValueError(f"tool observation mode mismatch: {identifier}:{name}")
            if name != "finish_trace" and "current_state_smiles" not in result:
                raise ValueError(f"authoritative current state missing: {identifier}:{name}")
            if name == "finish_trace" and "derived_precursor" not in result:
                raise ValueError(f"terminal endpoint missing: {identifier}")
        if finish != 1:
            raise ValueError(f"finish_trace count is {finish}: {identifier}")
        ids.add(identifier)
        rows += 1
        max_calls = max(max_calls, calls)
    return {
        "file": str(path),
        "rows": rows,
        "unique_ids": len(ids),
        "sha256": sha256(path),
        "max_tool_calls": max_calls,
        "tool_results": tool_results,
        "visible_audit_field_leaks": 0,
    }, ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("data/flower_inverse_tool_sft_compact_full_state_v1"),
    )
    parser.add_argument(
        "--proof-dir",
        type=Path,
        default=Path("data/mechet_proof_sft_flower_full_v4"),
    )
    args = parser.parse_args()

    proof_manifest_path = args.proof_dir / "manifest.json"
    proof_manifest = json.loads(proof_manifest_path.read_text(encoding="utf-8"))
    if int(proof_manifest.get("skipped_total") or 0) != sum(UPSTREAM_CORRUPT.values()):
        raise ValueError("strict proof manifest no longer matches the named corrupt set")

    splits: dict[str, dict[str, Any]] = {}
    ids: dict[str, set[str]] = {}
    for split, expected in STRICT_EXPECTED.items():
        report_path = args.trace_dir / f"{split}.report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("observation_mode") != "compact_full_state_v1":
            raise ValueError(f"report observation mode mismatch: {split}")
        if int(report.get("written") or 0) != expected or int(
            report.get("quarantined") or 0
        ):
            raise ValueError(f"strict compact conversion mismatch: {split}: {report}")
        splits[split], ids[split] = audit(args.trace_dir / f"{split}.jsonl")
        if int(splits[split]["rows"]) != expected:
            raise ValueError(f"strict denominator mismatch: {split}")

    overlap = {
        "train_valid": len(ids["train"] & ids["valid"]),
        "train_test": len(ids["train"] & ids["test"]),
        "valid_test": len(ids["valid"] & ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"split ID overlap: {overlap}")

    manifest = {
        "schema_version": 1,
        "artifact_type": "flower_strict_compact_full_state_trace_owned_tool_sft",
        "source_dataset": "FlowER flower_new_dataset",
        "paper_condition": "A7",
        "condition": "trace_no_knowledge_compact_full_state",
        "observation_mode": "compact_full_state_v1",
        "intermediate_state_model_visible": True,
        "strict_trace_universe_complete": True,
        "reaction_coverage_complete": False,
        "official_reaction_denominators": OFFICIAL_EXPECTED,
        "named_upstream_corrupt_rows_excluded": UPSTREAM_CORRUPT,
        "reaction_filtering": True,
        "reaction_filter_type": "strict_proof_eligibility",
        "mechanism_eligibility_restriction": (
            "named non_atom_conserving upstream endpoints only"
        ),
        "overlap_filtering": False,
        "token_length_filtering": False,
        "corpus_used": False,
        "endpoint_source": "environment_owned_finish_trace",
        "splits": splits,
        "split_id_overlap": overlap,
        "proof_manifest": {
            "file": str(proof_manifest_path),
            "sha256": sha256(proof_manifest_path),
        },
    }
    output = args.trace_dir / "training_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    status = {
        "artifact_id": "flower_compact_full_state_v1",
        "status": "validated",
        "training_allowed": True,
        "view": "strict_executable_compact_full_state",
        "rows": STRICT_EXPECTED,
        "full_official_denominator": OFFICIAL_EXPECTED,
        "excluded_named_upstream_corrupt_rows": UPSTREAM_CORRUPT,
    }
    (args.trace_dir / "ARTIFACT_STATUS.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
