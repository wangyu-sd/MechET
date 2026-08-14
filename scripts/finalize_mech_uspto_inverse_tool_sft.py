#!/usr/bin/env python3
"""Finalize and fingerprint the mech-USPTO inverse Tool-SFT artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ids_sha256(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mech_uspto_31k_inverse_tool_sft"),
    )
    parser.add_argument(
        "--training-config",
        default="configs/agent/tool_sft_mech_uspto_31k_inverse.yaml",
    )
    args = parser.parse_args()

    split_names = ("train", "valid", "test")
    split_ids: dict[str, set[str]] = {}
    split_manifest: dict[str, dict[str, object]] = {}
    validation_splits: dict[str, dict[str, object]] = {}
    tokenizer_audit_splits: dict[str, dict[str, object]] = {}
    total_forward = total_written = total_quarantined = 0

    for split in split_names:
        data = args.data_dir / f"{split}.jsonl"
        quarantine = args.data_dir / f"{split}.quarantine.jsonl"
        report_path = args.data_dir / f"{split}.report.json"
        validation_path = args.data_dir / f"{split}.validation.json"
        tokenizer_audit_path = args.data_dir / f"{split}.tokenizer_audit.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        tokenizer_audit = json.loads(
            tokenizer_audit_path.read_text(encoding="utf-8")
        )

        ids: set[str] = set()
        trace_steps = trace_moves = root_imports = rows = 0
        with data.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                stable_id = str(row["id"])
                if stable_id in ids:
                    raise ValueError(f"DUPLICATE_ID:{split}:{stable_id}")
                ids.add(stable_id)
                rows += 1
                metadata = row["metadata"]
                trace_steps += int(metadata["n_trace_steps"])
                trace_moves += int(metadata["n_trace_moves"])
                root_imports += int(metadata["n_trace_imports"])

        if rows != int(report["written"]):
            raise ValueError(f"REPORT_ROW_MISMATCH:{split}")
        if validation.get("input_sha256") != sha256(data):
            raise ValueError(f"VALIDATION_INPUT_HASH_MISMATCH:{split}")
        if not validation.get("passed") or int(validation.get("failures", -1)) != 0:
            raise ValueError(f"VALIDATION_FAILED:{split}")
        if int(validation.get("rows", -1)) != rows:
            raise ValueError(f"VALIDATION_ROW_MISMATCH:{split}")
        if tokenizer_audit.get("input_sha256") != sha256(data):
            raise ValueError(f"TOKENIZER_AUDIT_INPUT_HASH_MISMATCH:{split}")
        if not tokenizer_audit.get("passed"):
            raise ValueError(f"TOKENIZER_AUDIT_FAILED:{split}")
        if int(tokenizer_audit.get("rows", -1)) != rows:
            raise ValueError(f"TOKENIZER_AUDIT_ROW_MISMATCH:{split}")

        split_ids[split] = ids
        split_manifest[split] = {
            "rows": rows,
            "unique_ids": len(ids),
            "stable_ids_sha256": stable_ids_sha256(ids),
            "trace_steps": trace_steps,
            "trace_moves": trace_moves,
            "root_imports": root_imports,
            "file": str(data),
            "sha256": sha256(data),
            "quarantine": str(quarantine),
            "quarantine_sha256": sha256(quarantine),
            "report": str(report_path),
            "report_sha256": sha256(report_path),
            "validation": str(validation_path),
            "validation_sha256": sha256(validation_path),
            "tokenizer_audit": str(tokenizer_audit_path),
            "tokenizer_audit_sha256": sha256(tokenizer_audit_path),
        }
        validation_splits[split] = validation
        tokenizer_audit_splits[split] = tokenizer_audit
        total_forward += int(report["read"])
        total_written += int(report["written"])
        total_quarantined += int(report["quarantined"])

    overlap = {
        "train_valid": len(split_ids["train"] & split_ids["valid"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "valid_test": len(split_ids["valid"] & split_ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"SPLIT_ID_OVERLAP:{overlap}")

    validation_summary = {
        "artifact_type": "inverse_tool_sft_replay_validation_summary",
        "splits": validation_splits,
        "split_id_overlap": overlap,
        "rows": sum(int(value["rows"]) for value in validation_splits.values()),
        "replayed": sum(
            int(value["replayed"]) for value in validation_splits.values()
        ),
        "tool_calls": sum(
            int(value["tool_calls"]) for value in validation_splits.values()
        ),
        "failures": sum(
            int(value["failures"]) for value in validation_splits.values()
        ),
        "passed": all(bool(value["passed"]) for value in validation_splits.values()),
    }
    validation_output = args.data_dir / "validation.json"
    validation_output.write_text(
        json.dumps(validation_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    max_lengths = {
        int(value["configured_max_length"])
        for value in tokenizer_audit_splits.values()
    }
    if len(max_lengths) != 1:
        raise ValueError(f"TOKENIZER_MAX_LENGTH_MISMATCH:{sorted(max_lengths)}")
    tokenizer_audit_summary = {
        "artifact_type": "tool_sft_tokenizer_length_audit_summary",
        "splits": tokenizer_audit_splits,
        "rows": sum(int(value["rows"]) for value in tokenizer_audit_splits.values()),
        "total_input_tokens": sum(
            int(value["total_input_tokens"])
            for value in tokenizer_audit_splits.values()
        ),
        "total_supervised_tokens": sum(
            int(value["total_supervised_tokens"])
            for value in tokenizer_audit_splits.values()
        ),
        "max_input_tokens": max(
            int(value["max_input_tokens"])
            for value in tokenizer_audit_splits.values()
        ),
        "configured_max_length": next(iter(max_lengths)),
        "truncation_count": sum(
            int(value["truncation_count"])
            for value in tokenizer_audit_splits.values()
        ),
        "zero_supervision_count": sum(
            int(value["zero_supervision_count"])
            for value in tokenizer_audit_splits.values()
        ),
        "passed": all(
            bool(value["passed"]) for value in tokenizer_audit_splits.values()
        ),
    }
    tokenizer_audit_output = args.data_dir / "tokenizer_audit.json"
    tokenizer_audit_output.write_text(
        json.dumps(tokenizer_audit_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "artifact_type": "trace_owned_inverse_tool_sft",
        "scientific_contract": "mech_uspto_inverse_trace_owned_v2",
        "source": "mech_uspto_31k",
        "condition": "trace_no_knowledge",
        "target_source": "rxn_prod_min matched into globally mapped final state",
        "endpoint_source": "environment_owned_finish_trace",
        "corpus_used": False,
        "stereo_policy": (
            "clear only reacting tetrahedral tags absent from the final state"
        ),
        "splits": split_manifest,
        "split_id_overlap": overlap,
        "conversion": {
            "forward_globally_mapped_input": total_forward,
            "inverse_written": total_written,
            "inverse_quarantined": total_quarantined,
        },
        "validation": str(validation_output),
        "validation_sha256": sha256(validation_output),
        "validation_passed": validation_summary["passed"],
        "tokenizer_audit": str(tokenizer_audit_output),
        "tokenizer_audit_sha256": sha256(tokenizer_audit_output),
        "tokenizer_audit_passed": tokenizer_audit_summary["passed"],
        "training_config": args.training_config,
    }
    manifest_output = args.data_dir / "manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
