#!/usr/bin/env python3
"""Freeze the decontaminated FlowER-only subset of the mixed inverse dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SOURCE = Path("data/mixed_inverse_tool_sft")
OUTPUT = Path("data/flower_inverse_tool_sft")
INPUTS = {
    "train": SOURCE / "train.a100_20480.jsonl",
    "valid": SOURCE / "valid.jsonl",
    "test": SOURCE / "test.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ids_sha256(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def filter_split(source: Path, output: Path) -> tuple[dict[str, Any], set[str]]:
    ids: set[str] = set()
    rows = 0
    with source.open(encoding="utf-8") as reader, output.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = dict(row.get("metadata") or {})
            if metadata.get("mixture_source") != "flower":
                continue
            identifier = str(row.get("id") or "")
            if not identifier or identifier in ids:
                raise ValueError(f"invalid or duplicate FlowER ID: {identifier!r}")
            if metadata.get("corpus_used") is not False:
                raise ValueError(f"FlowER row unexpectedly uses corpus: {identifier}")
            ids.add(identifier)
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
    return {
        "file": str(output),
        "sha256": sha256(output),
        "rows": rows,
        "stable_ids_sha256": stable_ids_sha256(ids),
        "source_file": str(source),
        "source_sha256": sha256(source),
    }, ids


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    splits: dict[str, dict[str, Any]] = {}
    ids: dict[str, set[str]] = {}
    for split, source in INPUTS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        splits[split], ids[split] = filter_split(
            source, OUTPUT / f"{split}.jsonl"
        )
    overlap = {
        "train_valid": len(ids["train"] & ids["valid"]),
        "train_test": len(ids["train"] & ids["test"]),
        "valid_test": len(ids["valid"] & ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"FlowER split ID overlap: {overlap}")
    upstream = SOURCE / "training_manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "frozen_flower_inverse_tool_sft_training_set",
        "condition": "trace_no_knowledge",
        "source_dataset": "flower_mech_proof",
        "corpus_used": False,
        "selection": "mixture_source_equals_flower_v1",
        "splits": splits,
        "split_id_overlap": overlap,
        "upstream_training_manifest": {
            "file": str(upstream),
            "sha256": sha256(upstream),
        },
        "inherited_guarantees": [
            "exact_structural_train_heldout_overlap_zero",
            "a100_20480_zero_truncation_train",
            "trace_owned_environment_endpoint",
        ],
    }
    output = OUTPUT / "training_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
