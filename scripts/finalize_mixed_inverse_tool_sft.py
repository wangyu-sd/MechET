#!/usr/bin/env python3
"""Freeze the audited mixed inverse Tool-SFT artifacts into one manifest."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path("data/mixed_inverse_tool_sft")
OVERLAP = Path("outputs/mixed_inverse_overlap/decontaminated_heldout/overlap_summary.json")
VALID_TEST_OVERLAP = Path("outputs/mixed_inverse_overlap/valid_vs_test/overlap_summary.json")
CONFIG = Path("configs/agent/tool_sft_mixed_inverse_qwen3_8b.yaml")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_split(path: Path) -> tuple[dict[str, object], set[str]]:
    ids: set[str] = set()
    sources: Counter[str] = Counter()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = str(row["id"])
            if identifier in ids:
                raise ValueError(f"duplicate ID in {path}: {identifier}")
            ids.add(identifier)
            sources[str(row["metadata"]["mixture_source"])] += 1
            if row["metadata"].get("corpus_used") is not False:
                raise ValueError(f"corpus unexpectedly enabled: {identifier}")
            rows += 1
    return {
        "file": str(path),
        "sha256": sha256(path),
        "rows": rows,
        "source_rows": dict(sorted(sources.items())),
    }, ids


def main() -> int:
    overlap = json.loads(OVERLAP.read_text(encoding="utf-8"))
    valid_test_overlap = json.loads(
        VALID_TEST_OVERLAP.read_text(encoding="utf-8")
    )
    forbidden = ("exact_full", "exact_structural", "product")
    if any(int(overlap["overlap_counts"][key]) for key in forbidden):
        raise ValueError("post-decontamination train/heldout overlap is nonzero")

    splits: dict[str, object] = {}
    ids: dict[str, set[str]] = {}
    paths = {
        "train": ROOT / "train.a100_20480.jsonl",
        "valid": ROOT / "valid.jsonl",
        "test": ROOT / "test.jsonl",
    }
    audits: dict[str, object] = {}
    for split, path in paths.items():
        splits[split], ids[split] = inspect_split(path)
        audit_path = ROOT / f"{split}.tokenizer_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            not audit.get("passed")
            or audit.get("input_sha256") != sha256(path)
            or int(audit.get("configured_max_length", 0)) != 20480
        ):
            raise ValueError(f"failed or stale tokenizer audit: {audit_path}")
        audits[split] = {
            "file": str(audit_path),
            "sha256": sha256(audit_path),
            "max_input_tokens": audit["max_input_tokens"],
            "configured_max_length": audit["configured_max_length"],
            "passed": True,
        }

    split_overlap = {
        "train_valid": len(ids["train"] & ids["valid"]),
        "train_test": len(ids["train"] & ids["test"]),
        "valid_test": len(ids["valid"] & ids["test"]),
    }
    if any(split_overlap.values()):
        raise ValueError(f"split ID overlap: {split_overlap}")

    dependencies = {
        "raw_mixture_manifest": ROOT / "manifest.json",
        "decontamination_manifest": ROOT / "decontamination.json",
        "overlap_report": OVERLAP,
        "valid_test_overlap_report": VALID_TEST_OVERLAP,
        "training_config": CONFIG,
        "a100_token_budget_filter": ROOT / "train.a100_20480.filter.json",
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "frozen_mixed_inverse_tool_sft_training_set",
        "condition": "trace_no_knowledge",
        "corpus_used": False,
        "sources": ["flower_mech_proof", "mech_uspto_31k"],
        "splits": splits,
        "split_id_overlap": split_overlap,
        "decontamination_policy": ["exact_structural"],
        "post_decontamination_overlap_counts": overlap["overlap_counts"],
        "valid_test_overlap_counts": valid_test_overlap["overlap_counts"],
        "tokenizer_audits": audits,
        "model": {
            "name": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        },
        "dependencies": {
            name: {"file": str(path), "sha256": sha256(path)}
            for name, path in dependencies.items()
        },
    }
    output = ROOT / "training_manifest.json"
    output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
