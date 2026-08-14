#!/usr/bin/env python3
"""Build a frozen trace-only FlowER + mech-USPTO inverse Tool-SFT mixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.knowledge_ablation import strip_knowledge_messages  # noqa: E402
from train_tool_sft import validate_conversation  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_ids_sha256(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _order_key(row: dict[str, Any], *, seed: int, split: str) -> str:
    payload = f"{seed}:{split}:{row['id']}".encode()
    return hashlib.sha256(payload).hexdigest()


def prepare_flower(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    value = strip_knowledge_messages(row)
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "source_dataset": "flower_mech_proof",
            "source_split": split,
            "trace_condition": "trace_no_knowledge",
            "mixture_source": "flower",
            "corpus_used": False,
        }
    )
    value["metadata"] = metadata
    return value


def prepare_mech_uspto(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    value = dict(row)
    metadata = dict(value.get("metadata") or {})
    metadata.update(
        {
            "source_split": split,
            "trace_condition": "trace_no_knowledge",
            "mixture_source": "mech_uspto_31k",
            "corpus_used": False,
        }
    )
    value["metadata"] = metadata
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-dir", type=Path, default=Path("data/textbook_tool_sft"))
    parser.add_argument(
        "--mech-uspto-dir",
        type=Path,
        default=Path("data/mech_uspto_31k_inverse_tool_sft"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mixed_inverse_tool_sft"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_ids: dict[str, set[str]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    canonical_tools = ""
    for split in ("train", "valid", "test"):
        flower_path = args.flower_dir / f"{split}.jsonl"
        mech_path = args.mech_uspto_dir / f"{split}.jsonl"
        flower = [prepare_flower(row, split=split) for row in read_rows(flower_path)]
        mech = [prepare_mech_uspto(row, split=split) for row in read_rows(mech_path)]
        rows = flower + mech
        ids = {str(row.get("id") or "") for row in rows}
        if "" in ids or len(ids) != len(rows):
            raise ValueError(f"INVALID_OR_DUPLICATE_ID:{split}")
        for row in rows:
            validate_conversation(row, require_trace_owned=True)
            schema = json.dumps(row.get("tools") or [], sort_keys=True, separators=(",", ":"))
            if not canonical_tools:
                canonical_tools = schema
            elif schema != canonical_tools:
                raise ValueError(f"TOOL_SCHEMA_MISMATCH:{split}:{row['id']}")
        rows.sort(key=lambda row: _order_key(row, seed=args.seed, split=split))
        output = args.output_dir / f"{split}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        split_ids[split] = ids
        split_reports[split] = {
            "rows": len(rows),
            "source_rows": {"flower": len(flower), "mech_uspto_31k": len(mech)},
            "source_ratio": {
                "flower": len(flower) / len(rows),
                "mech_uspto_31k": len(mech) / len(rows),
            },
            "file": str(output),
            "sha256": sha256(output),
            "stable_ids_sha256": stable_ids_sha256(ids),
            "inputs": {
                "flower": {"file": str(flower_path), "sha256": sha256(flower_path)},
                "mech_uspto_31k": {"file": str(mech_path), "sha256": sha256(mech_path)},
            },
        }

    overlap = {
        "train_valid": len(split_ids["train"] & split_ids["valid"]),
        "train_test": len(split_ids["train"] & split_ids["test"]),
        "valid_test": len(split_ids["valid"] & split_ids["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"SPLIT_ID_OVERLAP:{overlap}")
    heldout_path = args.output_dir / "heldout.jsonl"
    with heldout_path.open("w", encoding="utf-8") as handle:
        for split in ("valid", "test"):
            with (args.output_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        handle.write(line)
    manifest = {
        "schema_version": 1,
        "artifact_type": "mixed_trace_owned_inverse_tool_sft",
        "condition": "trace_no_knowledge",
        "sources": ["flower_mech_proof", "mech_uspto_31k"],
        "corpus_used": False,
        "sampling_policy": "natural_frequency_deterministic_interleave_v1",
        "shuffle_seed": args.seed,
        "splits": split_reports,
        "split_id_overlap": overlap,
        "heldout": {
            "file": str(heldout_path),
            "rows": len(split_ids["valid"]) + len(split_ids["test"]),
            "sha256": sha256(heldout_path),
        },
        "tool_schema_sha256": hashlib.sha256(canonical_tools.encode()).hexdigest(),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
