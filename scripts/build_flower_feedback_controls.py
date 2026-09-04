#!/usr/bin/env python3
"""Build the matched B1, B3, and B5 FlowER internal controls.

All controls preserve the strict executable reaction-ID universe.  B1 removes
the inspect/enumeration interface, B3 replaces repeated observations from a
tool with that tool's first observation in the episode, and B5 gives a direct
answer model only the legal electron containers deterministically enumerable
from the product.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.forward_expert import enumerate_containers


DEFAULT_SOURCE = Path(
    "/aaa/fionafyang/buddy1/whaleywang/MechET/"
    "data/flower_inverse_tool_sft_compact_full_state_v1"
)
EXPECTED = {"train": 257_167, "valid": 2_890, "test": 28_967}


def _tool_name(schema: dict[str, Any]) -> str:
    return str((schema.get("function") or {}).get("name") or "")


def _metadata(row: dict[str, Any], control: str) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "paper_control": control,
            "control_source": "flower_compact_full_state_strict_program_v1",
        }
    )
    return metadata


def build_b1(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["tools"] = [
        dict(item)
        for item in row.get("tools") or []
        if _tool_name(item) != "inspect_state"
    ]
    messages: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        copied = dict(message)
        for call in copied.get("tool_calls") or []:
            if str((call.get("function") or {}).get("name") or "") == "inspect_state":
                raise ValueError(f"B1 source contains inspect_state target: {row.get('id')}")
        if copied.get("role") == "user":
            content = str(copied.get("content") or "")
            content = content.replace(
                "Use inspect_state before referencing atom maps.",
                "Reference atom maps only from TARGET and imported fragments.",
            ).replace(
                "Atom maps come from TARGET and imported fragments; inspect_state "
                "returns legal-action inventory without serializing the current state.",
                "Atom maps come only from TARGET and imported fragments.",
            )
            copied["content"] = content
        messages.append(copied)
    out["messages"] = messages
    out["task_type"] = "b1_no_source_sink_enumeration"
    out["metadata"] = _metadata(row, "B1_no_source_sink_enumeration")
    return out


def build_b3(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    first_visible: dict[str, str] = {}
    messages: list[dict[str, Any]] = []
    stale_replacements = 0
    for message in row.get("messages") or []:
        copied = dict(message)
        if copied.get("role") == "tool":
            name = str(copied.get("name") or "")
            raw = str(copied.get("content") or "")
            if name in first_visible:
                copied["content"] = first_visible[name]
                stale_replacements += 1
            else:
                first_visible[name] = raw
        messages.append(copied)
    out["messages"] = messages
    out["task_type"] = "b3_stale_action_feedback"
    metadata = _metadata(row, "B3_stale_action_feedback")
    metadata["stale_observation_policy"] = "first_visible_result_per_tool_v1"
    metadata["stale_observation_replacements"] = stale_replacements
    out["metadata"] = metadata
    return out


B5_SYSTEM = """You are a product-only retrosynthesis model.
Predict the complete structural precursor set for the target product. A legal
electron-container inventory computed deterministically from the product is
provided as product-derived structural information; it is not a reaction
answer. Return exactly one <answer> block containing mapped precursor SMILES."""


def build_b5(row: dict[str, Any]) -> dict[str, Any]:
    target = str(row.get("target_smiles") or "")
    sources, sinks = enumerate_containers(target)
    inventory = {
        "source": "deterministic_product_only_enumeration_v1",
        "sources": [item.to_dict() for item in sources],
        "sinks": [item.to_dict() for item in sinks],
    }
    answer = str(row.get("structural_precursor") or row.get("expected_precursor") or "")
    if not target or not answer:
        raise ValueError(f"B5 source lacks target/answer: {row.get('id')}")
    out = {
        key: row[key]
        for key in (
            "id",
            "source_id",
            "artifact_type",
            "target_smiles",
            "expected_precursor",
            "full_precursor_state",
            "structural_precursor",
            "auxiliary_fragments",
        )
        if key in row
    }
    source_metadata = dict(row.get("metadata") or {})
    out.update(
        {
            "task_type": "b5_direct_legal_actions",
            "messages": [
                {"role": "system", "content": B5_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"TARGET: {target}\n\nLEGAL-ACTION SUMMARY:\n"
                        + json.dumps(inventory, ensure_ascii=False, separators=(",", ":"))
                    ),
                },
                {"role": "assistant", "content": f"<answer>\n{answer}\n</answer>"},
            ],
            "metadata": {
                "paper_control": "B5_direct_plus_legal_actions",
                "control_source": "flower_compact_full_state_strict_program_v1",
                "source_trajectory_id": source_metadata.get("source_trajectory_id"),
                "source_mechanism_family": source_metadata.get("source_mechanism_family"),
                "original_proof_sha256": source_metadata.get("original_proof_sha256"),
                "legal_action_summary_source": "product_only",
                "legal_action_summary_dynamic": False,
                "trace_supervision": False,
            },
        }
    )
    out.pop("tools", None)
    return out


BUILDERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "b1_no_enumeration": build_b1,
    "b3_stale_feedback": build_b3,
    "b5_direct_legal_actions": build_b5,
}


def build_split(
    source: Path,
    destination: Path,
    builder: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    limit: int = 0,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    rows = 0
    with source.open(encoding="utf-8") as reader, temporary.open("wb") as writer:
        for line in reader:
            if not line.strip():
                continue
            encoded = (
                json.dumps(builder(json.loads(line)), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode()
            writer.write(encoded)
            digest.update(encoded)
            rows += 1
            if rows % 10_000 == 0:
                print(f"[meteor-data] split={destination.stem} rows={rows}", flush=True)
            if limit and rows >= limit:
                break
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "rows": rows,
        "bytes": destination.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=sorted(BUILDERS), required=True)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-root", type=Path, default=REPO / "data/iclr_feedback_controls_v1"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.output_root / args.control
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file() and not args.force and not args.limit:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(
            Path(manifest["splits"][split]["path"]).is_file()
            and int(manifest["splits"][split]["rows"]) == EXPECTED[split]
            for split in EXPECTED
        ):
            print(f"[meteor-data] reuse={manifest_path}", flush=True)
            return 0

    splits = {
        split: build_split(
            args.source_dir / f"{split}.jsonl",
            destination / f"{split}.jsonl",
            BUILDERS[args.control],
            limit=args.limit,
        )
        for split in ("train", "valid", "test")
    }
    if not args.limit:
        for split, expected in EXPECTED.items():
            if int(splits[split]["rows"]) != expected:
                raise ValueError(
                    f"{args.control}/{split}: {splits[split]['rows']} != {expected}"
                )
    manifest = {
        "artifact_type": "flower_matched_internal_control_v1",
        "paper_control": args.control,
        "source_dir": str(args.source_dir.resolve()),
        "source_universe": EXPECTED,
        "splits": splits,
        "tasks": {args.control: splits},
        "limited_smoke": bool(args.limit),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
