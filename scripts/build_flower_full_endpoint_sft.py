#!/usr/bin/env python3
"""Build the complete FlowER reaction-level inverse endpoint dataset.

This builder deliberately does not compile mechanisms into MECH_PROOF.  Every
row in ``flower_retro`` is retained, so endpoint coverage is identical to the
official reaction/trajectory split.  Executable trace data remains a separate
subset with its own coverage denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.mech_graph import compact_mapped_smiles, get_main_product, parse_flower_line


DEFAULT_DATA_ROOT = Path("/aaa/fionafyang/buddy1/whaleywang/datasets/retro/data")
EXPECTED_ROWS = {"train": 257_171, "valid": 2_890, "test": 28_971}
SOURCE_SPLITS = {"train": "train", "valid": "val", "test": "test"}
MAP_PATTERN = re.compile(r":(\d+)\]")
SYSTEM_PROMPT = (
    "Predict the atom-contributing structural precursor SMILES for the mapped "
    "main product. Return exactly one line beginning with PRECURSOR:."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _maps(smiles: str) -> set[int]:
    """Extract atom maps without requiring RDKit sanitization."""

    return {int(value) for value in MAP_PATTERN.findall(smiles or "")}


def _fragments(smiles: str) -> list[str]:
    return [part.strip() for part in (smiles or "").split(".") if part.strip()]


def _compact_fragments(fragments: list[str]) -> list[str]:
    # Canonicalize fragment-by-fragment.  compact_mapped_smiles is fail-soft,
    # which prevents unusual FlowER valence states from disappearing silently.
    return sorted(compact_mapped_smiles(fragment) for fragment in fragments)


def split_endpoint_roles(reactants: str, target: str) -> tuple[str, list[str], bool]:
    target_maps = _maps(target)
    structural: list[str] = []
    auxiliary: list[str] = []
    if not target_maps:
        structural = _fragments(reactants)
        mapped_roles = False
    else:
        mapped_roles = True
        for fragment in _fragments(reactants):
            (structural if _maps(fragment) & target_maps else auxiliary).append(fragment)
    return (
        ".".join(_compact_fragments(structural)),
        _compact_fragments(auxiliary),
        mapped_roles,
    )


def iter_trajectory_ids(path: Path) -> Iterator[str]:
    """Yield each trajectory ID once, preserving first-appearance order.

    FlowER rows are not strictly contiguous by trajectory ID, so transition-
    based grouping over-counts train/valid/test.  A set of integer-like IDs is
    small compared with the molecular states and gives the correct alignment
    with the prebuilt reaction-level ``flower_retro`` files.
    """

    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = parse_flower_line(line)
            if parsed is None:
                raise ValueError(f"malformed FlowER row at {path}:{line_number}")
            trajectory_id = parsed[2]
            if trajectory_id not in seen:
                seen.add(trajectory_id)
                yield trajectory_id


def _parse_reaction(line: str, *, path: Path, line_number: int) -> tuple[str, str]:
    text = line.strip()
    if not text or text.count(">>") != 1:
        raise ValueError(f"malformed reaction at {path}:{line_number}")
    reactants, products = (part.strip() for part in text.split(">>", 1))
    if not reactants or not products:
        raise ValueError(f"empty reaction side at {path}:{line_number}")
    return reactants, products


def build_split(data_root: Path, output_root: Path, split: str) -> dict[str, Any]:
    source_name = SOURCE_SPLITS[split]
    reaction_path = data_root / "flower_retro" / f"{source_name}.txt"
    trajectory_path = data_root / "flower_new_dataset" / f"{source_name}.txt"
    if not reaction_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError(f"missing FlowER sources: {reaction_path}, {trajectory_path}")

    output_path = output_root / f"{split}.jsonl"
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    trajectory_ids = iter_trajectory_ids(trajectory_path)
    seen_ids: set[str] = set()
    rows = 0
    mapped_role_rows = 0
    structural_fragments = 0
    auxiliary_fragments = 0

    with reaction_path.open("r", encoding="utf-8") as reader, temporary_path.open(
        "w", encoding="utf-8"
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            reactants, products = _parse_reaction(
                line, path=reaction_path, line_number=line_number
            )
            try:
                trajectory_id = next(trajectory_ids)
            except StopIteration as exc:
                raise ValueError(
                    f"more reaction rows than unique trajectory IDs in {split}"
                ) from exc
            stable_id = f"flower-full-endpoint:{split}:{trajectory_id}"
            if stable_id in seen_ids:
                raise ValueError(f"duplicate stable ID: {stable_id}")
            seen_ids.add(stable_id)

            target = compact_mapped_smiles(get_main_product(products))
            structural, auxiliary, mapped_roles = split_endpoint_roles(reactants, target)
            full_precursor = ".".join(_compact_fragments(_fragments(reactants)))
            full_product = ".".join(_compact_fragments(_fragments(products)))
            if not target or not structural:
                raise ValueError(
                    f"empty endpoint at {reaction_path}:{line_number}; no row may be dropped"
                )

            row = {
                "id": stable_id,
                "source_id": trajectory_id,
                "artifact_type": "supervision" if split == "train" else "evaluation_target",
                "task_type": "flower_endpoint_retro",
                "target_smiles": target,
                "structural_precursor": structural,
                "expected_precursor": structural,
                "full_precursor_state": full_precursor,
                "auxiliary_fragments": auxiliary,
                "full_product_state": full_product,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"TARGET: {target}"},
                    {"role": "assistant", "content": f"PRECURSOR: {structural}"},
                ],
                "metadata": {
                    "source_dataset": "flower_new_dataset",
                    "source_view": "flower_retro_reaction_endpoint",
                    "source_split": source_name,
                    "trajectory_id": trajectory_id,
                    "mapped_roles_supported": mapped_roles,
                    "endpoint_policy": "reactant_fragments_sharing_atom_map_with_main_product",
                    "mechanism_supervision": False,
                    "executable_trace": False,
                    "coverage_track": "full_endpoint",
                    "qwen_sft_format": "chat_messages_v1",
                    "assistant_only_loss": True,
                },
            }
            writer.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1
            mapped_role_rows += int(mapped_roles)
            structural_fragments += len(_fragments(structural))
            auxiliary_fragments += len(auxiliary)
            if rows % 25_000 == 0:
                print(f"[{split}] rows={rows:,}/{EXPECTED_ROWS[split]:,}", flush=True)

    try:
        extra_trajectory_id = next(trajectory_ids)
    except StopIteration:
        extra_trajectory_id = ""
    if extra_trajectory_id:
        raise ValueError(
            f"fewer reaction rows than unique trajectory IDs in {split}; "
            f"first extra ID={extra_trajectory_id}"
        )
    expected = EXPECTED_ROWS[split]
    if rows != expected:
        raise ValueError(f"unexpected {split} size: {rows} != {expected}")
    temporary_path.replace(output_path)

    return {
        "rows": rows,
        "expected_rows": expected,
        "coverage": rows / expected,
        "reaction_source": str(reaction_path),
        "reaction_source_sha256": sha256_file(reaction_path),
        "trajectory_source": str(trajectory_path),
        "trajectory_source_sha256": sha256_file(trajectory_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "stable_ids_sha256": hashlib.sha256(
            "\n".join(sorted(seen_ids)).encode("utf-8")
        ).hexdigest(),
        "mapped_role_rows": mapped_role_rows,
        "structural_fragments": structural_fragments,
        "auxiliary_fragments": auxiliary_fragments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "data/flower_full_endpoint_sft"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=tuple(SOURCE_SPLITS), default=list(SOURCE_SPLITS)
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        split: build_split(args.data_root, args.output_dir, split)
        for split in args.splits
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "frozen_flower_full_endpoint_inverse_sft",
        "task_type": "flower_endpoint_retro",
        "source_dataset": "FlowER flower_new_dataset",
        "model_input_policy": "mapped main product only",
        "primary_target": "atom-contributing structural precursor",
        "endpoint_policy": "whole reactant fragments sharing >=1 atom map with main product",
        "coverage_contract": (
            "100% reaction-level split coverage; no mechanism compiler or replay filtering"
        ),
        "mechanism_supervision": False,
        "executable_trace": False,
        "trace_subset": "data/flower_inverse_tool_sft (reported separately)",
        "splits": splits,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
