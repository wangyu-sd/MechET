#!/usr/bin/env python3
"""Derive the reaction-level ``flower_retro`` view from FlowER trajectories.

The public FlowER archive contains elementary transitions in
``flower_new_dataset/{train,val,test}.txt``.  The historical reaction-level
view used by retrosynthesis baselines is deterministic: for each trajectory ID
in first-appearance order, retain the left side of its first transition and
the right side of its last transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.mech_graph import parse_flower_line


SOURCE_SPLITS = {"train": "train", "valid": "val", "test": "test"}
EXPECTED_ROWS = {"train": 257_171, "valid": 2_890, "test": 28_971}
EXPECTED_SHA256 = {
    "train": "35ca595c362249c4922a9157a316c8c7fb839b5627d03c0d544abe46314615a5",
    "valid": "19ddcd0fd8ad5134994ce92e2556c011cba41466bb5bbde08c14f693bb138769",
    "test": "5fb019a128ad9f2ec62f23fea4be22f9d5baf226147c9ec0985fba0213988e6d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_reactions(source: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Collect first reactants and last products for every trajectory."""

    order: list[str] = []
    endpoints: dict[str, list[str]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = parse_flower_line(line)
            if parsed is None:
                raise ValueError(f"malformed FlowER row at {source}:{line_number}")
            reactants, products, trajectory_id = parsed
            if trajectory_id not in endpoints:
                order.append(trajectory_id)
                endpoints[trajectory_id] = [reactants, products]
            else:
                endpoints[trajectory_id][1] = products
    return order, endpoints


def build_split(
    flower_root: Path,
    output_dir: Path,
    split: str,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    source_name = SOURCE_SPLITS[split]
    source = flower_root / f"{source_name}.txt"
    if not source.is_file():
        raise FileNotFoundError(f"missing FlowER trajectory split: {source}")

    order, endpoints = derive_reactions(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source_name}.txt"
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as writer:
        for trajectory_id in order:
            reactants, products = endpoints[trajectory_id]
            writer.write(f"{reactants}>>{products}\n")

    rows = len(order)
    output_sha256 = sha256_file(temporary)
    if require_canonical:
        expected_rows = EXPECTED_ROWS[split]
        expected_sha256 = EXPECTED_SHA256[split]
        if rows != expected_rows or output_sha256 != expected_sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"noncanonical {split} flower_retro artifact: rows={rows} "
                f"sha256={output_sha256}; expected rows={expected_rows} "
                f"sha256={expected_sha256}"
            )
    temporary.replace(output)
    return {
        "split": split,
        "source_split": source_name,
        "rows": rows,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": output_sha256,
        "derivation": "first_transition_lhs_to_last_transition_rhs_by_trajectory_id",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flower-root",
        type=Path,
        required=True,
        help="directory containing flower_new_dataset train.txt/val.txt/test.txt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SOURCE_SPLITS),
        default=list(SOURCE_SPLITS),
    )
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="allow toy or alternate FlowER splits without official row/hash checks",
    )
    args = parser.parse_args()

    results = {
        split: build_split(
            args.flower_root,
            args.output_dir,
            split,
            require_canonical=not args.allow_noncanonical,
        )
        for split in args.splits
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": "flower_retro_reaction_endpoint_view",
        "source_dataset": "FlowER flower_new_dataset",
        "derivation": "first LHS and last RHS per trajectory ID in first-appearance order",
        "canonical_checks_required": not args.allow_noncanonical,
        "splits": results,
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
