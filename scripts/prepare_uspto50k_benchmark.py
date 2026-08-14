#!/usr/bin/env python3
"""Freeze Schneider USPTO-50K and build product-only MechET eval rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.data_audit import (  # noqa: E402
    sha256_file,
    split_reaction_smiles,
    split_structural_and_environment,
)


SOURCE_REPOSITORY = "https://github.com/coleygroup/rxn-ebm"
SOURCE_REVISION = "1919eeccdd31e16ec7a44478b756bcd974c35a3c"
SOURCE_TEMPLATE = (
    "https://raw.githubusercontent.com/coleygroup/rxn-ebm/"
    + SOURCE_REVISION
    + "/rxnebm/data/original_data/schneider50k_{split}.csv"
)
SOURCE_SHA256 = {
    "train": "69661b12baa44d5a0be6cfc7698af8b518341fcb4427780c60358e0d9dcd8e7f",
    "valid": "a52eb4cfd889820cf5172f65ac0e1ac124f3a36051674d5ca8bd63d037e149ee",
    "test": "f28ed322ebc35518da9d8b1c742b7fda9022fd1761529a0f2a221970e2ffa810",
}
EXPECTED_ROWS = {"train": 40008, "valid": 5001, "test": 5007}
RAW_REACTION_FIELDS = (
    "reactants>reagents>production",
    "reaction_smiles",
    "rxn_smiles",
    "reaction",
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    digest = hashlib.sha256()
    with urlopen(url) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
            handle.write(chunk)
    temporary.replace(destination)


def _reaction(row: dict[str, str]) -> str:
    for field in RAW_REACTION_FIELDS:
        value = str(row.get(field) or "").strip()
        if value:
            return value
    raise ValueError(f"reaction column absent; fields={sorted(row)}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_split(root: Path, split: str, *, download: bool) -> dict[str, Any]:
    source = root / "source" / f"schneider50k_{split}.csv"
    if download and not source.exists():
        _download(SOURCE_TEMPLATE.format(split=split), source)
    if not source.is_file():
        raise FileNotFoundError(f"missing {source}; rerun with --download")
    observed_hash = sha256_file(source)
    if observed_hash != SOURCE_SHA256[split]:
        raise ValueError(
            f"source hash mismatch for {split}: {observed_hash} != {SOURCE_SHA256[split]}"
        )

    table_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    counts = {
        "mapped_role_rows": 0,
        "unmapped_role_rows": 0,
        "structural_fragments": 0,
        "environment_fragments": 0,
    }
    with source.open("r", encoding="utf-8", newline="") as handle:
        for index, raw in enumerate(csv.DictReader(handle)):
            reaction = _reaction(raw)
            reactants, reagents, product = split_reaction_smiles(reaction)
            roles = split_structural_and_environment(reactants, product)
            stable_id = f"uspto50k-standard:{split}:{index:05d}"
            patent_id = str(raw.get("id") or "")
            reaction_class = str(raw.get("class") or "UNK")
            full_left = ".".join(part for part in (reactants, reagents) if part)
            auxiliary = ".".join(roles.environment)
            table_rows.append(
                {
                    "id": stable_id,
                    "patent_id": patent_id,
                    "reaction_class": reaction_class,
                    "reaction_smiles": reaction,
                    "source_row_index": index,
                }
            )
            eval_rows.append(
                {
                    "id": stable_id,
                    "source_id": patent_id or stable_id,
                    "artifact_type": "evaluation_target",
                    "target_smiles": product,
                    "structural_precursor": roles.structural_smiles,
                    "expected_precursor": roles.structural_smiles,
                    "full_precursor_state": full_left,
                    "auxiliary_fragments": auxiliary,
                    "metadata": {
                        "source_dataset": "schneider_uspto50k_standard",
                        "source_split": split,
                        "mixture_source": "uspto50k_standard",
                        "patent_id": patent_id,
                        "reaction_class": reaction_class,
                        "mapped_roles_supported": roles.mapped,
                    },
                }
            )
            counts["mapped_role_rows" if roles.mapped else "unmapped_role_rows"] += 1
            counts["structural_fragments"] += len(roles.structural)
            counts["environment_fragments"] += len(roles.environment)

    if len(table_rows) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"unexpected {split} size: {len(table_rows)} != {EXPECTED_ROWS[split]}"
        )
    if len({row["id"] for row in table_rows}) != len(table_rows):
        raise ValueError(f"duplicate stable IDs in {split}")

    table_path = root / f"{split}.csv"
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    eval_path = root / f"{split}.inverse_eval.jsonl"
    _write_jsonl(eval_path, eval_rows)
    return {
        "split": split,
        "rows": len(table_rows),
        "source_path": str(source),
        "source_sha256": observed_hash,
        "reaction_table": str(table_path),
        "reaction_table_sha256": sha256_file(table_path),
        "inverse_eval": str(eval_path),
        "inverse_eval_sha256": sha256_file(eval_path),
        **counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "data/benchmarks/uspto50k"
    )
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = [
        prepare_split(args.output_dir, split, download=args.download)
        for split in ("train", "valid", "test")
    ]
    manifest = {
        "artifact_type": "frozen_uspto50k_standard_benchmark",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_variant": "rxn-ebm original_data/schneider50k_{split}.csv",
        "endpoint_policy": (
            "whole left-side fragments sharing >=1 atom map with product"
        ),
        "model_input_policy": "mapped product only; no gold precursor or class",
        "gold_trace_available": False,
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
