#!/usr/bin/env python3
"""Build the complete 31,199-reaction mech-USPTO endpoint benchmark.

The upstream dataset stores one row per elementary mechanism step. Headline
retrosynthesis evaluation is reaction-level, so this builder groups by
``rxn_idx`` and keeps every upstream reaction. It deliberately performs no
executor replay, step-validity filtering, or global trace-stitching.

For each reaction, the precursor-side reference is ``elem_reac_min`` from the
first forward elementary step and the product is the reaction-level
``rxn_prod_min`` value. Both mapped and unmapped canonical views are frozen so
published baselines can use their native preprocessing without remapping the
reaction independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


EXPECTED_REACTIONS = {"train": 24_959, "valid": 3_120, "test": 3_120}
SOURCE_SPLITS = {"train": "train", "valid": "val", "test": "test"}
SYSTEM_PROMPT = (
    "Predict the precursor SMILES for the target product. Return exactly one "
    "line beginning with PRECURSOR:."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_smiles(value: Any) -> Chem.Mol:
    if value is None or pd.isna(value):
        raise ValueError("empty SMILES")
    text = str(value).strip()
    if not text:
        raise ValueError("empty SMILES")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None:
        raise ValueError(f"invalid SMILES: {text}")
    return mol


def canonical_mapped_smiles(value: Any) -> str:
    """Canonicalize while preserving all upstream atom-map labels."""
    mol = _parse_smiles(value)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_unmapped_smiles(value: Any) -> str:
    """Canonicalize the same structure after removing atom-map labels."""
    mol = _parse_smiles(value)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _first_forward_row(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(["step_idx_forward"], kind="stable")
    row = ordered.iloc[0]
    if int(row["step_idx_forward"]) != 0:
        raise ValueError(
            f"reaction {int(row['rxn_idx'])} has no forward step 0; "
            f"minimum={int(row['step_idx_forward'])}"
        )
    return row


def _unique_product(group: pd.DataFrame, *, mapped: bool) -> str:
    canonicalizer = canonical_mapped_smiles if mapped else canonical_unmapped_smiles
    values = {
        canonicalizer(value)
        for value in group["rxn_prod_min"].tolist()
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    if len(values) != 1:
        rxn_idx = int(group.iloc[0]["rxn_idx"])
        raise ValueError(
            f"reaction {rxn_idx} has {len(values)} distinct rxn_prod_min values"
        )
    return next(iter(values))


def build_split(source: Path, output: Path, split: str) -> dict[str, Any]:
    table = pd.read_parquet(
        source,
        columns=["rxn_idx", "step_idx_forward", "elem_reac_min", "rxn_prod_min"],
    )
    groups = table.groupby("rxn_idx", sort=True)
    expected = EXPECTED_REACTIONS[split]
    if groups.ngroups != expected:
        raise ValueError(
            f"unexpected {split} reaction count: {groups.ngroups} != {expected}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    stable_ids: list[str] = []
    rows = 0
    with output.open("w", encoding="utf-8") as handle:
        for rxn_idx, group in groups:
            first = _first_forward_row(group)
            precursor_mapped = canonical_mapped_smiles(first["elem_reac_min"])
            precursor_unmapped = canonical_unmapped_smiles(first["elem_reac_min"])
            product_mapped = _unique_product(group, mapped=True)
            product_unmapped = _unique_product(group, mapped=False)

            stable_id = f"mech-uspto31k-full:{split}:{int(rxn_idx)}"
            stable_ids.append(stable_id)
            row = {
                "id": stable_id,
                "source_id": str(int(rxn_idx)),
                "artifact_type": "supervision" if split == "train" else "evaluation_target",
                "task_type": "mech_uspto_31k_full_endpoint_retro",
                # Mapped fields preserve the source representation for methods
                # whose published preprocessing requires mapped reactions.
                "product_mapped": product_mapped,
                "precursor_mapped": precursor_mapped,
                # Unmapped fields are the common endpoint-evaluation view.
                "product_unmapped": product_unmapped,
                "precursor_unmapped": precursor_unmapped,
                "reaction_mapped": f"{precursor_mapped}>>{product_mapped}",
                "reaction_unmapped": f"{precursor_unmapped}>>{product_unmapped}",
                # Existing MechET data readers expect these aliases. Keep the
                # mapped target for address construction and the mapped source
                # reference; the shared evaluator removes maps structurally.
                "target_smiles": product_mapped,
                "structural_precursor": precursor_mapped,
                "expected_precursor": precursor_mapped,
                "full_precursor_state": precursor_mapped,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"TARGET: {product_mapped}"},
                    {"role": "assistant", "content": f"PRECURSOR: {precursor_mapped}"},
                ],
                "metadata": {
                    "source_dataset": "mech_uspto_31k",
                    "source_split": SOURCE_SPLITS[split],
                    "rxn_idx": int(rxn_idx),
                    "reaction_level_benchmark": True,
                    "benchmark_universe": "full_31199",
                    "endpoint_policy": "elem_reac_min_at_forward_step_0_to_rxn_prod_min",
                    "mapped_view_source": "upstream_mech_uspto_atom_maps",
                    "unmapped_view_source": "same_row_maps_removed_rdkit_canonical",
                    "mechanism_supervision": False,
                    "executable_trace_required": False,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    if rows != expected or len(set(stable_ids)) != expected:
        raise ValueError(f"failed to freeze {expected} unique {split} reactions")
    return {
        "rows": rows,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "stable_ids_sha256": hashlib.sha256(
            "\n".join(stable_ids).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/mech_uspto_31k/data"),
        help="directory containing train/val/test parquet shards",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mech_uspto_31k_full_endpoint_sft"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, dict[str, Any]] = {}
    for split, source_split in SOURCE_SPLITS.items():
        source = args.data_root / f"{source_split}-00000-of-00001.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        splits[split] = build_split(
            source, args.output_dir / f"{split}.jsonl", split
        )

    manifest = {
        "schema_version": 2,
        "artifact_type": "mech_uspto_31k_full_reaction_endpoint_benchmark",
        "source_dataset": "SchwallerGroup/mech_uspto_31k",
        "benchmark_universe": "all_31199_reactions",
        "reaction_counts": EXPECTED_REACTIONS,
        "total_reactions": sum(EXPECTED_REACTIONS.values()),
        "endpoint_policy": "elem_reac_min at step_idx_forward=0 -> rxn_prod_min",
        "representations": [
            "upstream mapped canonical SMILES",
            "same structures with maps removed and RDKit canonicalized",
        ],
        "coverage_contract": (
            "100% upstream reaction-level split coverage; no executor replay or "
            "trace-stitch filtering"
        ),
        "trace_subset": (
            "data/mech_uspto_31k_inverse_tool_sft; separate program-analysis view"
        ),
        "splits": splits,
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
