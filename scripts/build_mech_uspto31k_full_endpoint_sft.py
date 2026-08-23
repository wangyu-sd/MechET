#!/usr/bin/env python3
"""Build full mech-USPTO-31k endpoints from the original mapped reactions.

The Hugging Face elementary-step export is used only for the frozen split and
for joining/auditing ``rxn_prod_min``. It is not an atom-mapping source. The
mapped ``reaction`` column must come from the original Figshare reaction-level
CSV (or an equivalent lossless table).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.reaction_mapping import (  # noqa: E402
    assert_product_contained_in_reference,
    product_only_reindex_reaction,
)


EXPECTED = {"train": 24_959, "valid": 3_120, "test": 3_120}
PARQUET_SPLITS = {"train": "train", "valid": "val", "test": "test"}
SYSTEM_PROMPT = (
    "Predict the atom-contributing structural precursor SMILES for the mapped "
    "product. Return exactly one line beginning with PRECURSOR:."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_source_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^rxn[_-]?", "", text, flags=re.IGNORECASE)
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    if not re.fullmatch(r"\d+", text):
        raise ValueError(f"unsupported reaction identifier: {value!r}")
    return str(int(text))


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"unsupported mapped-reaction table: {path}")


def resolve_column(frame: pd.DataFrame, requested: str, candidates: tuple[str, ...]) -> str:
    normalized = {
        str(column).strip().lower().replace(" ", "_"): str(column)
        for column in frame.columns
    }
    if requested:
        key = requested.strip().lower().replace(" ", "_")
        if key not in normalized:
            raise ValueError(f"column {requested!r} absent; available={list(frame.columns)}")
        return normalized[key]
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"none of {candidates} found in {list(frame.columns)}")


def load_mapped_reactions(
    path: Path, *, id_column: str = "", reaction_column: str = ""
) -> tuple[dict[str, str], dict[str, Any]]:
    frame = read_table(path)
    id_key = resolve_column(
        frame, id_column, ("rxn_id", "reaction_id", "source", "id", "rxn_idx")
    )
    reaction_key = resolve_column(
        frame, reaction_column, ("reaction", "mapped_reaction", "mapped_rxn", "rxn_smiles")
    )
    output: dict[str, str] = {}
    for row_number, row in frame.iterrows():
        source_id = normalize_source_id(row[id_key])
        reaction = str(row[reaction_key] or "").strip()
        if not reaction:
            raise ValueError(f"empty mapped reaction at source row {row_number}")
        if source_id in output:
            raise ValueError(f"duplicate mapped reaction ID: {source_id}")
        output[source_id] = reaction
    return output, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "id_column": id_key,
        "reaction_column": reaction_key,
    }


def load_split_rows(parquet: Path) -> list[dict[str, str]]:
    frame = pd.read_parquet(
        parquet,
        columns=["rxn_idx", "source", "step_idx_forward", "rxn_prod_min"],
    )
    rows: list[dict[str, str]] = []
    for _, group in frame.groupby("rxn_idx", sort=True):
        ordered = group.sort_values("step_idx_forward")
        if int(ordered.iloc[0]["step_idx_forward"]) != 0:
            raise ValueError(f"reaction {ordered.iloc[0]['rxn_idx']} has no first step")
        sources = set(ordered["source"].astype(str))
        products = set(ordered["rxn_prod_min"].astype(str))
        if len(sources) != 1 or len(products) != 1:
            raise ValueError(f"reaction group has inconsistent source/product: {sources}")
        rows.append(
            {
                "rxn_idx": str(int(ordered.iloc[0]["rxn_idx"])),
                "source": next(iter(sources)),
                "rxn_prod_min": next(iter(products)),
            }
        )
    return rows


def build_split(
    *,
    split: str,
    parquet: Path,
    mapped_reactions: dict[str, str],
    endpoint_dir: Path,
    localretro_dir: Path,
) -> dict[str, Any]:
    source_rows = load_split_rows(parquet)
    if len(source_rows) != EXPECTED[split]:
        raise ValueError(f"{split}: expected {EXPECTED[split]} reactions, got {len(source_rows)}")

    endpoint_path = endpoint_dir / f"{split}.jsonl"
    localretro_path = localretro_dir / f"{split}.csv"
    endpoint_tmp = endpoint_path.with_suffix(".jsonl.part")
    localretro_tmp = localretro_path.with_suffix(".csv.part")
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    localretro_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    with endpoint_tmp.open("w", encoding="utf-8") as endpoint_handle, localretro_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as localretro_handle:
        localretro_writer = csv.DictWriter(
            localretro_handle, fieldnames=["class", "id", "rxn_smiles"]
        )
        localretro_writer.writeheader()
        for source in source_rows:
            join_id = normalize_source_id(source["source"])
            if join_id in seen:
                raise ValueError(f"{split}: duplicate joined source ID {join_id}")
            seen.add(join_id)
            try:
                original_reaction = mapped_reactions[join_id]
            except KeyError as exc:
                raise ValueError(f"{split}: original mapped reaction missing for {source['source']}") from exc
            mapped = product_only_reindex_reaction(original_reaction)
            assert_product_contained_in_reference(
                mapped.products, source["rxn_prod_min"]
            )
            stable_id = f"mech-uspto31k-full:{source['source']}"
            full_precursor = mapped.reactants
            row = {
                "id": stable_id,
                "source_id": source["source"],
                "artifact_type": "supervision" if split == "train" else "evaluation_target",
                "task_type": "mech_uspto31k_endpoint_retro",
                "target_smiles": mapped.products,
                "product_mapped": mapped.products,
                "product_unmapped": mapped.products_unmapped,
                "structural_precursor": mapped.structural_precursor,
                "expected_precursor": mapped.structural_precursor,
                "full_precursor_state": full_precursor,
                "reactants_mapped": mapped.reactants,
                "reactants_unmapped": mapped.reactants_unmapped,
                "auxiliary_fragments": list(mapped.auxiliary_fragments),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"TARGET: {mapped.products}"},
                    {
                        "role": "assistant",
                        "content": f"PRECURSOR: {mapped.structural_precursor}",
                    },
                ],
                "metadata": {
                    "source_dataset": "mech_uspto_31k_figshare_v2",
                    "split_source": "SchwallerGroup/mech_uspto_31k",
                    "rxn_idx": source["rxn_idx"],
                    "mapped_reaction_join_id": join_id,
                    "mapped_reaction_source": "original_reaction_column_not_rxn_prod_min",
                    "atom_map_policy": "product_only_canonical_reindex_synchronized_to_reactants",
                    "endpoint_policy": "original_reaction_product_and_atom_contributing_reactant_fragments",
                    "rxn_prod_min_role": "mechanism_final_mixture_join_audit_only",
                    "mapped_roles_supported": True,
                    "mechanism_supervision": False,
                    "coverage_track": "full_endpoint",
                    "reaction_class_used": False,
                },
            }
            endpoint_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            localretro_writer.writerow(
                {"class": "0", "id": stable_id, "rxn_smiles": mapped.reaction_smiles}
            )

    endpoint_tmp.replace(endpoint_path)
    localretro_tmp.replace(localretro_path)
    return {
        "rows": len(source_rows),
        "expected_rows": EXPECTED[split],
        "coverage": len(source_rows) / EXPECTED[split],
        "hf_elementary_source": str(parquet.resolve()),
        "hf_elementary_source_sha256": sha256_file(parquet),
        "endpoint_file": str(endpoint_path.resolve()),
        "endpoint_sha256": sha256_file(endpoint_path),
        "localretro_file": str(localretro_path.resolve()),
        "localretro_sha256": sha256_file(localretro_path),
        "stable_ids_sha256": hashlib.sha256(
            "\n".join(sorted(f"mech-uspto31k-full:rxn_{value}" for value in seen)).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapped-reactions", type=Path, required=True)
    parser.add_argument("--mapped-id-column", default="")
    parser.add_argument("--mapped-reaction-column", default="")
    parser.add_argument(
        "--hf-root", type=Path, default=Path("data/raw/mech_uspto_31k/data")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mech_uspto_31k_full_endpoint_sft")
    )
    parser.add_argument(
        "--localretro-dir", type=Path, default=Path("data/baselines/localretro_mech_uspto_31k")
    )
    parser.add_argument(
        "--splits", nargs="+", choices=tuple(EXPECTED), default=list(EXPECTED)
    )
    args = parser.parse_args()

    mapped_reactions, mapped_source = load_mapped_reactions(
        args.mapped_reactions,
        id_column=args.mapped_id_column,
        reaction_column=args.mapped_reaction_column,
    )
    split_reports = {}
    used_ids: set[str] = set()
    for split in args.splits:
        parquet_name = PARQUET_SPLITS[split]
        parquet = args.hf_root / f"{parquet_name}-00000-of-00001.parquet"
        split_reports[split] = build_split(
            split=split,
            parquet=parquet,
            mapped_reactions=mapped_reactions,
            endpoint_dir=args.output_dir,
            localretro_dir=args.localretro_dir,
        )
        frame = pd.read_parquet(parquet, columns=["source"])
        used_ids.update(normalize_source_id(value) for value in frame["source"].unique())
    missing = used_ids - set(mapped_reactions)
    if missing:
        raise ValueError(f"mapped source missing {len(missing)} used reactions")

    manifest = {
        "schema_version": 1,
        "artifact_type": "mech_uspto_31k_full_endpoint_sft",
        "protocol": "original_mapped_reaction_product_only_reindex_v1",
        "headline_eligible": True,
        "mapped_source": mapped_source,
        "forbidden_mapping_source": "rxn_prod_min (unmapped mechanism final mixture)",
        "model_input_policy": "original desired product with product-only canonical atom reindexing",
        "localretro_contract": "mapped reactants and products share atom identities; constant class=0 placeholder with reaction-class features disabled",
        "coverage_contract": "complete reaction-level splits; fail rather than filter",
        "splits": split_reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.localretro_dir.mkdir(parents=True, exist_ok=True)
    (args.localretro_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
