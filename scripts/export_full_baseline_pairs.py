#!/usr/bin/env python3
"""Export frozen full-corpus reaction pairs for external retrosynthesis baselines.

This is the only supported handoff from MechET dataset builders to external
baseline repositories.  It does not resplit, filter by executor compatibility,
or derive method-specific labels.  It freezes mapped and unmapped product /
precursor views plus stable source IDs so LocalRetro, ReactSeq, EditRetro,
R-SMILES, RetroBridge, RETRO SYNFLOW, and other published systems start from
exactly the same reaction universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from rdkit import Chem


EXPECTED = {
    "flower_full": {"train": 257_171, "valid": 2_890, "test": 28_971},
    "mech_uspto_31k_full": {"train": 24_959, "valid": 3_120, "test": 3_120},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield row


def canonical_unmapped(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    parts: list[str] = []
    for fragment in text.split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(fragment, params)
        if mol is None:
            raise ValueError(f"invalid SMILES: {fragment}")
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(parts))


def canonical_mapped(smiles: str) -> str:
    text = str(smiles or "").strip()
    if not text:
        return ""
    parts: list[str] = []
    for fragment in text.split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(fragment, params)
        if mol is None:
            raise ValueError(f"invalid SMILES: {fragment}")
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return ".".join(sorted(parts))


def normalize_row(row: dict[str, Any], *, dataset: str, split: str) -> dict[str, Any]:
    identifier = str(row.get("id") or "").strip()
    if not identifier:
        raise ValueError("row missing stable id")

    product_mapped = str(
        row.get("product_mapped") or row.get("target_smiles") or ""
    ).strip()
    precursor_mapped = str(
        row.get("precursor_mapped")
        or row.get("structural_precursor")
        or row.get("expected_precursor")
        or ""
    ).strip()
    if not product_mapped or not precursor_mapped:
        raise ValueError(f"{identifier}: missing product or precursor")

    product_mapped = canonical_mapped(product_mapped)
    precursor_mapped = canonical_mapped(precursor_mapped)
    product_unmapped = str(row.get("product_unmapped") or "").strip()
    precursor_unmapped = str(row.get("precursor_unmapped") or "").strip()
    product_unmapped = (
        canonical_unmapped(product_unmapped)
        if product_unmapped
        else canonical_unmapped(product_mapped)
    )
    precursor_unmapped = (
        canonical_unmapped(precursor_unmapped)
        if precursor_unmapped
        else canonical_unmapped(precursor_mapped)
    )

    metadata = dict(row.get("metadata") or {})
    source_reaction_id = str(
        metadata.get("trajectory_id")
        or metadata.get("rxn_idx")
        or row.get("source_id")
        or identifier
    )
    return {
        "stable_id": identifier,
        "dataset": dataset,
        "split": split,
        "source_reaction_id": source_reaction_id,
        "product_mapped": product_mapped,
        "precursor_mapped": precursor_mapped,
        "product_unmapped": product_unmapped,
        "precursor_unmapped": precursor_unmapped,
        "reaction_mapped": f"{precursor_mapped}>>{product_mapped}",
        "reaction_unmapped": f"{precursor_unmapped}>>{product_unmapped}",
        "full_precursor_mapped": str(row.get("full_precursor_state") or precursor_mapped),
        "auxiliary_fragments_mapped": row.get("auxiliary_fragments") or [],
        "source_dataset": str(metadata.get("source_dataset") or dataset),
    }


def export_dataset(
    source_dir: Path,
    output_dir: Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_reports: dict[str, dict[str, Any]] = {}
    universes: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        source = source_dir / f"{split}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows = [normalize_row(row, dataset=dataset, split=split) for row in read_jsonl(source)]
        expected = EXPECTED[dataset][split]
        if len(rows) != expected:
            raise ValueError(f"{dataset}/{split}: {len(rows)} rows != {expected}")
        ids = [row["stable_id"] for row in rows]
        if len(set(ids)) != expected:
            raise ValueError(f"{dataset}/{split}: duplicate stable IDs")
        universes[split] = set(ids)
        output = output_dir / f"{split}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        split_reports[split] = {
            "rows": len(rows),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "output": str(output),
            "output_sha256": sha256_file(output),
            "stable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }

    overlap = {
        "train_valid": len(universes["train"] & universes["valid"]),
        "train_test": len(universes["train"] & universes["test"]),
        "valid_test": len(universes["valid"] & universes["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"{dataset}: split ID overlap: {overlap}")

    manifest = {
        "schema_version": 1,
        "artifact_type": "external_baseline_full_reaction_pairs",
        "dataset": dataset,
        "benchmark_universe": "complete_reaction_level_split",
        "executor_filtering": False,
        "representations": ["mapped", "unmapped"],
        "common_task": "product_to_precursor",
        "split_id_overlap": overlap,
        "splits": split_reports,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flower-dir",
        type=Path,
        default=Path("data/flower_full_endpoint_sft"),
    )
    parser.add_argument(
        "--mech-uspto-dir",
        type=Path,
        default=Path("data/mech_uspto_31k_full_endpoint_sft"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/external_baselines"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(EXPECTED),
        default=sorted(EXPECTED),
        help=(
            "Datasets to export. Select flower_full alone while the mapped "
            "mech-USPTO full endpoint source is unavailable."
        ),
    )
    args = parser.parse_args()

    manifests: dict[str, Any] = {}
    if "flower_full" in args.datasets:
        manifests["flower_full"] = export_dataset(
            args.flower_dir,
            args.output_root / "flower_full",
            dataset="flower_full",
        )
    if "mech_uspto_31k_full" in args.datasets:
        manifests["mech_uspto_31k_full"] = export_dataset(
            args.mech_uspto_dir,
            args.output_root / "mech_uspto_31k_full",
            dataset="mech_uspto_31k_full",
        )
    print(json.dumps(manifests, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
