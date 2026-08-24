#!/usr/bin/env python3
"""Build the full mech-USPTO-31k external-baseline handoff from HF endpoints.

This is the public-source fallback when the original Figshare reaction table is
unavailable.  It groups the official Hugging Face elementary-step parquet by
reaction, takes the first forward ``elem_reac_spe`` as the complete precursor
and the invariant reaction-level ``rxn_prod_min`` as the product endpoint, and
maps that pair exactly once with a pinned RXNMapper environment.  Every
external baseline consumes the resulting shared mapping; downstream methods
must not remap reactions independently.

The builder is fail-closed and resumable.  It never filters reactions for
mapping, proof, replay, or sequence-length compatibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.reaction_mapping import product_only_reindex_reaction  # noqa: E402


EXPECTED = {"train": 24_959, "valid": 3_120, "test": 3_120}
PARQUET_SPLITS = {"train": "train", "valid": "val", "test": "test"}
SYSTEM_PROMPT = (
    "Predict the precursor SMILES for the target product. Return exactly one "
    "line beginning with PRECURSOR:."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_unmapped(smiles: Any) -> str:
    text = str(smiles or "").strip()
    if not text:
        raise ValueError("empty endpoint SMILES")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ValueError(f"invalid endpoint SMILES: {text}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def select_main_product(final_mixture: str) -> str:
    """Select one deterministic desired-product proxy from the HF final state.

    The HF ``rxn_prod_min`` field is a final mixture and can contain salts and
    byproducts.  Standard one-step retrosynthesis predicts the principal
    product, so use the largest organic fragment with deterministic tie breaks.
    """
    candidates: list[tuple[tuple[int, int, int, float, str], str]] = []
    for fragment in str(final_mixture or "").split("."):
        fragment = fragment.strip()
        if not fragment:
            continue
        canonical = canonical_unmapped(fragment)
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"invalid final-mixture fragment: {fragment}")
        heavy = int(mol.GetNumHeavyAtoms())
        carbons = sum(int(atom.GetAtomicNum() == 6) for atom in mol.GetAtoms())
        mass = float(sum(atom.GetMass() for atom in mol.GetAtoms()))
        candidates.append(((int(carbons > 0), heavy, carbons, mass, canonical), canonical))
    if not candidates:
        raise ValueError("final mixture contains no product fragment")
    return max(candidates, key=lambda item: item[0])[1]


def load_reaction_rows(
    hf_root: Path, splits: Iterable[str], *, limit_reactions: int = 0
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in splits:
        source = hf_root / f"{PARQUET_SPLITS[split]}-00000-of-00001.parquet"
        frame = pd.read_parquet(
            source,
            columns=["rxn_idx", "step_idx_forward", "elem_reac_spe", "rxn_prod_min"],
        )
        groups = frame.groupby("rxn_idx", sort=True)
        expected = EXPECTED[split]
        if groups.ngroups != expected:
            raise ValueError(f"{split}: {groups.ngroups} reactions != {expected}")
        emitted = 0
        for rxn_idx, group in groups:
            first = group.sort_values("step_idx_forward", kind="stable").iloc[0]
            if int(first["step_idx_forward"]) != 0:
                raise ValueError(f"{split}/{rxn_idx}: missing forward step zero")
            final_mixtures = {
                canonical_unmapped(value)
                for value in group["rxn_prod_min"].tolist()
                if str(value or "").strip()
            }
            if len(final_mixtures) != 1:
                raise ValueError(f"{split}/{rxn_idx}: {len(final_mixtures)} final mixtures")
            precursor = canonical_unmapped(first["elem_reac_spe"])
            final_mixture = next(iter(final_mixtures))
            product = select_main_product(final_mixture)
            stable_id = f"mech-uspto31k-full:{split}:{int(rxn_idx)}"
            output.append(
                {
                    "stable_id": stable_id,
                    "split": split,
                    "rxn_idx": int(rxn_idx),
                    "precursor_unmapped": precursor,
                    "product_unmapped": product,
                    "final_mixture_unmapped": final_mixture,
                    "reaction_unmapped": f"{precursor}>>{product}",
                    "hf_source": str(source.resolve()),
                }
            )
            emitted += 1
            if limit_reactions and emitted >= limit_reactions:
                break
    return output


def read_mapping_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            stable_id = str(row.get("stable_id") or "")
            if not stable_id:
                raise ValueError(f"invalid mapping cache row {line_number}")
            # A later row supersedes an earlier mapping when endpoint-selection
            # policy changes during a resumable build.
            output[stable_id] = row
    return output


def map_missing_rows(
    rows: list[dict[str, Any]], cache_path: Path, *, batch_size: int
) -> dict[str, dict[str, Any]]:
    cache = read_mapping_cache(cache_path)
    pending = [
        row
        for row in rows
        if row["stable_id"] not in cache
        or cache[row["stable_id"]].get("reaction_unmapped") != row["reaction_unmapped"]
    ]
    if not pending:
        return cache
    try:
        from rxnmapper import RXNMapper
    except ImportError as exc:
        raise RuntimeError("install the isolated mechet[mapping] environment") from exc

    mapper = RXNMapper()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            reactions = [row["reaction_unmapped"] for row in chunk]
            try:
                results = mapper.get_attention_guided_atom_maps(reactions)
            except Exception:
                if len(chunk) == 1:
                    raise
                results = []
                for reaction in reactions:
                    results.extend(
                        mapper.get_attention_guided_atom_maps([reaction])
                    )
            if len(results) != len(chunk):
                raise ValueError("RXNMapper returned the wrong batch cardinality")
            for row, result in zip(chunk, results):
                mapped_reaction = str(
                    result.get("mapped_rxn") or result.get("mapped_reaction") or ""
                ).strip()
                if ">>" not in mapped_reaction:
                    raise ValueError(f"{row['stable_id']}: mapper returned no reaction")
                reactants, products = mapped_reaction.split(">>", 1)
                if canonical_unmapped(reactants) != row["precursor_unmapped"]:
                    raise ValueError(f"{row['stable_id']}: mapped precursor changed")
                if canonical_unmapped(products) != row["product_unmapped"]:
                    raise ValueError(f"{row['stable_id']}: mapped product changed")
                # This also verifies unique positive product maps and complete
                # cross-side product-map transport before writing the cache.
                try:
                    normalized = product_only_reindex_reaction(mapped_reaction)
                except Exception as exc:
                    raise ValueError(f"{row['stable_id']}: invalid mapped endpoint: {exc}") from exc
                cache_row = {
                    "stable_id": row["stable_id"],
                    "reaction_unmapped": row["reaction_unmapped"],
                    "reaction_mapped": normalized.reaction_smiles,
                    "mapping_confidence": result.get("confidence"),
                }
                handle.write(json.dumps(cache_row, sort_keys=True) + "\n")
                handle.flush()
                cache[row["stable_id"]] = cache_row
            print(
                f"mapped {min(start + len(chunk), len(pending))}/{len(pending)} "
                f"pending endpoints (cache={len(cache)})",
                flush=True,
            )
    return cache


def write_outputs(
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    localretro_dir: Path,
    hf_root: Path,
    cache_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    localretro_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for split in sorted({row["split"] for row in rows}):
        split_rows = [row for row in rows if row["split"] == split]
        endpoint_path = output_dir / f"{split}.jsonl"
        localretro_path = localretro_dir / f"{split}.csv"
        ids: list[str] = []
        confidences: list[float] = []
        with endpoint_path.open("w", encoding="utf-8") as endpoint_handle, localretro_path.open(
            "w", encoding="utf-8", newline=""
        ) as localretro_handle:
            writer = csv.DictWriter(
                localretro_handle, fieldnames=["class", "id", "rxn_smiles"]
            )
            writer.writeheader()
            for source in split_rows:
                mapped_row = cache.get(source["stable_id"])
                if mapped_row is None:
                    raise ValueError(f"missing mapping: {source['stable_id']}")
                mapped = product_only_reindex_reaction(mapped_row["reaction_mapped"])
                ids.append(source["stable_id"])
                confidence = mapped_row.get("mapping_confidence")
                if confidence is not None:
                    confidences.append(float(confidence))
                endpoint = {
                    "id": source["stable_id"],
                    "source_id": str(source["rxn_idx"]),
                    "artifact_type": "supervision" if split == "train" else "evaluation_target",
                    "task_type": "mech_uspto31k_hf_endpoint_retro",
                    "target_smiles": mapped.products,
                    "product_mapped": mapped.products,
                    "product_unmapped": mapped.products_unmapped,
                    "structural_precursor": mapped.structural_precursor,
                    "expected_precursor": mapped.structural_precursor,
                    "precursor_mapped": mapped.structural_precursor,
                    "precursor_unmapped": canonical_unmapped(mapped.structural_precursor),
                    "full_precursor_state": mapped.reactants,
                    "reactants_mapped": mapped.reactants,
                    "reactants_unmapped": mapped.reactants_unmapped,
                    "reaction_mapped": mapped.reaction_smiles,
                    "reaction_unmapped": source["reaction_unmapped"],
                    "final_mixture_unmapped": source["final_mixture_unmapped"],
                    "auxiliary_fragments": list(mapped.auxiliary_fragments),
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"TARGET: {mapped.products}"},
                        {"role": "assistant", "content": f"PRECURSOR: {mapped.structural_precursor}"},
                    ],
                    "metadata": {
                        "source_dataset": "SchwallerGroup/mech_uspto_31k",
                        "source_split": PARQUET_SPLITS[split],
                        "rxn_idx": source["rxn_idx"],
                        "endpoint_policy": "elem_reac_spe_step_0_to_largest_organic_rxn_prod_min_fragment",
                        "mapping_source": "rxnmapper_recomputed_once_from_hf_endpoint_pair",
                        "mapping_confidence": confidence,
                        "atom_map_policy": "product_only_canonical_reindex_synchronized_to_reactants",
                        "mechanism_supervision": False,
                        "executable_trace_required": False,
                    },
                }
                endpoint_handle.write(json.dumps(endpoint, ensure_ascii=False) + "\n")
                writer.writerow(
                    {"class": "0", "id": source["stable_id"], "rxn_smiles": mapped.reaction_smiles}
                )
        expected = len(split_rows) if smoke else EXPECTED[split]
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"{split}: output coverage is not {expected}/{expected}")
        reports[split] = {
            "rows": len(ids),
            "expected_rows": expected,
            "endpoint_file": str(endpoint_path.resolve()),
            "endpoint_sha256": sha256_file(endpoint_path),
            "localretro_file": str(localretro_path.resolve()),
            "localretro_sha256": sha256_file(localretro_path),
            "stable_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "mapping_confidence": {
                "count": len(confidences),
                "min": min(confidences) if confidences else None,
                "mean": sum(confidences) / len(confidences) if confidences else None,
            },
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "mech_uspto_31k_full_hf_endpoint_rxnmapper",
        "benchmark_universe": "complete_hf_reaction_level_split",
        "source_repository": "SchwallerGroup/mech_uspto_31k",
        "source_files": {
            name: {
                "path": str((hf_root / f"{source}-00000-of-00001.parquet").resolve()),
                "sha256": sha256_file(hf_root / f"{source}-00000-of-00001.parquet"),
            }
            for name, source in PARQUET_SPLITS.items()
            if name in reports
        },
        "endpoint_policy": "elem_reac_spe at step 0 -> deterministic largest organic rxn_prod_min fragment",
        "mapping_policy": "one shared RXNMapper mapping then product-only canonical reindex",
        "mapping_runtime": {
            package: importlib_metadata.version(package)
            for package in ("rxnmapper", "transformers", "tokenizers")
        },
        "mapping_cache": str(cache_path.resolve()),
        "executor_filtering": False,
        "mapping_failure_filtering": False,
        "smoke": smoke,
        "splits": reports,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (localretro_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-root", type=Path, default=Path("data/raw/mech_uspto_31k/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mech_uspto_31k_full_endpoint_rxnmapper")
    )
    parser.add_argument(
        "--localretro-dir", type=Path, default=Path("data/baselines/localretro_mech_uspto_31k_rxnmapper")
    )
    parser.add_argument("--mapping-cache", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=tuple(EXPECTED), default=list(EXPECTED))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-reactions", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit_reactions < 0:
        parser.error("batch size must be positive and limit must be non-negative")
    cache_path = args.mapping_cache or args.output_dir / "rxnmapper_cache.jsonl"
    rows = load_reaction_rows(
        args.hf_root, args.splits, limit_reactions=args.limit_reactions
    )
    cache = map_missing_rows(rows, cache_path, batch_size=args.batch_size)
    manifest = write_outputs(
        rows,
        cache,
        output_dir=args.output_dir,
        localretro_dir=args.localretro_dir,
        hf_root=args.hf_root,
        cache_path=cache_path,
        smoke=bool(args.limit_reactions),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
