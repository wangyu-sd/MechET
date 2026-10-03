#!/usr/bin/env python3
"""Audit frozen NMI split overlap without loading multi-GB JSONL rows.

Morgan/Tanimoto near-duplicate classification uses an exact bit-count bound:
if Tanimoto(a,b) >= t, then t*a <= b <= a/t. Only candidates in that range
need full similarity evaluation; no above-threshold pair is discarded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Any, Iterator

from rdkit import DataStructs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.structural_overlap import (
    _fingerprint,
    canonical_unmapped_smiles,
    family_label,
    murcko_scaffold_key,
    reaction_center_context_features,
    structural_precursor_smiles,
    target_smiles,
)


def _feature(item: tuple[str, str, dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
    identifier, split, row = item
    product_smiles = target_smiles(row)
    product = canonical_unmapped_smiles(product_smiles)
    precursor = canonical_unmapped_smiles(structural_precursor_smiles(row))
    try:
        center_features = reaction_center_context_features(row, preserve_mapped_h=True)
        center = center_features["ordered_context_key"]
        local_centers = center_features["local_structural_centers"]
        center_error = None
    except ValueError as exc:
        if not str(exc).startswith("REACTION_CENTER_ATOMS_MISSING:"):
            raise
        # Do not quarantine or silently count an undefined center as unseen.
        # Keep the reaction in its frozen split and expose the diagnostic.
        center, local_centers, center_error = None, (), str(exc)
    return identifier, split, {
        "product_key": product,
        "reaction_key": f"{product}>>{precursor}",
        "scaffold_key": murcko_scaffold_key(product_smiles),
        "reaction_center_key": center,
        "local_structural_centers": local_centers,
        "reaction_center_error": center_error,
        "family": family_label(row),
        "product_fingerprint": _fingerprint(product_smiles),
    }


def _candidate_bitcounts(bits: int, threshold: float) -> range:
    lower = int(threshold * bits)
    if lower < threshold * bits:
        lower += 1
    upper = int(bits / threshold)
    return range(lower, upper + 1)


def is_near_duplicate(fp, buckets: dict[int, list[Any]], threshold: float) -> bool:
    for size in _candidate_bitcounts(fp.GetNumOnBits(), threshold):
        candidates = buckets.get(size)
        if candidates and any(
            value >= threshold for value in DataStructs.BulkTanimotoSimilarity(fp, candidates)
        ):
            return True
    return False


def _load_split_ids(split_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifest = json.loads((split_dir / "manifest.json").read_text())
    id_to_split: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        path = split_dir / f"{split}.ids.txt"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["split_id_sha256"][split]:
            raise ValueError(f"split ID SHA mismatch: {split}")
        ids = content.decode().splitlines()
        if len(ids) != manifest["rows"][split]:
            raise ValueError(f"split ID count mismatch: {split}")
        for identifier in ids:
            if identifier in id_to_split:
                raise ValueError(f"split ID appears twice: {identifier}")
            id_to_split[identifier] = split
    return id_to_split, manifest


def audit_overlap(
    source: Path, split_dir: Path, *, workers: int = 8,
    similarity_threshold: float = 0.9, progress_every: int = 10000,
    limit: int = 0,
) -> dict[str, Any]:
    if not 0 < similarity_threshold <= 1 or workers < 1:
        raise ValueError("invalid similarity threshold or worker count")
    id_to_split, manifest = _load_split_ids(split_dir)
    source_digest = hashlib.sha256()
    scanned = 0
    train_product: set[str] = set()
    train_reaction: set[str] = set()
    train_scaffold: set[str] = set()
    train_center: set[str] = set()
    train_local_center_frequency: Counter[str] = Counter()
    center_errors: dict[str, str] = {}
    train_family: set[str] = set()
    fp_buckets: dict[int, list[Any]] = defaultdict(list)
    heldout: dict[str, list[tuple[str, dict[str, Any]]]] = {"valid": [], "test": []}

    def items() -> Iterator[tuple[str, str, dict[str, Any]]]:
        nonlocal scanned
        with source.open("rb") as handle:
            for line in handle:
                if limit and scanned >= limit:
                    break
                source_digest.update(line)
                row = json.loads(line)
                identifier = str(row.get("source_id") or "")
                split = id_to_split.get(identifier)
                if split is None:
                    raise ValueError(f"source ID absent from frozen split: {identifier}")
                metadata = row.get("metadata") or {}
                reduced = {
                    "id": identifier,
                    "target_smiles": row.get("target_smiles"),
                    "structural_precursor": row.get("structural_precursor"),
                    "metadata": {
                        "trace_plan": metadata.get("trace_plan"),
                        "reaction_family": metadata.get("reaction_family"),
                        "mechanism_family": metadata.get("mechanism_family"),
                    },
                }
                scanned += 1
                yield identifier, split, reduced

    with Pool(processes=workers) as pool:
        for index, (identifier, split, feature) in enumerate(
            pool.imap(_feature, items(), chunksize=32), start=1
        ):
            if split == "train":
                train_product.add(feature["product_key"])
                train_reaction.add(feature["reaction_key"])
                train_scaffold.add(feature["scaffold_key"])
                if feature["reaction_center_key"] is not None:
                    train_center.add(feature["reaction_center_key"])
                    train_local_center_frequency.update(set(feature["local_structural_centers"]))
                if feature["family"]:
                    train_family.add(feature["family"])
                fp = feature["product_fingerprint"]
                fp_buckets[fp.GetNumOnBits()].append(fp)
            else:
                heldout[split].append((identifier, feature))
            if feature["reaction_center_error"]:
                center_errors[identifier] = feature["reaction_center_error"]
            if progress_every and index % progress_every == 0:
                print(json.dumps({"phase": "features", "rows": index}), flush=True)
    if not limit:
        if scanned != manifest["n_rows"] or source_digest.hexdigest() != manifest["source_sha256"]:
            raise ValueError("structural audit source contract mismatch")
        if scanned != len(id_to_split):
            raise ValueError("structural audit did not cover all split IDs")

    reports: dict[str, Any] = {}
    for split, rows in heldout.items():
        counts: Counter[str] = Counter()
        annotations: dict[str, dict[str, Any]] = {}
        for index, (identifier, feature) in enumerate(rows, start=1):
            product_seen = feature["product_key"] in train_product
            exact_reaction_seen = feature["reaction_key"] in train_reaction
            scaffold_seen = feature["scaffold_key"] in train_scaffold
            center_seen = (
                feature["reaction_center_key"] in train_center
                if feature["reaction_center_key"] is not None else None
            )
            local_centers = feature["local_structural_centers"]
            local_center_frequencies = [train_local_center_frequency[item] for item in local_centers]
            local_center_fraction_seen = (
                sum(value > 0 for value in local_center_frequencies) / len(local_center_frequencies)
                if local_center_frequencies else None
            )
            fp = feature["product_fingerprint"]
            near_duplicate = product_seen or is_near_duplicate(fp, fp_buckets, similarity_threshold)
            family = feature["family"]
            family_seen = family in train_family if family else None
            counts.update({
                "exact_product_seen": product_seen,
                "exact_reaction_seen": exact_reaction_seen,
                "scaffold_seen": scaffold_seen,
                "reaction_center_seen": center_seen is True,
                "reaction_center_undefined": center_seen is None,
                "local_center_any_seen": any(value > 0 for value in local_center_frequencies),
                "local_center_all_seen": bool(local_center_frequencies) and all(value > 0 for value in local_center_frequencies),
                "near_duplicate": near_duplicate,
                "family_available": bool(family),
                "family_seen": family_seen is True,
            })
            annotations[identifier] = {
                "exact_product_seen_in_train": product_seen,
                "exact_reaction_seen_in_train": exact_reaction_seen,
                "murcko_scaffold_seen_in_train": scaffold_seen,
                "reaction_center_context_seen_in_train": center_seen,
                "local_center_fraction_seen_in_train": local_center_fraction_seen,
                "local_center_min_train_reaction_frequency": min(local_center_frequencies) if local_center_frequencies else None,
                "local_center_any_seen_in_train": any(value > 0 for value in local_center_frequencies),
                "local_center_all_seen_in_train": bool(local_center_frequencies) and all(value > 0 for value in local_center_frequencies),
                "reaction_center_error": feature["reaction_center_error"],
                "near_duplicate_at_threshold": near_duplicate,
                "family_seen_in_train": family_seen,
            }
            if progress_every and index % progress_every == 0:
                print(json.dumps({"phase": "near_duplicates", "split": split, "rows": index}), flush=True)
        reports[split] = {
            "n": len(rows),
            "counts": dict(counts),
            "rates": {key: value / len(rows) if rows else 0.0 for key, value in counts.items()},
            "annotations": annotations,
        }
    return {
        "artifact_type": "nmi_mechcomp_structural_overlap_v2",
        "split_manifest_sha256": hashlib.sha256((split_dir / "manifest.json").read_bytes()).hexdigest(),
        "source_sha256": source_digest.hexdigest() if not limit else None,
        "source_rows_scanned": scanned,
        "scope": "full_frozen_split" if not limit else "diagnostic_prefix_only",
        "similarity": {
            "fingerprint": "Morgan radius=2 2048-bit largest product fragment",
            "metric": "Tanimoto",
            "threshold": similarity_threshold,
            "bitcount_bound_exact_for_threshold": True,
        },
        "reaction_center_context_definition": "step_state_plus_edge_imports_v3_explicit_h",
        "local_center_definition": "per_step_unmapped_radius1_state_context_excluding_move_labels_and_order_v1",
        "reaction_center_undefined_count": len(center_errors),
        "reaction_center_errors": center_errors,
        "split": reports,
        "gates": {
            "test_zero_exact_reaction_overlap": reports["test"]["counts"].get("exact_reaction_seen", 0) == 0,
            "valid_zero_exact_reaction_overlap": reports["valid"]["counts"].get("exact_reaction_seen", 0) == 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--similarity-threshold", type=float, default=0.9)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing structural audit: {args.output}")
    report = audit_overlap(
        args.source, args.split_dir, workers=args.workers,
        similarity_threshold=args.similarity_threshold,
        progress_every=args.progress_every, limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "source_rows_scanned": report["source_rows_scanned"],
        "scope": report["scope"],
        "test": {key: value for key, value in report["split"]["test"].items() if key != "annotations"},
        "gates": report["gates"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
