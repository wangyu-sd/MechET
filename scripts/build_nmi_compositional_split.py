#!/usr/bin/env python3
"""Freeze a composition-disjoint split of the strict-executable train pool.

The split is reaction-level and intentionally separate from official FlowER
valid/test. Exact structural reactions and move compositions are grouped before
selection; model training must filter all representations by these source IDs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mechet.proof_splits import ProofSplitFeatures, _select_groups, extract_split_features
from mechet.structural_overlap import canonical_unmapped_smiles, structural_precursor_smiles, target_smiles


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def select_split(
    identifiers: list[str], features: list[ProofSplitFeatures], reaction_keys: list[str],
    *, seed: int, test_fraction: float, valid_fraction: float,
    min_train_primitive_count: int,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    if not (0 < test_fraction < 1 and 0 <= valid_fraction < 1 and test_fraction + valid_fraction < 1):
        raise ValueError("invalid split fractions")
    if len(identifiers) != len(features) or len(features) != len(reaction_keys):
        raise ValueError("precomputed feature lengths differ")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("source IDs are not unique")
    union = UnionFind()
    first_composition: dict[str, int] = {}
    first_reaction: dict[str, int] = {}
    primitive_counts: Counter[str] = Counter()
    for index, (feature, reaction) in enumerate(zip(features, reaction_keys, strict=True)):
        assert union.add() == index
        if feature.composition in first_composition:
            union.union(index, first_composition[feature.composition])
        else:
            first_composition[feature.composition] = index
        if reaction in first_reaction:
            union.union(index, first_reaction[reaction])
        else:
            first_reaction[reaction] = index
        primitive_counts.update(feature.primitives)
    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(len(identifiers)):
        groups[str(union.find(index))].append(index)
    available = set(range(len(identifiers)))
    rng = random.Random(seed)
    test = _select_groups(
        groups, features, available, primitive_counts,
        target_n=round(len(identifiers) * test_fraction),
        min_remaining_primitive_count=min_train_primitive_count, rng=rng,
    )
    available.difference_update(test)
    valid = _select_groups(
        groups, features, available, primitive_counts,
        target_n=round(len(identifiers) * valid_fraction),
        min_remaining_primitive_count=min_train_primitive_count, rng=rng,
    ) if valid_fraction else set()
    available.difference_update(valid)
    split_indices = {"train": available, "valid": valid, "test": test}
    train_primitives = set().union(*(set(features[index].primitives) for index in available)) if available else set()
    train_compositions = {features[index].composition for index in available}
    train_reactions = {reaction_keys[index] for index in available}
    heldout_primitives = set().union(*(set(features[index].primitives) for index in (valid | test))) if valid or test else set()
    valid_compositions = {features[index].composition for index in valid}
    test_compositions = {features[index].composition for index in test}
    valid_reactions = {reaction_keys[index] for index in valid}
    test_reactions = {reaction_keys[index] for index in test}
    train_counts: Counter[str] = Counter()
    for index in available:
        train_counts.update(features[index].primitives)
    gates = {
        "nonempty_test": bool(test),
        "zero_train_test_composition_overlap": not bool(train_compositions & test_compositions),
        "zero_valid_test_composition_overlap": not bool(valid_compositions & test_compositions),
        "zero_train_test_exact_reaction_overlap": not bool(train_reactions & test_reactions),
        "zero_train_valid_exact_reaction_overlap": not bool(train_reactions & valid_reactions),
        "all_heldout_primitives_seen_in_train": heldout_primitives.issubset(train_primitives),
        "heldout_primitives_at_declared_min_frequency": all(
            train_counts[primitive] >= min_train_primitive_count for primitive in heldout_primitives
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"split invariant failed: {gates}")
    report = {
        "seed": seed,
        "requested_fraction": {"valid": valid_fraction, "test": test_fraction},
        "requested_min_train_primitive_count": min_train_primitive_count,
        "n_rows": len(identifiers),
        "n_union_components": len(groups),
        "largest_union_component": max(map(len, groups.values()), default=0),
        "rows": {split: len(indices) for split, indices in split_indices.items()},
        "unique_train_primitives": len(train_primitives),
        "unique_train_compositions": len(train_compositions),
        "heldout_unique_primitives": len(heldout_primitives),
        "min_heldout_primitive_train_frequency": min(
            (train_counts[primitive] for primitive in heldout_primitives), default=0
        ),
        "gates": gates,
    }
    return split_indices, report


def load_features(
    source: Path, *, expected_sha256: str, expected_rows: int,
    verify_first: int = 32,
) -> tuple[list[str], list[ProofSplitFeatures], list[str], str]:
    identifiers: list[str] = []
    features: list[ProofSplitFeatures] = []
    reaction_keys: list[str] = []
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            identifier = str(row.get("source_id") or "")
            if not identifier or metadata.get("executor_replayed") is not True:
                raise ValueError(f"missing source ID or replay flag: {identifier}")
            feature = ProofSplitFeatures(
                composition=str(metadata.get("execution_composition_signature") or ""),
                primitives=tuple(metadata.get("execution_primitive_signatures") or ()),
            )
            if len(feature.composition) != 64 or not feature.primitives:
                raise ValueError(f"missing execution signature: {identifier}")
            if len(identifiers) < verify_first and feature != extract_split_features(row):
                raise ValueError(f"precomputed feature mismatch: {identifier}")
            product = canonical_unmapped_smiles(target_smiles(row))
            precursor = canonical_unmapped_smiles(structural_precursor_smiles(row))
            identifiers.append(identifier)
            features.append(feature)
            reaction_keys.append(f"{product}>>{precursor}")
    actual_sha256 = digest.hexdigest()
    if len(identifiers) != expected_rows or actual_sha256 != expected_sha256:
        raise ValueError("source row-count/SHA contract mismatch")
    return identifiers, features, reaction_keys, actual_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--min-train-primitive-count", type=int, default=5)
    parser.add_argument("--verify-first", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen split: {args.output_dir}")
    contract = json.loads(args.manifest.read_text())["splits"]["train"]
    if Path(contract["file"]).name != args.source.name:
        raise ValueError("manifest source basename mismatch")
    identifiers, features, reaction_keys, source_sha256 = load_features(
        args.source, expected_sha256=str(contract["sha256"]),
        expected_rows=int(contract["rows"]), verify_first=args.verify_first,
    )
    splits, report = select_split(
        identifiers, features, reaction_keys, seed=args.seed,
        test_fraction=args.test_fraction, valid_fraction=args.valid_fraction,
        min_train_primitive_count=args.min_train_primitive_count,
    )
    args.output_dir.mkdir(parents=True)
    split_hashes = {}
    for split, indices in splits.items():
        content = "".join(identifier + "\n" for index, identifier in enumerate(identifiers) if index in indices)
        path = args.output_dir / f"{split}.ids.txt"
        path.write_text(content)
        split_hashes[split] = hashlib.sha256(content.encode()).hexdigest()
    report.update({
        "artifact_type": "nmi_mechcomp_split_ids_v1",
        "parent_scope": "strict_executable_flower_training_pool_not_official_test",
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "primitive_basis": "source_to_sink_execution_moves_v1",
        "composition_basis": "execution_composition_signature_ordered_moves_v1",
        "exact_reaction_key": "canonical_unmapped_product_and_structural_precursor",
        "split_id_sha256": split_hashes,
        "test_use_policy": "freeze_before_training; no final-test model selection",
        "pending_audits": [
            "product_scaffold_overlap", "step_state_reaction_center_overlap",
            "morgan_product_near_duplicate", "primitive_unseen_negative_control",
        ],
    })
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
