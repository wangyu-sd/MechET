"""Composition-disjoint splits for executable retrosynthesis proofs."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Iterable

from mechet.proof_equivalence import (
    composition_signature,
    primitive_signatures,
)
from mechet.proof_program import ProofProgramError, extract_proof_body


@dataclass(frozen=True)
class ProofSplitFeatures:
    composition: str
    primitives: tuple[str, ...]


def _assistant_text(row: dict[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def extract_split_features(row: dict[str, Any]) -> ProofSplitFeatures:
    text = _assistant_text(row)
    if not extract_proof_body(text):
        raise ProofProgramError(
            f"row {row.get('id')} has no MECH_PROOF assistant output"
        )
    return ProofSplitFeatures(
        composition=composition_signature(text),
        primitives=primitive_signatures(text),
    )


def _annotate_row(
    row: dict[str, Any],
    features: ProofSplitFeatures,
    split: str,
) -> dict[str, Any]:
    out = deepcopy(row)
    metadata = dict(out.get("metadata") or {})
    metadata.update(
        {
            "mechcomp_split": split,
            "proof_composition_signature": features.composition,
            "proof_primitive_signatures": list(features.primitives),
        }
    )
    out["metadata"] = metadata
    return out


def _select_groups(
    groups: dict[str, list[int]],
    features: list[ProofSplitFeatures],
    available: set[int],
    primitive_counts: Counter[str],
    *,
    target_n: int,
    min_remaining_primitive_count: int,
    rng: random.Random,
) -> set[int]:
    candidates = list(groups.items())
    rng.shuffle(candidates)
    # Prefer compact composition groups so the requested fraction can be met
    # without sacrificing primitive coverage.
    candidates.sort(
        key=lambda item: len(
            [index for index in item[1] if index in available]
        )
    )
    selected: set[int] = set()
    for _composition, indices in candidates:
        active = [index for index in indices if index in available]
        if not active:
            continue
        if selected and len(selected) >= target_n:
            break
        removal_counts: Counter[str] = Counter()
        for index in active:
            removal_counts.update(features[index].primitives)
        if any(
            primitive_counts[primitive] - count
            < min_remaining_primitive_count
            for primitive, count in removal_counts.items()
        ):
            continue
        selected.update(active)
        primitive_counts.subtract(removal_counts)
    return selected


def build_compositional_ood_split(
    rows: Iterable[dict[str, Any]],
    *,
    test_fraction: float = 0.1,
    valid_fraction: float = 0.1,
    min_train_primitive_count: int = 2,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Create composition-disjoint train/valid/test proof splits.

    Complete composition groups are held out. Every primitive appearing in a
    held-out split remains represented in train at least
    ``min_train_primitive_count`` times.
    """
    if not 0 <= test_fraction < 1 or not 0 <= valid_fraction < 1:
        raise ValueError("split fractions must be in [0, 1)")
    if test_fraction + valid_fraction >= 1:
        raise ValueError("test_fraction + valid_fraction must be < 1")
    if min_train_primitive_count < 1:
        raise ValueError("min_train_primitive_count must be >= 1")

    rows_list = list(rows)
    features = [extract_split_features(row) for row in rows_list]
    groups: dict[str, list[int]] = defaultdict(list)
    primitive_counts: Counter[str] = Counter()
    for index, feature in enumerate(features):
        groups[feature.composition].append(index)
        primitive_counts.update(feature.primitives)

    rng = random.Random(seed)
    available = set(range(len(rows_list)))
    test_indices = _select_groups(
        groups,
        features,
        available,
        primitive_counts,
        target_n=round(len(rows_list) * test_fraction),
        min_remaining_primitive_count=min_train_primitive_count,
        rng=rng,
    )
    available.difference_update(test_indices)

    valid_indices = _select_groups(
        groups,
        features,
        available,
        primitive_counts,
        target_n=round(len(rows_list) * valid_fraction),
        min_remaining_primitive_count=min_train_primitive_count,
        rng=rng,
    )
    available.difference_update(valid_indices)
    train_indices = available

    index_sets = {
        "train": train_indices,
        "valid": valid_indices,
        "test": test_indices,
    }
    splits = {
        split: [
            _annotate_row(rows_list[index], features[index], split)
            for index in sorted(indices)
        ]
        for split, indices in index_sets.items()
    }

    composition_sets = {
        split: {features[index].composition for index in indices}
        for split, indices in index_sets.items()
    }
    primitive_sets = {
        split: {
            primitive
            for index in indices
            for primitive in features[index].primitives
        }
        for split, indices in index_sets.items()
    }
    train_primitive_counts: Counter[str] = Counter()
    for index in train_indices:
        train_primitive_counts.update(features[index].primitives)

    manifest = {
        "seed": seed,
        "n_total": len(rows_list),
        "n_train": len(splits["train"]),
        "n_valid": len(splits["valid"]),
        "n_test": len(splits["test"]),
        "n_compositions": len(groups),
        "composition_overlap": {
            "train_valid": len(
                composition_sets["train"] & composition_sets["valid"]
            ),
            "train_test": len(
                composition_sets["train"] & composition_sets["test"]
            ),
            "valid_test": len(
                composition_sets["valid"] & composition_sets["test"]
            ),
        },
        "heldout_primitive_coverage": {
            split: (
                len(primitive_sets[split] & primitive_sets["train"])
                / max(len(primitive_sets[split]), 1)
            )
            for split in ("valid", "test")
        },
        "min_train_primitive_count": min(
            train_primitive_counts.values(),
            default=0,
        ),
        "requested": {
            "test_fraction": test_fraction,
            "valid_fraction": valid_fraction,
            "min_train_primitive_count": min_train_primitive_count,
        },
    }
    return splits, manifest
