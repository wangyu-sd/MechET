"""Deterministic paired and multi-seed statistics for MechET evaluations."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import mean, median, stdev
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class BinaryPair:
    left: bool
    right: bool


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    """Return the two-sided exact binomial McNemar p-value."""

    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2.0 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_binary_contrast(
    left: Iterable[bool],
    right: Iterable[bool],
    *,
    bootstrap_samples: int = 5000,
    bootstrap_seed: int = 17,
    confidence: float = 0.95,
) -> dict[str, float | int | list[float]]:
    """Compare two binary outcomes over the same frozen identifier universe.

    The reported effect is ``left_rate - right_rate``. Bootstrap resampling is
    paired by identifier, and the exact McNemar test is computed from discordant
    pairs without requiring SciPy.
    """

    left_values = [bool(value) for value in left]
    right_values = [bool(value) for value in right]
    if len(left_values) != len(right_values):
        raise ValueError("paired outcomes must have identical lengths")
    if not left_values:
        raise ValueError("paired outcomes cannot be empty")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    pairs = [BinaryPair(a, b) for a, b in zip(left_values, right_values)]
    n = len(pairs)
    left_correct = sum(pair.left for pair in pairs)
    right_correct = sum(pair.right for pair in pairs)
    left_only = sum(pair.left and not pair.right for pair in pairs)
    right_only = sum(not pair.left and pair.right for pair in pairs)
    delta = (left_correct - right_correct) / n

    rng = random.Random(int(bootstrap_seed))
    bootstrap: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled = [pairs[rng.randrange(n)] for _ in range(n)]
        bootstrap.append(
            (
                sum(pair.left for pair in sampled)
                - sum(pair.right for pair in sampled)
            )
            / n
        )
    alpha = 1.0 - confidence
    lower = _quantile(bootstrap, alpha / 2.0)
    upper = _quantile(bootstrap, 1.0 - alpha / 2.0)

    return {
        "n_pairs": n,
        "left_rate": left_correct / n,
        "right_rate": right_correct / n,
        "delta_left_minus_right": delta,
        "paired_bootstrap_confidence": confidence,
        "paired_bootstrap_samples": int(bootstrap_samples),
        "paired_bootstrap_seed": int(bootstrap_seed),
        "paired_bootstrap_ci": [lower, upper],
        "left_correct_right_incorrect": left_only,
        "left_incorrect_right_correct": right_only,
        "discordant_pairs": left_only + right_only,
        "mcnemar_exact_p_value": _exact_mcnemar_p_value(left_only, right_only),
        "matched_odds_ratio_continuity_corrected": (
            (left_only + 0.5) / (right_only + 0.5)
        ),
    }


def holm_adjust(
    p_values: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Apply Holm's step-down family-wise error correction."""

    valid = sorted(
        ((name, float(value)) for name, value in p_values.items() if value is not None),
        key=lambda item: item[1],
    )
    adjusted: dict[str, float | None] = {name: None for name in p_values}
    running = 0.0
    total = len(valid)
    for rank, (name, value) in enumerate(valid):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def aggregate_seed_effects(
    effects: Mapping[str, float],
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 23,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Aggregate one precomputed effect per independent training seed."""

    values = {str(seed): float(value) for seed, value in effects.items()}
    if not values:
        raise ValueError("at least one seed effect is required")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    keys = sorted(values)
    observed = [values[key] for key in keys]
    rng = random.Random(int(bootstrap_seed))
    bootstrap = [
        mean(values[keys[rng.randrange(len(keys))]] for _ in keys)
        for _ in range(int(bootstrap_samples))
    ]
    alpha = 1.0 - confidence
    return {
        "n_seeds": len(keys),
        "seeds": keys,
        "effects_by_seed": values,
        "mean_effect": mean(observed),
        "median_effect": median(observed),
        "sample_standard_deviation": stdev(observed) if len(observed) > 1 else 0.0,
        "positive_seed_fraction": sum(value > 0 for value in observed) / len(observed),
        "negative_seed_fraction": sum(value < 0 for value in observed) / len(observed),
        "zero_seed_fraction": sum(value == 0 for value in observed) / len(observed),
        "seed_bootstrap_confidence": confidence,
        "seed_bootstrap_samples": int(bootstrap_samples),
        "seed_bootstrap_seed": int(bootstrap_seed),
        "seed_bootstrap_ci": [
            _quantile(bootstrap, alpha / 2.0),
            _quantile(bootstrap, 1.0 - alpha / 2.0),
        ],
    }


def hierarchical_paired_binary_contrast(
    pairs_by_seed: Mapping[str, tuple[Sequence[bool], Sequence[bool]]],
    *,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 29,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap independent seeds, then paired rows within each sampled seed."""

    normalized: dict[str, tuple[list[bool], list[bool]]] = {}
    for seed, pair in pairs_by_seed.items():
        left, right = [bool(v) for v in pair[0]], [bool(v) for v in pair[1]]
        if not left or len(left) != len(right):
            raise ValueError(f"invalid paired outcomes for seed {seed}")
        normalized[str(seed)] = (left, right)
    if not normalized:
        raise ValueError("at least one seed is required")

    per_seed_effects = {
        seed: (sum(left) - sum(right)) / len(left)
        for seed, (left, right) in normalized.items()
    }
    seed_keys = sorted(normalized)
    rng = random.Random(int(bootstrap_seed))
    bootstrap: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled_seed_effects: list[float] = []
        for _seed_slot in seed_keys:
            seed = seed_keys[rng.randrange(len(seed_keys))]
            left, right = normalized[seed]
            n = len(left)
            indices = [rng.randrange(n) for _ in range(n)]
            sampled_seed_effects.append(
                (
                    sum(left[index] for index in indices)
                    - sum(right[index] for index in indices)
                )
                / n
            )
        bootstrap.append(mean(sampled_seed_effects))
    alpha = 1.0 - confidence
    return {
        **aggregate_seed_effects(
            per_seed_effects,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence=confidence,
        ),
        "hierarchical_bootstrap_ci": [
            _quantile(bootstrap, alpha / 2.0),
            _quantile(bootstrap, 1.0 - alpha / 2.0),
        ],
        "hierarchical_bootstrap_definition": (
            "resample independent seeds, then paired identifiers within each sampled seed"
        ),
    }
