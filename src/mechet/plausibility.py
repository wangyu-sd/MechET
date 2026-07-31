"""Typed external evidence for chemical plausibility ranking.

Formal execution is a hard prerequisite. Missing evidence remains missing; this
module never invents energies, conditions, precedents, or expert support.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PlausibilityEvidence:
    precedent_score: float | None = None
    condition_score: float | None = None
    energy_score: float | None = None
    expert_score: float | None = None
    uncertainty: float | None = None
    sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def combine_evidence(
    evidence: PlausibilityEvidence,
    *,
    weights: Mapping[str, float] | None = None,
) -> float | None:
    weights = dict(weights or {
        "precedent_score": 1.0,
        "condition_score": 1.0,
        "energy_score": 1.0,
        "expert_score": 1.0,
    })
    numerator = denominator = 0.0
    for name, weight in weights.items():
        value = getattr(evidence, name, None)
        if value is None or float(weight) == 0.0:
            continue
        numerator += float(weight) * float(value)
        denominator += abs(float(weight))
    if denominator == 0.0:
        return None
    return numerator / denominator


def load_oracle(spec: str) -> Callable[[dict[str, Any]], PlausibilityEvidence]:
    """Load ``module:function`` returning PlausibilityEvidence or a mapping."""
    if ":" not in spec:
        raise ValueError("oracle spec must be module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)

    def wrapped(payload: dict[str, Any]) -> PlausibilityEvidence:
        value = function(payload)
        if isinstance(value, PlausibilityEvidence):
            return value
        if isinstance(value, Mapping):
            return PlausibilityEvidence(**dict(value))
        raise TypeError("plausibility oracle must return PlausibilityEvidence or mapping")

    return wrapped
