"""Deterministic mixed-horizon curriculum for endpoint process RLVR."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Mapping


PHASE_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "A": {"product": 0.20, "middle": 0.30, "near_end": 0.50},
    "B": {"product": 0.50, "middle": 0.25, "near_end": 0.25},
    "C": {"product": 0.80, "middle": 0.10, "near_end": 0.10},
}


@dataclass(frozen=True)
class HorizonStart:
    phase: str
    horizon: str
    prefix_events: int
    remaining_gold_events: int


class MixedHorizonSampler:
    def __init__(self, *, seed: int = 17, phase: str = "A") -> None:
        if phase not in PHASE_WEIGHTS:
            raise ValueError(f"unknown curriculum phase: {phase}")
        self.seed = int(seed)
        self.phase = phase

    def set_phase(self, phase: str) -> None:
        if phase not in PHASE_WEIGHTS:
            raise ValueError(f"unknown curriculum phase: {phase}")
        self.phase = phase

    def sample(
        self, *, identifier: str, n_events: int, draw_index: int
    ) -> HorizonStart:
        if n_events < 1:
            raise ValueError("mixed-horizon rows require at least one event")
        digest = hashlib.sha256(
            f"{self.seed}\0{self.phase}\0{identifier}\0{draw_index}".encode()
        ).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        value = rng.random()
        cumulative = 0.0
        horizon = "near_end"
        for name, weight in PHASE_WEIGHTS[self.phase].items():
            cumulative += weight
            if value < cumulative:
                horizon = name
                break
        if horizon == "product":
            prefix = 0
        elif horizon == "middle":
            prefix = (
                rng.randint(1, n_events - 2)
                if n_events >= 3
                else max(0, n_events - 1)
            )
        else:
            suffix = rng.choice((1, 2))
            prefix = max(0, n_events - suffix)
        return HorizonStart(
            phase=self.phase,
            horizon=horizon,
            prefix_events=prefix,
            remaining_gold_events=n_events - prefix,
        )


@dataclass
class CurriculumController:
    """Advance one continuous run using frozen promotion observations."""

    phase: str = "A"
    near_end_threshold: float = 0.60
    stable_product_checks: int = 2
    _positive_product_checks: int = 0

    def observe(
        self,
        *,
        near_end_exact_rate: float | None = None,
        product_monitor_exact_rate: float | None = None,
    ) -> str:
        if self.phase == "A" and near_end_exact_rate is not None:
            if near_end_exact_rate >= self.near_end_threshold:
                self.phase = "B"
        if self.phase == "B" and product_monitor_exact_rate is not None:
            if product_monitor_exact_rate > 0:
                self._positive_product_checks += 1
            else:
                self._positive_product_checks = 0
            if self._positive_product_checks >= self.stable_product_checks:
                self.phase = "C"
        return self.phase
