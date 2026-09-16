"""Local-credit helpers for natural-language electron-event post-training.

The policy is Markovian: every generated completion is exactly one tool call
conditioned on the original product and the executor-owned current state.  A
training-only reference prefix may be replayed to obtain an exact reset state,
but neither the hidden suffix nor the expected precursor is put in the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from .in_place_grounded_flow import deterministic_unmapped_state


VERSION = "natural_language_verified_anchor_branch_rl_v1"


@dataclass(frozen=True)
class NaturalLanguageAnchorTask:
    reaction_id: str
    target: str
    anchor_state: str
    expected_precursor: str
    horizon: int
    total_events: int
    prefix_events: int
    state_hash: str

    @property
    def is_full_episode(self) -> bool:
        return self.prefix_events == 0


def task_from_episode(episode: Mapping[str, Any]) -> NaturalLanguageAnchorTask:
    total = int(episode["total_events"])
    horizon = int(episode["horizon"])
    if not 1 <= horizon <= total:
        raise ValueError(f"invalid anchor horizon: {horizon}/{total}")
    state = str(episode["start_state"])
    return NaturalLanguageAnchorTask(
        reaction_id=str(episode["reaction_id"]),
        target=str(episode["target"]),
        anchor_state=state,
        expected_precursor=str(episode["expected_precursor"]),
        horizon=horizon,
        total_events=total,
        prefix_events=total - horizon,
        state_hash=hashlib.sha256(state.encode("utf-8")).hexdigest(),
    )


def successor_fingerprint(
    *,
    prompt_mode: str,
    action_name: str,
    successor_state: str,
    terminal: bool,
    invalid_text: str = "",
) -> str:
    """Pool surface-different actions that reach the same chemical successor."""

    if not successor_state:
        digest = hashlib.sha256(str(invalid_text).encode("utf-8")).hexdigest()[:16]
        return f"{prompt_mode}:INVALID:{digest}"
    visible = deterministic_unmapped_state(successor_state).text
    payload = f"{action_name}:{int(bool(terminal))}:{visible}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prompt_mode}:{digest}"


def assign_local_advantages(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate first-action values without comparing incompatible prompt modes."""

    if not records:
        raise ValueError("empty natural-language anchor group")
    anchors = {(row["id"], row["anchor"]["state_hash"]) for row in records}
    if len(anchors) != 1:
        raise ValueError("anchor group mixes reactions or reset states")
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_mode.setdefault(str(row["prompt_mode"]), []).append(row)

    effective = False
    action_values: dict[str, float] = {}
    for mode, mode_rows in by_mode.items():
        rewards: dict[str, list[float]] = {}
        for row in mode_rows:
            rewards.setdefault(str(row["action_fingerprint"]), []).append(
                float(row["reward"])
            )
        q_values = {
            key: sum(values) / len(values) for key, values in rewards.items()
        }
        baseline = sum(q_values.values()) / len(q_values)
        variance = sum((value - baseline) ** 2 for value in q_values.values()) / max(
            len(q_values) - 1, 1
        )
        scale = math.sqrt(variance)
        mode_effective = len(q_values) > 1 and scale > 1e-8
        effective = effective or mode_effective
        for key, value in q_values.items():
            action_values[f"{mode}:{key}"] = value
        for row in mode_rows:
            q_value = q_values[str(row["action_fingerprint"])]
            row["anchor_action_q"] = q_value
            row["anchor_baseline"] = baseline
            row["advantage"] = (
                (q_value - baseline) / (scale + 1e-4) if mode_effective else 0.0
            )
    return {
        "unique_actions": len(action_values),
        "effective": effective,
        "endpoint_success": any(bool(row["score"]["correct"]) for row in records),
        "candidate_success_rate": sum(
            bool(row["score"]["correct"]) for row in records
        )
        / len(records),
    }


def task_record(task: NaturalLanguageAnchorTask) -> dict[str, Any]:
    return {
        "version": VERSION,
        "state_hash": task.state_hash,
        "horizon": task.horizon,
        "total_steps": task.total_events,
        "prefix_steps": task.prefix_events,
        "is_full_episode": task.is_full_episode,
    }


def stable_rng(seed: int, round_index: int, reaction_id: str) -> random.Random:
    value = f"{int(seed)}:{int(round_index)}:{reaction_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(value).digest()[:8], "big"))
