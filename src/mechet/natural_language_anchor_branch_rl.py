"""Local-credit helpers for natural-language electron-event post-training.

The policy is Markovian: every generated completion is exactly one tool call
conditioned on the original product and the executor-owned current state.  A
training-only reference prefix may be replayed to obtain an exact reset state,
but neither the hidden suffix nor the expected precursor is put in the prompt.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .in_place_grounded_flow import deterministic_unmapped_state


VERSION = "natural_language_verified_anchor_branch_rl_v1"
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


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


@lru_cache(maxsize=32768)
def _endpoint_components(state: str) -> tuple[tuple[str, int, Any], ...]:
    """Return canonical, heavy-atom-weighted component fingerprints.

    The actor never sees this representation.  It is used only by the training
    reward, where the frozen precursor endpoint is already an allowed private
    label.  Atom maps are deliberately removed before comparison.
    """

    raw = str(state)
    try:
        visible = deterministic_unmapped_state(raw).text
    except ValueError:
        # Frozen endpoints and executor observations are already unmapped.  The
        # reward accepts both those public states and private mapped reset states.
        public = Chem.MolFromSmiles(raw)
        if public is None:
            return ()
        for atom in public.GetAtoms():
            atom.SetAtomMapNum(0)
        visible = Chem.MolToSmiles(public, canonical=True)
    mol = Chem.MolFromSmiles(visible)
    if mol is None:
        return ()
    output = []
    for component in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True):
        canonical = Chem.MolToSmiles(component, canonical=True)
        weight = max(int(component.GetNumHeavyAtoms()), 1)
        output.append((canonical, weight, _MORGAN.GetFingerprint(component)))
    return tuple(sorted(output, key=lambda item: (item[0], item[1])))


def endpoint_potential(state: str, expected_precursor: str) -> dict[str, float]:
    """Measure graded progress toward the frozen endpoint without map leakage.

    Large structural components dominate spectator ions and solvents.  Extra
    hallucinated components increase the denominator, so importing arbitrary
    fragments cannot improve the score for free.
    """

    predicted = list(_endpoint_components(str(state)))
    expected = list(_endpoint_components(str(expected_precursor)))
    if not predicted or not expected:
        return {"fingerprint": 0.0, "component_exact": 0.0, "combined": 0.0}

    expected_weight = sum(item[1] for item in expected)
    predicted_weight = sum(item[1] for item in predicted)
    denominator = float(max(expected_weight, predicted_weight, 1))

    # Match larger reference components first.  This keeps the score focused on
    # the synthetic substrate while still accounting for small endpoint context.
    remaining = set(range(len(predicted)))
    fingerprint_mass = 0.0
    for _, weight, fingerprint in sorted(expected, key=lambda item: -item[1]):
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda index: DataStructs.TanimotoSimilarity(
                fingerprint, predicted[index][2]
            ),
        )
        similarity = float(
            DataStructs.TanimotoSimilarity(fingerprint, predicted[best][2])
        )
        fingerprint_mass += float(weight) * similarity
        remaining.remove(best)

    predicted_counts = Counter(item[0] for item in predicted)
    expected_counts = Counter(item[0] for item in expected)
    predicted_weights = {item[0]: item[1] for item in predicted}
    expected_weights = {item[0]: item[1] for item in expected}
    exact_mass = sum(
        min(predicted_counts[key], expected_counts[key])
        * float(max(predicted_weights[key], expected_weights[key]))
        for key in predicted_counts.keys() & expected_counts.keys()
    )
    fingerprint_score = min(max(fingerprint_mass / denominator, 0.0), 1.0)
    exact_score = min(max(exact_mass / denominator, 0.0), 1.0)
    combined = 0.8 * fingerprint_score + 0.2 * exact_score
    return {
        "fingerprint": fingerprint_score,
        "component_exact": exact_score,
        "combined": min(max(combined, 0.0), 1.0),
    }


def endpoint_shaped_reward(
    *,
    correct: bool,
    terminal: bool,
    anchor_state: str,
    first_successor_state: str,
    final_state: str,
    expected_precursor: str,
    invalid_penalty: float,
    wrong_terminal_penalty: float,
    endpoint_similarity_weight: float,
    first_successor_progress_weight: float,
    nonexact_reward_ceiling: float,
) -> dict[str, float | str]:
    """Give dense endpoint credit while keeping exactness the unique success.

    Every non-exact trajectory remains strictly below zero.  Consequently this
    shaping cannot turn a merely executable or chemically similar precursor
    into a benchmark success, but it does let the policy rank useful successors
    when exact endpoint hits are initially sparse.
    """

    anchor = endpoint_potential(anchor_state, expected_precursor)["combined"]
    first = endpoint_potential(
        first_successor_state or anchor_state, expected_precursor
    )["combined"]
    final = endpoint_potential(final_state or first_successor_state or anchor_state,
                               expected_precursor)["combined"]
    progress = first - anchor
    if correct:
        reward = 1.0
        outcome = "exact_endpoint"
    elif terminal:
        reward = (
            -abs(float(wrong_terminal_penalty))
            + float(endpoint_similarity_weight) * final
            + float(first_successor_progress_weight) * progress
        )
        reward = min(reward, -abs(float(nonexact_reward_ceiling)))
        outcome = "wrong_endpoint"
    else:
        reward = (
            -abs(float(invalid_penalty))
            + float(first_successor_progress_weight) * progress
        )
        reward = min(reward, -abs(float(nonexact_reward_ceiling)))
        outcome = "invalid_or_incomplete"
    return {
        "reward": float(reward),
        "outcome": outcome,
        "anchor_potential": float(anchor),
        "first_successor_potential": float(first),
        "final_potential": float(final),
        "first_successor_progress": float(progress),
    }
