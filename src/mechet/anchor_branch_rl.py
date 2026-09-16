"""Verified, same-state branching for long-horizon electron-program RL.

The environment may use a reference program to *reset* training episodes to an
intermediate state.  Reference suffix actions are never included in an RL
prompt or an RL reward.  Candidate suffixes are sampled from the same chemical
state, replayed by the deterministic executor, and receive credit only on the
tokens that select their first electron step.

This is deliberately separate from the historical whole-trajectory GRPO code:
the latter broadcasts one scalar over every generated action, whereas this
module estimates a local action value at an exact executor state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Any, Mapping, Sequence

from .forward_expert import verify_electron_step
from .endpoints import split_precursor_endpoints, structural_exact
from .python_continual import supervised_record
from .python_program import PythonElectronProgram
from .proof_program import sides_equal
from .python_template_slots import (
    execute_template_slots,
    format_template_slots,
    parse_template_slots,
)


ANCHOR_BRANCH_VERSION = "verified_anchor_branch_rl_v1"

ANCHOR_SYSTEM = (
    "Continue inverse electron flow from an executor-verified CURRENT_STATE. "
    "Output only one complete STEPS list, beginning with the next electron "
    "step. The executor derives and verifies the precursor endpoint. Do not "
    "output prose, Markdown, a precursor answer, or the reference trajectory."
)


@dataclass(frozen=True)
class AnchorTask:
    """One training-only suffix problem rooted at an exact executor state."""

    reaction_id: str
    original_target: str
    anchor_state: str
    expected_precursor: str
    horizon: int
    total_steps: int
    prefix_steps: int
    gold_suffix: str
    state_hash: str

    @property
    def is_full_episode(self) -> bool:
        return self.prefix_steps == 0


def _assistant_text(row: Mapping[str, Any]) -> str:
    answers = [
        str(message.get("content") or "")
        for message in row.get("messages") or ()
        if str(message.get("role") or "") == "assistant"
    ]
    if len(answers) != 1 or not answers[0]:
        raise ValueError(f"{row.get('id')}: expected exactly one assistant program")
    return answers[0]


def reference_states(row: Mapping[str, Any]) -> tuple[PythonElectronProgram, tuple[str, ...]]:
    """Replay the frozen reference into resettable states, without filtering."""

    target = str(row.get("target_smiles") or "")
    if not target:
        raise ValueError(f"{row.get('id')}: empty target_smiles")
    program = parse_template_slots(_assistant_text(row), target_smiles=target)
    states = [target]
    state = target
    for index, step in enumerate(program.steps):
        augmented = ".".join((state, *step.imports))
        result = verify_electron_step(augmented, step.moves)
        if not result.get("ok"):
            raise ValueError(
                f"{row.get('id')}: reference step {index} failed: "
                f"{result.get('code')} {result.get('message')}"
            )
        state = str(result["state_smiles"])
        states.append(state)
    return program, tuple(states)


def build_anchor_task(row: Mapping[str, Any], horizon: int) -> AnchorTask:
    """Create a suffix task ``horizon`` steps from the reference endpoint."""

    program, states = reference_states(row)
    total = len(program.steps)
    if total < 1:
        raise ValueError(f"{row.get('id')}: reference program has no steps")
    horizon = max(1, min(int(horizon), total))
    prefix = total - horizon
    anchor = states[prefix]
    suffix = PythonElectronProgram(anchor, program.steps[prefix:])
    digest = hashlib.sha256(anchor.encode("utf-8")).hexdigest()
    return AnchorTask(
        reaction_id=str(row.get("id") or row.get("source_id") or ""),
        original_target=str(row["target_smiles"]),
        anchor_state=anchor,
        expected_precursor=str(row["expected_precursor"]),
        horizon=horizon,
        total_steps=total,
        prefix_steps=prefix,
        gold_suffix=format_template_slots(suffix),
        state_hash=digest,
    )


def choose_horizon(
    total_steps: int,
    frontier: int,
    rng: random.Random,
    *,
    full_episode_fraction: float = 0.2,
) -> int:
    """Sample near the current competence frontier, with full-task rehearsal."""

    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= full_episode_fraction <= 1.0:
        raise ValueError("full_episode_fraction must be in [0, 1]")
    if rng.random() < full_episode_fraction:
        return total_steps
    center = max(1, min(int(frontier), total_steps))
    band = [max(1, center - 1), center, min(total_steps, center + 1)]
    # Preserve weights after clipping at the ends of the trajectory.
    merged: dict[int, float] = {}
    for value, weight in zip(band, (0.2, 0.6, 0.2), strict=True):
        merged[value] = merged.get(value, 0.0) + weight
    draw = rng.random() * sum(merged.values())
    cumulative = 0.0
    for value, weight in sorted(merged.items()):
        cumulative += weight
        if draw <= cumulative:
            return value
    return max(merged)


def anchor_messages(task: AnchorTask) -> list[dict[str, str]]:
    """Gold-free prompt for an intermediate reset state."""

    return [
        {"role": "system", "content": ANCHOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"ORIGINAL_PRODUCT: {task.original_target}\n"
                f"CURRENT_STATE: {task.anchor_state}\n"
                "Complete the remaining inverse electron-flow steps."
            ),
        },
    ]


def first_step_fingerprint(text: str, *, target_smiles: str) -> str:
    """Canonical first action used to pool continuation noise."""

    try:
        program = parse_template_slots(text, target_smiles=target_smiles)
        first = PythonElectronProgram(target_smiles, (program.steps[0],))
        return format_template_slots(first)
    except (ValueError, KeyError, TypeError):
        return "__INVALID_FIRST_STEP__"


def _first_step_character_end(text: str) -> int | None:
    """Return the exclusive character end of the first ``step(...)`` call."""

    start = text.find("step(")
    if start < 0:
        return None
    depth = 0
    quote = ""
    escaped = False
    for index in range(start + len("step"), len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def first_step_token_mask(tokenizer: Any, token_ids: Sequence[int], text: str) -> list[int]:
    """Mask only the sampled prefix through the first complete electron step.

    Invalid generations retain a full completion mask so syntax failures can
    receive a small negative constraint signal.  Valid suffix tokens after the
    first action are context for its endpoint value, not optimization targets.
    """

    ids = list(map(int, token_ids))
    end = _first_step_character_end(text)
    if end is None:
        return [1] * len(ids)
    stop = len(ids)
    for index in range(1, len(ids) + 1):
        decoded = tokenizer.decode(ids[:index], skip_special_tokens=True)
        if len(decoded) >= end:
            stop = index
            break
    return [1 if index < stop else 0 for index in range(len(ids))]


def score_completion(
    task: AnchorTask,
    text: str,
    *,
    terminated: bool,
    invalid_penalty: float = 0.1,
) -> dict[str, Any]:
    """Score a suffix without exposing the expected endpoint to the policy."""

    formal_execute = False
    correct = False
    noop = False
    precursor = ""
    diagnostics: list[dict[str, Any]] = []
    if not terminated:
        diagnostics = [{"code": "TRUNCATED_SUFFIX", "message": "generation did not terminate"}]
    else:
        execution = execute_template_slots(text, target_smiles=task.anchor_state)
        formal_execute = bool(execution.ok)
        precursor = str(execution.precursor_smiles or "")
        diagnostics = list(execution.diagnostics)
        if formal_execute:
            try:
                # Structural identity must remain defined by the original
                # product maps. The anchor can contain already imported atoms;
                # treating those maps as product atoms creates false endpoint
                # matches and false no-op diagnoses.
                predicted_structural = split_precursor_endpoints(
                    precursor, task.original_target
                ).structural
                expected_structural = split_precursor_endpoints(
                    task.expected_precursor, task.original_target
                ).structural
                structural_match = structural_exact(
                    predicted_structural, expected_structural
                )
                full_match = sides_equal(
                    precursor, task.expected_precursor, ignore_maps=True
                )
                reference_is_structural_noop = structural_exact(
                    expected_structural, task.original_target
                )
                # Some frozen references change only auxiliary/protonation
                # state while retaining the product-connected scaffold. Those
                # rows are learnable only with a full-endpoint discriminator;
                # a bare no-op still fails because it cannot match that state.
                correct = full_match if reference_is_structural_noop else structural_match
                noop = (
                    structural_exact(predicted_structural, task.original_target)
                    and not full_match
                )
            except (ValueError, KeyError, TypeError) as exc:
                diagnostics.append(
                    {"code": "ANCHOR_ENDPOINT_FAILED", "message": str(exc)}
                )
    correct = bool(formal_execute and correct and not noop)
    if correct:
        reward = 1.0
    elif formal_execute and not noop:
        reward = 0.0
    else:
        reward = -abs(float(invalid_penalty))
    return {
        "feedback": {"ok": formal_execute, "diagnostics": diagnostics},
        "formal_execute": formal_execute,
        "correct": correct,
        "noop": noop,
        "precursor_smiles": precursor,
        "effective_program": text,
        "tool_repairs": [],
        "reward": reward,
        "action_fingerprint": first_step_fingerprint(
            text, target_smiles=task.anchor_state
        ),
    }


def assign_anchor_advantages(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Estimate local advantages from alternative actions at one exact state."""

    if not records:
        raise ValueError("empty anchor group")
    keys = {(row["id"], row["anchor"]["state_hash"]) for row in records}
    if len(keys) != 1:
        raise ValueError("anchor group mixes reaction IDs or executor states")
    grouped: dict[str, list[float]] = {}
    for row in records:
        grouped.setdefault(row["action_fingerprint"], []).append(float(row["reward"]))
    action_q = {key: sum(values) / len(values) for key, values in grouped.items()}
    values = list(action_q.values())
    baseline = sum(values) / len(values)
    variance = sum((value - baseline) ** 2 for value in values) / max(len(values) - 1, 1)
    scale = math.sqrt(variance)
    effective = len(action_q) > 1 and scale > 1e-8
    for row in records:
        q_value = action_q[row["action_fingerprint"]]
        row["anchor_action_q"] = q_value
        row["anchor_baseline"] = baseline
        row["advantage"] = (q_value - baseline) / (scale + 1e-4) if effective else 0.0
    return {
        "unique_actions": len(action_q),
        "effective": effective,
        "baseline": baseline,
        "endpoint_success": any(bool(row["score"]["correct"]) for row in records),
        "candidate_success_rate": sum(bool(row["score"]["correct"]) for row in records)
        / len(records),
    }


def update_frontier(
    frontier: int,
    summaries: Sequence[Mapping[str, Any]],
    *,
    promote_pass_rate: float = 0.55,
    min_effective_groups: int = 32,
    maximum: int = 64,
) -> tuple[int, dict[str, Any]]:
    """Promote only after enough informative same-state comparisons."""

    relevant = [row for row in summaries if not bool(row.get("is_full_episode"))]
    effective = [row for row in relevant if bool(row.get("effective"))]
    pass_rate = (
        sum(bool(row.get("endpoint_success")) for row in relevant) / len(relevant)
        if relevant
        else 0.0
    )
    promoted = len(effective) >= int(min_effective_groups) and pass_rate >= float(
        promote_pass_rate
    )
    next_frontier = min(int(maximum), int(frontier) + int(promoted))
    return next_frontier, {
        "frontier_before": int(frontier),
        "frontier_after": next_frontier,
        "groups": len(relevant),
        "effective_groups": len(effective),
        "pass_at_group": pass_rate,
        "promoted": promoted,
    }


def anchor_supervised_record(tokenizer: Any, prompt_ids: Sequence[int], task: AnchorTask) -> dict[str, Any]:
    """Optional verified rehearsal; never used as an RL reward or observation."""

    row = supervised_record(
        tokenizer,
        list(prompt_ids),
        task.gold_suffix,
        task.reaction_id,
        "anchor_reference_replay",
    )
    row["anchor"] = {
        "version": ANCHOR_BRANCH_VERSION,
        "state_hash": task.state_hash,
        "horizon": task.horizon,
        "prefix_steps": task.prefix_steps,
        "is_full_episode": task.is_full_episode,
    }
    return row


def task_record(task: AnchorTask) -> dict[str, Any]:
    """Serializable non-oracle anchor metadata for rollout artifacts."""

    return {
        "version": ANCHOR_BRANCH_VERSION,
        "state_hash": task.state_hash,
        "horizon": task.horizon,
        "total_steps": task.total_steps,
        "prefix_steps": task.prefix_steps,
        "is_full_episode": task.is_full_episode,
    }


def stable_row_rng(seed: int, round_index: int, reaction_id: str) -> random.Random:
    material = f"{int(seed)}:{int(round_index)}:{reaction_id}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def dumps_summary(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
