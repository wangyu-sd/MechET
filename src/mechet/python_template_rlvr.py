"""Verifier rewards for fixed-template inverse electron programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .electron_flow_trace import ElectronFlowTrace
from .endpoints import reference_structural_precursor, split_precursor_endpoints, structural_exact
from .forward_expert import verify_electron_step
from .proof_program import ProofProgramError
from .python_program import execute_python_program
from .python_template_slots import parse_template_slots


@dataclass(frozen=True)
class TemplateRLVRRewardConfig:
    """Fail-closed staged rewards; only the final term uses the gold endpoint."""

    parse_failure: float = -1.0
    parsed: float = 0.0
    executable_prefix: float = 1.0
    formal_execution: float = 2.0
    structural_precursor: float = 4.0


def _completion_text(value: Any) -> str:
    """Normalize TRL plain or conversational completion representations."""

    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in reversed(value):
            if isinstance(item, Mapping) and str(item.get("role") or "") == "assistant":
                content = item.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, Sequence):
                    return "".join(
                        str(part.get("text") or "")
                        for part in content
                        if isinstance(part, Mapping)
                    )
        if len(value) == 1 and isinstance(value[0], Mapping):
            return str(value[0].get("content") or "")
    return str(value or "")


def score_template_rlvr_candidate(
    row: Mapping[str, Any],
    completion: Any,
    *,
    config: TemplateRLVRRewardConfig | None = None,
) -> dict[str, Any]:
    """Score syntax, executable prefix, full execution, then structural endpoint.

    Prefix credit is normalized by the model-declared step count. This gives
    useful signal when a later step fails without rewarding programs that pad a
    valid first action with many invalid actions.
    """

    cfg = config or TemplateRLVRRewardConfig()
    target = str(row.get("target_smiles") or "")
    text = _completion_text(completion)
    result: dict[str, Any] = {
        "reward": float(cfg.parse_failure),
        "parse_ok": False,
        "executed_steps": 0,
        "declared_steps": 0,
        "executable_prefix_fraction": 0.0,
        "formal_execute": False,
        "structural_precursor_exact": False,
        "precursor_smiles": "",
        "diagnostics": [],
    }
    try:
        program = parse_template_slots(text, target_smiles=target)
    except (ProofProgramError, ValueError, KeyError, TypeError) as exc:
        result["diagnostics"] = [
            {"code": "TEMPLATE_RLVR_PARSE_FAILED", "message": str(exc)}
        ]
        return result

    result["parse_ok"] = True
    result["declared_steps"] = len(program.steps)
    current_state = target
    trace = ElectronFlowTrace(target)
    for index, item in enumerate(program.steps):
        augmented = ".".join((current_state, *item.imports))
        replay = verify_electron_step(augmented, item.moves)
        if not replay.get("ok"):
            result["diagnostics"] = [
                {
                    "code": str(replay.get("code") or "ELECTRON_STEP_FAILED"),
                    "message": f"electron step {index} failed: {replay.get('message', '')}".strip(),
                }
            ]
            break
        next_state = str(replay.get("state_smiles") or "")
        trace.append(
            state_before=current_state,
            state_after=next_state,
            moves=item.moves,
            imports=item.imports,
        )
        current_state = next_state
        result["executed_steps"] += 1

    declared = max(int(result["declared_steps"]), 1)
    prefix_fraction = int(result["executed_steps"]) / declared
    result["executable_prefix_fraction"] = prefix_fraction
    reward = float(cfg.parsed) + float(cfg.executable_prefix) * prefix_fraction

    if int(result["executed_steps"]) == int(result["declared_steps"]):
        execution = execute_python_program(program, target_smiles=target)
        result["formal_execute"] = bool(execution.ok)
        result["diagnostics"] = list(execution.diagnostics)
        if execution.ok:
            reward += float(cfg.formal_execution)
            predicted = str(execution.precursor_smiles or "")
            result["precursor_smiles"] = predicted
            try:
                predicted_structural = split_precursor_endpoints(predicted, target).structural
                expected_structural = reference_structural_precursor(dict(row))
                endpoint_ok = structural_exact(predicted_structural, expected_structural)
                result["structural_precursor_exact"] = bool(endpoint_ok)
                if endpoint_ok:
                    reward += float(cfg.structural_precursor)
            except (ProofProgramError, ValueError, KeyError, TypeError) as exc:
                result["diagnostics"].append(
                    {"code": "STRUCTURAL_ENDPOINT_FAILED", "message": str(exc)}
                )

    result["reward"] = float(reward)
    return result


def template_rlvr_rewards(
    prompts: Sequence[Any],
    completions: Sequence[Any],
    *,
    target_smiles: Sequence[str],
    expected_precursor: Sequence[str],
    **_: Any,
) -> list[float]:
    """TRL-compatible reward function with no teacher-path matching."""

    if not (
        len(prompts)
        == len(completions)
        == len(target_smiles)
        == len(expected_precursor)
    ):
        raise ValueError("RLVR reward columns have inconsistent lengths")
    rewards: list[float] = []
    for completion, target, expected in zip(
        completions, target_smiles, expected_precursor, strict=True
    ):
        scored = score_template_rlvr_candidate(
            {"target_smiles": target, "expected_precursor": expected}, completion
        )
        rewards.append(float(scored["reward"]))
    return rewards
