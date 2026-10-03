"""Parse and execute an OPEN_FLOW v1 program only after full generation.

The model receives no intermediate observations. Parsing/execution is an
offline evaluation operation, never an in-loop repair or gold-guided search.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .agent_env import AgentEnvConfig
from .trace_agent_env import TraceOwnedAgentEnv


class OpenFlowFormatError(ValueError):
    pass


_STEP = re.compile(r"^STEP ([0-9]+) (\[.*\])$")


def parse_open_flow(text: str, *, max_characters: int = 20000) -> tuple[list[str], list[list[dict[str, Any]]]]:
    if len(text) > max_characters:
        raise OpenFlowFormatError("FLOW_TOO_LONG")
    if text.count("<flow>") != 1 or text.count("</flow>") != 1:
        raise OpenFlowFormatError("FLOW_TAGS_INVALID")
    body = text.split("<flow>", 1)[1].split("</flow>", 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) < 3 or lines[0] != "OPEN_FLOW v1" or lines[-1] != "EXECUTE":
        raise OpenFlowFormatError("FLOW_HEADER_OR_TERMINAL_INVALID")
    imports: list[str] = []
    steps: list[list[dict[str, Any]]] = []
    seen_step = False
    for line in lines[1:-1]:
        if line.startswith("IMPORT "):
            if seen_step:
                raise OpenFlowFormatError("IMPORT_AFTER_STEP")
            fragment = line.removeprefix("IMPORT ").strip()
            if not fragment:
                raise OpenFlowFormatError("EMPTY_IMPORT")
            imports.append(fragment)
            continue
        match = _STEP.fullmatch(line)
        if match is None:
            raise OpenFlowFormatError("FLOW_LINE_INVALID")
        seen_step = True
        if int(match.group(1)) != len(steps):
            raise OpenFlowFormatError("STEP_INDEX_INVALID")
        try:
            moves = json.loads(match.group(2))
        except json.JSONDecodeError as exc:
            raise OpenFlowFormatError("STEP_JSON_INVALID") from exc
        if not isinstance(moves, list) or not moves or any(not isinstance(item, dict) for item in moves):
            raise OpenFlowFormatError("STEP_MOVES_INVALID")
        steps.append(moves)
    if not steps:
        raise OpenFlowFormatError("NO_STEPS")
    return imports, steps


def execute_open_flow(
    text: str, target_smiles: str, *, max_tool_calls: int = 40,
) -> dict[str, Any]:
    """Return the gold-independent formal result of the complete program."""
    try:
        imports, steps = parse_open_flow(text)
    except OpenFlowFormatError as exc:
        return {"execute_ok": False, "failure_code": str(exc), "derived_precursor": ""}
    if len(imports) + len(steps) + 1 > max_tool_calls:
        return {"execute_ok": False, "failure_code": "TOOL_BUDGET_EXCEEDED", "derived_precursor": ""}
    try:
        env = TraceOwnedAgentEnv(config=AgentEnvConfig(
            max_tool_calls=max_tool_calls, observation_mode="action_delta",
        ))
        env.reset(target_smiles=target_smiles)
        for fragment in imports:
            result = json.loads(env.import_fragment(fragment))
            if not result.get("ok"):
                return {"execute_ok": False, "failure_code": "IMPORT_FAILED",
                        "detail": result.get("message"), "derived_precursor": ""}
        for index, moves in enumerate(steps):
            result = json.loads(env.apply_coupled_electron_moves(json.dumps(moves)))
            if not result.get("ok"):
                return {"execute_ok": False, "failure_code": str(result.get("code") or "MOVE_FAILED"),
                        "failed_step": index, "derived_precursor": ""}
        terminal = json.loads(env.finish_trace())
        if not terminal.get("ok"):
            return {"execute_ok": False,
                    "failure_code": str(terminal.get("code") or "FINISH_FAILED"),
                    "derived_precursor": ""}
        return {
            "execute_ok": True,
            "failure_code": "",
            "derived_precursor": str(env.final_result.get("derived_precursor") or ""),
            "n_imports": len(imports),
            "n_steps": len(steps),
            "trace_digest": str(env.final_result.get("trace_digest") or ""),
        }
    except Exception as exc:
        return {"execute_ok": False, "failure_code": "EXECUTION_EXCEPTION",
                "detail": str(exc), "derived_precursor": ""}
