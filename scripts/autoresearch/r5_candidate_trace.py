"""Candidate-conditioned R5 trace execution, with an executor-owned endpoint.

This is an evaluation-only bridge from the existing natural-language action
protocol to TraceOwnedAgentEnv. The proposed precursors are shown to the
policy, but are never passed as the environment's expected precursor or used
to decide whether an electron move is accepted.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from mechet.agent_env import AgentEnvConfig
from mechet.in_place_grounded_flow import map_unmapped_fragment, mapped_atom_numbers
from mechet.natural_language_electron_flow import compile_event_arguments
from mechet.trace_agent_env import TraceOwnedAgentEnv
from scripts.autoresearch.prepare_r3_repair_prompts import mapped_for_inventory
from scripts.autoresearch.stratified_manifest import product_key
from scripts.run_natural_language_value_search import (
    PROMPT_SUFFIX, normal_smiles, policy_prompt, visible,
)


def candidate_conditioned_prompt(product: str, proposed_precursors: str,
                                 mapped_state: str,
                                 accepted_actions: list[dict[str, Any]]) -> str:
    """Render the Stage-II observation plus the external candidate, not a label."""

    prompt = policy_prompt(product, mapped_state, include_inventory=True,
                           actions=accepted_actions, compact_history=True)
    if not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError("R5 Stage-II prompt suffix drifted")
    return (prompt[:-len(PROMPT_SUFFIX)]
            + "\n\nEXTERNAL PRECURSOR PROPOSAL (not a reference answer): "
            + proposed_precursors
            + "\nTest this proposal through explicit inverse electron-flow actions. "
              "It may be wrong; do not claim support unless finish_trace derives "
              "the proposed precursor. The executor owns every state and endpoint."
            + PROMPT_SUFFIX)


class CandidateTraceSession:
    """One model-visible candidate and one private, reference-free executor."""

    def __init__(self, product: str, proposed_precursors: str,
                 *, max_tool_calls: int = 48, max_imports: int = 32) -> None:
        canonical_product = product_key(product)
        canonical_candidate = product_key(proposed_precursors)
        if (product != canonical_product or proposed_precursors != canonical_candidate
                or max_tool_calls < 1 or max_imports < 0):
            raise ValueError("R5 candidate input or budgets are invalid")
        self.model_input = {"product_smiles": product,
                            "proposed_precursors": proposed_precursors}
        self.env = TraceOwnedAgentEnv(config=AgentEnvConfig(
            observation_mode="full_state", max_tool_calls=max_tool_calls))
        mapped = mapped_for_inventory(product)
        self.env.reset(target_smiles=mapped, expected_precursor="")
        self.next_map = max(mapped_atom_numbers(mapped), default=0) + 1
        self.max_imports = max_imports
        self.imported = 0
        self.accepted_actions: list[dict[str, Any]] = []
        self.last_result: dict[str, Any] | None = None
        self.termination_reason: str | None = None

    def prompt(self) -> str:
        if self.termination_reason is not None:
            raise RuntimeError("R5 trace attempt is already terminated")
        return candidate_conditioned_prompt(
            self.model_input["product_smiles"],
            self.model_input["proposed_precursors"],
            self.env.current_state, self.accepted_actions)

    def _import_fragments(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if set(arguments) != {"fragments"} or not isinstance(arguments["fragments"], list):
            raise ValueError("IMPORT_SCHEMA_INVALID")
        fragments = arguments["fragments"]
        if not fragments:
            raise ValueError("IMPORT_LIST_EMPTY")
        expanded: list[str] = []
        for item in fragments:
            if not isinstance(item, dict) or set(item) != {"smiles", "count", "purpose"}:
                raise ValueError("IMPORT_SCHEMA_INVALID")
            count = item["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("IMPORT_COUNT_INVALID")
            if item["purpose"] != "electron_participant":
                # TraceOwnedAgentEnv cannot compile uncommitted endpoint-only
                # imports; never relabel these as a formal trace.
                raise ValueError("ENDPOINT_CONTEXT_IMPORT_UNSUPPORTED")
            expanded.extend([normal_smiles(item["smiles"])] * count)
        if self.imported + len(expanded) > self.max_imports:
            raise ValueError("IMPORT_BUDGET_EXCEEDED")
        mapped: list[str] = []
        next_map = self.next_map
        for fragment in expanded:
            mapped_fragment, next_map = map_unmapped_fragment(fragment, first_map=next_map)
            mapped.append(mapped_fragment)
        for fragment in mapped:
            result = json.loads(self.env.import_fragment(fragment))
            if result.get("ok") is not True:
                return result
        self.next_map = next_map
        self.imported += len(mapped)
        return {"ok": True, "code": "PASS", "current_state": visible(self.env.current_state)}

    def step(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.termination_reason is not None:
            raise RuntimeError("R5 trace attempt is already terminated")
        if not isinstance(arguments, Mapping):
            self.termination_reason = "invalid_action"
            return {"ok": False, "code": "ACTION_ARGUMENTS_INVALID"}
        try:
            if name == "import_fragments":
                result = self._import_fragments(arguments)
            elif name == "apply_electron_flow":
                moves = compile_event_arguments(self.env.current_state, arguments)
                result = json.loads(self.env.apply_coupled_electron_moves(
                    json.dumps(moves, sort_keys=True)))
                if result.get("ok") is True:
                    result = {"ok": True, "code": "PASS",
                              "current_state": visible(self.env.current_state)}
            elif name == "finish_trace":
                if arguments:
                    raise ValueError("FINISH_ARGUMENTS_NOT_EMPTY")
                result = json.loads(self.env.finish_trace())
                if result.get("ok") is True:
                    self.last_result = dict(self.env.final_result)
                    self.termination_reason = "terminal_tool"
            else:
                raise ValueError("UNKNOWN_TOOL")
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            result = {"ok": False, "code": f"{type(exc).__name__}:{exc}"}
        if result.get("ok") is True and name != "finish_trace":
            self.accepted_actions.append({"name": name, "arguments": dict(arguments),
                                          "result": result})
        elif result.get("ok") is not True:
            self.termination_reason = "tool_rejected"
        return result

    def attempt_record(self) -> dict[str, Any]:
        return {"termination_reason": self.termination_reason or "decision_budget",
                "final_result": self.last_result,
                "accepted_action_count": len(self.accepted_actions),
                "imported_fragment_count": self.imported}
