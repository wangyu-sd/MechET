"""Framework-neutral stateful environment for inverse electron-flow agents.

Training and inference frameworks must wrap this contract rather than expose all
public implementation helpers as model tools. Invalid calls consume tool budget,
and frozen forward experts are cached per process instead of reloaded per rollout.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .endpoints import split_precursor_endpoints
from .forward_expert import (
    ForwardElectronExpert,
    enumerate_containers,
    verify_electron_step,
)
from .forward_rewards import score_inverse_proof_forward
from .proof_program import execute_proof, parse_proof_program, sides_equal


@lru_cache(maxsize=8)
def _cached_forward_expert(checkpoint: str, device: str):
    """Load one frozen forward expert per checkpoint/device/process."""

    return ForwardElectronExpert.load(checkpoint, device=device)


@dataclass(frozen=True)
class AgentEnvConfig:
    """Reward and interaction settings shared by all agent-RL backends."""

    max_tool_calls: int = 12
    formal_success: float = 1.0
    formal_failure: float = -4.0
    endpoint_exact: float = 2.0
    successful_step: float = 0.05
    successful_step_cap: int = 3
    failed_step_penalty: float = 0.25
    tool_use_bonus: float = 0.25
    no_tool_penalty: float = 0.25
    forward_reward_scale: float = 0.5
    abstain_reward: float = -0.25
    unfinished_reward: float = -4.0
    require_tool_use: bool = False
    observation_mode: str = "full_state"
    reaction_center_radius: int = 1


class MechETAgentEnv:
    """Stateful chemistry implementation used behind explicit tool facades."""

    def __init__(
        self,
        *,
        config: AgentEnvConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
    ) -> None:
        self.config = (
            config
            if isinstance(config, AgentEnvConfig)
            else AgentEnvConfig(**dict(config or {}))
        )
        checkpoint = str(forward_checkpoint or "").strip()
        self.forward_model = (
            _cached_forward_expert(checkpoint, str(forward_device))
            if checkpoint
            else None
        )
        self._clear()

    def _clear(self) -> None:
        self.target_smiles = ""
        self.expected_precursor = ""
        self.competitor_products: list[str] = []
        self.conditions: Any = None
        self.current_state = ""
        self.tool_calls = 0
        self.successful_steps = 0
        self.failed_steps = 0
        self.finalized = False
        self.abstained = False
        self.reward = self.config.unfinished_reward
        self.trace: list[dict[str, Any]] = []
        self.visited_states: set[str] = set()
        self.final_result: dict[str, Any] = {}

    @staticmethod
    def _decode_list(value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            if isinstance(decoded, list):
                return [str(item) for item in decoded if item]
            return [str(decoded)]
        if isinstance(value, (tuple, list)):
            return [str(item) for item in value if item]
        return [str(value)]

    @staticmethod
    def _target_from_prompt(prompt: Any) -> str:
        if isinstance(prompt, list):
            text = "\n".join(
                str(item.get("content") or "")
                for item in prompt
                if isinstance(item, dict)
            )
        else:
            text = str(prompt or "")
        for line in text.splitlines():
            if line.strip().upper().startswith("TARGET:"):
                return line.split(":", 1)[1].strip()
        return ""

    def reset(
        self,
        target_smiles: str = "",
        expected_precursor: str = "",
        competitor_products: Any = None,
        conditions: Any = None,
        prompt: Any = None,
        **kwargs: Any,
    ) -> str:
        """Reset one rollout and return the initial chemistry observation."""

        del kwargs
        self._clear()
        self.target_smiles = str(target_smiles or "").strip() or self._target_from_prompt(
            prompt
        )
        if not self.target_smiles:
            raise ValueError("reset requires an atom-mapped target_smiles")
        self.expected_precursor = str(expected_precursor or "").strip()
        self.competitor_products = self._decode_list(competitor_products)
        self.conditions = conditions
        self.current_state = self.target_smiles
        self.visited_states.add(self.current_state)
        observation = {
            "task": "inverse_electron_flow",
            "target_smiles": self.target_smiles,
            "max_tool_calls": self.config.max_tool_calls,
            "instructions": [
                "Use inspect_state before inventing atom maps.",
                "Use explicit electron-flow tools for each claimed state change.",
                "Use the terminal operation defined by the selected experiment.",
                "Use abstain when chemical support is insufficient.",
            ],
        }
        self.trace.append({"event": "reset", "observation": observation})
        return json.dumps(observation, ensure_ascii=False)

    def _consume_call(self) -> None:
        if self.finalized:
            raise RuntimeError("episode is already finalized")
        self.tool_calls += 1
        if self.tool_calls > self.config.max_tool_calls:
            self.finalized = True
            self.reward = self.config.formal_failure
            raise RuntimeError("TOOL_BUDGET_EXCEEDED")

    def _invalid_call(self, event: str, code: str, message: str = "") -> str:
        """Consume budget and record malformed calls consistently."""

        try:
            self._consume_call()
        except Exception as exc:
            result = {
                "ok": False,
                "code": "TOOL_BUDGET_EXCEEDED",
                "message": str(exc),
                "remaining_tool_calls": 0,
            }
            self.trace.append({"event": event, "result": result})
            return json.dumps(result, ensure_ascii=False)
        self.failed_steps += 1
        result = {
            "ok": False,
            "code": code,
            "message": message or code,
            "remaining_tool_calls": max(self.config.max_tool_calls - self.tool_calls, 0),
        }
        self.trace.append({"event": event, "result": result})
        return json.dumps(result, ensure_ascii=False)

    def inspect_state(self) -> str:
        self._consume_call()
        try:
            sources, sinks = enumerate_containers(self.current_state)
            result = {
                "ok": True,
                "state_smiles": self.current_state,
                "sources": [item.to_dict() for item in sources],
                "sinks": [item.to_dict() for item in sinks],
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
        except Exception as exc:
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "STATE_INSPECTION_FAILED",
                "message": str(exc),
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
        self.trace.append({"event": "inspect_state", "result": result})
        return json.dumps(result, ensure_ascii=False)

    def _apply_moves(self, moves: list[dict[str, Any]]) -> str:
        self._consume_call()
        result = verify_electron_step(self.current_state, moves)
        if result.get("ok"):
            next_state = str(result.get("state_smiles") or "")
            if not next_state or next_state in self.visited_states:
                self.failed_steps += 1
                result = {
                    "ok": False,
                    "code": "STATE_CYCLE",
                    "message": "the move returns to a previously visited molecular state",
                }
            else:
                self.current_state = next_state
                self.visited_states.add(next_state)
                self.successful_steps += 1
        else:
            self.failed_steps += 1
        result["remaining_tool_calls"] = self.config.max_tool_calls - self.tool_calls
        self.trace.append({"event": "apply_moves", "moves": moves, "result": result})
        return json.dumps(result, ensure_ascii=False)

    def apply_electron_move(
        self,
        source_kind: str,
        source_atoms: list[int],
        sink_kind: str,
        sink_atoms: list[int],
    ) -> str:
        return self._apply_moves(
            [
                {
                    "source": {"kind": source_kind, "atoms": source_atoms},
                    "sink": {"kind": sink_kind, "atoms": sink_atoms},
                    "electrons": 2,
                }
            ]
        )

    def apply_coupled_electron_moves(self, moves_json: str) -> str:
        """Execute coupled arrows; malformed calls still consume tool budget."""

        try:
            moves = json.loads(moves_json)
        except (json.JSONDecodeError, TypeError) as exc:
            return self._invalid_call(
                "apply_moves", "MOVE_JSON_INVALID", str(exc)
            )
        if not isinstance(moves, list) or not moves:
            return self._invalid_call("apply_moves", "MOVE_LIST_EMPTY")
        if any(not isinstance(item, dict) for item in moves):
            return self._invalid_call("apply_moves", "MOVE_LIST_INVALID")
        return self._apply_moves([dict(item) for item in moves])

    def submit_proof(self, proof: str) -> str:
        self._consume_call()
        output: dict[str, Any]
        try:
            program = parse_proof_program(proof)
            if not sides_equal(
                program.target_smiles, self.target_smiles, ignore_maps=True
            ):
                raise ValueError("PROOF_TARGET_MISMATCH")
            execution = execute_proof(program)
            if not execution.ok:
                output = {
                    "formal_execute": False,
                    "diagnostics": execution.diagnostics,
                }
                terminal_reward = self.config.formal_failure
            else:
                endpoint_exact = bool(
                    self.expected_precursor
                    and sides_equal(
                        execution.precursor_smiles,
                        self.expected_precursor,
                        ignore_maps=True,
                    )
                )
                endpoints = split_precursor_endpoints(
                    execution.precursor_smiles, self.target_smiles
                )
                output = {
                    "formal_execute": True,
                    "derived_precursor": execution.precursor_smiles,
                    **endpoints.to_dict(),
                    "endpoint_exact": endpoint_exact,
                }
                terminal_reward = self.config.formal_success
                if endpoint_exact:
                    terminal_reward += self.config.endpoint_exact
                if self.forward_model is not None:
                    forward = score_inverse_proof_forward(
                        self.forward_model,
                        proof,
                        competitor_products=self.competitor_products,
                        conditions=self.conditions,
                    )
                    output["forward"] = forward
                    terminal_reward += self.config.forward_reward_scale * float(
                        forward.get("forward_reward") or 0.0
                    )
        except Exception as exc:
            output = {
                "formal_execute": False,
                "diagnostics": [
                    {"code": "PROOF_SUBMISSION_FAILED", "message": str(exc)}
                ],
            }
            terminal_reward = self.config.formal_failure

        process_reward = (
            self.config.successful_step
            * min(self.successful_steps, self.config.successful_step_cap)
            - self.config.failed_step_penalty * self.failed_steps
        )
        if self.successful_steps:
            process_reward += self.config.tool_use_bonus
        elif self.config.require_tool_use:
            process_reward -= self.config.no_tool_penalty
        self.reward = float(terminal_reward + process_reward)
        self.finalized = True
        output["reward"] = self.reward
        output["tool_calls"] = self.tool_calls
        output["successful_steps"] = self.successful_steps
        output["failed_steps"] = self.failed_steps
        self.final_result = output
        self.trace.append({"event": "submit_proof", "result": output})
        return json.dumps(output, ensure_ascii=False)

    def _submit_preexecuted_proof(
        self, *, proof: str, precursor_smiles: str
    ) -> str:
        """Finalize a proof already executed by a trusted environment compiler.

        This is intentionally not exposed as a model tool.  Trace-owned
        environments call it only after ``compile_trace_to_proof`` has executed
        the generated program and checked every intermediate state.
        """

        self._consume_call()
        endpoint_exact = bool(
            self.expected_precursor
            and sides_equal(
                precursor_smiles,
                self.expected_precursor,
                ignore_maps=True,
            )
        )
        endpoints = split_precursor_endpoints(
            precursor_smiles, self.target_smiles
        )
        output: dict[str, Any] = {
            "formal_execute": True,
            "derived_precursor": precursor_smiles,
            **endpoints.to_dict(),
            "endpoint_exact": endpoint_exact,
        }
        terminal_reward = self.config.formal_success
        if endpoint_exact:
            terminal_reward += self.config.endpoint_exact
        if self.forward_model is not None:
            forward = score_inverse_proof_forward(
                self.forward_model,
                proof,
                competitor_products=self.competitor_products,
                conditions=self.conditions,
            )
            output["forward"] = forward
            terminal_reward += self.config.forward_reward_scale * float(
                forward.get("forward_reward") or 0.0
            )
        process_reward = (
            self.config.successful_step
            * min(self.successful_steps, self.config.successful_step_cap)
            - self.config.failed_step_penalty * self.failed_steps
        )
        if self.successful_steps:
            process_reward += self.config.tool_use_bonus
        elif self.config.require_tool_use:
            process_reward -= self.config.no_tool_penalty
        self.reward = float(terminal_reward + process_reward)
        self.finalized = True
        output["reward"] = self.reward
        output["tool_calls"] = self.tool_calls
        output["successful_steps"] = self.successful_steps
        output["failed_steps"] = self.failed_steps
        self.final_result = output
        self.trace.append({"event": "submit_proof", "result": output})
        return json.dumps(output, ensure_ascii=False)

    def abstain(self, reason: str) -> str:
        self._consume_call()
        self.finalized = True
        self.abstained = True
        self.reward = self.config.abstain_reward
        self.final_result = {
            "abstained": True,
            "reason": reason,
            "reward": self.reward,
        }
        self.trace.append({"event": "abstain", "result": self.final_result})
        return json.dumps(self.final_result, ensure_ascii=False)

    def get_reward(self) -> float:
        return float(self.reward if self.finalized else self.config.unfinished_reward)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "target_smiles": self.target_smiles,
            "expected_precursor": self.expected_precursor,
            "current_state": self.current_state,
            "tool_calls": self.tool_calls,
            "successful_steps": self.successful_steps,
            "failed_steps": self.failed_steps,
            "finalized": self.finalized,
            "abstained": self.abstained,
            "reward": self.reward,
            "final_result": self.final_result,
            "trace": self.trace,
        }
