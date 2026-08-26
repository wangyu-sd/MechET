"""Faithful inverse-agent environment whose trace uniquely determines proof."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from .agent_env import AgentEnvConfig, MechETAgentEnv
from .compact_observation import (
    compact_failure_observation,
    compact_terminal_observation,
    compact_transition_observation,
)
from .electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from .proof_program import _canonical_mapped, _combine_smiles


class TraceOwnedAgentEnv(MechETAgentEnv):
    """Stateful inverse environment with causal trace-to-proof binding."""

    def __init__(
        self,
        *,
        config: AgentEnvConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
    ) -> None:
        super().__init__(
            config=config,
            forward_checkpoint=forward_checkpoint,
            forward_device=forward_device,
        )

    def _clear(self) -> None:
        super()._clear()
        self.flow_trace = ElectronFlowTrace("")
        self.last_committed_state = ""
        self.pending_imports: list[str] = []

    def reset(self, *args: Any, **kwargs: Any) -> str:
        if self.config.observation_mode not in {
            "full_state",
            "action_delta",
            "reaction_center_delta",
        }:
            raise ValueError(
                f"unsupported observation_mode: {self.config.observation_mode}"
            )
        observation = json.loads(super().reset(*args, **kwargs))
        self.flow_trace = ElectronFlowTrace(self.target_smiles)
        self.last_committed_state = self.target_smiles
        self.pending_imports = []
        observation["task"] = "trace_owned_inverse_electron_flow"
        observation["faithfulness_contract"] = {
            "free_form_proof_submission": False,
            "endpoint_source": "environment_owned_trace",
            "final_tool": "finish_trace",
            "declared_moves_replayed_before_compilation": True,
            "observation_mode": self.config.observation_mode,
        }
        inspect_instruction = (
            "Use inspect_state before referencing atom maps."
            if self.config.observation_mode == "full_state"
            else (
                "Atom maps come from TARGET and imported fragments; inspect_state "
                "returns legal-action inventory without serializing the current state."
            )
        )
        observation["instructions"] = [
            inspect_instruction,
            "Use import_fragment when a required mapped precursor fragment is absent.",
            "Use explicit electron-flow actions for every claimed transition.",
            "Call finish_trace; the environment replays moves and compiles MECH_PROOF v1.",
            "Free-form submit_proof is disabled in this environment.",
            "Use abstain when chemical support is insufficient.",
        ]
        self.trace[-1]["observation"] = observation
        return json.dumps(observation, ensure_ascii=False)

    def inspect_state(self) -> str:
        """Inspect legal containers without leaking compact-mode state SMILES."""

        raw = json.loads(super().inspect_state())
        if self.config.observation_mode == "full_state" or not raw.get("ok"):
            return json.dumps(raw, ensure_ascii=False)
        raw.pop("state_smiles", None)
        raw["observation_mode"] = f"{self.config.observation_mode}_v1"
        return json.dumps(raw, ensure_ascii=False)

    @staticmethod
    def _atom_maps(smiles: str) -> set[int]:
        params = Chem.SmilesParserParams()
        params.removeHs = False
        mol = Chem.MolFromSmiles(str(smiles or ""), params)
        if mol is None:
            raise ValueError("IMPORT_SMILES_INVALID")
        maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        if any(value <= 0 for value in maps) or len(maps) != len(set(maps)):
            raise ValueError("IMPORT_REQUIRES_UNIQUE_POSITIVE_MAPS")
        return set(maps)

    def import_fragment(self, fragment_smiles: str) -> str:
        self._consume_call()
        try:
            fragment = _canonical_mapped(fragment_smiles)
            existing = self._atom_maps(self.current_state)
            incoming = self._atom_maps(fragment)
            overlap = sorted(existing & incoming)
            if overlap:
                raise ValueError(f"IMPORT_ATOM_MAP_COLLISION: {overlap}")
            combined = _canonical_mapped(
                _combine_smiles(self.current_state, [fragment])
            )
            self.current_state = combined
            self.pending_imports.append(fragment)
            result = {
                "ok": True,
                "state_smiles": self.current_state,
                "imported_fragment": fragment,
                "pending_imports": list(self.pending_imports),
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
        except Exception as exc:
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "IMPORT_FAILED",
                "message": str(exc),
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
        self.trace.append({"event": "import_fragment", "result": result})
        if self.config.observation_mode != "full_state":
            if not result.get("ok"):
                return json.dumps(
                    compact_failure_observation(
                        result, observation_mode=self.config.observation_mode
                    ),
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "ok": True,
                    "observation_mode": f"{self.config.observation_mode}_v1",
                    "pending_import_count": len(self.pending_imports),
                    "remaining_tool_calls": result["remaining_tool_calls"],
                },
                ensure_ascii=False,
            )
        return json.dumps(result, ensure_ascii=False)

    def _apply_moves(self, moves: list[dict[str, Any]]) -> str:
        state_before = self.last_committed_state
        execution_state_before = self.current_state
        imports = tuple(self.pending_imports)
        result = json.loads(super()._apply_moves(moves))
        if result.get("ok"):
            transition = self.flow_trace.append(
                state_before=state_before,
                state_after=self.current_state,
                moves=moves,
                imports=imports,
            )
            self.last_committed_state = self.current_state
            self.pending_imports = []
            result["trace_step"] = transition.to_dict()
            result["trace_digest"] = self.flow_trace.digest()
            result["move_sequence_digest"] = self.flow_trace.move_sequence_digest()
            result["trace_bound"] = True
            if self.trace and self.trace[-1].get("event") == "apply_moves":
                self.trace[-1]["authoritative_transition"] = transition.to_dict()
                self.trace[-1]["result"] = result
        if self.config.observation_mode != "full_state":
            return json.dumps(
                compact_transition_observation(
                    result=result,
                    state_before=execution_state_before,
                    state_after=self.current_state,
                    moves=moves,
                    radius=self.config.reaction_center_radius,
                    include_local_state=(
                        self.config.observation_mode == "reaction_center_delta"
                    ),
                ),
                ensure_ascii=False,
            )
        return json.dumps(result, ensure_ascii=False)

    def submit_proof(self, proof: str) -> str:
        del proof
        self._consume_call()
        self.failed_steps += 1
        result = {
            "ok": False,
            "code": "FREE_FORM_PROOF_DISABLED",
            "message": "Call finish_trace; the proof must be compiled from committed actions.",
            "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
        }
        self.trace.append({"event": "submit_proof_rejected", "result": result})
        return json.dumps(result, ensure_ascii=False)

    def finish_trace(self) -> str:
        if self.pending_imports:
            self._consume_call()
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "UNCOMMITTED_IMPORTS",
                "message": "Imported fragments must participate in a transition before finishing.",
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
            self.trace.append({"event": "finish_trace_failed", "result": result})
            if self.config.observation_mode != "full_state":
                result = compact_failure_observation(
                    result, observation_mode=self.config.observation_mode
                )
            return json.dumps(result, ensure_ascii=False)
        try:
            # Every transition was already replayed successfully by
            # ``_apply_moves`` before it entered ``flow_trace``.  Compilation
            # still derives and executes the proof, but need not replay the
            # same moves a second time.
            compilation = compile_trace_to_proof(
                self.flow_trace, declared_moves_already_verified=True
            )
        except Exception as exc:
            self._consume_call()
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "TRACE_COMPILATION_FAILED",
                "message": str(exc),
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
            self.trace.append({"event": "finish_trace_failed", "result": result})
            if self.config.observation_mode != "full_state":
                result = compact_failure_observation(
                    result, observation_mode=self.config.observation_mode
                )
            return json.dumps(result, ensure_ascii=False)

        result = json.loads(
            self._submit_preexecuted_proof(
                proof=compilation.proof,
                precursor_smiles=compilation.precursor_smiles,
            )
        )
        result.update(
            {
                "ok": bool(result.get("formal_execute")),
                "trace_bound": True,
                "trace_digest": compilation.trace_digest,
                "move_sequence_digest": compilation.move_sequence_digest,
                "compiled_proof": compilation.proof,
                "n_trace_transitions": compilation.n_transitions,
                "endpoint_source": "environment_owned_trace",
                "declared_moves_replayed": True,
            }
        )
        self.final_result = result
        if self.trace and self.trace[-1].get("event") == "submit_proof":
            self.trace[-1]["event"] = "finish_trace"
            self.trace[-1]["result"] = result
        if self.config.observation_mode != "full_state":
            return json.dumps(
                compact_terminal_observation(
                    result, observation_mode=self.config.observation_mode
                ),
                ensure_ascii=False,
            )
        return json.dumps(result, ensure_ascii=False)

    def state_dict(self) -> dict[str, Any]:
        value = super().state_dict()
        value.update(
            {
                "flow_trace": self.flow_trace.to_dict(),
                "last_committed_state": self.last_committed_state,
                "pending_imports": list(self.pending_imports),
                "trace_bound": True,
            }
        )
        return value
