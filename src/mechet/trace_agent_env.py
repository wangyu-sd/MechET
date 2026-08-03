"""Faithful inverse-agent environment whose trace uniquely determines proof.

``TraceOwnedAgentEnv`` is the main-method environment. The legacy
``MechETAgentEnv`` remains available as a complete-proof baseline, but this
class disables free-form proof submission and exposes ``finish_trace`` instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rdkit import Chem

from .agent_env import AgentEnvConfig, MechETAgentEnv
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
        observation = json.loads(super().reset(*args, **kwargs))
        self.flow_trace = ElectronFlowTrace(self.target_smiles)
        self.last_committed_state = self.target_smiles
        self.pending_imports = []
        observation["task"] = "trace_owned_inverse_electron_flow"
        observation["faithfulness_contract"] = {
            "free_form_proof_submission": False,
            "endpoint_source": "environment_owned_trace",
            "final_tool": "finish_trace",
        }
        observation["instructions"] = [
            "Use inspect_state before referencing atom maps.",
            "Use import_fragment when a required mapped precursor fragment is absent.",
            "Use apply_electron_move or apply_coupled_electron_moves for every claimed arrow.",
            "Call finish_trace; the environment compiles the committed trace into MECH_PROOF v1.",
            "Free-form submit_proof is disabled in this environment.",
            "Use abstain when chemical support is insufficient.",
        ]
        self.trace[-1]["observation"] = observation
        return json.dumps(observation, ensure_ascii=False)

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
        """Add a mapped fragment required by the next inverse transition.

        Imports are recorded in the environment trace and compiled into the next
        proof edge. They cannot be silently introduced in the final answer.
        """

        self._consume_call()
        try:
            fragment = _canonical_mapped(fragment_smiles)
            existing = self._atom_maps(self.current_state)
            incoming = self._atom_maps(fragment)
            overlap = sorted(existing & incoming)
            if overlap:
                raise ValueError(f"IMPORT_ATOM_MAP_COLLISION: {overlap}")
            combined = _canonical_mapped(_combine_smiles(self.current_state, [fragment]))
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
        return json.dumps(result, ensure_ascii=False)

    def _apply_moves(self, moves: list[dict[str, Any]]) -> str:
        state_before = self.last_committed_state
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
            result["trace_bound"] = True
            if self.trace and self.trace[-1].get("event") == "apply_moves":
                self.trace[-1]["authoritative_transition"] = transition.to_dict()
                self.trace[-1]["result"] = result
        return json.dumps(result, ensure_ascii=False)

    def submit_proof(self, proof: str) -> str:
        """Reject an independent proof in the trace-owned main-method path."""

        del proof
        self._consume_call()
        self.failed_steps += 1
        result = {
            "ok": False,
            "code": "FREE_FORM_PROOF_DISABLED",
            "message": "Call finish_trace; the proof must be compiled from committed tool actions.",
            "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
        }
        self.trace.append({"event": "submit_proof_rejected", "result": result})
        return json.dumps(result, ensure_ascii=False)

    def finish_trace(self) -> str:
        """Compile the committed trace, execute it, and derive the precursor."""

        if self.pending_imports:
            self._consume_call()
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "UNCOMMITTED_IMPORTS",
                "message": "Imported fragments must participate in an executed transition before finishing.",
            }
            self.trace.append({"event": "finish_trace_failed", "result": result})
            return json.dumps(result, ensure_ascii=False)
        try:
            compilation = compile_trace_to_proof(self.flow_trace)
        except Exception as exc:
            self._consume_call()
            self.failed_steps += 1
            result = {
                "ok": False,
                "code": "TRACE_COMPILATION_FAILED",
                "message": str(exc),
            }
            self.trace.append({"event": "finish_trace_failed", "result": result})
            return json.dumps(result, ensure_ascii=False)

        # Reuse the legacy terminal scoring implementation only after the proof
        # has been deterministically compiled from the authoritative trace.
        result = json.loads(MechETAgentEnv.submit_proof(self, compilation.proof))
        result.update(
            {
                "ok": bool(result.get("formal_execute")),
                "trace_bound": True,
                "trace_digest": compilation.trace_digest,
                "compiled_proof": compilation.proof,
                "n_trace_transitions": compilation.n_transitions,
                "endpoint_source": "environment_owned_trace",
            }
        )
        self.final_result = result
        if self.trace and self.trace[-1].get("event") == "submit_proof":
            self.trace[-1]["event"] = "finish_trace"
            self.trace[-1]["result"] = result
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
