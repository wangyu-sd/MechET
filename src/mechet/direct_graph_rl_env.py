"""Candidate-free stage-3 environment adapter for the graph policy track."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .agent_env import AgentEnvConfig
from .electron_policy_protocol import CompressedTrajectory
from .graph_fragment_actions import (
    ReactiveFragmentProgram,
    allocate_reactive_fragment_maps,
    bind_reactive_import,
)
from .trace_agent_env import TraceOwnedAgentEnv


@dataclass(frozen=True)
class DirectGraphObservation:
    target: str
    current: str
    history: CompressedTrajectory
    remaining_steps: int


@dataclass(frozen=True)
class DirectGraphTransition:
    observation: DirectGraphObservation
    action: Mapping[str, Any]
    next_observation: DirectGraphObservation
    reward: float
    done: bool
    accepted: bool
    result: Mapping[str, Any]


def _maximum_atom_map(smiles: str) -> int:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("invalid executor state")
    return max((int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()), default=0)


class DirectGraphElectronEnv:
    """Execute directly sampled canonical actions; never enumerate choices."""

    def __init__(self, *, max_steps: int = 32) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.max_steps = int(max_steps)
        self._env = TraceOwnedAgentEnv(
            config=AgentEnvConfig(
                max_tool_calls=max_steps,
                observation_mode="full_state",
            )
        )
        self._history = CompressedTrajectory()
        self._steps = 0
        self._done = False
        self._reactive_guards = []

    @property
    def observation(self) -> DirectGraphObservation:
        return DirectGraphObservation(
            target=self._env.target_smiles,
            current=self._env.current_state,
            history=self._history,
            remaining_steps=max(self.max_steps - self._steps, 0),
        )

    def reset(
        self,
        *,
        target: str,
        expected_precursor: str = "",
        competitor_products: Sequence[str] = (),
    ) -> DirectGraphObservation:
        self._env.reset(
            target_smiles=target,
            expected_precursor=expected_precursor,
            competitor_products=list(competitor_products),
        )
        self._history = CompressedTrajectory()
        self._steps = 0
        self._done = False
        self._reactive_guards = []
        return self.observation

    @staticmethod
    def _program(action: Mapping[str, Any]) -> ReactiveFragmentProgram:
        value = action.get("program")
        if isinstance(value, ReactiveFragmentProgram):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("import action requires a fragment graph program")
        return ReactiveFragmentProgram.from_dict(value)

    def _import(self, action: Mapping[str, Any]) -> dict[str, Any]:
        program = self._program(action)
        first_map = _maximum_atom_map(self._env.current_state) + 1
        fragment, assigned = allocate_reactive_fragment_maps(
            program, first_map=first_map
        )
        result = json.loads(self._env.import_fragment(fragment))
        if result.get("ok") and program.role != "ENVIRONMENT":
            self._reactive_guards.append(bind_reactive_import(program, assigned))
        return result

    def _flow(self, action: Mapping[str, Any]) -> dict[str, Any]:
        moves = list(action.get("moves") or ())
        if not moves:
            return {"ok": False, "code": "EMPTY_ELECTRON_EVENT"}
        for guard in self._reactive_guards:
            checked = guard.validate(moves)
            if not checked.get("ok"):
                return dict(checked)
        result = json.loads(
            self._env.apply_coupled_electron_moves(
                json.dumps(moves, separators=(",", ":"))
            )
        )
        if result.get("ok"):
            self._reactive_guards = []
        return result

    def step(self, action: Mapping[str, Any]) -> DirectGraphTransition:
        if self._done:
            raise RuntimeError("episode is already terminal")
        before = self.observation
        family = str(action.get("kind") or action.get("action") or "")
        canonical = dict(action)
        canonical["kind"] = family
        self._steps += 1
        try:
            if family in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
                result = self._import(canonical)
            elif family in {"FLOW", "BE_DELTA"}:
                result = self._flow(canonical)
            elif family == "FINISH":
                result = json.loads(self._env.finish_trace())
            else:
                result = {"ok": False, "code": "UNKNOWN_ACTION_FAMILY"}
        except Exception as exc:
            result = {"ok": False, "code": "ACTION_EXECUTION_ERROR", "message": str(exc)}

        accepted = bool(result.get("ok"))
        if accepted:
            self._history = self._history.append(canonical)
        terminal = family == "FINISH" or self._steps >= self.max_steps
        self._done = terminal
        if terminal:
            reward = float(self._env.get_reward()) if family == "FINISH" else float(
                self._env.config.unfinished_reward
            )
        else:
            reward = (
                float(self._env.config.successful_step)
                if accepted
                else -float(self._env.config.failed_step_penalty)
            )
        after = self.observation
        return DirectGraphTransition(
            observation=before,
            action=canonical,
            next_observation=after,
            reward=reward,
            done=terminal,
            accepted=accepted,
            result=result,
        )
