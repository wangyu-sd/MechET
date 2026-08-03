"""TRL-facing MechET environments with explicit public tool surfaces.

TRL exposes every public environment method other than ``reset`` and
``get_reward`` as a model tool. These facades keep the chemistry implementation
private and expose only the scientifically declared methods. Invalid calls and
causal interventions consume the same environment budget as ordinary tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .agent_env import AgentEnvConfig, MechETAgentEnv
from .knowledge_agent_env import KnowledgeAgentConfig, KnowledgeAugmentedAgentEnv
from .trace_agent_env import TraceOwnedAgentEnv


class TraceOwnedTRLEnvironment:
    """TRL facade for the trace-owned main method."""

    def __init__(
        self,
        *,
        config: AgentEnvConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
        intervention: str = "none",
        shuffled_observations: dict[str, list[str]] | None = None,
    ) -> None:
        self._env = TraceOwnedAgentEnv(
            config=config,
            forward_checkpoint=forward_checkpoint,
            forward_device=forward_device,
        )
        self._intervention = str(intervention or "none")
        self._shuffled_observations = {
            str(name): list(values)
            for name, values in dict(shuffled_observations or {}).items()
        }
        self._shuffle_offsets: dict[str, int] = {}
        self._last_visible_tool_result = ""

    def reset(self, **kwargs: Any) -> str:
        """Reset one rollout and return the initial task observation."""

        self._last_visible_tool_result = ""
        self._shuffle_offsets = {}
        return self._env.reset(**kwargs)

    def get_reward(self) -> float:
        """Return the environment-owned terminal reward."""

        return self._env.get_reward()

    def _reject(self, tool_name: str, code: str, message: str = "") -> str:
        """Record an invalid or disabled tool call without bypassing budget."""

        try:
            self._env._consume_call()
        except Exception as exc:
            result = {
                "ok": False,
                "code": "TOOL_BUDGET_EXCEEDED",
                "message": str(exc),
                "remaining_tool_calls": 0,
            }
            self._env.trace.append(
                {"event": f"{tool_name}_rejected", "result": result}
            )
            return json.dumps(result, ensure_ascii=False)
        self._env.failed_steps += 1
        result = {
            "ok": False,
            "code": code,
            "message": message or code,
            "remaining_tool_calls": max(
                self._env.config.max_tool_calls - self._env.tool_calls, 0
            ),
        }
        self._env.trace.append(
            {"event": f"{tool_name}_rejected", "result": result}
        )
        return json.dumps(result, ensure_ascii=False)

    def _visible(self, tool_name: str, raw: str) -> str:
        if self._intervention == "remove_tool_observations":
            visible = json.dumps(
                {
                    "ok": True,
                    "intervention": "observation_removed",
                    "tool": tool_name,
                },
                ensure_ascii=False,
            )
        elif (
            self._intervention == "stale_tool_observations"
            and self._last_visible_tool_result
        ):
            visible = self._last_visible_tool_result
        elif self._intervention == "shuffle_tool_observations":
            values = self._shuffled_observations.get(tool_name) or []
            if values:
                offset = self._shuffle_offsets.get(tool_name, 0)
                visible = values[offset % len(values)]
                self._shuffle_offsets[tool_name] = offset + 1
            else:
                visible = json.dumps(
                    {
                        "ok": False,
                        "code": "SHUFFLED_OBSERVATION_UNAVAILABLE",
                        "tool": tool_name,
                    },
                    ensure_ascii=False,
                )
        else:
            visible = raw
        if (
            self._intervention != "stale_tool_observations"
            or not self._last_visible_tool_result
        ):
            self._last_visible_tool_result = visible
        return visible

    def inspect_state(self) -> str:
        """Inspect the mapped state and enumerate legal electron containers.

        Returns:
            JSON observation containing the current state, sources, and sinks.
        """

        if self._intervention == "disable_inspect_state":
            return self._visible(
                "inspect_state",
                self._reject(
                    "inspect_state", "INSPECT_STATE_DISABLED_BY_INTERVENTION"
                ),
            )
        return self._visible("inspect_state", self._env.inspect_state())

    def import_fragment(self, fragment_smiles: str) -> str:
        """Import a mapped fragment for the next inverse transition.

        Args:
            fragment_smiles: Atom-mapped fragment SMILES with unique positive maps.

        Returns:
            JSON result containing the augmented state or a stable failure code.
        """

        if not str(fragment_smiles or "").strip():
            return self._visible(
                "import_fragment",
                self._reject("import_fragment", "IMPORT_FRAGMENT_EMPTY"),
            )
        return self._visible(
            "import_fragment", self._env.import_fragment(fragment_smiles)
        )

    def apply_electron_move(
        self,
        source_kind: str,
        source_atoms: list[int],
        sink_kind: str,
        sink_atoms: list[int],
    ) -> str:
        """Apply one explicit two-electron move.

        Args:
            source_kind: ``LP`` or ``BOND``.
            source_atoms: One source atom map for LP or two for BOND.
            sink_kind: ``ATOM``, ``LP``, or ``BOND``.
            sink_atoms: One sink atom map for ATOM/LP or two for BOND.

        Returns:
            JSON execution result and the sanitized next state.
        """

        if self._intervention == "disable_intermediate_execution":
            return self._visible(
                "apply_electron_move",
                self._reject(
                    "apply_electron_move", "INTERMEDIATE_EXECUTION_DISABLED"
                ),
            )
        if not isinstance(source_atoms, list) or not isinstance(sink_atoms, list):
            return self._visible(
                "apply_electron_move",
                self._reject("apply_electron_move", "MOVE_ATOMS_INVALID"),
            )
        raw = self._env.apply_electron_move(
            source_kind, source_atoms, sink_kind, sink_atoms
        )
        return self._visible("apply_electron_move", raw)

    def apply_coupled_electron_moves(self, moves: list[dict[str, Any]]) -> str:
        """Apply coupled two-electron moves atomically.

        Args:
            moves: Non-empty list of source/sink move objects.

        Returns:
            JSON execution result and the sanitized next state.
        """

        if self._intervention == "disable_intermediate_execution":
            return self._visible(
                "apply_coupled_electron_moves",
                self._reject(
                    "apply_coupled_electron_moves",
                    "INTERMEDIATE_EXECUTION_DISABLED",
                ),
            )
        if not isinstance(moves, list) or not moves:
            return self._visible(
                "apply_coupled_electron_moves",
                self._reject(
                    "apply_coupled_electron_moves", "MOVE_LIST_EMPTY"
                ),
            )
        if any(not isinstance(item, dict) for item in moves):
            return self._visible(
                "apply_coupled_electron_moves",
                self._reject(
                    "apply_coupled_electron_moves", "MOVE_LIST_INVALID"
                ),
            )
        raw = self._env.apply_coupled_electron_moves(
            json.dumps(moves, ensure_ascii=False)
        )
        return self._visible("apply_coupled_electron_moves", raw)

    def finish_trace(self) -> str:
        """Compile, execute, and finish the environment-owned trace.

        Returns:
            JSON terminal result containing the compiled proof and precursor.
        """

        return self._visible("finish_trace", self._env.finish_trace())

    def abstain(self, reason: str) -> str:
        """Terminate without an unsupported precursor.

        Args:
            reason: Concise reason that available evidence is insufficient.

        Returns:
            JSON abstention result and terminal reward.
        """

        return self._visible("abstain", self._env.abstain(reason))

    def _snapshot(self) -> dict[str, Any]:
        return self._env.state_dict()


class TextbookTraceOwnedTRLEnvironment(TraceOwnedTRLEnvironment):
    """TRL facade for trace ownership with textbook evidence."""

    def __init__(
        self,
        *,
        config: KnowledgeAgentConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
        intervention: str = "none",
        shuffled_observations: dict[str, list[str]] | None = None,
    ) -> None:
        self._env = KnowledgeAugmentedAgentEnv(
            config=config,
            forward_checkpoint=forward_checkpoint,
            forward_device=forward_device,
        )
        self._intervention = str(intervention or "none")
        self._shuffled_observations = {
            str(name): list(values)
            for name, values in dict(shuffled_observations or {}).items()
        }
        self._shuffle_offsets = {}
        self._last_visible_tool_result = ""

    def retrieve_textbook_guidance(
        self,
        query: str = "",
        top_k: int = 0,
        max_characters: int = 0,
    ) -> str:
        """Retrieve bounded textbook evidence for the current state.

        Args:
            query: Optional inference-available chemistry query.
            top_k: Number of passages to retrieve; zero uses the frozen default.
            max_characters: Context character limit; zero uses the frozen default.

        Returns:
            JSON bounded evidence cards with provenance and retrieval scores.
        """

        raw = self._env.retrieve_textbook_guidance(
            query=query, top_k=top_k, max_characters=max_characters
        )
        return self._visible("retrieve_textbook_guidance", raw)


class TextbookAnchorTraceOwnedTRLEnvironment(
    TextbookTraceOwnedTRLEnvironment
):
    """TRL facade for the combined textbook-plus-anchor condition."""

    def retrieve_primitives(self, query: str = "", top_k: int = 0) -> str:
        """Retrieve structured mechanistic knowledge anchors.

        Args:
            query: Optional inference-available chemistry query.
            top_k: Number of anchor matches; zero uses the frozen default.

        Returns:
            JSON anchor matches, role bindings, warnings, and provenance.
        """

        raw = self._env.retrieve_primitives(query=query, top_k=top_k)
        return self._visible("retrieve_primitives", raw)


class LegacyProofTRLEnvironment:
    """TRL facade for the legacy loose-trace/complete-proof baseline."""

    def __init__(
        self,
        *,
        config: AgentEnvConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
    ) -> None:
        self._env = MechETAgentEnv(
            config=config,
            forward_checkpoint=forward_checkpoint,
            forward_device=forward_device,
        )

    def reset(self, **kwargs: Any) -> str:
        """Reset one legacy baseline rollout."""

        return self._env.reset(**kwargs)

    def get_reward(self) -> float:
        """Return the environment-owned terminal reward."""

        return self._env.get_reward()

    def _reject(self, tool_name: str, code: str) -> str:
        try:
            self._env._consume_call()
        except Exception as exc:
            result = {
                "ok": False,
                "code": "TOOL_BUDGET_EXCEEDED",
                "message": str(exc),
                "remaining_tool_calls": 0,
            }
            self._env.trace.append(
                {"event": f"{tool_name}_rejected", "result": result}
            )
            return json.dumps(result, ensure_ascii=False)
        self._env.failed_steps += 1
        result = {
            "ok": False,
            "code": code,
            "remaining_tool_calls": max(
                self._env.config.max_tool_calls - self._env.tool_calls, 0
            ),
        }
        self._env.trace.append(
            {"event": f"{tool_name}_rejected", "result": result}
        )
        return json.dumps(result, ensure_ascii=False)

    def inspect_state(self) -> str:
        """Inspect the current mapped state.

        Returns:
            JSON state and electron-container inventory.
        """

        return self._env.inspect_state()

    def apply_electron_move(
        self,
        source_kind: str,
        source_atoms: list[int],
        sink_kind: str,
        sink_atoms: list[int],
    ) -> str:
        """Apply one explicit two-electron move.

        Args:
            source_kind: ``LP`` or ``BOND``.
            source_atoms: Source atom maps.
            sink_kind: ``ATOM``, ``LP``, or ``BOND``.
            sink_atoms: Sink atom maps.

        Returns:
            JSON execution result.
        """

        return self._env.apply_electron_move(
            source_kind, source_atoms, sink_kind, sink_atoms
        )

    def apply_coupled_electron_moves(self, moves: list[dict[str, Any]]) -> str:
        """Apply coupled two-electron moves atomically.

        Args:
            moves: Non-empty list of source/sink move objects.

        Returns:
            JSON execution result.
        """

        if not isinstance(moves, list) or not moves:
            return self._reject("apply_coupled_electron_moves", "MOVE_LIST_EMPTY")
        if any(not isinstance(item, dict) for item in moves):
            return self._reject("apply_coupled_electron_moves", "MOVE_LIST_INVALID")
        return self._env.apply_coupled_electron_moves(
            json.dumps(moves, ensure_ascii=False)
        )

    def submit_proof(self, proof: str) -> str:
        """Submit one complete executable proof in the legacy baseline.

        Args:
            proof: Complete ``MECH_PROOF v1`` text.

        Returns:
            JSON formal and endpoint evaluation result.
        """

        return self._env.submit_proof(proof)

    def abstain(self, reason: str) -> str:
        """Terminate the baseline without a prediction.

        Args:
            reason: Concise abstention reason.

        Returns:
            JSON abstention record.
        """

        return self._env.abstain(reason)

    def _snapshot(self) -> dict[str, Any]:
        return self._env.state_dict()
