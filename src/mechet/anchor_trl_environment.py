"""Anchor-only TRL facade for the H3 structured-evidence condition."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .knowledge_agent_env import KnowledgeAgentConfig, KnowledgeAugmentedAgentEnv
from .trl_environments import TraceOwnedTRLEnvironment


class AnchorTraceOwnedTRLEnvironment(TraceOwnedTRLEnvironment):
    """Trace-owned environment exposing anchors but no textbook retrieval tool."""

    def __init__(
        self,
        *,
        config: KnowledgeAgentConfig | dict[str, Any] | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
        intervention: str = "none",
        shuffled_observations: dict[str, list[str]] | None = None,
    ) -> None:
        cfg = (
            config
            if isinstance(config, KnowledgeAgentConfig)
            else KnowledgeAgentConfig(**dict(config or {}))
        )
        if not cfg.enable_structured_primitives:
            raise ValueError(
                "AnchorTraceOwnedTRLEnvironment requires enable_structured_primitives=true"
            )
        self._env = KnowledgeAugmentedAgentEnv(
            config=cfg,
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
