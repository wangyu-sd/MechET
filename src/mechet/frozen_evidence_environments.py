"""Inference facades that replay frozen evidence results for matched H3 tests."""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .anchor_trl_environment import AnchorTraceOwnedTRLEnvironment
from .trl_environments import (
    TextbookAnchorTraceOwnedTRLEnvironment,
    TextbookTraceOwnedTRLEnvironment,
)


class _FrozenEvidenceMixin:
    _frozen_textbook_result: dict[str, Any] | None = None
    _frozen_anchor_result: dict[str, Any] | None = None

    def _consume_frozen_result(
        self, tool_name: str, value: dict[str, Any]
    ) -> str:
        self._env._consume_call()
        result = deepcopy(value)
        result["ok"] = bool(result.get("ok", True))
        result["state_smiles"] = self._env.current_state
        result["soft_evidence_only"] = True
        result["direct_reward"] = False
        result["frozen_evidence_replay"] = True
        result["remaining_tool_calls"] = (
            self._env.config.max_tool_calls - self._env.tool_calls
        )
        if tool_name == "retrieve_textbook_guidance":
            self._env.textbook_retrievals.append(result)
        elif tool_name == "retrieve_primitives":
            self._env.primitive_retrievals.append(result)
        self._env.trace.append({"event": tool_name, "result": result})
        return json.dumps(result, ensure_ascii=False)


class FrozenTextbookTraceOwnedTRLEnvironment(
    _FrozenEvidenceMixin, TextbookTraceOwnedTRLEnvironment
):
    """Textbook condition that replays one frozen bounded evidence result."""

    def reset(
        self,
        frozen_textbook_result: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Reset and attach the row-specific frozen textbook result."""

        self._frozen_textbook_result = (
            deepcopy(frozen_textbook_result)
            if isinstance(frozen_textbook_result, dict)
            else None
        )
        return super().reset(**kwargs)

    def retrieve_textbook_guidance(
        self,
        query: str = "",
        top_k: int = 0,
        max_characters: int = 0,
    ) -> str:
        """Return frozen textbook evidence or fall back to online retrieval.

        Args:
            query: Optional inference-available query; ignored by frozen replay.
            top_k: Requested result count for online fallback.
            max_characters: Context budget for online fallback.

        Returns:
            JSON bounded evidence result.
        """

        if self._frozen_textbook_result is not None:
            return self._visible(
                "retrieve_textbook_guidance",
                self._consume_frozen_result(
                    "retrieve_textbook_guidance",
                    self._frozen_textbook_result,
                ),
            )
        return super().retrieve_textbook_guidance(
            query=query, top_k=top_k, max_characters=max_characters
        )


class FrozenAnchorTraceOwnedTRLEnvironment(
    _FrozenEvidenceMixin, AnchorTraceOwnedTRLEnvironment
):
    """Anchor-only condition that replays frozen anchor matches."""

    def reset(
        self,
        frozen_anchor_result: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Reset and attach the row-specific frozen anchor result."""

        self._frozen_anchor_result = (
            deepcopy(frozen_anchor_result)
            if isinstance(frozen_anchor_result, dict)
            else None
        )
        return super().reset(**kwargs)

    def retrieve_primitives(self, query: str = "", top_k: int = 0) -> str:
        """Return frozen anchors or fall back to online anchor retrieval.

        Args:
            query: Optional inference-available query; ignored by frozen replay.
            top_k: Requested result count for online fallback.

        Returns:
            JSON anchor result.
        """

        if self._frozen_anchor_result is not None:
            return self._visible(
                "retrieve_primitives",
                self._consume_frozen_result(
                    "retrieve_primitives", self._frozen_anchor_result
                ),
            )
        return super().retrieve_primitives(query=query, top_k=top_k)


class FrozenTextbookAnchorTraceOwnedTRLEnvironment(
    _FrozenEvidenceMixin, TextbookAnchorTraceOwnedTRLEnvironment
):
    """Combined condition replaying both textbook and anchor evidence."""

    def reset(
        self,
        frozen_textbook_result: dict[str, Any] | None = None,
        frozen_anchor_result: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Reset and attach row-specific frozen evidence results."""

        self._frozen_textbook_result = (
            deepcopy(frozen_textbook_result)
            if isinstance(frozen_textbook_result, dict)
            else None
        )
        self._frozen_anchor_result = (
            deepcopy(frozen_anchor_result)
            if isinstance(frozen_anchor_result, dict)
            else None
        )
        return super().reset(**kwargs)

    def retrieve_textbook_guidance(
        self,
        query: str = "",
        top_k: int = 0,
        max_characters: int = 0,
    ) -> str:
        """Return frozen textbook evidence or use online retrieval.

        Args:
            query: Optional inference-available query.
            top_k: Requested result count for online fallback.
            max_characters: Context budget for online fallback.

        Returns:
            JSON bounded textbook result.
        """

        if self._frozen_textbook_result is not None:
            return self._visible(
                "retrieve_textbook_guidance",
                self._consume_frozen_result(
                    "retrieve_textbook_guidance",
                    self._frozen_textbook_result,
                ),
            )
        return TextbookTraceOwnedTRLEnvironment.retrieve_textbook_guidance(
            self,
            query=query,
            top_k=top_k,
            max_characters=max_characters,
        )

    def retrieve_primitives(self, query: str = "", top_k: int = 0) -> str:
        """Return frozen anchors or use online retrieval.

        Args:
            query: Optional inference-available query.
            top_k: Requested result count for online fallback.

        Returns:
            JSON anchor result.
        """

        if self._frozen_anchor_result is not None:
            return self._visible(
                "retrieve_primitives",
                self._consume_frozen_result(
                    "retrieve_primitives", self._frozen_anchor_result
                ),
            )
        return TextbookAnchorTraceOwnedTRLEnvironment.retrieve_primitives(
            self, query=query, top_k=top_k
        )
