"""Primitive-retrieval augmentation for the inverse agent environment.

The reference library is exposed as a tool and an optional soft process reward.
It never replaces the deterministic executor and never hard-rejects a formally
valid move merely because no reviewed primitive matched it.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .agent_env import AgentEnvConfig, MechETAgentEnv
from .primitive_library import PrimitiveLibrary


@dataclass(frozen=True)
class PrimitiveAgentConfig(AgentEnvConfig):
    primitive_library_path: str = "knowledge/primitives/core_polar_primitives.yaml"
    primitive_source_registry_path: str = "knowledge/source_registry.yaml"
    primitive_context_top_k: int = 6
    primitive_reward_scale: float = 0.0
    primitive_reward_cap: float = 1.0
    auto_retrieve_on_reset: bool = True


class PrimitiveAugmentedAgentEnv(MechETAgentEnv):
    """MechET environment with provenance-aware primitive retrieval."""

    def __init__(
        self,
        *,
        config: PrimitiveAgentConfig | dict[str, Any] | None = None,
        primitive_library_path: str | Path | None = None,
        primitive_source_registry_path: str | Path | None = None,
        forward_checkpoint: str | Path | None = None,
        forward_device: str = "cpu",
    ) -> None:
        cfg = config if isinstance(config, PrimitiveAgentConfig) else PrimitiveAgentConfig(**dict(config or {}))
        self.primitive_config = cfg
        self.primitive_library = PrimitiveLibrary.load(
            primitive_library_path or cfg.primitive_library_path,
            source_registry=primitive_source_registry_path or cfg.primitive_source_registry_path,
        )
        super().__init__(
            config=cfg,
            forward_checkpoint=forward_checkpoint,
            forward_device=forward_device,
        )

    def _clear(self) -> None:
        super()._clear()
        self.primitive_support_total = 0.0
        self.primitive_support_events: list[dict[str, Any]] = []

    def reset(self, *args: Any, **kwargs: Any) -> str:
        observation = json.loads(super().reset(*args, **kwargs))
        observation["primitive_library"] = {
            "enabled": True,
            "n_primitives": len(self.primitive_library.primitives),
            "soft_guidance_only": True,
            "tool": "retrieve_primitives",
        }
        observation["instructions"].append(
            "Use retrieve_primitives for reviewed motifs, role bindings, competing pathways, and warnings. Primitive matches are soft guidance."
        )
        if self.primitive_config.auto_retrieve_on_reset:
            observation["initial_primitive_candidates"] = [
                item.to_dict()
                for item in self.primitive_library.retrieve(
                    self.current_state,
                    top_k=self.primitive_config.primitive_context_top_k,
                )
            ]
        self.trace[-1]["observation"] = observation
        return json.dumps(observation, ensure_ascii=False)

    def retrieve_primitives(self, query: str = "", top_k: int = 0) -> str:
        """Retrieve reviewed primitive records for the current mapped state.

        Args:
            query: Optional mechanism or reaction-family hint.
            top_k: Maximum returned matches; zero uses the configured default.
        """
        self._consume_call()
        try:
            matches = self.primitive_library.retrieve(
                self.current_state,
                query=query,
                top_k=int(top_k or self.primitive_config.primitive_context_top_k),
            )
            result = {
                "ok": True,
                "state_smiles": self.current_state,
                "query": query,
                "matches": [item.to_dict() for item in matches],
                "soft_guidance_only": True,
                "remaining_tool_calls": self.config.max_tool_calls - self.tool_calls,
            }
        except Exception as exc:
            result = {"ok": False, "code": "PRIMITIVE_RETRIEVAL_FAILED", "message": str(exc)}
        self.trace.append({"event": "retrieve_primitives", "result": result})
        return json.dumps(result, ensure_ascii=False)

    def _apply_moves(self, moves: list[dict[str, Any]]) -> str:
        state_before = self.current_state
        support = self.primitive_library.support_moves(state_before, moves)
        result = json.loads(super()._apply_moves(moves))
        result["primitive_support"] = support
        if result.get("ok") and support.get("supported"):
            score = float(support.get("support_score") or 0.0)
            self.primitive_support_total += score
            self.primitive_support_events.append({
                "state_smiles": state_before,
                "moves": moves,
                "support": support,
            })
        if self.trace and self.trace[-1].get("event") == "apply_moves":
            self.trace[-1]["result"] = result
            self.trace[-1]["primitive_support"] = support
        return json.dumps(result, ensure_ascii=False)

    def submit_proof(self, proof: str) -> str:
        result = json.loads(super().submit_proof(proof))
        bonus = self.primitive_config.primitive_reward_scale * min(
            self.primitive_support_total,
            self.primitive_config.primitive_reward_cap,
        )
        if bonus:
            self.reward += float(bonus)
            result["reward"] = self.reward
        result.update(
            primitive_reward=float(bonus),
            primitive_support_total=float(self.primitive_support_total),
            primitive_support_events=self.primitive_support_events,
            primitive_support_is_soft=True,
        )
        self.final_result = result
        if self.trace and self.trace[-1].get("event") == "submit_proof":
            self.trace[-1]["result"] = result
        return json.dumps(result, ensure_ascii=False)

    def state_dict(self) -> dict[str, Any]:
        value = super().state_dict()
        value.update(
            primitive_library=self.primitive_library.manifest(),
            primitive_support_total=self.primitive_support_total,
            primitive_support_events=self.primitive_support_events,
        )
        return value
