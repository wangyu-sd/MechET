"""Compact, runtime-reconstructible history for Markov tool decisions.

The executor-owned current molecular state is the chemical sufficient state.
This capsule retains only control-flow facts that are not reliably recoverable
from that SMILES.  It deliberately excludes gold horizon and endpoint fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ALLOWED_ACTIONS = {"import_fragments", "apply_electron_flow", "finish_trace"}


@dataclass(frozen=True)
class TrajectoryHistory:
    accepted_action_types: tuple[str, ...] = ()
    import_batches: int = 0
    imported_fragments: int = 0
    electron_events: int = 0
    last_result_code: str = "START"

    def render(self) -> str:
        action_path = (
            ">".join(self.accepted_action_types)
            if self.accepted_action_types
            else "START"
        )
        last_action = self.accepted_action_types[-1] if self.accepted_action_types else "START"
        return (
            "TRAJECTORY HISTORY (executor-owned; past actions only)\n"
            f"accepted_actions: {len(self.accepted_action_types)}\n"
            f"accepted_action_types: {action_path}\n"
            f"import_batches_committed: {self.import_batches}\n"
            f"imported_fragments_committed: {self.imported_fragments}\n"
            f"electron_events_committed: {self.electron_events}\n"
            f"last_action: {last_action}\n"
            f"last_result: {self.last_result_code}"
        )

    def accept(
        self,
        name: str,
        arguments: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> "TrajectoryHistory":
        if name not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported history action: {name}")
        if result.get("ok") is not True:
            raise ValueError("standard trajectory history accepts only passing gold actions")
        code = str(result.get("code") or "PASS")
        imported = 0
        import_batches = self.import_batches
        events = self.electron_events
        if name == "import_fragments":
            fragments = arguments.get("fragments") or []
            if not isinstance(fragments, list):
                raise ValueError("import fragments must be a list")
            imported = sum(int(item.get("count", 1)) for item in fragments)
            import_batches += 1
        elif name == "apply_electron_flow":
            events += 1
        return TrajectoryHistory(
            accepted_action_types=self.accepted_action_types + (name,),
            import_batches=import_batches,
            imported_fragments=self.imported_fragments + imported,
            electron_events=events,
            last_result_code=code,
        )

