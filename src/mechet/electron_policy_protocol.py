"""Shared three-stage protocol for LLM and graph electron policies.

The two policy tracks consume different surface representations, but they
predict the same canonical next action.  Stage 1 is Markov behavior cloning,
stage 2 adds a compact accepted-action history, and stage 3 samples directly
from the policy for executor-verified reinforcement learning.  No stage asks a
chemistry rule engine to enumerate candidate actions.

The compact history deliberately omits atom-map identifiers, intermediate
state snapshots, the reference horizon, and the expected precursor.  Exact
chemistry lives in the executor-owned current state; history supplies only
non-Markov control-flow information and a small typed action ledger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "electron_policy_two_track_three_stage_v1"

TRACK_LLM = "llm"
TRACK_GRAPH = "graph"
POLICY_TRACKS = (TRACK_LLM, TRACK_GRAPH)

STAGE_STATE_BC = "state_bc"
STAGE_TRAJECTORY_BC = "trajectory_bc"
STAGE_ONLINE_RL = "online_rl"
POLICY_STAGES = (STAGE_STATE_BC, STAGE_TRAJECTORY_BC, STAGE_ONLINE_RL)

ACTION_FAMILIES = (
    "IMPORT_ENV",
    "IMPORT_REACTIVE",
    "FLOW",
    "BE_DELTA",
    "FINISH",
)
CONTAINER_KINDS = ("NONE", "LP", "ATOM", "BOND", "RADICAL_PAIR")
IMPORT_ROLES = (
    "NONE",
    "ENVIRONMENT",
    "NUCLEOPHILE",
    "ELECTROPHILE",
    "BOND_DONOR",
    "LEAVING_GROUP",
    "PROTON_TRANSFER",
    "RADICAL",
    "REDOX",
    "OTHER_REACTIVE",
)


@dataclass(frozen=True)
class CompactElectronEvent:
    """Map-free summary of one accepted canonical action."""

    family: str
    source_kinds: tuple[str, ...] = ()
    sink_kinds: tuple[str, ...] = ()
    electron_moves: int = 0
    import_role: str = "NONE"
    fragment_atoms: int = 0
    active_atoms: int = 0
    extra_bonds: int = 0
    bond_edits: int = 0
    charge_edits: int = 0

    def __post_init__(self) -> None:
        if self.family not in ACTION_FAMILIES:
            raise ValueError(f"unknown action family: {self.family}")
        if any(kind not in CONTAINER_KINDS for kind in self.source_kinds):
            raise ValueError(f"unknown source kind: {self.source_kinds}")
        if any(kind not in CONTAINER_KINDS for kind in self.sink_kinds):
            raise ValueError(f"unknown sink kind: {self.sink_kinds}")
        if self.import_role not in IMPORT_ROLES:
            raise ValueError(f"unknown import role: {self.import_role}")
        counts = (
            self.electron_moves,
            self.fragment_atoms,
            self.active_atoms,
            self.extra_bonds,
            self.bond_edits,
            self.charge_edits,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("compact event counts must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CompactElectronEvent":
        return cls(
            family=str(value["family"]),
            source_kinds=tuple(str(item) for item in value.get("source_kinds") or ()),
            sink_kinds=tuple(str(item) for item in value.get("sink_kinds") or ()),
            electron_moves=int(value.get("electron_moves") or 0),
            import_role=str(value.get("import_role") or "NONE"),
            fragment_atoms=int(value.get("fragment_atoms") or 0),
            active_atoms=int(value.get("active_atoms") or 0),
            extra_bonds=int(value.get("extra_bonds") or 0),
            bond_edits=int(value.get("bond_edits") or 0),
            charge_edits=int(value.get("charge_edits") or 0),
        )

    def render(self) -> str:
        if self.family in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
            return (
                f"{self.family}(role={self.import_role},atoms={self.fragment_atoms},"
                f"active={self.active_atoms},rings={self.extra_bonds})"
            )
        if self.family == "FLOW":
            signatures = ",".join(
                f"{source}>{sink}"
                for source, sink in zip(self.source_kinds, self.sink_kinds)
            )
            return f"FLOW({signatures};arrows={self.electron_moves})"
        if self.family == "BE_DELTA":
            return f"BE_DELTA(bonds={self.bond_edits},charges={self.charge_edits})"
        return "FINISH"


@dataclass(frozen=True)
class CompressedTrajectory:
    """Ordered accepted-action ledger with no repeated molecular states."""

    events: tuple[CompactElectronEvent, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "CompressedTrajectory":
        if not value:
            return cls()
        version = value.get("version")
        if version not in (None, PROTOCOL_VERSION):
            raise ValueError(f"unsupported trajectory protocol: {version}")
        return cls(
            tuple(
                CompactElectronEvent.from_dict(item)
                for item in value.get("events") or ()
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "events": [asdict(item) for item in self.events],
        }

    def append(self, decision: Mapping[str, Any]) -> "CompressedTrajectory":
        return CompressedTrajectory(self.events + (event_from_decision(decision),))

    def render(self) -> str:
        if not self.events:
            return "accepted_events: START"
        rows = [f"E{index:02d} {event.render()}" for index, event in enumerate(self.events, 1)]
        return "accepted_events:\n" + "\n".join(rows)

    @property
    def family_path(self) -> tuple[str, ...]:
        return tuple(item.family for item in self.events)


def event_from_decision(decision: Mapping[str, Any]) -> CompactElectronEvent:
    """Project an executor action into the shared, map-invariant history."""

    family = str(decision["kind"])
    if family in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
        program = decision.get("program") or {}
        atoms: Sequence[Any] = program.get("atoms") or ()
        active: Sequence[Any] = program.get("active_atoms") or ()
        extra: Sequence[Any] = program.get("extra_bonds") or ()
        role = str(program.get("role") or ("ENVIRONMENT" if family == "IMPORT_ENV" else "OTHER_REACTIVE"))
        return CompactElectronEvent(
            family=family,
            import_role=role,
            fragment_atoms=len(atoms),
            active_atoms=len(active),
            extra_bonds=len(extra),
        )
    if family == "FLOW":
        moves = list(decision.get("moves") or ())
        return CompactElectronEvent(
            family=family,
            source_kinds=tuple(str((item.get("source") or {}).get("kind") or "NONE") for item in moves),
            sink_kinds=tuple(str((item.get("sink") or {}).get("kind") or "NONE") for item in moves),
            electron_moves=len(moves),
        )
    if family == "BE_DELTA":
        moves = list(decision.get("moves") or ())
        payload = moves[0] if moves else {}
        return CompactElectronEvent(
            family=family,
            bond_edits=len(payload.get("bond_deltas") or ()),
            charge_edits=len(payload.get("charge_actions") or ()),
        )
    if family == "FINISH":
        return CompactElectronEvent(family=family)
    raise ValueError(f"unknown decision family: {family}")


def canonical_action_target(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Return the common action target used by both policy tracks."""

    family = str(decision["kind"])
    output: dict[str, Any] = {"action": family}
    if family in {"FLOW", "BE_DELTA"}:
        output["moves"] = list(decision.get("moves") or ())
    elif family in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
        program = dict(decision.get("program") or {})
        # This string is only a builder/debug checksum.  Both policies predict
        # the graph program itself, so do not expose a redundant answer field.
        program.pop("source_unmapped_smiles", None)
        output["program"] = program
    elif family != "FINISH":
        raise ValueError(f"unknown decision family: {family}")
    return output


def llm_training_record(
    decision: Mapping[str, Any], *, stage: str
) -> dict[str, Any]:
    """Serialize one common decision into the LLM track's teacher-forcing view."""

    if stage not in {STAGE_STATE_BC, STAGE_TRAJECTORY_BC}:
        raise ValueError("LLM training records are defined only for BC stages")
    # Imports are local to avoid making the canonical protocol depend on the
    # LLM surface adapter during graph-only jobs.
    from .in_place_grounded_flow import deterministic_unmapped_state
    from .natural_language_electron_flow import build_inventory, render_event_arguments

    history = CompressedTrajectory.from_dict(decision.get("history"))
    inventory = build_inventory(str(decision["current"]))
    target = deterministic_unmapped_state(str(decision["target"])).text
    sections = [
        "TASK: predict exactly one retrosynthetic electron event.",
        f"TARGET_PRODUCT: {target}",
        inventory.prompt,
    ]
    if stage == STAGE_TRAJECTORY_BC:
        sections.append("COMPRESSED_TRAJECTORY:\n" + history.render())
    sections.append(
        "Return one canonical action: IMPORT_ENV, IMPORT_REACTIVE, FLOW, "
        "BE_DELTA, or FINISH. Do not enumerate alternatives."
    )
    family = str(decision["kind"])
    if family in {"FLOW", "BE_DELTA"}:
        llm_target: dict[str, Any] = {
            "action": family,
            "arguments": render_event_arguments(
                str(decision["current"]), list(decision.get("moves") or ())
            ),
        }
    elif family in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
        program = dict(decision.get("program") or {})
        program.pop("source_unmapped_smiles", None)
        llm_target = {"action": family, "program": program}
    elif family == "FINISH":
        llm_target = {"action": family}
    else:
        raise ValueError(f"unknown decision family: {family}")
    return {
        "id": str(decision.get("reaction_id") or ""),
        "track": TRACK_LLM,
        "stage": stage,
        "input": "\n".join(sections),
        "output": json.dumps(llm_target, separators=(",", ":")),
    }
