"""Compact chemistry observations for trace-owned inverse execution.

The executor keeps the authoritative complete molecular state. Model-facing
observations expose either no molecular state, one authoritative current state,
or a mapped reaction-centre neighbourhood for a controlled ablation. All modes
remove duplicated transition/audit serialization.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem


def move_atom_maps(moves: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    """Collect every mapped atom addressed by electron-flow actions."""

    selected: set[int] = set()
    for move in moves:
        for endpoint in (move.get("source"), move.get("sink")):
            if isinstance(endpoint, Mapping):
                selected.update(int(value) for value in endpoint.get("atoms") or ())
        for item in move.get("bond_deltas") or ():
            selected.update(int(value) for value in item.get("atoms") or ())
        for item in move.get("charge_actions") or ():
            value = int(item.get("atom_map") or 0)
            if value > 0:
                selected.add(value)
    return tuple(sorted(value for value in selected if value > 0))


def mapped_reaction_center_smiles(
    state_smiles: str,
    center_maps: Iterable[int],
    *,
    radius: int = 1,
) -> str:
    """Serialize only mapped atoms near ``center_maps`` from a complete state.

    ``radius=1`` includes the directly addressed atoms and their immediate
    neighbours, which preserves local valence and substituent context without
    repeating remote scaffold atoms or spectator fragments.
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(state_smiles or ""), params)
    if mol is None:
        raise ValueError("COMPACT_STATE_SMILES_INVALID")
    requested = {int(value) for value in center_maps if int(value) > 0}
    selected = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomMapNum() in requested
    }
    if requested and not selected:
        raise ValueError("COMPACT_STATE_CENTER_MAPS_ABSENT")
    frontier = set(selected)
    for _ in range(max(int(radius), 0)):
        neighbours = {
            neighbour.GetIdx()
            for index in frontier
            for neighbour in mol.GetAtomWithIdx(index).GetNeighbors()
        }
        frontier = neighbours - selected
        selected.update(neighbours)
    if not selected:
        return ""
    bonds = [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in selected and bond.GetEndAtomIdx() in selected
    ]
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(selected),
        bondsToUse=bonds,
        canonical=True,
        isomericSmiles=True,
        kekuleSmiles=True,
    )


def compact_failure_observation(
    result: Mapping[str, Any],
    *,
    observation_mode: str,
    current_state_smiles: str | None = None,
) -> dict[str, Any]:
    """Return only stable control fields from a failed executor call."""

    compact = {
        key: result[key]
        for key in ("ok", "code", "remaining_tool_calls")
        if key in result
    }
    compact["observation_mode"] = f"{observation_mode}_v1"
    if observation_mode == "compact_full_state":
        compact["current_state_smiles"] = str(current_state_smiles or "")
    return compact


def compact_full_state_observation(
    result: Mapping[str, Any],
    *,
    current_state_smiles: str,
    include_inventory: bool = False,
) -> dict[str, Any]:
    """Expose one authoritative state without duplicated transition audit data."""

    if not result.get("ok"):
        return compact_failure_observation(
            result,
            observation_mode="compact_full_state",
            current_state_smiles=current_state_smiles,
        )
    compact: dict[str, Any] = {
        "ok": True,
        "code": result.get("code", "PASS"),
        "observation_mode": "compact_full_state_v1",
        "current_state_smiles": current_state_smiles,
        "remaining_tool_calls": result.get("remaining_tool_calls"),
    }
    if include_inventory:
        for key in ("sources", "sinks"):
            if key in result:
                compact[key] = result[key]
    for key in ("pending_import_count", "trace_bound"):
        if key in result:
            compact[key] = result[key]
    return compact


def compact_transition_observation(
    *,
    result: Mapping[str, Any],
    state_before: str,
    state_after: str,
    moves: Sequence[Mapping[str, Any]],
    radius: int = 1,
    include_local_state: bool = False,
) -> dict[str, Any]:
    """Build an action result, optionally with a local-state observation."""

    if not result.get("ok"):
        return compact_failure_observation(
            result,
            observation_mode=(
                "reaction_center_delta" if include_local_state else "action_delta"
            ),
        )
    centers = move_atom_maps(moves)
    compact: dict[str, Any] = {
        "ok": True,
        "code": result.get("code", "PASS"),
        "observation_mode": (
            "reaction_center_delta_v1" if include_local_state else "action_delta_v1"
        ),
        "remaining_tool_calls": result.get("remaining_tool_calls"),
    }
    if include_local_state:
        compact.update(
            {
                "changed_atom_maps": list(centers),
                "local_state_before": mapped_reaction_center_smiles(
                    state_before, centers, radius=radius
                ),
                "local_state_after": mapped_reaction_center_smiles(
                    state_after, centers, radius=radius
                ),
            }
        )
    for key in ("trace_bound",):
        if key in result:
            compact[key] = result[key]
    return compact


def compact_terminal_observation(
    result: Mapping[str, Any], *, observation_mode: str = "action_delta"
) -> dict[str, Any]:
    """Expose the executed endpoint once while retaining full audit data inside."""

    keep = [
        "ok",
        "formal_execute",
        "endpoint_exact",
        "derived_precursor",
        "trace_bound",
        "trace_digest",
        "move_sequence_digest",
        "n_trace_transitions",
        "endpoint_source",
        "declared_moves_replayed",
        "reward",
        "tool_calls",
        "successful_steps",
        "failed_steps",
    ]
    if observation_mode == "compact_full_state":
        keep.remove("trace_digest")
        keep.remove("move_sequence_digest")
    compact = {key: result[key] for key in keep if key in result}
    compact["observation_mode"] = f"{observation_mode}_v1"
    return compact
