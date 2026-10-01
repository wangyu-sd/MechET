"""Successor-level value supervision for executable electron-flow search.

The critic judges a *transition* rather than a state in isolation.  Positive
labels are executor-produced reference successors.  Negative labels are
distinct executable successors proposed by the current actor at the same
state.  The frozen precursor endpoint is never part of the model input.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from rdkit import Chem

from .in_place_grounded_flow import deterministic_unmapped_state


VERSION = "natural_language_successor_value_v1"
SUCCESSOR_VALUE_SYSTEM = (
    "You are the MechET transition-value critic. Judge whether the candidate "
    "executor-produced successor is a productive next state for retrosynthetic "
    "electron-flow reasoning from the target product. Reply with exactly P for "
    "a productive successor or N for a lower-value successor. Do not explain."
)
REACHABILITY_VALUE_SYSTEM = (
    "You are the MechET bounded-reachability critic. Given an executor-produced "
    "candidate successor and a remaining decision budget, estimate whether "
    "the current search policy can reach a valid precursor endpoint within "
    "that budget. Reply with exactly P or N. This is a search-survival estimate, "
    "not a claim of chemical impossibility."
)


def _visible(state: str) -> str:
    raw = str(state)
    mol = Chem.MolFromSmiles(raw)
    if mol is None:
        raise ValueError("invalid successor-value molecular state")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if maps and all(value > 0 for value in maps) and len(set(maps)) == len(maps):
        return deterministic_unmapped_state(raw).text
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def successor_value_prompt(
    target: str,
    current_state: str,
    successor_state: str,
    *,
    terminal: bool,
) -> str:
    return (
        f"TARGET PRODUCT SMILES: {_visible(target)}\n"
        f"CURRENT STATE SMILES: {_visible(current_state)}\n"
        f"CANDIDATE SUCCESSOR SMILES: {_visible(successor_state)}\n"
        f"CANDIDATE TERMINAL: {'yes' if terminal else 'no'}\n\n"
        "Return P or N."
    )


def reachability_value_prompt(
    target: str,
    current_state: str,
    successor_state: str,
    *,
    remaining_decisions: int,
    terminal: bool,
) -> str:
    if remaining_decisions < 0:
        raise ValueError("remaining decision budget cannot be negative")
    return (
        f"TARGET PRODUCT SMILES: {_visible(target)}\n"
        f"CURRENT STATE SMILES: {_visible(current_state)}\n"
        f"CANDIDATE SUCCESSOR SMILES: {_visible(successor_state)}\n"
        f"CANDIDATE TERMINAL: {'yes' if terminal else 'no'}\n"
        f"REMAINING DECISIONS: {remaining_decisions}\n\n"
        "Return P or N."
    )


def reachability_value_row(
    *, reaction_id: str, state_hash: str, target: str,
    current_state: str, successor_state: str, terminal: bool,
    remaining_decisions: int, label: str, provenance: str,
    observations: int,
) -> dict[str, Any]:
    if label not in {"P", "N"} or observations < 1:
        raise ValueError("invalid reachability label/observation count")
    prompt = reachability_value_prompt(
        target, current_state, successor_state,
        remaining_decisions=remaining_decisions, terminal=terminal,
    )
    digest = hashlib.sha256(
        f"{state_hash}:{remaining_decisions}:{int(terminal)}:{_visible(successor_state)}".encode()
    ).hexdigest()[:20]
    return {
        "id": f"{reaction_id}::reachability::{digest}",
        "source_id": reaction_id,
        "artifact_type": "supervision",
        "task_type": "bounded_on_policy_successor_reachability_v1",
        "messages": [
            {"role": "system", "content": REACHABILITY_VALUE_SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label},
        ],
        "tools": [],
        "metadata": {
            "label": label, "provenance": provenance,
            "anchor_state_hash": state_hash,
            "remaining_decisions": remaining_decisions,
            "observations": observations,
            "executor_successor": True,
            "reference_endpoint_model_visible": False,
            "negative_means_chemically_impossible": False,
        },
    }


def successor_value_row(
    *,
    reaction_id: str,
    state_hash: str,
    target: str,
    current_state: str,
    successor_state: str,
    terminal: bool,
    label: str,
    provenance: str,
) -> dict[str, Any]:
    if label not in {"P", "N"}:
        raise ValueError(f"invalid successor-value label: {label}")
    visible_successor = _visible(successor_state)
    digest = hashlib.sha256(
        f"{state_hash}:{int(terminal)}:{visible_successor}:{label}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "id": f"{reaction_id}::successor_value::{digest}",
        "source_id": reaction_id,
        "artifact_type": "supervision",
        "task_type": VERSION,
        "messages": [
            {"role": "system", "content": SUCCESSOR_VALUE_SYSTEM},
            {
                "role": "user",
                "content": successor_value_prompt(
                    target,
                    current_state,
                    successor_state,
                    terminal=terminal,
                ),
            },
            {"role": "assistant", "content": label},
        ],
        "tools": [],
        "metadata": {
            "label": label,
            "provenance": provenance,
            "anchor_state_hash": state_hash,
            "executor_successor": True,
            "model_visible_atom_maps": False,
            "reference_endpoint_model_visible": False,
        },
    }


def successor_value_margin(label_logps: Mapping[str, float]) -> float:
    if set(label_logps) != {"P", "N"}:
        raise ValueError("successor value requires exactly P/N log probabilities")
    return float(label_logps["P"]) - float(label_logps["N"])
