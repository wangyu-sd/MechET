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


def visible_successor_state(state: str) -> str:
    """Canonical public SMILES for either mapped executor or public rollout state."""
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
        f"TARGET PRODUCT SMILES: {visible_successor_state(target)}\n"
        f"CURRENT STATE SMILES: {visible_successor_state(current_state)}\n"
        f"CANDIDATE SUCCESSOR SMILES: {visible_successor_state(successor_state)}\n"
        f"CANDIDATE TERMINAL: {'yes' if terminal else 'no'}\n\n"
        "Return P or N."
    )


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
    visible_successor = visible_successor_state(successor_state)
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
