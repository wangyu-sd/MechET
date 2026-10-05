"""Gold-independent action-family observations for the PR81 Phase-1a pilot.

The three-way decision is deliberately separate from fragment generation:
IMPORT is a route choice, not a closed-vocabulary fragment prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import torch
from torch import nn
from rdkit import Chem


ANNOTATED_STATE = re.compile(r"^ANNOTATED CURRENT STATE: (.+)$", re.MULTILINE)


def parse_action_atom_names(user_content: str, row_id: str) -> tuple[str, ...]:
    """Read visible atom handles, retaining explicit hydrogens in final states."""
    match = ANNOTATED_STATE.search(user_content)
    if not match:
        raise ValueError(f"{row_id}: missing annotated current state")
    annotated = match.group(1)
    names = tuple(token[1:-1] for token in re.findall(r"<A\d+>", annotated))
    if names != tuple(f"A{i + 1:02d}" for i in range(len(names))):
        raise ValueError(f"{row_id}: nonconsecutive atom inventory")
    smiles = re.sub(r"<A\d+>", "", annotated)
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None or molecule.GetNumAtoms() != len(names):
        raise ValueError(f"{row_id}: inventory does not parse to matching molecule")
    return names


ACTION_NAMES = ("apply_electron_flow", "import_fragments", "finish_trace")
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
DECISION_TO_ACTION = {
    "event": "apply_electron_flow",
    "import": "import_fragments",
    "finish": "finish_trace",
}


@dataclass(frozen=True)
class ActionFamilyExample:
    row_id: str
    reaction_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    atom_names: tuple[str, ...]
    label: int
    history_accepted_actions: int


def parse_action_family_example(row: dict[str, Any]) -> ActionFamilyExample:
    """Remove the answer/tool result before constructing the policy input."""
    row_id = str(row["id"])
    messages = row["messages"]
    assistant_positions = [i for i, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if len(assistant_positions) != 1:
        raise ValueError(f"{row_id}: expected exactly one supervised assistant action")
    assistant_index = assistant_positions[0]
    assistant = messages[assistant_index]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"{row_id}: expected exactly one supervised tool call")
    action = calls[0]["function"]["name"]
    expected = DECISION_TO_ACTION.get(row["metadata"]["decision_type"])
    if action not in ACTION_TO_INDEX or action != expected:
        raise ValueError(f"{row_id}: action family disagrees with the source contract")
    prefix = messages[:assistant_index]
    if not prefix or prefix[-1].get("role") != "user":
        raise ValueError(f"{row_id}: decision prefix does not end in a user observation")
    if any(message.get("role") in ("assistant", "tool") for message in prefix):
        raise ValueError(f"{row_id}: decision prefix contains a previous answer/tool result")
    atom_names = parse_action_atom_names(prefix[-1]["content"], row_id)
    metadata = row["metadata"]
    history = int(metadata["history_accepted_actions"])
    if history < 0:
        raise ValueError(f"{row_id}: negative accepted-action history")
    reaction_id = str(metadata["reaction_id"])
    if not row_id.startswith(reaction_id + "::"):
        raise ValueError(f"{row_id}: reaction ID differs from decision ID")
    return ActionFamilyExample(
        row_id=row_id,
        reaction_id=reaction_id,
        messages=prefix,
        tools=row["tools"],
        atom_names=atom_names,
        label=ACTION_TO_INDEX[action],
        history_accepted_actions=history,
    )


class ActionFamilyHead(nn.Module):
    """Tiny route-choice head over a frozen post-state Qwen hidden vector."""

    def __init__(self, hidden_size: int, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, width),
            nn.GELU(),
            nn.Linear(width, len(ACTION_NAMES)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state.float())
