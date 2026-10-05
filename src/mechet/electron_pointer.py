"""Protocol-v2 electron-flow localization labels for the PR71 pointer pilot.

The candidate universe is derived only from the current, model-visible SMILES.
Gold tool calls label candidates; they never enter the model input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rdkit import Chem


ATOM = re.compile(r"<A(\d+)>|A(\d+)")
ANNOTATED = re.compile(r"^ANNOTATED CURRENT STATE: (.+)$", re.MULTILINE)
PHRASE_ATOMS = re.compile(r"A(\d+)")


class UnsupportedPointerEvent(ValueError):
    """Electron-flow label outside the frozen pointer candidate universe."""


@dataclass(frozen=True)
class PointerObservation:
    row_id: str
    annotated: str
    atom_names: tuple[str, ...]
    atom_spans: tuple[tuple[int, int], ...]
    bonds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PointerExample(PointerObservation):
    messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    tools: list[dict[str, Any]]
    source_targets: tuple[tuple[str, int, int], ...]
    sink_targets: tuple[tuple[str, int, int], ...]


def _phrase_target(phrase: str) -> tuple[str, int, int]:
    names = [int(x) - 1 for x in PHRASE_ATOMS.findall(phrase)]
    if "radical pair" in phrase.lower():
        raise UnsupportedPointerEvent(f"radical-pair pointer is outside the pilot universe: {phrase}")
    if "bond" in phrase:
        if len(names) != 2 or names[0] == names[1]:
            raise ValueError(f"invalid bond pointer: {phrase}")
        return ("bond", *sorted(names))
    if len(names) != 1:
        raise ValueError(f"invalid atom pointer: {phrase}")
    return ("atom", names[0], names[0])


def parse_pointer_observation(user_content: str, *, row_id: str = "runtime") -> PointerObservation:
    match = ANNOTATED.search(user_content)
    if not match:
        raise ValueError(f"{row_id}: missing annotated current state")
    annotated = match.group(1)
    marker_matches = list(re.finditer(r"<A\d+>", annotated))
    names = tuple(marker.group()[1:-1] for marker in marker_matches)
    if names != tuple(f"A{i + 1:02d}" for i in range(len(names))):
        raise ValueError(f"{row_id}: nonconsecutive atom inventory")
    smiles = re.sub(r"<A\d+>", "", annotated)
    params = Chem.SmilesParserParams()
    # build_inventory() deliberately keeps explicit H atoms when assigning
    # temporary Axx handles. Dropping them here makes a valid off-reference
    # state unaddressable after an IMPORT or electron step.
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or mol.GetNumAtoms() != len(names):
        raise ValueError(f"{row_id}: inventory does not parse to matching molecule")
    bonds = tuple(sorted(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()))
    return PointerObservation(
        row_id=row_id, annotated=annotated, atom_names=names,
        atom_spans=tuple((m.start(), m.end()) for m in marker_matches),
        bonds=bonds,
    )


def parse_pointer_example(row: dict[str, Any]) -> PointerExample | None:
    """Extract every source/sink target; imports and finishes have no pointers."""
    messages = row["messages"]
    assistant = next(msg for msg in reversed(messages) if msg["role"] == "assistant")
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "apply_electron_flow":
        return None
    user = next(msg for msg in reversed(messages) if msg["role"] == "user")
    observation = parse_pointer_observation(user["content"], row_id=row["id"])
    moves = calls[0]["function"]["arguments"].get("electron_flow") or []
    if not moves:
        # Stage-II can encode valid bond/charge-delta-only events through the
        # same apply_electron_flow tool. They are valid chemistry but have no
        # source/sink target for this pointer pilot.
        raise UnsupportedPointerEvent(f"{row['id']}: event has no source/sink electron-flow move")
    sources, sinks = action_pointer_targets(calls[0]["function"]["arguments"])
    n = len(observation.atom_names)
    if any(max(a, b) >= n for _, a, b in (*sources, *sinks)):
        raise ValueError(f"{row['id']}: target outside inventory")
    if any((a, b) not in observation.bonds for typ, a, b in sources if typ == "bond"):
        raise ValueError(f"{row['id']}: source bond absent from current state")
    return PointerExample(
        row_id=row["id"], annotated=observation.annotated,
        atom_names=observation.atom_names, atom_spans=observation.atom_spans,
        bonds=observation.bonds, messages=messages[: messages.index(assistant)],
        assistant_message=assistant,
        tools=row["tools"], source_targets=sources, sink_targets=sinks,
    )


def action_pointer_targets(
    arguments: dict[str, Any],
) -> tuple[tuple[tuple[str, int, int], ...], tuple[tuple[str, int, int], ...]]:
    moves = arguments.get("electron_flow") or []
    return (
        tuple(_phrase_target(move["source"]) for move in moves),
        tuple(_phrase_target(move["destination"]) for move in moves),
    )


def candidate_keys(n_atoms: int, bonds: tuple[tuple[int, int], ...], *, source: bool) -> list[tuple[str, int, int]]:
    """Source: atoms and present bonds. Sink: atoms and any unordered pair."""
    atoms = [("atom", i, i) for i in range(n_atoms)]
    pairs = bonds if source else tuple((i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms))
    return atoms + [("bond", i, j) for i, j in pairs]


def target_indices(example: PointerExample, *, source: bool) -> list[int]:
    candidates = candidate_keys(len(example.atom_names), example.bonds, source=source)
    indices = {key: i for i, key in enumerate(candidates)}
    targets = example.source_targets if source else example.sink_targets
    return sorted({indices[target] for target in targets})


def locate_marker_tokens(tokenizer: Any, prefix: str, example: PointerObservation) -> list[int]:
    """Use fast-tokenizer offsets to find each marker in the *rendered* prefix."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("pointer localization requires a fast tokenizer")
    anchor = "ANNOTATED CURRENT STATE: " + example.annotated
    start = prefix.rfind(anchor)
    if start < 0:
        raise ValueError(f"{example.row_id}: rendered prefix lost inventory")
    marker_base = start + len("ANNOTATED CURRENT STATE: ")
    offsets = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    output: list[int] = []
    for local_start, local_end in example.atom_spans:
        absolute_end = marker_base + local_end
        positions = [i for i, (a, b) in enumerate(offsets) if a < absolute_end <= b]
        if len(positions) != 1:
            raise ValueError(f"{example.row_id}: cannot locate marker token at {absolute_end}")
        output.append(positions[0])
    return output
