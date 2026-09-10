"""In-place grounded electron-flow serialization and exact round-trip audit.

The model sees deterministic unmapped molecular strings.  Event-local role
markers are inserted immediately before atom occurrences and compile back to
the existing mapped ``ElectronMove`` representation only inside the executor.
"""
from __future__ import annotations

from ast import literal_eval
from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem

from .a7_rescue import canonical_event, canonical_mapped_state
from .forward_expert import ElectronContainer, ElectronMove, verify_electron_step


ROLE_NAMES = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))
MARKER_RE = re.compile(r"<([A-Z])>")
STANDARD_CLAUSE_RE = re.compile(r"^(\*?[A-Z]{1,2}\*?)>(\*?[A-Z]{1,2}\*?)$")
BOND_DELTA_RE = re.compile(r"^BOND ([A-Z]{2}) ([+-][123])$")
CHARGE_DELTA_RE = re.compile(r"^CHARGE ([A-Z]) (-?\d+) (-?\d+)$")


@dataclass(frozen=True)
class MolecularSerialization:
    text: str
    atom_maps: tuple[int, ...]
    atom_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.atom_maps) != len(self.atom_spans):
            raise ValueError("atom-map/output-span length mismatch")


@dataclass(frozen=True)
class GroundedEvent:
    marked_state: str
    flow: str
    role_to_map: Mapping[str, int]
    compiled_moves: tuple[dict[str, Any], ...]


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("invalid mapped molecular state")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError("all atoms require unique positive maps")
    return mol


def mapped_atom_numbers(smiles: str) -> set[int]:
    return {int(atom.GetAtomMapNum()) for atom in _mapped_mol(smiles).GetAtoms()}


def atom_token_spans(smiles: str) -> tuple[tuple[int, int], ...]:
    """Return atom-token spans without treating bracket contents as atoms."""

    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            end = smiles.find("]", index + 1)
            if end < 0:
                raise ValueError("unclosed bracket atom")
            spans.append((index, end + 1))
            index = end + 1
            continue
        pair = smiles[index : index + 2]
        if pair in {"Cl", "Br", "se", "as"}:
            spans.append((index, index + 2))
            index += 2
            continue
        if char in "BCNOPSFIKbcnops*":
            spans.append((index, index + 1))
        index += 1
    return tuple(spans)


def deterministic_unmapped_state(mapped_smiles: str) -> MolecularSerialization:
    """Serialize without maps while retaining a private occurrence-to-map table."""

    mol = _mapped_mol(mapped_smiles)
    index_to_map = tuple(int(atom.GetAtomMapNum()) for atom in mol.GetAtoms())
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    text = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        output_order = tuple(int(value) for value in literal_eval(
            mol.GetProp("_smilesAtomOutputOrder")
        ))
    except Exception as exc:
        raise ValueError("RDKit did not expose deterministic SMILES atom order") from exc
    spans = atom_token_spans(text)
    if len(output_order) != mol.GetNumAtoms() or len(spans) != mol.GetNumAtoms():
        raise ValueError(
            "serialized atom count mismatch: "
            f"order={len(output_order)} spans={len(spans)} atoms={mol.GetNumAtoms()}"
        )
    return MolecularSerialization(
        text=text,
        atom_maps=tuple(index_to_map[index] for index in output_order),
        atom_spans=spans,
    )


def _maps_in_event(moves: Sequence[Mapping[str, Any]]) -> set[int]:
    output: set[int] = set()
    for raw in moves:
        if raw.get("mode") == "BE_DELTA":
            for item in raw.get("bond_deltas") or []:
                output.update(int(value) for value in item.get("atoms") or [])
            for item in raw.get("charge_actions") or []:
                output.add(int(item["atom_map"]))
            continue
        move = ElectronMove.parse(raw)
        output.update(move.source.atoms)
        output.update(move.sink.atoms)
    return output


def _insert_markers(
    serialization: MolecularSerialization, role_to_map: Mapping[str, int]
) -> str:
    map_to_role = {int(atom_map): role for role, atom_map in role_to_map.items()}
    if len(map_to_role) != len(role_to_map):
        raise ValueError("one atom cannot have multiple event roles")
    pieces: list[str] = []
    cursor = 0
    for atom_map, (start, end) in zip(
        serialization.atom_maps, serialization.atom_spans
    ):
        pieces.append(serialization.text[cursor:start])
        role = map_to_role.get(atom_map)
        if role:
            pieces.append(f"<{role}>")
        pieces.append(serialization.text[start:end])
        cursor = end
    pieces.append(serialization.text[cursor:])
    return "".join(pieces)


def _container_token(container: ElectronContainer, map_to_role: Mapping[int, str]) -> str:
    labels = "".join(sorted(map_to_role[int(value)] for value in container.atoms))
    if container.kind == "RADICAL_PAIR":
        return f"*{labels}*"
    return labels


def compact_flow(
    moves: Sequence[Mapping[str, Any]], map_to_role: Mapping[int, str]
) -> str:
    clauses: list[str] = []
    if any(raw.get("mode") == "BE_DELTA" for raw in moves):
        if len(moves) != 1 or moves[0].get("mode") != "BE_DELTA":
            raise ValueError("BE_DELTA must be the sole event payload")
        raw = moves[0]
        for item in raw.get("bond_deltas") or []:
            labels = "".join(
                sorted(map_to_role[int(value)] for value in item.get("atoms") or [])
            )
            clauses.append(f"BOND {labels} {int(item['delta']):+d}")
        for item in raw.get("charge_actions") or []:
            role = map_to_role[int(item["atom_map"])]
            clauses.append(f"CHARGE {role} {int(item['q0'])} {int(item['q1'])}")
        if not clauses:
            raise ValueError("empty BE_DELTA event")
        return "DELTA " + " ; ".join(clauses)

    for raw in moves:
        move = ElectronMove.parse(raw)
        clauses.append(
            f"{_container_token(move.source, map_to_role)}>"
            f"{_container_token(move.sink, map_to_role)}"
        )
    if not clauses:
        raise ValueError("empty electron-flow event")
    return " ; ".join(clauses)


def _marked_bindings(
    marked_state: str, authoritative: MolecularSerialization
) -> dict[str, int]:
    clean_parts: list[str] = []
    markers_at: dict[int, str] = {}
    source_cursor = clean_cursor = 0
    for match in MARKER_RE.finditer(marked_state):
        clean_parts.append(marked_state[source_cursor : match.start()])
        clean_cursor += match.start() - source_cursor
        role = match.group(1)
        if role in markers_at.values():
            raise ValueError(f"duplicate event role marker: {role}")
        if clean_cursor in markers_at:
            raise ValueError("multiple markers at one position")
        markers_at[clean_cursor] = role
        source_cursor = match.end()
    clean_parts.append(marked_state[source_cursor:])
    clean = "".join(clean_parts)
    if clean != authoritative.text:
        raise ValueError("marked state is not insertion-only over authoritative state")
    span_to_map = {
        start: atom_map
        for atom_map, (start, _) in zip(
            authoritative.atom_maps, authoritative.atom_spans
        )
    }
    if set(markers_at) - set(span_to_map):
        raise ValueError("a marker is not immediately before an atom occurrence")
    return {role: span_to_map[position] for position, role in markers_at.items()}


def _parse_container(token: str, *, source: bool, bindings: Mapping[str, int]) -> ElectronContainer:
    radical = token.startswith("*") and token.endswith("*")
    labels = token[1:-1] if radical else token
    if not labels or any(label not in bindings for label in labels):
        raise ValueError(f"unknown role in container: {token}")
    atoms = tuple(bindings[label] for label in labels)
    if radical:
        kind = "RADICAL_PAIR"
    elif len(labels) == 2:
        kind = "BOND"
    elif len(labels) == 1:
        kind = "LP" if source else "ATOM"
    else:
        raise ValueError(f"invalid compact container: {token}")
    return ElectronContainer(kind, atoms)


def compile_flow(
    *,
    mapped_state: str,
    marked_state: str,
    flow: str,
) -> list[dict[str, Any]]:
    authoritative = deterministic_unmapped_state(mapped_state)
    bindings = _marked_bindings(marked_state, authoritative)
    value = str(flow or "").strip()
    if value.startswith("DELTA "):
        bond_deltas: list[dict[str, Any]] = []
        charge_actions: list[dict[str, Any]] = []
        for clause in (part.strip() for part in value[6:].split(";")):
            bond_match = BOND_DELTA_RE.fullmatch(clause)
            if bond_match:
                labels, delta = bond_match.groups()
                bond_deltas.append(
                    {
                        "atoms": sorted(bindings[label] for label in labels),
                        "delta": int(delta),
                    }
                )
                continue
            charge_match = CHARGE_DELTA_RE.fullmatch(clause)
            if charge_match:
                role, q0, q1 = charge_match.groups()
                charge_actions.append(
                    {"atom_map": bindings[role], "q0": int(q0), "q1": int(q1)}
                )
                continue
            raise ValueError(f"invalid DELTA clause: {clause}")
        return [
            {
                "mode": "BE_DELTA",
                "bond_deltas": bond_deltas,
                "charge_actions": charge_actions,
            }
        ]

    moves: list[dict[str, Any]] = []
    for clause in (part.strip() for part in value.split(";")):
        match = STANDARD_CLAUSE_RE.fullmatch(clause)
        if not match:
            raise ValueError(f"invalid FLOW clause: {clause}")
        source = _parse_container(match.group(1), source=True, bindings=bindings)
        sink = _parse_container(match.group(2), source=False, bindings=bindings)
        moves.append(ElectronMove(source, sink).to_dict())
    if not moves:
        raise ValueError("FLOW has no moves")
    return moves


def encode_grounded_event(
    mapped_state: str, moves: Sequence[Mapping[str, Any]]
) -> GroundedEvent:
    serialization = deterministic_unmapped_state(mapped_state)
    used_maps = _maps_in_event(moves)
    missing = used_maps - set(serialization.atom_maps)
    if missing:
        raise ValueError(f"event references absent atoms: {sorted(missing)}")
    ordered = [value for value in serialization.atom_maps if value in used_maps]
    if len(ordered) > len(ROLE_NAMES):
        raise ValueError(f"event needs {len(ordered)} roles; maximum is {len(ROLE_NAMES)}")
    role_to_map = dict(zip(ROLE_NAMES, ordered))
    map_to_role = {atom_map: role for role, atom_map in role_to_map.items()}
    marked = _insert_markers(serialization, role_to_map)
    flow = compact_flow(moves, map_to_role)
    compiled = compile_flow(
        mapped_state=mapped_state,
        marked_state=marked,
        flow=flow,
    )
    if canonical_event(compiled) != canonical_event(moves):
        raise ValueError("compact FLOW move round-trip mismatch")
    return GroundedEvent(marked, flow, role_to_map, tuple(compiled))


def merge_mapped_fragments(state: str, fragments: Iterable[str]) -> str:
    pieces = [str(state)]
    present = mapped_atom_numbers(state)
    for fragment in fragments:
        incoming = mapped_atom_numbers(fragment)
        overlap = present & incoming
        if overlap:
            raise ValueError(f"fragment map overlap: {sorted(overlap)}")
        pieces.append(str(fragment))
        present.update(incoming)
    return canonical_mapped_state(".".join(pieces))


def extract_import_fragments(row: Mapping[str, Any]) -> list[str]:
    fragments: list[str] = []
    for message in row.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "import_fragment":
                continue
            fragment = str((function.get("arguments") or {}).get("fragment_smiles") or "")
            if not fragment:
                raise ValueError("empty mapped import fragment")
            fragments.append(fragment)
    return fragments


def schedule_imports(
    fragments: Sequence[str], steps: Sequence[Mapping[str, Any]]
) -> list[list[str]]:
    if not steps:
        raise ValueError("reaction has no electron-flow events")
    used_by_step = [_maps_in_event(list(step.get("moves") or [])) for step in steps]
    scheduled: list[list[str]] = [[] for _ in steps]
    for fragment in fragments:
        fragment_maps = mapped_atom_numbers(fragment)
        index = next(
            (i for i, used in enumerate(used_by_step) if fragment_maps & used),
            len(steps) - 1,
        )
        scheduled[index].append(fragment)
    return scheduled


def _tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "apply_grounded_event",
                "description": (
                    "Import any first-use fragments, insert event-local role markers "
                    "into the exact current state, and execute one coupled electron-flow event."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "imports": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "marked_state": {"type": "string"},
                        "flow": {"type": "string"},
                    },
                    "required": ["imports", "marked_state", "flow"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_trace",
                "description": "Replay the committed events and derive the precursor endpoint.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]


def convert_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one strict executable trace without filtering or map exposure."""

    identifier = str(row.get("id") or "")
    metadata = dict(row.get("metadata") or {})
    plan = dict(metadata.get("trace_plan") or {})
    steps = [dict(item) for item in plan.get("steps") or []]
    target = str(row.get("target_smiles") or plan.get("target_smiles") or "")
    expected = str(row.get("full_precursor_state") or row.get("expected_precursor") or "")
    if not identifier or not target or not expected or not steps:
        raise ValueError(f"{identifier}: incomplete strict trace row")

    fragments = extract_import_fragments(row)
    target_maps = mapped_atom_numbers(target)
    expected_import_maps = mapped_atom_numbers(expected) - target_maps
    observed_import_maps = set().union(
        *(mapped_atom_numbers(fragment) for fragment in fragments)
    ) if fragments else set()
    if observed_import_maps != expected_import_maps:
        raise ValueError(
            f"{identifier}: imported maps do not match endpoint-only maps: "
            f"missing={sorted(expected_import_maps - observed_import_maps)} "
            f"extra={sorted(observed_import_maps - expected_import_maps)}"
        )
    scheduled = schedule_imports(fragments, steps)
    current = canonical_mapped_state(target)
    target_visible = deterministic_unmapped_state(current).text
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Infer the precursor by executing a sequence of grounded electron-flow events. "
                "For each event, import only first-use unmapped fragments; copy the exact CURRENT "
                "STATE and only insert event-local <A>..<Z> markers immediately before atom "
                "occurrences; then emit compact FLOW clauses. A>AB means LP(A)->BOND(A,B), "
                "AB>B means BOND(A,B)->ATOM(B), and AB>BC means a bond shift. "
                "Starred pairs encode radical-pair containers; DELTA BOND/CHARGE "
                "clauses preserve executor-native bond/charge delta events. "
                "The environment owns every state transition and the final precursor."
            ),
        },
        {
            "role": "user",
            "content": f"TARGET: {target_visible}\nCURRENT STATE: {target_visible}",
        },
    ]

    event_audits: list[dict[str, Any]] = []
    for event_index, (step, event_imports) in enumerate(zip(steps, scheduled)):
        current = merge_mapped_fragments(current, event_imports)
        moves = [dict(item) for item in step.get("moves") or []]
        grounded = encode_grounded_event(current, moves)
        replay = verify_electron_step(current, list(grounded.compiled_moves))
        if not replay.get("ok"):
            raise ValueError(
                f"{identifier}: event {event_index} replay failed: {replay}"
            )
        successor = canonical_mapped_state(str(replay["state_smiles"]))
        call_id = f"event_{event_index:03d}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "apply_grounded_event",
                            "arguments": {
                                "imports": [
                                    deterministic_unmapped_state(fragment).text
                                    for fragment in event_imports
                                ],
                                "marked_state": grounded.marked_state,
                                "flow": grounded.flow,
                            },
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "apply_grounded_event",
                "content": json.dumps(
                    {
                        "ok": True,
                        "code": "PASS",
                        "current_state": deterministic_unmapped_state(successor).text,
                        "remaining_events": len(steps) - event_index - 1,
                    },
                    separators=(",", ":"),
                ),
            }
        )
        event_audits.append(
            {
                "event_index": event_index,
                "n_moves": len(moves),
                "n_roles": len(grounded.role_to_map),
                "n_imports": len(event_imports),
                "move_roundtrip": True,
                "successor_replay": True,
                "be_delta": any(item.get("mode") == "BE_DELTA" for item in moves),
            }
        )
        current = successor

    if canonical_mapped_state(current) != canonical_mapped_state(expected):
        raise ValueError(f"{identifier}: final endpoint mismatch after rescheduled replay")
    finish_id = "finish_trace"
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": finish_id,
                        "type": "function",
                        "function": {"name": "finish_trace", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": finish_id,
                "name": "finish_trace",
                "content": json.dumps(
                    {
                        "ok": True,
                        "formal_execute": True,
                        "endpoint_exact": True,
                        "derived_precursor": deterministic_unmapped_state(expected).text,
                        "endpoint_source": "environment_owned_trace",
                    },
                    separators=(",", ":"),
                ),
            },
            {
                "role": "assistant",
                "content": "The grounded electron-flow trace executed successfully.",
            },
        ]
    )
    visible_payload = json.dumps(
        {"messages": messages, "tools": _tool_schema()}, ensure_ascii=False
    )
    if re.search(r":\d+\]", visible_payload):
        raise ValueError(f"{identifier}: atom map leaked into model-visible payload")
    return {
        "id": identifier,
        "source_id": str(row.get("source_id") or identifier),
        "artifact_type": "supervision",
        "task_type": "in_place_grounded_flow_v1",
        "target_smiles": target_visible,
        "expected_precursor": deterministic_unmapped_state(expected).text,
        "messages": messages,
        "tools": _tool_schema(),
        "metadata": {
            "representation": "in_place_grounded_flow_v1",
            "endpoint_source": "environment_owned_trace",
            "executor_replayed": True,
            "trace_digest": str(metadata.get("trace_digest") or "source-trace"),
            "move_sequence_digest": str(metadata.get("move_sequence_digest") or ""),
            "n_events": len(steps),
            "n_import_fragments": len(fragments),
            "mapping_model_visible": False,
            "marker_edit_contract": "insertion_only",
            "event_audits": event_audits,
        },
    }
