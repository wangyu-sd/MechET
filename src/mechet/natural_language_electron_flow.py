"""Natural-language electron-flow actions with executor-owned atom grounding.

The model sees an unmapped SMILES string plus a compact atom/bond inventory.
Temporary ``Axx``/``Bxx`` names are regenerated from the current state at every
decision and are never molecular atom maps.  A model describes electron flow
using ordinary chemical phrases; the executor resolves those phrases back to
private mapped electron containers and performs the authoritative replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .forward_expert import ElectronContainer, ElectronMove, verify_electron_step
from .in_place_grounded_flow import deterministic_unmapped_state


def _mapped_mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles or ""), params)
    if mol is None:
        raise ValueError("invalid mapped state")
    maps = [int(atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    if any(value <= 0 for value in maps) or len(set(maps)) != len(maps):
        raise ValueError("state requires unique positive private atom maps")
    return mol


@dataclass(frozen=True)
class NaturalLanguageInventory:
    """One deterministic public inventory and its private resolver tables."""

    visible_smiles: str
    atom_to_map: Mapping[str, int]
    map_to_atom: Mapping[int, str]
    bond_to_maps: Mapping[str, tuple[int, int]]
    maps_to_bond: Mapping[tuple[int, int], str]
    prompt: str


def build_inventory(mapped_state: str) -> NaturalLanguageInventory:
    """Annotate every SMILES atom token without exposing private maps."""

    mol = _mapped_mol(mapped_state)
    serialization = deterministic_unmapped_state(mapped_state)
    atom_to_map = {
        f"A{index:02d}": int(atom_map)
        for index, atom_map in enumerate(serialization.atom_maps, 1)
    }
    map_to_atom = {value: key for key, value in atom_to_map.items()}
    pairs: list[tuple[str, str, int, int, Chem.Bond]] = []
    for bond in mol.GetBonds():
        left_map = int(bond.GetBeginAtom().GetAtomMapNum())
        right_map = int(bond.GetEndAtom().GetAtomMapNum())
        left, right = sorted((map_to_atom[left_map], map_to_atom[right_map]))
        pairs.append((left, right, left_map, right_map, bond))
    pairs.sort(key=lambda item: (item[0], item[1]))
    bond_to_maps: dict[str, tuple[int, int]] = {}
    maps_to_bond: dict[tuple[int, int], str] = {}
    for index, (left, right, left_map, right_map, _) in enumerate(pairs, 1):
        alias = f"B{index:02d}"
        key = tuple(sorted((left_map, right_map)))
        bond_to_maps[alias] = key
        maps_to_bond[key] = alias

    map_to_alias = {value: key for key, value in atom_to_map.items()}
    pieces: list[str] = []
    cursor = 0
    for atom_map, (start, end) in zip(serialization.atom_maps, serialization.atom_spans):
        pieces.append(serialization.text[cursor:start])
        pieces.append(f"<{map_to_alias[int(atom_map)]}>")
        pieces.append(serialization.text[start:end])
        cursor = end
    pieces.append(serialization.text[cursor:])
    annotated = "".join(pieces)
    prompt = (
        "Every atom token is preceded by a temporary name in the annotated SMILES. "
        "Read elements, charges, branches, rings, and bonds directly from this structure.\n"
        f"ANNOTATED CURRENT STATE: {annotated}"
    )
    return NaturalLanguageInventory(
        visible_smiles=serialization.text,
        atom_to_map=atom_to_map,
        map_to_atom=map_to_atom,
        bond_to_maps=bond_to_maps,
        maps_to_bond=maps_to_bond,
        prompt=prompt,
    )


def _atom_phrase(inventory: NaturalLanguageInventory, atom_map: int) -> str:
    return f"atom {inventory.map_to_atom[int(atom_map)]}"


def _bond_phrase(
    inventory: NaturalLanguageInventory, atoms: Sequence[int], *, source: bool
) -> str:
    pair = tuple(sorted(int(value) for value in atoms))
    alias = inventory.maps_to_bond.get(pair)
    if alias:
        left, right = sorted(inventory.map_to_atom[value] for value in pair)
        return f"the bond between atoms {left} and {right}"
    if source:
        raise ValueError(f"electron-source bond is absent: {pair}")
    left, right = (inventory.map_to_atom[value] for value in pair)
    return f"the bond to form between atoms {left} and {right}"


def _container_phrase(
    inventory: NaturalLanguageInventory,
    container: ElectronContainer,
    *,
    source: bool,
) -> str:
    if container.kind == "LP":
        if not source or len(container.atoms) != 1:
            raise ValueError("invalid lone-pair container")
        return f"a lone pair on {_atom_phrase(inventory, container.atoms[0])}"
    if container.kind == "ATOM":
        if source or len(container.atoms) != 1:
            raise ValueError("invalid atom container")
        return _atom_phrase(inventory, container.atoms[0])
    if container.kind == "BOND":
        return _bond_phrase(inventory, container.atoms, source=source)
    if container.kind == "RADICAL_PAIR":
        left, right = sorted(inventory.map_to_atom[int(v)] for v in container.atoms)
        return f"a radical pair on atoms {left} and {right}"
    raise ValueError(f"unsupported electron container: {container.kind}")


def render_event_arguments(
    mapped_state: str, moves: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Render a reference event into constrained natural-language fields."""

    inventory = build_inventory(mapped_state)
    if any(item.get("mode") == "BE_DELTA" for item in moves):
        if len(moves) != 1 or moves[0].get("mode") != "BE_DELTA":
            raise ValueError("BE_DELTA must be the sole event payload")
        raw = moves[0]
        bond_changes = []
        for item in raw.get("bond_deltas") or []:
            left, right = sorted(
                inventory.map_to_atom[int(value)] for value in item.get("atoms") or []
            )
            delta = int(item["delta"])
            verb = "increase" if delta > 0 else "decrease"
            bond_changes.append(
                {
                    "atoms": [left, right],
                    "delta": delta,
                    "instruction": (
                        f"{verb.capitalize()} the bond order between atoms {left} "
                        f"and {right} by {abs(delta)}."
                    ),
                }
            )
        charge_changes = []
        for item in raw.get("charge_actions") or []:
            atom = inventory.map_to_atom[int(item["atom_map"])]
            q0, q1 = int(item["q0"]), int(item["q1"])
            charge_changes.append(
                {
                    "atom": atom,
                    "from": q0,
                    "to": q1,
                    "instruction": (
                        f"Change the formal charge on atom {atom} from {q0:+d} "
                        f"to {q1:+d}."
                    ),
                }
            )
        return {
            "direction": "retrosynthetic",
            "electron_flow": [],
            "bond_order_changes": bond_changes,
            "charge_changes": charge_changes,
        }

    arrows = []
    for index, raw in enumerate(moves, 1):
        move = ElectronMove.parse(raw)
        source = _container_phrase(inventory, move.source, source=True)
        destination = _container_phrase(inventory, move.sink, source=False)
        arrows.append(
            {
                "source": source,
                "destination": destination,
                "instruction": (
                    f"Move {index}: transfer the electron pair from {source} "
                    f"to {destination}."
                ),
            }
        )
    if not arrows:
        raise ValueError("event contains no electron moves")
    return {
        "direction": "retrosynthetic",
        "electron_flow": arrows,
        "bond_order_changes": [],
        "charge_changes": [],
    }


def _resolve_source(text: str, inventory: NaturalLanguageInventory) -> ElectronContainer:
    value = str(text)
    prefix = "a lone pair on atom "
    if value.startswith(prefix):
        alias = value[len(prefix) :]
        return ElectronContainer("LP", (inventory.atom_to_map[alias],))
    prefix = "the bond between atoms "
    if value.startswith(prefix):
        left, right = value[len(prefix) :].split(" and ")
        pair = tuple(sorted((inventory.atom_to_map[left], inventory.atom_to_map[right])))
        if pair not in inventory.maps_to_bond:
            raise ValueError(f"electron-source bond is absent: {left},{right}")
        return ElectronContainer("BOND", pair)
    prefix = "a radical pair on atoms "
    if value.startswith(prefix):
        left, right = value[len(prefix) :].split(" and ")
        return ElectronContainer(
            "RADICAL_PAIR",
            (inventory.atom_to_map[left], inventory.atom_to_map[right]),
        )
    raise ValueError(f"unrecognized natural-language electron source: {value}")


def _resolve_destination(
    text: str, inventory: NaturalLanguageInventory
) -> ElectronContainer:
    value = str(text)
    prefix = "atom "
    if value.startswith(prefix):
        return ElectronContainer("ATOM", (inventory.atom_to_map[value[len(prefix) :]],))
    prefix = "the bond between atoms "
    if value.startswith(prefix):
        left, right = value[len(prefix) :].split(" and ")
        pair = tuple(sorted((inventory.atom_to_map[left], inventory.atom_to_map[right])))
        if pair not in inventory.maps_to_bond:
            raise ValueError(f"destination bond is absent: {left},{right}")
        return ElectronContainer("BOND", pair)
    prefix = "the bond to form between atoms "
    if value.startswith(prefix):
        left, right = value[len(prefix) :].split(" and ")
        return ElectronContainer(
            "BOND", (inventory.atom_to_map[left], inventory.atom_to_map[right])
        )
    prefix = "a radical pair on atoms "
    if value.startswith(prefix):
        left, right = value[len(prefix) :].split(" and ")
        return ElectronContainer(
            "RADICAL_PAIR",
            (inventory.atom_to_map[left], inventory.atom_to_map[right]),
        )
    raise ValueError(f"unrecognized natural-language electron destination: {value}")


def compile_event_arguments(
    mapped_state: str, arguments: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Resolve constrained natural language to executor-native moves."""

    if arguments.get("direction") != "retrosynthetic":
        raise ValueError("event direction must be retrosynthetic")
    inventory = build_inventory(mapped_state)
    arrows = list(arguments.get("electron_flow") or [])
    bond_changes = list(arguments.get("bond_order_changes") or [])
    charge_changes = list(arguments.get("charge_changes") or [])
    if arrows and (bond_changes or charge_changes):
        raise ValueError("ordinary arrows and BE_DELTA changes cannot be mixed")
    if arrows:
        output = []
        for arrow in arrows:
            move = ElectronMove(
                _resolve_source(str(arrow["source"]), inventory),
                _resolve_destination(str(arrow["destination"]), inventory),
            )
            output.append(move.to_dict())
        return output
    if not bond_changes and not charge_changes:
        raise ValueError("natural-language event is empty")
    return [
        {
            "mode": "BE_DELTA",
            "bond_deltas": [
                {
                    "atoms": sorted(
                        inventory.atom_to_map[str(alias)]
                        for alias in item.get("atoms") or []
                    ),
                    "delta": int(item["delta"]),
                }
                for item in bond_changes
            ],
            "charge_actions": [
                {
                    "atom_map": inventory.atom_to_map[str(item["atom"])],
                    "q0": int(item["from"]),
                    "q1": int(item["to"]),
                }
                for item in charge_changes
            ],
        }
    ]


def execute_event_arguments(
    mapped_state: str, arguments: Mapping[str, Any]
) -> dict[str, Any]:
    """Compile and replay one natural-language event without hidden correction."""

    moves = compile_event_arguments(mapped_state, arguments)
    return verify_electron_step(mapped_state, moves)
