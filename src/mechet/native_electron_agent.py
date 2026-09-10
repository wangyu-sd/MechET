"""Transactional electron-flow tools for native LLM tool calling.

The model selects executor-owned positions and electron moves. Generated text is
never executed as Python, and reference precursors are not part of this runtime.
"""
from __future__ import annotations

import copy
from typing import Any

from rdkit import Chem

from .electron_flow_trace import ElectronFlowTrace, compile_trace_to_proof
from .forward_expert import enumerate_containers, verify_electron_step
from .reaction_mapping import fragment_multiset_unmapped


SYSTEM_PROMPT = """You are a retrosynthesis scientist. Starting only from the
product, construct an inverse electron-flow pathway through tools. The endpoint
must come from execution, never from a separately written reactant answer.
Positions pN are persistent executor identities, not chemical properties.
First inspect the real structure. For each next elementary step, decide whether
a concrete nucleophile, electrophile, acid, base, or redox participant is
missing and import it only when needed. Use apply_step for all concerted arrows
of one elementary step. A bond source sent to an atom breaks or lowers that bond;
a lone-pair source sent to a bond creates or raises that bond. This pilot supports
electron pairs, not single-electron redox. After success, reason from the returned
state. After failure, inspect and repair or undo a wrong accepted step. Formal
execution is not proof of experimental viability. Call finish only after a
nontrivial plausible pathway. Make exactly one tool call per turn."""

NONTHINKING_PROMPT = SYSTEM_PROMPT + """
Operate in non-thinking mode. Do not write a mechanism essay. First call inspect.
Then choose only the next operation. Position arrays are flat strings, for
example ["p4", "p6"], never [["p4", "p6"]]. A bond container is
{"kind":"bond","positions":["p4","p6"]}. Do not reuse example positions
without checking the actual index."""


def _schema(name: str, description: str, properties: dict, required=()) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


_STRING = {"type": "string"}
_CONTAINER = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["atom", "lp", "bond"]},
        "positions": {
            "type": "array",
            "items": _STRING,
            "minItems": 1,
            "maxItems": 2,
        },
    },
    "required": ["kind", "positions"],
    "additionalProperties": False,
}

TOOLS = [
    _schema(
        "inspect",
        "Read a complete paginated position index, or mark one position in its "
        "full connected-component SMILES. The index exposes executor-legal "
        "electron sources; it does not select a reaction center.",
        {"offset": {"type": "integer", "minimum": 0}, "position": _STRING},
    ),
    _schema(
        "import_fragment",
        "Introduce one concrete participant for the next electron step. The "
        "tool maps it and returns new persistent positions.",
        {
            "state_id": _STRING,
            "smiles": _STRING,
            "role": {
                "type": "string",
                "enum": [
                    "nucleophile",
                    "electrophile",
                    "acid",
                    "base",
                    "redox_participant",
                ],
            },
            "reason": _STRING,
        },
        ("state_id", "smiles", "role", "reason"),
    ),
    _schema(
        "apply_step",
        "Atomically execute all simultaneous two-electron moves in one "
        "elementary step. Failure leaves the state unchanged.",
        {
            "state_id": _STRING,
            "reason": _STRING,
            "moves": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {"source": _CONTAINER, "sink": _CONTAINER},
                    "required": ["source", "sink"],
                    "additionalProperties": False,
                },
            },
        },
        ("state_id", "reason", "moves"),
    ),
    _schema(
        "undo_step",
        "Discard pending imports, or undo the last accepted step and the "
        "fragments introduced for that step.",
        {"state_id": _STRING},
        ("state_id",),
    ),
    _schema(
        "finish",
        "Replay the nonempty committed trajectory, compile its proof, and "
        "derive the precursor. This does not compare with a reference answer.",
        {"state_id": _STRING, "reason": _STRING},
        ("state_id", "reason"),
    ),
]


def _mol(smiles: str) -> Chem.Mol:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(smiles), params)
    if mol is None:
        raise ValueError("INVALID_SMILES")
    return mol


def _unmap(smiles: str) -> str:
    mol = _mol(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


class NativeElectronAgent:
    """Gold-free state machine behind the five native tool schemas."""

    def __init__(self, product_smiles: str):
        mol = _mol(product_smiles)
        if any(atom.GetAtomMapNum() for atom in mol.GetAtoms()):
            raise ValueError("PRODUCT_MUST_BE_UNMAPPED")
        for index, atom in enumerate(mol.GetAtoms(), 1):
            atom.SetAtomMapNum(index)
        self.target = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
        self.current_state = self.target
        self.next_map = mol.GetNumAtoms() + 1
        self.pending_imports: list[str] = []
        self.pending_roles: list[dict[str, str]] = []
        self.trace = ElectronFlowTrace(self.target)
        self.revision = 0
        self.finished = False
        self.precursor_smiles = ""

    @property
    def state_id(self) -> str:
        return f"s{self.revision}"

    def augmented_state(self) -> str:
        return ".".join((self.current_state, *self.pending_imports))

    def observation(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "current_smiles": _unmap(self.augmented_state()),
            "accepted_steps": len(self.trace.transitions),
            "pending_imports": len(self.pending_imports),
        }

    @staticmethod
    def _position(value: Any, available: set[int]) -> int:
        if not isinstance(value, str) or not value.startswith("p") or not value[1:].isdigit():
            raise ValueError("Use a position returned by inspect, for example p4")
        number = int(value[1:])
        if number not in available:
            raise ValueError(f"UNKNOWN_POSITION {value}; call inspect")
        return number

    def inspect(self, offset: int = 0, position: str | None = None) -> dict[str, Any]:
        mol = _mol(self.augmented_state())
        available = {atom.GetAtomMapNum() for atom in mol.GetAtoms()}
        if position is not None:
            selected = self._position(position, available)
            for fragment in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
                fragment_maps = {atom.GetAtomMapNum() for atom in fragment.GetAtoms()}
                if selected not in fragment_maps:
                    continue
                for atom in fragment.GetAtoms():
                    atom.SetAtomMapNum(1 if atom.GetAtomMapNum() == selected else 0)
                return {
                    "ok": True,
                    **self.observation(),
                    "position": position,
                    "marked_component_smiles": Chem.MolToSmiles(
                        fragment, canonical=True, isomericSmiles=True
                    ),
                }
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer")
        sources, _ = enumerate_containers(self.augmented_state())
        lone_pairs = {item.atoms[0] for item in sources if item.kind == "LP"}
        source_bonds = {
            tuple(sorted(item.atoms)) for item in sources if item.kind == "BOND"
        }
        atoms = sorted(mol.GetAtoms(), key=lambda atom: atom.GetAtomMapNum())
        entries = []
        for atom in atoms[offset : offset + 24]:
            atom_map = atom.GetAtomMapNum()
            entries.append(
                {
                    "position": f"p{atom_map}",
                    "element": atom.GetSymbol(),
                    "charge": atom.GetFormalCharge(),
                    "aromatic": atom.GetIsAromatic(),
                    "hydrogens": atom.GetTotalNumHs(includeNeighbors=True),
                    "lone_pair_source": atom_map in lone_pairs,
                    "neighbors": [
                        {
                            "position": f"p{bond.GetOtherAtom(atom).GetAtomMapNum()}",
                            "bond": str(bond.GetBondType()),
                            "bond_source": tuple(
                                sorted((atom_map, bond.GetOtherAtom(atom).GetAtomMapNum()))
                            )
                            in source_bonds,
                        }
                        for bond in atom.GetBonds()
                    ],
                }
            )
        return {
            "ok": True,
            **self.observation(),
            "positions": entries,
            "total_positions": len(atoms),
            "next_offset": offset + 24 if offset + 24 < len(atoms) else None,
        }

    def _container(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"kind", "positions"}:
            raise ValueError("Electron container requires kind and positions")
        kind = value["kind"]
        positions = value["positions"]
        if kind not in {"atom", "lp", "bond"}:
            raise ValueError("Container kind must be atom, lp, or bond")
        if not isinstance(positions, list) or any(
            not isinstance(item, str) for item in positions
        ):
            raise ValueError(
                'positions must be a flat string list such as ["p4", "p6"]'
            )
        available = {
            atom.GetAtomMapNum() for atom in _mol(self.augmented_state()).GetAtoms()
        }
        atoms = [self._position(item, available) for item in positions]
        required = 2 if kind == "bond" else 1
        if len(atoms) != required or len(set(atoms)) != len(atoms):
            raise ValueError(f"{kind} requires {required} distinct position(s)")
        return {"kind": kind.upper(), "atoms": atoms}

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        snapshot = copy.deepcopy(self.__dict__)
        try:
            matching = [
                item["function"]["parameters"]
                for item in TOOLS
                if item["function"]["name"] == name
            ]
            if not matching:
                raise ValueError(f"UNKNOWN_TOOL {name}")
            schema = matching[0]
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            if set(arguments) - set(schema["properties"]):
                raise ValueError("UNKNOWN_TOOL_ARGUMENTS")
            if set(schema["required"]) - set(arguments):
                raise ValueError("MISSING_TOOL_ARGUMENTS")
            if name == "inspect":
                return self.inspect(**arguments)
            if self.finished:
                raise ValueError("ALREADY_FINISHED")
            if arguments["state_id"] != self.state_id:
                raise ValueError(f"STALE_STATE: current state_id is {self.state_id}")
            if name == "import_fragment":
                if arguments["role"] not in {
                    "nucleophile",
                    "electrophile",
                    "acid",
                    "base",
                    "redox_participant",
                }:
                    raise ValueError("INVALID_IMPORT_ROLE")
                if len(self.pending_imports) >= 4:
                    raise ValueError("PENDING_IMPORT_BUDGET_EXCEEDED")
                fragment = _mol(arguments["smiles"])
                if any(atom.GetAtomMapNum() for atom in fragment.GetAtoms()):
                    raise ValueError("IMPORT_MUST_BE_UNMAPPED")
                if fragment.GetNumAtoms() > 64:
                    raise ValueError("IMPORT_ATOM_BUDGET_EXCEEDED")
                start = self.next_map
                for atom in fragment.GetAtoms():
                    atom.SetAtomMapNum(self.next_map)
                    self.next_map += 1
                self.pending_imports.append(
                    Chem.MolToSmiles(fragment, canonical=False, isomericSmiles=True)
                )
                self.pending_roles.append(
                    {"role": arguments["role"], "reason": arguments["reason"]}
                )
                self.revision += 1
                return {
                    "ok": True,
                    **self.observation(),
                    "new_positions": [f"p{i}" for i in range(start, self.next_map)],
                }
            if name == "apply_step":
                if len(self.trace.transitions) >= 16:
                    raise ValueError("STEP_BUDGET_EXCEEDED")
                if not isinstance(arguments["moves"], list):
                    raise ValueError("moves must be an array")
                if not 1 <= len(arguments["moves"]) <= 12:
                    raise ValueError("MOVE_BUDGET_INVALID")
                moves = []
                for move in arguments["moves"]:
                    if not isinstance(move, dict) or set(move) != {"source", "sink"}:
                        raise ValueError("Each move requires source and sink")
                    moves.append(
                        {
                            "source": self._container(move["source"]),
                            "sink": self._container(move["sink"]),
                            "electrons": 2,
                        }
                    )
                before = self.augmented_state()
                result = verify_electron_step(before, moves)
                if not result.get("ok"):
                    raise ValueError(
                        f"STEP_FAILED {result.get('code')}: {result.get('message')}"
                    )
                after = str(result["state_smiles"])
                if fragment_multiset_unmapped(before) == fragment_multiset_unmapped(after):
                    raise ValueError("NO_OP: molecular state did not change")
                self.trace.append(
                    state_before=self.current_state,
                    state_after=after,
                    moves=moves,
                    imports=tuple(self.pending_imports),
                )
                self.current_state = after
                self.pending_imports = []
                self.pending_roles = []
            elif name == "undo_step":
                high_water = self.next_map
                if self.pending_imports:
                    self.pending_imports = []
                    self.pending_roles = []
                elif self.trace.transitions:
                    transition = self.trace.transitions.pop()
                    self.current_state = transition.state_before
                else:
                    raise ValueError("NOTHING_TO_UNDO")
                self.next_map = high_water
            elif name == "finish":
                if self.pending_imports:
                    raise ValueError("PENDING_IMPORTS_MUST_BE_USED_OR_UNDONE")
                compilation = compile_trace_to_proof(self.trace)
                if fragment_multiset_unmapped(compilation.precursor_smiles) == fragment_multiset_unmapped(self.target):
                    raise ValueError("ENDPOINT_EQUALS_INITIAL_PRODUCT")
                self.precursor_smiles = compilation.precursor_smiles
                self.finished = True
            else:
                raise ValueError("UNKNOWN_TOOL")
            self.revision += 1
            return {
                "ok": True,
                **self.observation(),
                "finished": self.finished,
                "precursor_smiles": (
                    _unmap(self.precursor_smiles) if self.finished else None
                ),
            }
        except Exception as exc:
            self.__dict__ = snapshot
            return {
                "ok": False,
                "error": str(exc),
                "state_unchanged": True,
                **self.observation(),
            }
