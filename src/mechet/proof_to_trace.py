"""Conservative conversion from executable proofs to source-sink tool traces.

Only linear proofs with uniquely pairable two-electron changes are converted.
Ambiguous electron pairing is rejected rather than invented. Root-level imports
are preserved explicitly so replay uses the same initial state as proof execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from rdkit import Chem

from .proof_program import (
    ProofEdge,
    _canonical_mapped,
    _combine_smiles,
    execute_proof,
    parse_proof_program,
    sides_equal,
)


@dataclass(frozen=True)
class TracePlanStep:
    step_index: int
    state_before: str
    state_after: str
    imports: tuple[str, ...]
    moves: tuple[dict[str, Any], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TracePlanStep":
        return cls(
            step_index=int(value.get("step_index", 0)),
            state_before=str(value.get("state_before") or ""),
            state_after=str(value.get("state_after") or ""),
            imports=tuple(str(item) for item in value.get("imports") or ()),
            moves=tuple(dict(item) for item in value.get("moves") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["imports"] = list(self.imports)
        value["moves"] = list(self.moves)
        return value


@dataclass(frozen=True)
class ProofTracePlan:
    target_smiles: str
    expected_precursor: str
    initial_imports: tuple[str, ...]
    steps: tuple[TracePlanStep, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProofTracePlan":
        return cls(
            target_smiles=str(value.get("target_smiles") or ""),
            expected_precursor=str(value.get("expected_precursor") or ""),
            initial_imports=tuple(
                str(item) for item in value.get("initial_imports") or ()
            ),
            steps=tuple(
                TracePlanStep.from_dict(item) for item in value.get("steps") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "expected_precursor": self.expected_precursor,
            "initial_imports": list(self.initial_imports),
            "steps": [item.to_dict() for item in self.steps],
        }


def _units(edge: ProofEdge):
    decreased: list[tuple[int, int]] = []
    increased: list[tuple[int, int]] = []
    for left, right, delta in edge.bonds:
        pair = tuple(sorted((int(left), int(right))))
        target = increased if delta > 0 else decreased
        target.extend([pair] * abs(int(delta)))
    lp_gain: list[int] = []
    lp_loss: list[int] = []
    for atom_map, delta in edge.lone_pairs:
        if int(delta) % 2:
            raise ValueError("ODD_LONE_PAIR_DELTA")
        target = lp_gain if delta > 0 else lp_loss
        target.extend([int(atom_map)] * (abs(int(delta)) // 2))
    return decreased, increased, lp_gain, lp_loss


def infer_moves_from_edge(edge: ProofEdge) -> tuple[dict[str, Any], ...]:
    """Infer a unique source-sink pairing from one executable proof edge."""

    decreased, increased, lp_gain, lp_loss = _units(edge)
    moves: list[dict[str, Any]] = []

    remaining_increased: list[tuple[int, int]] = []
    for pair in increased:
        donors = [atom for atom in lp_loss if atom in pair]
        if len(donors) == 1:
            donor = donors[0]
            lp_loss.remove(donor)
            moves.append(
                {
                    "source": {"kind": "LP", "atoms": [donor]},
                    "sink": {"kind": "BOND", "atoms": list(pair)},
                    "electrons": 2,
                }
            )
        else:
            remaining_increased.append(pair)

    remaining_decreased: list[tuple[int, int]] = []
    for pair in decreased:
        acceptors = [atom for atom in lp_gain if atom in pair]
        if len(acceptors) == 1:
            acceptor = acceptors[0]
            lp_gain.remove(acceptor)
            moves.append(
                {
                    "source": {"kind": "BOND", "atoms": list(pair)},
                    "sink": {"kind": "ATOM", "atoms": [acceptor]},
                    "electrons": 2,
                }
            )
        else:
            remaining_decreased.append(pair)

    while remaining_decreased or remaining_increased:
        candidates: list[tuple[int, int]] = []
        for old_index, old_pair in enumerate(remaining_decreased):
            for new_index, new_pair in enumerate(remaining_increased):
                if len(set(old_pair) & set(new_pair)) == 1:
                    candidates.append((old_index, new_index))
        if len(candidates) != 1:
            raise ValueError(
                "AMBIGUOUS_ELECTRON_PAIRING: "
                f"decreased={remaining_decreased} increased={remaining_increased}"
            )
        old_index, new_index = candidates[0]
        old_pair = remaining_decreased.pop(old_index)
        new_pair = remaining_increased.pop(new_index)
        moves.append(
            {
                "source": {"kind": "BOND", "atoms": list(old_pair)},
                "sink": {"kind": "BOND", "atoms": list(new_pair)},
                "electrons": 2,
            }
        )

    if lp_gain or lp_loss:
        raise ValueError(
            f"UNPAIRED_LONE_PAIR_DELTA: gain={lp_gain} loss={lp_loss}"
        )
    if not moves:
        raise ValueError("EDGE_HAS_NO_INFERABLE_MOVES")
    return tuple(moves)


def _linear_edges(program) -> tuple[str, list[ProofEdge]]:
    if len(program.roots) != 1:
        raise ValueError("NONLINEAR_PROOF_UNSUPPORTED: multiple roots")
    root_id = next(iter(program.roots))
    current = root_id
    output: list[ProofEdge] = []
    remaining = list(program.edges)
    visited = {current}
    while current != program.precursor_state_id:
        candidates = [edge for edge in remaining if edge.src == current]
        if len(candidates) != 1:
            raise ValueError(
                f"NONLINEAR_PROOF_UNSUPPORTED at {current}: {len(candidates)} outgoing edges"
            )
        edge = candidates[0]
        if edge.dst in visited:
            raise ValueError("CYCLIC_PROOF_UNSUPPORTED")
        output.append(edge)
        remaining.remove(edge)
        current = edge.dst
        visited.add(current)
    if remaining:
        raise ValueError("NONLINEAR_PROOF_UNSUPPORTED: unused branch edges")
    return root_id, output


def proof_to_trace_plan(proof: str) -> ProofTracePlan:
    program = parse_proof_program(proof)
    execution = execute_proof(program)
    if not execution.ok:
        raise ValueError(f"PROOF_NOT_EXECUTABLE: {execution.diagnostics}")
    root_id, edges = _linear_edges(program)
    initial_imports = tuple(program.roots.get(root_id) or ())
    steps: list[TracePlanStep] = []
    for index, edge in enumerate(edges):
        state_before = execution.states.get(edge.src, "")
        state_after = execution.states.get(edge.dst, "")
        if not state_before or not state_after:
            raise ValueError(f"PROOF_STATE_MISSING at edge {edge.src}->{edge.dst}")
        steps.append(
            TracePlanStep(
                step_index=index,
                state_before=state_before,
                state_after=state_after,
                imports=tuple(edge.imports),
                moves=infer_moves_from_edge(edge),
            )
        )
    return ProofTracePlan(
        target_smiles=program.target_smiles,
        expected_precursor=execution.precursor_smiles,
        initial_imports=initial_imports,
        steps=tuple(steps),
    )


def replay_trace_plan(env, plan: ProofTracePlan) -> dict[str, Any]:
    """Replay a plan through a trace-owned environment and return terminal output."""

    env.reset(
        target_smiles=plan.target_smiles,
        expected_precursor=plan.expected_precursor,
    )
    events: list[dict[str, Any]] = []
    for fragment in plan.initial_imports:
        result = json.loads(env.import_fragment(fragment))
        events.append(
            {
                "tool": "import_fragment",
                "arguments": {"fragment_smiles": fragment},
                "result": result,
                "import_scope": "root",
            }
        )
        if not result.get("ok"):
            raise ValueError(f"ROOT_IMPORT_REPLAY_FAILED: {result}")
    for step in plan.steps:
        for fragment in step.imports:
            result = json.loads(env.import_fragment(fragment))
            events.append(
                {
                    "tool": "import_fragment",
                    "arguments": {"fragment_smiles": fragment},
                    "result": result,
                    "import_scope": "edge",
                }
            )
            if not result.get("ok"):
                raise ValueError(f"IMPORT_REPLAY_FAILED: {result}")
        result = json.loads(
            env.apply_coupled_electron_moves(json.dumps(list(step.moves)))
        )
        events.append(
            {
                "tool": "apply_coupled_electron_moves",
                "arguments": {"moves": list(step.moves)},
                "result": result,
            }
        )
        if not result.get("ok"):
            raise ValueError(f"MOVE_REPLAY_FAILED: {result}")
        if not sides_equal(
            result.get("state_smiles", ""), step.state_after, ignore_maps=False
        ):
            raise ValueError(f"MOVE_REPLAY_STATE_MISMATCH at step {step.step_index}")
    terminal = json.loads(env.finish_trace())
    events.append({"tool": "finish_trace", "arguments": {}, "result": terminal})
    if not terminal.get("ok") or not terminal.get("endpoint_exact"):
        raise ValueError(f"TRACE_TERMINAL_REPLAY_FAILED: {terminal}")
    return {"events": events, "terminal": terminal, "state": env.state_dict()}


def _atom_descriptor(mol: Chem.Mol, atom_map: int) -> tuple[int, int, int]:
    for atom in mol.GetAtoms():
        if int(atom.GetAtomMapNum()) == int(atom_map):
            return (
                int(atom.GetAtomicNum()),
                int(atom.GetFormalCharge()),
                int(atom.GetIsAromatic()),
            )
    return (0, 0, 0)


def _bond_order(mol: Chem.Mol, atoms: Sequence[int]) -> int:
    if len(atoms) != 2:
        return 0
    indices = {
        int(atom.GetAtomMapNum()): atom.GetIdx()
        for atom in mol.GetAtoms()
        if int(atom.GetAtomMapNum()) > 0
    }
    if int(atoms[0]) not in indices or int(atoms[1]) not in indices:
        return 0
    bond = mol.GetBondBetweenAtoms(indices[int(atoms[0])], indices[int(atoms[1])])
    return int(round(bond.GetBondTypeAsDouble())) if bond is not None else 0


def execution_primitive_signature(
    move: Mapping[str, Any], source_state: str, imports: Sequence[str] = ()
) -> str:
    """Return a map-label-independent source-to-sink execution primitive label."""

    augmented = _canonical_mapped(_combine_smiles(source_state, imports))
    mol = Chem.MolFromSmiles(augmented)
    if mol is None:
        raise ValueError("EXECUTION_PRIMITIVE_STATE_INVALID")
    source = dict(move.get("source") or {})
    sink = dict(move.get("sink") or {})
    source_atoms = [int(item) for item in source.get("atoms") or []]
    sink_atoms = [int(item) for item in sink.get("atoms") or []]
    payload = {
        "source_kind": str(source.get("kind") or ""),
        "sink_kind": str(sink.get("kind") or ""),
        "source_atoms": sorted(_atom_descriptor(mol, item) for item in source_atoms),
        "sink_atoms": sorted(_atom_descriptor(mol, item) for item in sink_atoms),
        "source_bond_order": _bond_order(mol, source_atoms),
        "sink_bond_order": _bond_order(mol, sink_atoms),
        "electrons": int(move.get("electrons", 2)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def execution_primitive_signatures(plan: ProofTracePlan) -> tuple[str, ...]:
    signatures: set[str] = set()
    for step in plan.steps:
        for move in step.moves:
            signatures.add(
                execution_primitive_signature(move, step.state_before, step.imports)
            )
    return tuple(sorted(signatures))


def execution_composition_signature(plan: ProofTracePlan) -> str:
    """Digest the ordered source-to-sink move composition of a linear trace."""

    sequence = [
        [
            execution_primitive_signature(move, step.state_before, step.imports)
            for move in step.moves
        ]
        for step in plan.steps
    ]
    return hashlib.sha256(
        json.dumps(sequence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
