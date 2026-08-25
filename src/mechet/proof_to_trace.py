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
    radical_gain: list[int] = []
    radical_loss: list[int] = []
    for atom_map, delta in edge.lone_pairs:
        odd_target = radical_gain if delta > 0 else radical_loss
        if int(delta) % 2:
            odd_target.append(int(atom_map))
        target = lp_gain if delta > 0 else lp_loss
        target.extend([int(atom_map)] * (abs(int(delta)) // 2))
    return decreased, increased, lp_gain, lp_loss, radical_gain, radical_loss


def _be_delta_move(edge: ProofEdge) -> tuple[dict[str, Any], ...]:
    """Exact executable fallback for non-pairwise FlowER electron events."""
    return (
        {
            "mode": "BE_DELTA",
            "bond_deltas": [
                {"atoms": [int(left), int(right)], "delta": int(delta)}
                for left, right, delta in edge.bonds
            ],
            "charge_actions": [
                {
                    "atom_map": int(action.atom_map),
                    "q0": int(action.q0),
                    "q1": int(action.q1),
                }
                for action in edge.charges
            ],
        },
    )


def infer_moves_from_edge(edge: ProofEdge) -> tuple[dict[str, Any], ...]:
    """Infer a replay-exact local pairing of all two-electron units.

    A sparse BE delta fixes electron sources and sinks but can admit several
    arrow pairings.  Rejecting every non-unique bipartite matching discarded
    valid FlowER states.  We instead enumerate *local* pairings permitted by
    the executor and retain the canonical first matching whose aggregate
    formal-charge delta is exactly the one declared by the proof edge.  Thus
    the tie-break changes no molecular state and invents no net edit.
    """

    (
        decreased,
        increased,
        lp_gain,
        lp_loss,
        radical_gain,
        radical_loss,
    ) = _units(edge)
    selected: list[dict[str, Any]] = []

    # FlowER includes organometallic coordination and occasional homolysis as
    # two one-electron diagonal changes coupled to one bond-order change.  Keep
    # that elementary event atomic with an explicit RADICAL_PAIR container.
    for pair in list(increased):
        if all(atom in radical_loss for atom in pair):
            increased.remove(pair)
            for atom in pair:
                radical_loss.remove(atom)
            selected.append(
                {
                    "source": {"kind": "RADICAL_PAIR", "atoms": list(pair)},
                    "sink": {"kind": "BOND", "atoms": list(pair)},
                    "electrons": 2,
                }
            )
    for pair in list(decreased):
        if all(atom in radical_gain for atom in pair):
            decreased.remove(pair)
            for atom in pair:
                radical_gain.remove(atom)
            selected.append(
                {
                    "source": {"kind": "BOND", "atoms": list(pair)},
                    "sink": {"kind": "RADICAL_PAIR", "atoms": list(pair)},
                    "electrons": 2,
                }
            )
    if radical_gain or radical_loss:
        return _be_delta_move(edge)
    sources = sorted(
        [("LP", (atom,)) for atom in lp_loss]
        + [("BOND", pair) for pair in decreased]
    )
    sinks = sorted(
        [("ATOM", (atom,)) for atom in lp_gain]
        + [("BOND", pair) for pair in increased]
    )
    if not sources and not sinks and selected:
        return tuple(selected)
    if not sources and not sinks:
        raise ValueError("EDGE_HAS_NO_INFERABLE_MOVES")
    if len(sources) != len(sinks):
        return _be_delta_move(edge)

    expected_charge = {
        int(action.atom_map): int(action.q1) - int(action.q0)
        for action in edge.charges
        if int(action.q1) != int(action.q0)
    }

    def candidate(
        source: tuple[str, tuple[int, ...]],
        sink: tuple[str, tuple[int, ...]],
    ) -> tuple[dict[str, Any], dict[int, int]] | None:
        source_kind, source_atoms = source
        sink_kind, sink_atoms = sink
        charge: dict[int, int] = {}
        if source_kind == "LP" and sink_kind == "BOND":
            donor = source_atoms[0]
            if donor not in sink_atoms:
                return None
            acceptor = next(atom for atom in sink_atoms if atom != donor)
            charge = {donor: 1, acceptor: -1}
        elif source_kind == "BOND" and sink_kind == "ATOM":
            target = sink_atoms[0]
            if target not in source_atoms:
                return None
            other = next(atom for atom in source_atoms if atom != target)
            charge = {target: -1, other: 1}
        elif source_kind == sink_kind == "BOND":
            shared = set(source_atoms) & set(sink_atoms)
            if len(shared) != 1:
                return None
            centre = next(iter(shared))
            old = next(atom for atom in source_atoms if atom != centre)
            new = next(atom for atom in sink_atoms if atom != centre)
            charge = {old: 1, new: -1}
        else:
            return None
        move = {
            "source": {"kind": source_kind, "atoms": list(source_atoms)},
            "sink": {"kind": sink_kind, "atoms": list(sink_atoms)},
            "electrons": 2,
        }
        return move, charge

    options = [
        [(sink_index, candidate(source, sink)) for sink_index, sink in enumerate(sinks)]
        for source in sources
    ]
    options = [
        [(sink_index, value) for sink_index, value in row if value is not None]
        for row in options
    ]
    if any(not row for row in options):
        return _be_delta_move(edge)

    def search(
        source_index: int,
        used_sinks: set[int],
        charge_delta: dict[int, int],
    ) -> bool:
        if source_index == len(sources):
            normalized = {
                atom: delta for atom, delta in charge_delta.items() if delta
            }
            return normalized == expected_charge
        for sink_index, value in options[source_index]:
            if sink_index in used_sinks:
                continue
            assert value is not None
            move, contribution = value
            updated = dict(charge_delta)
            for atom, delta in contribution.items():
                updated[atom] = updated.get(atom, 0) + delta
            selected.append(move)
            used_sinks.add(sink_index)
            if search(source_index + 1, used_sinks, updated):
                return True
            used_sinks.remove(sink_index)
            selected.pop()
        return False

    if not search(0, set(), {}):
        return _be_delta_move(edge)
    return tuple(selected)


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
