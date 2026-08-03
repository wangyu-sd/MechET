"""Conservative conversion from executable proofs to source-sink tool traces.

Only linear proofs with uniquely pairable two-electron changes are converted.
Ambiguous electron pairing is rejected rather than invented.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .proof_program import ProofEdge, execute_proof, parse_proof_program, sides_equal


@dataclass(frozen=True)
class TracePlanStep:
    step_index: int
    state_before: str
    state_after: str
    imports: tuple[str, ...]
    moves: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["imports"] = list(self.imports)
        value["moves"] = list(self.moves)
        return value


@dataclass(frozen=True)
class ProofTracePlan:
    target_smiles: str
    expected_precursor: str
    steps: tuple[TracePlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "expected_precursor": self.expected_precursor,
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
            moves.append({"source": {"kind": "LP", "atoms": [donor]}, "sink": {"kind": "BOND", "atoms": list(pair)}, "electrons": 2})
        else:
            remaining_increased.append(pair)
    remaining_decreased: list[tuple[int, int]] = []
    for pair in decreased:
        acceptors = [atom for atom in lp_gain if atom in pair]
        if len(acceptors) == 1:
            acceptor = acceptors[0]
            lp_gain.remove(acceptor)
            moves.append({"source": {"kind": "BOND", "atoms": list(pair)}, "sink": {"kind": "ATOM", "atoms": [acceptor]}, "electrons": 2})
        else:
            remaining_decreased.append(pair)
    while remaining_decreased or remaining_increased:
        candidates: list[tuple[int, int]] = []
        for old_index, old_pair in enumerate(remaining_decreased):
            for new_index, new_pair in enumerate(remaining_increased):
                if len(set(old_pair) & set(new_pair)) == 1:
                    candidates.append((old_index, new_index))
        if len(candidates) != 1:
            raise ValueError("AMBIGUOUS_ELECTRON_PAIRING: " f"decreased={remaining_decreased} increased={remaining_increased}")
        old_index, new_index = candidates[0]
        old_pair = remaining_decreased.pop(old_index)
        new_pair = remaining_increased.pop(new_index)
        moves.append({"source": {"kind": "BOND", "atoms": list(old_pair)}, "sink": {"kind": "BOND", "atoms": list(new_pair)}, "electrons": 2})
    if lp_gain or lp_loss:
        raise ValueError(f"UNPAIRED_LONE_PAIR_DELTA: gain={lp_gain} loss={lp_loss}")
    if not moves:
        raise ValueError("EDGE_HAS_NO_INFERABLE_MOVES")
    return tuple(moves)


def _linear_edges(program) -> list[ProofEdge]:
    if len(program.roots) != 1:
        raise ValueError("NONLINEAR_PROOF_UNSUPPORTED: multiple roots")
    current = next(iter(program.roots))
    output: list[ProofEdge] = []
    remaining = list(program.edges)
    visited = {current}
    while current != program.precursor_state_id:
        candidates = [edge for edge in remaining if edge.src == current]
        if len(candidates) != 1:
            raise ValueError(f"NONLINEAR_PROOF_UNSUPPORTED at {current}: {len(candidates)} outgoing edges")
        edge = candidates[0]
        if edge.dst in visited:
            raise ValueError("CYCLIC_PROOF_UNSUPPORTED")
        output.append(edge)
        remaining.remove(edge)
        current = edge.dst
        visited.add(current)
    if remaining:
        raise ValueError("NONLINEAR_PROOF_UNSUPPORTED: unused branch edges")
    return output


def proof_to_trace_plan(proof: str) -> ProofTracePlan:
    program = parse_proof_program(proof)
    execution = execute_proof(program)
    if not execution.ok:
        raise ValueError(f"PROOF_NOT_EXECUTABLE: {execution.diagnostics}")
    steps: list[TracePlanStep] = []
    for index, edge in enumerate(_linear_edges(program)):
        state_before = execution.states.get(edge.src, "")
        state_after = execution.states.get(edge.dst, "")
        if not state_before or not state_after:
            raise ValueError(f"PROOF_STATE_MISSING at edge {edge.src}->{edge.dst}")
        steps.append(TracePlanStep(step_index=index, state_before=state_before, state_after=state_after, imports=tuple(edge.imports), moves=infer_moves_from_edge(edge)))
    return ProofTracePlan(target_smiles=program.target_smiles, expected_precursor=execution.precursor_smiles, steps=tuple(steps))


def replay_trace_plan(env, plan: ProofTracePlan) -> dict[str, Any]:
    """Replay a plan through a trace-owned environment and return terminal output."""
    import json
    env.reset(target_smiles=plan.target_smiles, expected_precursor=plan.expected_precursor)
    events: list[dict[str, Any]] = []
    for step in plan.steps:
        for fragment in step.imports:
            result = json.loads(env.import_fragment(fragment))
            events.append({"tool": "import_fragment", "arguments": {"fragment_smiles": fragment}, "result": result})
            if not result.get("ok"):
                raise ValueError(f"IMPORT_REPLAY_FAILED: {result}")
        result = json.loads(env.apply_coupled_electron_moves(json.dumps(list(step.moves))))
        events.append({"tool": "apply_coupled_electron_moves", "arguments": {"moves_json": json.dumps(list(step.moves))}, "result": result})
        if not result.get("ok"):
            raise ValueError(f"MOVE_REPLAY_FAILED: {result}")
        if not sides_equal(result.get("state_smiles", ""), step.state_after, ignore_maps=False):
            raise ValueError(f"MOVE_REPLAY_STATE_MISMATCH at step {step.step_index}")
    terminal = json.loads(env.finish_trace())
    events.append({"tool": "finish_trace", "arguments": {}, "result": terminal})
    if not terminal.get("ok") or not terminal.get("endpoint_exact"):
        raise ValueError(f"TRACE_TERMINAL_REPLAY_FAILED: {terminal}")
    return {"events": events, "terminal": terminal, "state": env.state_dict()}
