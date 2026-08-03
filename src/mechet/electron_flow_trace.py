"""Environment-owned electron-flow traces and deterministic proof compilation.

The main agent path records every successful import and electron-flow action in
an immutable trace. The final ``MECH_PROOF v1`` program is compiled from that
trace; the model cannot submit an unrelated free-form proof.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

from .proof_program import (
    ProofEdge,
    ProofProgram,
    _get_be_delta,
    execute_proof,
    format_proof_output,
    sides_equal,
)


@dataclass(frozen=True)
class ElectronFlowTransition:
    """One committed elementary transition in an inverse rollout."""

    step_index: int
    state_before: str
    state_after: str
    moves: tuple[dict[str, Any], ...]
    imports: tuple[str, ...] = ()

    @classmethod
    def parse(cls, row: Mapping[str, Any]) -> "ElectronFlowTransition":
        return cls(
            step_index=int(row.get("step_index", 0)),
            state_before=str(row.get("state_before") or ""),
            state_after=str(row.get("state_after") or ""),
            moves=tuple(dict(item) for item in row.get("moves") or ()),
            imports=tuple(str(item) for item in row.get("imports") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["moves"] = list(self.moves)
        value["imports"] = list(self.imports)
        return value


@dataclass
class ElectronFlowTrace:
    """The authoritative action history for one inverse episode."""

    target_smiles: str
    transitions: list[ElectronFlowTransition] = field(default_factory=list)

    def append(
        self,
        *,
        state_before: str,
        state_after: str,
        moves: Sequence[Mapping[str, Any]],
        imports: Sequence[str] = (),
    ) -> ElectronFlowTransition:
        transition = ElectronFlowTransition(
            step_index=len(self.transitions),
            state_before=str(state_before),
            state_after=str(state_after),
            moves=tuple(dict(item) for item in moves),
            imports=tuple(str(item) for item in imports),
        )
        self.transitions.append(transition)
        return transition

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "transitions": [item.to_dict() for item in self.transitions],
            "digest": self.digest(),
        }

    def digest(self) -> str:
        payload = {
            "target_smiles": self.target_smiles,
            "transitions": [item.to_dict() for item in self.transitions],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class TraceCompilation:
    proof: str
    program: ProofProgram
    precursor_smiles: str
    trace_digest: str
    n_transitions: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof": self.proof,
            "precursor_smiles": self.precursor_smiles,
            "trace_digest": self.trace_digest,
            "n_transitions": self.n_transitions,
        }


def _augmented_state(state: str, imports: Sequence[str]) -> str:
    fragments = [str(state).strip(), *(str(item).strip() for item in imports)]
    return ".".join(item for item in fragments if item)


def compile_trace_to_proof(trace: ElectronFlowTrace) -> TraceCompilation:
    """Compile and replay an environment-owned trace as ``MECH_PROOF v1``.

    The compiler derives BOND/LP/CHARGE operations from committed state changes.
    It refuses discontinuous traces and verifies that proof execution reproduces
    every committed state and the final precursor exactly, including atom maps.
    """

    if not trace.target_smiles:
        raise ValueError("trace requires target_smiles")
    if not trace.transitions:
        raise ValueError("trace contains no committed electron-flow transition")

    previous = trace.target_smiles
    edges: list[ProofEdge] = []
    for index, transition in enumerate(trace.transitions):
        if transition.step_index != index:
            raise ValueError("trace step indices must be contiguous")
        if not sides_equal(previous, transition.state_before, ignore_maps=False):
            raise ValueError(f"TRACE_STATE_DISCONTINUITY at step {index}")
        source_augmented = _augmented_state(
            transition.state_before,
            transition.imports,
        )
        bonds, lone_pairs, charges = _get_be_delta(
            source_augmented,
            transition.state_after,
        )
        if not bonds and not lone_pairs and not charges:
            raise ValueError(f"TRACE_NO_STATE_CHANGE at step {index}")
        edges.append(
            ProofEdge(
                src=f"s{index}",
                dst=f"s{index + 1}",
                imports=list(transition.imports),
                bonds=bonds,
                lone_pairs=lone_pairs,
                charges=charges,
            )
        )
        previous = transition.state_after

    program = ProofProgram(
        target_smiles=trace.target_smiles,
        roots={"s0": []},
        precursor_state_id=f"s{len(edges)}",
        edges=edges,
    )
    execution = execute_proof(program)
    if not execution.ok:
        message = "; ".join(
            str(item.get("message") or item) for item in execution.diagnostics
        )
        raise ValueError(f"TRACE_COMPILED_PROOF_FAILED: {message}")

    for index, transition in enumerate(trace.transitions, start=1):
        actual = execution.states.get(f"s{index}", "")
        if not sides_equal(actual, transition.state_after, ignore_maps=False):
            raise ValueError(f"TRACE_PROOF_STATE_MISMATCH at step {index - 1}")
    if not sides_equal(
        execution.precursor_smiles,
        trace.transitions[-1].state_after,
        ignore_maps=False,
    ):
        raise ValueError("TRACE_PROOF_ENDPOINT_MISMATCH")

    return TraceCompilation(
        proof=format_proof_output(program),
        program=program,
        precursor_smiles=execution.precursor_smiles,
        trace_digest=trace.digest(),
        n_transitions=len(trace.transitions),
    )
