"""Formal catalytic-cycle bookkeeping over executable proof steps.

The verifier checks proof execution, catalyst continuity/regeneration, optional
oxidation-state closure, and a declared net-reaction ledger. It does not claim
energetic, kinetic, spin-state, or experimental feasibility.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from mechet.data_audit import (
    NormalizationConfig,
    canonical_smiles,
    split_reaction_smiles,
)
from mechet.proof_program import execute_proof, parse_proof_program, sides_equal
from mechet.proof_routes import structural_precursors


@dataclass(frozen=True)
class CatalyticCycleStep:
    label: str
    proof: str
    catalyst_before: str
    catalyst_after: str
    reversible: bool = False
    oxidation_state_before: int | None = None
    oxidation_state_after: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalyticCycle:
    cycle_id: str
    catalyst_initial: str
    steps: tuple[CatalyticCycleStep, ...]
    net_reaction: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["steps"] = [step.to_dict() for step in self.steps]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CatalyticCycle":
        return cls(
            cycle_id=str(payload.get("cycle_id") or payload.get("id") or ""),
            catalyst_initial=str(payload.get("catalyst_initial") or ""),
            steps=tuple(
                CatalyticCycleStep(
                    label=str(item.get("label") or f"step_{index}"),
                    proof=str(item.get("proof") or ""),
                    catalyst_before=str(item.get("catalyst_before") or ""),
                    catalyst_after=str(item.get("catalyst_after") or ""),
                    reversible=bool(item.get("reversible")),
                    oxidation_state_before=item.get("oxidation_state_before"),
                    oxidation_state_after=item.get("oxidation_state_after"),
                    metadata=dict(item.get("metadata") or {}),
                )
                for index, item in enumerate(payload.get("steps") or [])
            ),
            net_reaction=str(payload.get("net_reaction") or ""),
            conditions=dict(payload.get("conditions") or {}),
        )


@dataclass(frozen=True)
class CatalyticCycleVerification:
    ok: bool
    proof_execution_ok: bool
    catalyst_continuity_ok: bool
    catalyst_regenerated: bool
    oxidation_state_closed: bool
    net_reaction_ok: bool | None
    n_steps: int
    errors: tuple[str, ...] = ()
    derived_net_reaction: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_species(smiles: str) -> str:
    return canonical_smiles(
        smiles,
        NormalizationConfig(remove_atom_maps=True, keep_stereo=True),
    )


def _cancel_ledger(left: Sequence[str], right: Sequence[str]) -> tuple[list[str], list[str]]:
    left_counts = Counter(_canonical_species(value) for value in left if value)
    right_counts = Counter(_canonical_species(value) for value in right if value)
    for key in set(left_counts) | set(right_counts):
        cancel = min(left_counts[key], right_counts[key])
        left_counts[key] -= cancel
        right_counts[key] -= cancel
    left_out = sorted(key for key, count in left_counts.items() for _ in range(count))
    right_out = sorted(key for key, count in right_counts.items() for _ in range(count))
    return left_out, right_out


def derive_cycle_net_reaction(cycle: CatalyticCycle) -> str:
    """Sum executable forward steps and cancel intermediates/catalyst states."""
    forward_left: list[str] = []
    forward_right: list[str] = []
    catalyst_states: list[str] = [cycle.catalyst_initial]
    for step in cycle.steps:
        result = execute_proof(step.proof)
        if not result.ok:
            raise ValueError(f"non-executable cycle step: {step.label}")
        program = parse_proof_program(step.proof)
        forward_left.extend(structural_precursors(program.target_smiles, result.precursor_smiles))
        forward_right.append(program.target_smiles)
        catalyst_states.extend([step.catalyst_before, step.catalyst_after])
    catalyst_keys = {_canonical_species(value) for value in catalyst_states if value}
    left, right = _cancel_ledger(forward_left, forward_right)
    left = [value for value in left if value not in catalyst_keys]
    right = [value for value in right if value not in catalyst_keys]
    return ".".join(left) + ">>" + ".".join(right)


def verify_catalytic_cycle(cycle: CatalyticCycle) -> CatalyticCycleVerification:
    errors: list[str] = []
    proof_ok = True
    continuity_ok = True
    oxidation_ok = True
    if not cycle.steps:
        errors.append("cycle has no steps")
    expected_catalyst = cycle.catalyst_initial
    previous_oxidation: int | None = None
    for index, step in enumerate(cycle.steps):
        execution = execute_proof(step.proof)
        if not execution.ok:
            proof_ok = False
            errors.append(f"step {index} ({step.label}) is not executable")
        if expected_catalyst and step.catalyst_before and not sides_equal(
            expected_catalyst,
            step.catalyst_before,
            ignore_maps=True,
        ):
            continuity_ok = False
            errors.append(f"catalyst discontinuity before step {index}")
        expected_catalyst = step.catalyst_after
        if (
            previous_oxidation is not None
            and step.oxidation_state_before is not None
            and previous_oxidation != step.oxidation_state_before
        ):
            oxidation_ok = False
            errors.append(f"oxidation-state discontinuity before step {index}")
        if step.oxidation_state_after is not None:
            previous_oxidation = int(step.oxidation_state_after)
    regenerated = bool(
        cycle.steps
        and cycle.catalyst_initial
        and expected_catalyst
        and sides_equal(cycle.catalyst_initial, expected_catalyst, ignore_maps=True)
    )
    if cycle.steps and not regenerated:
        errors.append("catalyst is not regenerated")
    first_oxidation = cycle.steps[0].oxidation_state_before if cycle.steps else None
    if first_oxidation is not None and previous_oxidation is not None:
        oxidation_ok = oxidation_ok and int(first_oxidation) == int(previous_oxidation)
        if int(first_oxidation) != int(previous_oxidation):
            errors.append("oxidation state does not close")

    derived = ""
    net_ok: bool | None = None
    if proof_ok:
        try:
            derived = derive_cycle_net_reaction(cycle)
        except Exception as exc:
            errors.append(f"net reaction derivation failed: {exc}")
    if cycle.net_reaction:
        net_ok = False
        try:
            declared_left, _, declared_right = split_reaction_smiles(cycle.net_reaction)
            derived_left, _, derived_right = split_reaction_smiles(derived)
            net_ok = sides_equal(declared_left, derived_left, ignore_maps=True) and sides_equal(
                declared_right,
                derived_right,
                ignore_maps=True,
            )
        except Exception:
            net_ok = False
        if not net_ok:
            errors.append("declared net reaction does not match cycle ledger")
    ok = (
        proof_ok
        and continuity_ok
        and regenerated
        and oxidation_ok
        and (net_ok is not False)
        and not errors
    )
    return CatalyticCycleVerification(
        ok=ok,
        proof_execution_ok=proof_ok,
        catalyst_continuity_ok=continuity_ok,
        catalyst_regenerated=regenerated,
        oxidation_state_closed=oxidation_ok,
        net_reaction_ok=net_ok,
        n_steps=len(cycle.steps),
        errors=tuple(errors),
        derived_net_reaction=derived,
    )
