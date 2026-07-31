"""Proof-carrying multistep route verification and best-first search."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import itertools
import math
from typing import Any, Callable, Iterable, Sequence

from mechet.data_audit import split_structural_and_environment
from mechet.proof_program import (
    execute_proof,
    parse_proof_program,
    sides_equal,
)


@dataclass(frozen=True)
class RouteStep:
    product: str
    proof: str
    precursors: tuple[str, ...]
    model_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["precursors"] = list(self.precursors)
        return payload


@dataclass
class RouteCandidate:
    target: str
    open_molecules: tuple[str, ...]
    steps: tuple[RouteStep, ...] = ()
    cost: float = 0.0
    depth: int = 0
    lineage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "open_molecules": list(self.open_molecules),
            "steps": [step.to_dict() for step in self.steps],
            "cost": self.cost,
            "depth": self.depth,
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True)
class RouteVerification:
    ok: bool
    n_steps: int
    n_executable_steps: int
    leaves: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def structural_precursors(product: str, derived_precursor: str) -> tuple[str, ...]:
    roles = split_structural_and_environment(derived_precursor, product)
    return tuple(sorted(roles.structural))


def step_from_proof(
    product: str,
    proof: str,
    *,
    model_score: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> RouteStep | None:
    result = execute_proof(proof)
    if not result.ok:
        return None
    program = parse_proof_program(proof)
    if not sides_equal(program.target_smiles, product, ignore_maps=True):
        return None
    precursors = structural_precursors(product, result.precursor_smiles)
    if not precursors:
        return None
    return RouteStep(
        product=product,
        proof=proof,
        precursors=precursors,
        model_score=float(model_score),
        metadata=dict(metadata or {}),
    )


def verify_route(
    target: str,
    steps: Sequence[RouteStep],
    *,
    is_building_block: Callable[[str], bool] | None = None,
) -> RouteVerification:
    """Verify every edge and the route's product/precursor connectivity."""
    errors: list[str] = []
    executable = 0
    produced: list[str] = [target]
    consumed: list[str] = []
    for index, step in enumerate(steps):
        result = execute_proof(step.proof)
        if not result.ok:
            errors.append(f"step {index} proof is not executable")
            continue
        executable += 1
        program = parse_proof_program(step.proof)
        if not sides_equal(program.target_smiles, step.product, ignore_maps=True):
            errors.append(f"step {index} target does not match proof")
        expected = structural_precursors(step.product, result.precursor_smiles)
        if sorted(expected) != sorted(step.precursors):
            errors.append(f"step {index} precursor declaration mismatch")
        if not any(sides_equal(step.product, item, ignore_maps=True) for item in produced):
            errors.append(f"step {index} product was not produced by the route")
        consumed.append(step.product)
        produced.extend(step.precursors)

    leaves: list[str] = []
    for molecule in produced:
        if any(sides_equal(molecule, item, ignore_maps=True) for item in consumed):
            continue
        if not any(sides_equal(molecule, item, ignore_maps=True) for item in leaves):
            leaves.append(molecule)
    if is_building_block is not None:
        missing = [leaf for leaf in leaves if not is_building_block(leaf)]
        if missing:
            errors.append(f"non-building-block leaves: {len(missing)}")
    return RouteVerification(
        ok=not errors,
        n_steps=len(steps),
        n_executable_steps=executable,
        leaves=tuple(leaves),
        errors=tuple(errors),
    )


def _same_molecule(left: str, right: str) -> bool:
    return sides_equal(left, right, ignore_maps=True)


def _contains(values: Iterable[str], query: str) -> bool:
    return any(_same_molecule(value, query) for value in values)


def best_first_route_search(
    target: str,
    *,
    expand: Callable[[str], Iterable[dict[str, Any]]],
    is_building_block: Callable[[str], bool],
    max_nodes: int = 1000,
    max_depth: int = 8,
    max_routes: int = 10,
    branch_limit: int = 32,
) -> tuple[list[RouteCandidate], dict[str, int]]:
    """Search routes using only executor-verified proof expansions.

    ``expand(molecule)`` yields dictionaries with ``proof`` and optional
    ``model_score``/``cost`` fields. Invalid proofs never enter the frontier.
    """
    counter = itertools.count()
    start = RouteCandidate(target, (target,), (), 0.0, 0, (target,))
    frontier: list[tuple[float, int, RouteCandidate]] = [(0.0, next(counter), start)]
    solved: list[RouteCandidate] = []
    stats = {
        "nodes_popped": 0,
        "expansions_considered": 0,
        "invalid_expansions": 0,
        "cycle_rejections": 0,
        "verified_expansions": 0,
    }
    while frontier and stats["nodes_popped"] < max_nodes and len(solved) < max_routes:
        _, _, node = heapq.heappop(frontier)
        stats["nodes_popped"] += 1
        unsolved = [mol for mol in node.open_molecules if not is_building_block(mol)]
        if not unsolved:
            solved.append(node)
            continue
        if node.depth >= max_depth:
            continue
        molecule = unsolved[0]
        candidates = list(expand(molecule))[:branch_limit]
        for candidate in candidates:
            stats["expansions_considered"] += 1
            proof = str(candidate.get("proof") or candidate.get("prediction") or "")
            score = float(candidate.get("model_score", candidate.get("score", 0.0)) or 0.0)
            step = step_from_proof(
                molecule,
                proof,
                model_score=score,
                metadata={key: value for key, value in candidate.items() if key not in {"proof", "prediction"}},
            )
            if step is None:
                stats["invalid_expansions"] += 1
                continue
            if any(_contains(node.lineage, precursor) for precursor in step.precursors):
                stats["cycle_rejections"] += 1
                continue
            stats["verified_expansions"] += 1
            remaining = list(node.open_molecules)
            removed = False
            kept: list[str] = []
            for value in remaining:
                if not removed and _same_molecule(value, molecule):
                    removed = True
                    continue
                kept.append(value)
            kept.extend(step.precursors)
            edge_cost = float(candidate.get("cost", -score if score else 1.0))
            if not math.isfinite(edge_cost):
                continue
            child = RouteCandidate(
                target=target,
                open_molecules=tuple(kept),
                steps=node.steps + (step,),
                cost=node.cost + edge_cost,
                depth=node.depth + 1,
                lineage=node.lineage + step.precursors,
            )
            heapq.heappush(frontier, (child.cost, next(counter), child))
    return solved, stats
