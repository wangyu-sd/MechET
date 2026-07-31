"""Executable proof hypergraphs for reaction-network exploration.

This module supplies deterministic graph bookkeeping. Energies, rates, and
experimental plausibility remain external attributes and are never inferred by
the formal executor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Iterable

from mechet.proof_equivalence import composition_signature
from mechet.proof_program import execute_proof, parse_proof_program, sides_equal
from mechet.proof_routes import structural_precursors


@dataclass(frozen=True)
class ReactionHyperedge:
    edge_id: str
    product: str
    precursors: tuple[str, ...]
    proof: str
    composition_digest: str
    reversible: bool = False
    model_score: float = 0.0
    plausibility_score: float = 0.0
    energy: float | None = None
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["precursors"] = list(self.precursors)
        return payload


@dataclass
class ReactionNetwork:
    species: list[str] = field(default_factory=list)
    edges: list[ReactionHyperedge] = field(default_factory=list)

    def _species_key(self, smiles: str) -> str:
        for value in self.species:
            if sides_equal(value, smiles, ignore_maps=True):
                return value
        self.species.append(smiles)
        return smiles

    def add_proof(
        self,
        proof: str,
        *,
        edge_id: str = "",
        reversible: bool = False,
        model_score: float = 0.0,
        plausibility_score: float = 0.0,
        energy: float | None = None,
        uncertainty: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> ReactionHyperedge | None:
        result = execute_proof(proof)
        if not result.ok:
            return None
        program = parse_proof_program(proof)
        product = self._species_key(program.target_smiles)
        precursors = tuple(
            self._species_key(value)
            for value in structural_precursors(product, result.precursor_smiles)
        )
        digest = composition_signature(proof)
        edge_id = edge_id or hashlib.sha256(
            (product + "|" + ".".join(sorted(precursors)) + "|" + digest).encode("utf-8")
        ).hexdigest()[:20]
        for existing in self.edges:
            if existing.edge_id == edge_id:
                return existing
        edge = ReactionHyperedge(
            edge_id=edge_id,
            product=product,
            precursors=precursors,
            proof=proof,
            composition_digest=digest,
            reversible=bool(reversible),
            model_score=float(model_score),
            plausibility_score=float(plausibility_score),
            energy=None if energy is None else float(energy),
            uncertainty=float(uncertainty),
            metadata=dict(metadata or {}),
        )
        self.edges.append(edge)
        return edge

    def to_dict(self) -> dict[str, Any]:
        return {
            "species": list(self.species),
            "edges": [edge.to_dict() for edge in self.edges],
            "summary": {
                "n_species": len(self.species),
                "n_edges": len(self.edges),
                "n_reversible": sum(edge.reversible for edge in self.edges),
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReactionNetwork":
        network = cls(species=list(payload.get("species") or []))
        for item in payload.get("edges") or []:
            network.edges.append(
                ReactionHyperedge(
                    edge_id=str(item["edge_id"]),
                    product=str(item["product"]),
                    precursors=tuple(item.get("precursors") or []),
                    proof=str(item.get("proof") or ""),
                    composition_digest=str(item.get("composition_digest") or ""),
                    reversible=bool(item.get("reversible")),
                    model_score=float(item.get("model_score") or 0.0),
                    plausibility_score=float(item.get("plausibility_score") or 0.0),
                    energy=item.get("energy"),
                    uncertainty=float(item.get("uncertainty") or 0.0),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        return network


def frontier_score(
    candidate: ReactionHyperedge,
    network: ReactionNetwork,
    *,
    novelty_weight: float = 1.0,
    uncertainty_weight: float = 1.0,
    plausibility_weight: float = 1.0,
    energy_weight: float = 0.0,
) -> float:
    """Rank candidates for expensive external evaluation.

    The score is only a frontier policy. It is not a proof of chemical
    feasibility and cannot override executor failure.
    """
    known_compositions = {edge.composition_digest for edge in network.edges}
    known_species = network.species
    new_species = sum(
        not any(sides_equal(item, value, ignore_maps=True) for value in known_species)
        for item in (candidate.product, *candidate.precursors)
    )
    novel_composition = int(candidate.composition_digest not in known_compositions)
    novelty = float(new_species + novel_composition)
    energy_term = 0.0
    if candidate.energy is not None:
        energy_term = -float(candidate.energy)
    return (
        novelty_weight * novelty
        + uncertainty_weight * float(candidate.uncertainty)
        + plausibility_weight * float(candidate.plausibility_score)
        + energy_weight * energy_term
    )


def rank_frontier(
    candidates: Iterable[ReactionHyperedge],
    network: ReactionNetwork,
    **weights: float,
) -> list[dict[str, Any]]:
    ranked = [
        {
            "score": frontier_score(candidate, network, **weights),
            "candidate": candidate.to_dict(),
        }
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _adjacency(network: ReactionNetwork) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {value: set() for value in network.species}
    for edge in network.edges:
        for precursor in edge.precursors:
            graph.setdefault(precursor, set()).add(edge.product)
            if edge.reversible:
                graph.setdefault(edge.product, set()).add(precursor)
    return graph


def find_species_cycles(
    network: ReactionNetwork,
    *,
    max_length: int = 8,
) -> list[list[str]]:
    """Find simple species-level cycles induced by proof hyperedges."""
    graph = _adjacency(network)
    cycles: set[tuple[str, ...]] = set()

    def canonical_cycle(values: list[str]) -> tuple[str, ...]:
        body = values[:-1]
        rotations = [tuple(body[index:] + body[:index]) for index in range(len(body))]
        return min(rotations)

    def visit(start: str, current: str, path: list[str]) -> None:
        if len(path) > max_length:
            return
        for nxt in graph.get(current, set()):
            if nxt == start and len(path) >= 2:
                cycles.add(canonical_cycle(path + [start]))
            elif nxt not in path:
                visit(start, nxt, path + [nxt])

    for species in graph:
        visit(species, species, [species])
    return [list(cycle) + [cycle[0]] for cycle in sorted(cycles)]


def network_digest(network: ReactionNetwork) -> str:
    payload = network.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
