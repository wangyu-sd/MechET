"""Hypothesis-set scoring, deduplication, ranking, and survival analysis."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from typing import Any, Iterable

from mechet.proof_diagnostics import diagnose_proof
from mechet.proof_equivalence import (
    canonical_partial_order_signature,
    composition_signature,
    proofs_equivalent,
)
from mechet.proof_program import execute_proof, sides_equal, verify_proof


@dataclass
class ProofHypothesis:
    proof: str
    source_index: int = 0
    model_logprob: float = 0.0
    plausibility_score: float = 0.0
    novelty_score: float = 0.0
    format_ok: bool = False
    execute_ok: bool = False
    endpoint_exact: bool = False
    derived_precursor: str = ""
    equivalence_digest: str = ""
    composition_digest: str = ""
    failure_code: str = ""
    failure_edge: str = ""
    repaired: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HypothesisSetSummary:
    n_generated: int
    n_parseable: int
    n_executable: int
    n_endpoint_exact: int
    n_equivalence_classes: int
    n_compositions: int
    n_unique_endpoints: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def score_hypothesis(
    proof: str,
    *,
    source_index: int = 0,
    expected_precursor: str | None = None,
    gold_proof: str | None = None,
    model_logprob: float = 0.0,
    plausibility_score: float = 0.0,
    novelty_score: float = 0.0,
    repaired: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ProofHypothesis:
    verified = verify_proof(proof, expected_precursor=expected_precursor)
    item = ProofHypothesis(
        proof=proof,
        source_index=source_index,
        model_logprob=float(model_logprob),
        plausibility_score=float(plausibility_score),
        novelty_score=float(novelty_score),
        format_ok=bool(verified.get("format_ok")),
        execute_ok=bool(verified.get("execute_ok")),
        endpoint_exact=bool(verified.get("endpoint_exact")),
        derived_precursor=str(verified.get("derived_precursor") or ""),
        repaired=bool(repaired),
        metadata=dict(metadata or {}),
    )
    if item.execute_ok:
        signature = canonical_partial_order_signature(proof)
        item.equivalence_digest = signature.digest()
        item.composition_digest = composition_signature(proof)
        if gold_proof:
            try:
                item.metadata["proof_equivalent_to_gold"] = proofs_equivalent(
                    proof,
                    gold_proof,
                )
            except Exception:
                item.metadata["proof_equivalent_to_gold"] = False
    else:
        certificate = diagnose_proof(proof)
        if certificate:
            item.failure_code = certificate.code
            item.failure_edge = certificate.edge
    return item


def deduplicate_hypotheses(
    hypotheses: Iterable[ProofHypothesis],
) -> list[ProofHypothesis]:
    """Keep the best representative of each executable proof class.

    Invalid hypotheses are deduplicated by exact proof digest, because no
    chemistry equivalence relation is defined for a non-executable program.
    """
    best: dict[str, ProofHypothesis] = {}
    for item in hypotheses:
        key = (
            f"eq:{item.equivalence_digest}"
            if item.execute_ok and item.equivalence_digest
            else "raw:" + hashlib.sha256(item.proof.encode("utf-8")).hexdigest()
        )
        previous = best.get(key)
        if previous is None or hypothesis_rank_key(item) > hypothesis_rank_key(previous):
            best[key] = item
    return sorted(best.values(), key=hypothesis_rank_key, reverse=True)


def hypothesis_rank_key(item: ProofHypothesis) -> tuple[Any, ...]:
    """Lexicographic ranking; formal validity always precedes soft scores."""
    return (
        int(item.execute_ok),
        int(item.endpoint_exact),
        float(item.plausibility_score),
        float(item.model_logprob),
        float(item.novelty_score) if item.execute_ok else 0.0,
        -int(item.source_index),
    )


def rank_hypotheses(
    hypotheses: Iterable[ProofHypothesis],
    *,
    executable_only: bool = False,
) -> list[ProofHypothesis]:
    values = [item for item in hypotheses if item.execute_ok or not executable_only]
    return sorted(values, key=hypothesis_rank_key, reverse=True)


def summarize_hypotheses(
    hypotheses: Iterable[ProofHypothesis],
) -> HypothesisSetSummary:
    values = list(hypotheses)
    executable = [item for item in values if item.execute_ok]
    endpoints = {
        item.derived_precursor
        for item in executable
        if item.derived_precursor
    }
    return HypothesisSetSummary(
        n_generated=len(values),
        n_parseable=sum(item.format_ok for item in values),
        n_executable=len(executable),
        n_endpoint_exact=sum(item.endpoint_exact for item in executable),
        n_equivalence_classes=len(
            {item.equivalence_digest for item in executable if item.equivalence_digest}
        ),
        n_compositions=len(
            {item.composition_digest for item in executable if item.composition_digest}
        ),
        n_unique_endpoints=len(endpoints),
    )


def survival_curve(hypotheses: Iterable[ProofHypothesis]) -> dict[str, int]:
    values = list(hypotheses)
    parseable = [item for item in values if item.format_ok]
    executable = [item for item in parseable if item.execute_ok]
    endpoint = [item for item in executable if item.endpoint_exact]
    plausible = [item for item in endpoint if item.plausibility_score > 0.0]
    return {
        "generated": len(values),
        "parseable": len(parseable),
        "executable": len(executable),
        "endpoint_compatible": len(endpoint),
        "plausibility_supported": len(plausible),
    }


def endpoint_groups(
    hypotheses: Iterable[ProofHypothesis],
) -> list[dict[str, Any]]:
    groups: dict[str, list[ProofHypothesis]] = {}
    for item in hypotheses:
        if not item.execute_ok or not item.derived_precursor:
            continue
        key = item.derived_precursor
        for existing in groups:
            if sides_equal(existing, item.derived_precursor):
                key = existing
                break
        groups.setdefault(key, []).append(item)
    output = []
    for endpoint, values in groups.items():
        ranked = rank_hypotheses(values)
        output.append(
            {
                "endpoint": endpoint,
                "n_hypotheses": len(values),
                "n_equivalence_classes": len(
                    {item.equivalence_digest for item in values}
                ),
                "best": ranked[0].to_dict(),
            }
        )
    return sorted(output, key=lambda item: item["n_hypotheses"], reverse=True)
