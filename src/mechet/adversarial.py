"""Hard-negative mining for alternating inverse/forward training."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .forward_expert import ForwardElectronExpert, score_reaction
from .proof_program import execute_proof, parse_proof_program


@dataclass(frozen=True)
class MiningConfig:
    minimum_target_score: float = 0.65
    maximum_selectivity_margin: float | None = None
    require_endpoint_error: bool = True


def iter_hypotheses(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for row in rows:
        for hypothesis in row.get("hypotheses") or []:
            if isinstance(hypothesis, dict):
                yield {"parent": row, **hypothesis}


def mine_forward_hard_negatives(
    model: ForwardElectronExpert,
    rows: Iterable[dict[str, Any]],
    *,
    config: MiningConfig | None = None,
) -> list[dict[str, Any]]:
    """Find formally valid actor proposals that fool the forward expert.

    Returned rows carry ``label=0``. They are used only as reaction-level hard
    negatives and are never interpreted as gold electron-flow mechanisms.
    """
    cfg = config or MiningConfig()
    output: list[dict[str, Any]] = []
    for item in iter_hypotheses(rows):
        proof = str(item.get("proof") or "")
        if not proof:
            continue
        execution = execute_proof(proof)
        if not execution.ok:
            continue
        endpoint_exact = bool(item.get("endpoint_exact"))
        if cfg.require_endpoint_error and endpoint_exact:
            continue
        program = parse_proof_program(proof)
        parent = item.get("parent") or {}
        metadata = item.get("metadata") or {}
        competitors = parent.get("competitor_products") or metadata.get(
            "competitor_products"
        ) or []
        evidence = score_reaction(
            model,
            execution.precursor_smiles,
            program.target_smiles,
            competitors,
            conditions=metadata.get("conditions"),
        )
        if evidence.target_score is None or evidence.target_score < cfg.minimum_target_score:
            continue
        if (
            cfg.maximum_selectivity_margin is not None
            and evidence.selectivity_margin is not None
            and evidence.selectivity_margin > cfg.maximum_selectivity_margin
        ):
            continue
        source_id = str(parent.get("id") or item.get("source_index") or "")
        output.append(
            {
                "id": f"actor-hard-negative:{source_id}:{len(output)}",
                "source": "inverse_actor_hard_negative",
                "reactants": execution.precursor_smiles,
                "products": program.target_smiles,
                "conditions": metadata.get("conditions"),
                "competitor_products": list(competitors),
                "steps": [],
                "label": 0,
                "split": "train",
                "metadata": {
                    "proof": proof,
                    "actor_model_logprob": item.get("model_logprob"),
                    "endpoint_exact": endpoint_exact,
                    "forward_evidence": evidence.to_dict(),
                },
            }
        )
    return output


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
