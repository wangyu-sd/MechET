"""Forward-expert rewards for complete inverse proof programs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .forward_expert import ForwardElectronExpert, score_reaction
from .proof_program import execute_proof, parse_proof_program


@dataclass(frozen=True)
class ForwardRewardConfig:
    target_weight: float = 2.0
    selectivity_weight: float = 1.0
    uncertainty_weight: float = 0.25
    formal_failure: float = -4.0
    low_margin: float = 0.10


def score_inverse_proof_forward(
    model: ForwardElectronExpert,
    proof: str,
    *,
    competitor_products: Iterable[str] = (),
    conditions: Any = None,
    config: ForwardRewardConfig | None = None,
) -> dict[str, Any]:
    """Execute an inverse proof, then independently score precursor -> target."""
    cfg = config or ForwardRewardConfig()
    execution = execute_proof(proof)
    if not execution.ok:
        return {
            "formal_execute": False,
            "forward_reward": cfg.formal_failure,
            "diagnostics": execution.diagnostics,
        }
    program = parse_proof_program(proof)
    evidence = score_reaction(
        model,
        execution.precursor_smiles,
        program.target_smiles,
        competitor_products,
        conditions=conditions,
    )
    reward = cfg.target_weight * float(evidence.target_score or 0.0)
    if evidence.selectivity_margin is not None:
        reward += cfg.selectivity_weight * float(evidence.selectivity_margin)
    if evidence.uncertainty is not None:
        reward -= cfg.uncertainty_weight * float(evidence.uncertainty)
    return {
        "formal_execute": True,
        "forward_reward": reward,
        "derived_precursor": execution.precursor_smiles,
        "target": program.target_smiles,
        "evidence": evidence.to_dict(),
        "config": asdict(cfg),
    }
