"""Adapter from the compact forward expert to MechET's plausibility oracle API."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .forward_expert import ForwardElectronExpert, score_reaction
from .plausibility import PlausibilityEvidence
from .proof_program import execute_proof

_MODEL: ForwardElectronExpert | None = None
_MODEL_PATH: str = ""


def _load() -> ForwardElectronExpert:
    global _MODEL, _MODEL_PATH
    path = os.environ.get("MECHET_FORWARD_EXPERT_PATH", "")
    if not path:
        raise RuntimeError("set MECHET_FORWARD_EXPERT_PATH to a trained checkpoint directory")
    if _MODEL is None or _MODEL_PATH != path:
        _MODEL = ForwardElectronExpert.load(
            Path(path),
            device=os.environ.get("MECHET_FORWARD_EXPERT_DEVICE", "cpu"),
        )
        _MODEL_PATH = path
    return _MODEL


def score_payload(payload: dict[str, Any]) -> PlausibilityEvidence:
    """Score a proof/precursor candidate without inventing missing evidence.

    Required payload fields are ``reactants`` or ``precursors`` and ``target`` or
    ``product``. Optional ``competitor_products`` enables selectivity scoring.
    A supplied proof must execute before the learned score is returned.
    """
    proof = str(payload.get("proof") or "")
    if proof:
        execution = execute_proof(proof)
        if not execution.ok:
            return PlausibilityEvidence(
                expert_score=0.0,
                uncertainty=1.0,
                sources=("mechet-formal-executor",),
                metadata={
                    "formal_execute": False,
                    "diagnostics": execution.diagnostics,
                },
            )
    reactants = str(payload.get("reactants") or payload.get("precursors") or "")
    target = str(payload.get("target") or payload.get("product") or "")
    if not reactants or not target:
        raise ValueError(
            "forward oracle requires reactants/precursors and target/product"
        )
    evidence = score_reaction(
        _load(),
        reactants,
        target,
        payload.get("competitor_products") or (),
        conditions=payload.get("conditions"),
    )
    return PlausibilityEvidence(
        expert_score=evidence.target_score,
        uncertainty=evidence.uncertainty,
        sources=("mechet-forward-electron-expert",),
        metadata={
            "formal_execute": True,
            "target_rank": evidence.target_rank,
            "best_competitor_score": evidence.best_competitor_score,
            "selectivity_margin": evidence.selectivity_margin,
            "verdict": evidence.verdict,
        },
    )
