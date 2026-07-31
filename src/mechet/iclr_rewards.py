"""ICLR-stage proof rewards using structural precursors and weak composition supervision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mechet.data_audit import split_structural_and_environment
from mechet.proof_equivalence import composition_signature
from mechet.proof_program import extract_proof_body, sides_equal, verify_proof


@dataclass(frozen=True)
class CoreProofRewardConfig:
    missing_proof: float = -4.0
    invalid_proof: float = -2.0
    execute: float = 2.5
    endpoint_core_exact: float = 4.0
    composition_match: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CoreProofRewardConfig":
        if not value:
            return cls()
        known = {field: value[field] for field in cls.__dataclass_fields__ if field in value}
        return cls(**known)


def _product(row: Mapping[str, Any]) -> str:
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith("TARGET:"):
            return content.split("\n", 1)[0].replace("TARGET:", "", 1).strip()
    return ""


def _assistant(row: Mapping[str, Any]) -> str:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _full_gold(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(metadata.get("derived_precursor") or metadata.get("initial_reactants") or "")


def core_gold(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    explicit = str(metadata.get("core_precursor") or "")
    if explicit:
        return explicit
    return split_structural_and_environment(_full_gold(row), _product(row)).structural_smiles


def compute_core_proof_reward(
    row: Mapping[str, Any],
    prediction: str,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reward deterministic execution and the atom-contributing precursor.

    Composition matching is deliberately a weak auxiliary reward: it does not
    require exact state ids, atom-map labels, serialization, or teacher trace.
    """
    cfg = CoreProofRewardConfig.from_mapping(config)
    product = _product(row)
    gold_core = core_gold(row)
    proof_body = extract_proof_body(prediction)
    if not proof_body:
        return {
            "gate_ok": False,
            "rlvr_total": cfg.missing_proof,
            "execute_ok": False,
            "endpoint_core_exact": False,
            "composition_match": False,
            "reward_mode": "mech_proof_core_v1",
            "diagnostics": [{"code": "MISSING_PROOF", "message": "missing <proof> block"}],
        }

    verified = verify_proof(prediction)
    if not verified.get("execute_ok"):
        return {
            **verified,
            "gate_ok": False,
            "rlvr_total": cfg.invalid_proof,
            "endpoint_core_exact": False,
            "composition_match": False,
            "reward_mode": "mech_proof_core_v1",
        }

    derived_full = str(verified.get("derived_precursor") or "")
    derived_core = split_structural_and_environment(derived_full, product).structural_smiles
    endpoint_core_exact = bool(gold_core and sides_equal(derived_core, gold_core))
    composition_match = False
    gold_proof = _assistant(row)
    if gold_proof and extract_proof_body(gold_proof):
        try:
            composition_match = composition_signature(prediction) == composition_signature(gold_proof)
        except Exception:
            composition_match = False

    total = cfg.execute
    if endpoint_core_exact:
        total += cfg.endpoint_core_exact
    if composition_match:
        total += cfg.composition_match
    return {
        **verified,
        "gate_ok": True,
        "rlvr_total": float(total),
        "derived_core_precursor": derived_core,
        "gold_core_precursor": gold_core,
        "endpoint_core_exact": endpoint_core_exact,
        "composition_match": composition_match,
        "reward_mode": "mech_proof_core_v1",
    }
