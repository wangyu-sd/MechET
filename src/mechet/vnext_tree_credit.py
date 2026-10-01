"""Verified same-state credit and search-teacher targets for MechET vNext.

This module never runs or replaces the chemistry executor. It consumes only
executor-produced successor states, rejects invalid branches, pools chemically
equivalent states, and refuses positive relative updates in all-negative groups.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from rdkit import Chem


def chemical_state_key(state: str, *, terminal: bool = False) -> str:
    """Canonical unmapped graph key, preserving isotopes/stereochemistry/charge."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(str(state or ""), params)
    if mol is None:
        raise ValueError("cannot pool an invalid chemical state")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return hashlib.sha256(f"{int(terminal)}:{canonical}".encode()).hexdigest()


def _first_successor(record: Mapping[str, Any]) -> tuple[str, bool, bool]:
    score = dict(record.get("score") or {})
    successor = str(score.get("first_successor_state") or "")
    failure = str(score.get("failure") or "").upper()
    valid = bool(successor) and not any(
        marker in failure for marker in ("STATE_CYCLE", "NO_OP", "TARGET_RETAINED_NO_TRANSFORM")
    )
    return successor, bool(score.get("first_successor_terminal")), valid


def assign_sibling_advantages(
    records: Sequence[dict[str, Any]],
    *,
    method: str = "tree",
    dynamic_all_negative: bool = True,
    allow_private_reference: bool = False,
) -> dict[str, Any]:
    """Assign advantages over *unique executed successors* of one state.

    `grpo` uses reward-normalized sibling outcomes; `gspo` uses the same
    advantages with sequence-level policy ratios in the optimizer; `tree`
    credits verified endpoint/reference-surviving branches plus an optional
    bounded successor-reachability score. Private reference matching is an
    explicit training condition, never an inference feature.
    """
    if method not in {"grpo", "gspo", "tree"}:
        raise ValueError(f"unknown sibling credit method: {method}")
    if not records:
        raise ValueError("empty sibling group")
    anchors = {(str(r["id"]), str(r["anchor"]["state_hash"]), str(r.get("prompt_mode", ""))) for r in records}
    if len(anchors) != 1:
        raise ValueError("sibling credit requires one reaction, state and prompt mode")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = 0
    for record in records:
        score = dict(record.get("score") or {})
        # The collector may retain a private audit label.  It contributes to
        # credit only under the explicit opt-in below, never by mere presence.
        successor, terminal, valid = _first_successor(record)
        record["advantage"] = 0.0
        record["update_eligible"] = False
        record["credit_mode"] = method
        record["ratio_mode"] = "sequence" if method == "gspo" else "token"
        record["credit_private_reference_used"] = bool(allow_private_reference)
        if not valid:
            record["successor_equivalence_key"] = "INVALID"
            invalid += 1
            continue
        key = chemical_state_key(successor, terminal=terminal)
        record["successor_equivalence_key"] = key
        grouped[key].append(record)

    positives = {
        key: any(
            bool(r["score"].get("correct"))
            or (allow_private_reference and bool(r["score"].get("reference_first_successor_exact")))
            for r in siblings
        )
        for key, siblings in grouped.items()
    }
    has_positive = any(positives.values())
    all_negative = not has_positive
    q: dict[str, float] = {}
    if has_positive:
        for key, siblings in grouped.items():
            if method == "tree":
                reachability = [
                    max(0.0, min(1.0, float(r["score"].get("successor_reachability", 0.0))))
                    for r in siblings
                ]
                endpoint = any(bool(r["score"].get("correct")) for r in siblings)
                q[key] = 1.0 if endpoint else (
                    0.6 + 0.4 * sum(reachability) / len(reachability)
                    if positives[key] else 0.4 * sum(reachability) / len(reachability)
                )
            else:
                q[key] = sum(float(r.get("reward", 0.0)) for r in siblings) / len(siblings)
    effective = has_positive and len(q) > 1
    if effective:
        mean = sum(q.values()) / len(q)
        variance = sum((v - mean) ** 2 for v in q.values()) / len(q)
        scale = math.sqrt(variance)
        effective = scale > 1e-8
        if effective:
            for key, siblings in grouped.items():
                advantage = (q[key] - mean) / (scale + 1e-4)
                for record in siblings:
                    record["advantage"] = advantage
                    record["update_eligible"] = True
                    record["sibling_q"] = q[key]
    return {
        "method": method,
        "uses_private_reference": bool(allow_private_reference),
        "ratio_mode": "sequence" if method == "gspo" else "token",
        "unique_successors": len(grouped),
        "invalid_branches": invalid,
        "has_verified_positive": has_positive,
        "all_negative": all_negative,
        "dynamic_resample": bool(dynamic_all_negative and all_negative),
        "effective": bool(effective),
        "eligible_records": sum(bool(r["update_eligible"]) for r in records),
        "equivalent_duplicates": sum(len(v) - 1 for v in grouped.values()),
    }


def search_teacher_distribution(
    records: Sequence[Mapping[str, Any]], *, temperature: float = 1.0
) -> list[dict[str, Any]]:
    """Build a positive-only action teacher from verified search siblings.

    A failed search gives no distillation target. Equivalent successors share
    one probability mass instead of being rewarded for duplicate surface forms.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    by_state: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not bool(record.get("update_eligible")):
            continue
        key = str(record.get("successor_equivalence_key") or "")
        if not key or key == "INVALID":
            continue
        score = dict(record.get("score") or {})
        if not (
            score.get("correct")
            or (record.get("credit_private_reference_used") and score.get("reference_first_successor_exact"))
        ):
            continue
        incumbent = by_state.get(key)
        if incumbent is None or float(record.get("sibling_q", 0)) > float(incumbent.get("sibling_q", 0)):
            by_state[key] = record
    if not by_state:
        return []
    chosen = list(by_state.values())
    scaled = [float(r.get("sibling_q", 0.0)) / temperature for r in chosen]
    shift = max(scaled)
    weights = [math.exp(v - shift) for v in scaled]
    total = sum(weights)
    return [
        {"action_fingerprint": str(r["action_fingerprint"]),
         "successor_key": str(r["successor_equivalence_key"]),
         "candidate_index": int(r["candidate_index"]),
         "probability": weight / total,
         "training_private_reference": bool(r.get("credit_private_reference_used") and r["score"].get("reference_first_successor_exact"))}
        for r, weight in zip(chosen, weights, strict=True)
    ]
