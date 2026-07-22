"""MechET process reward verifier (MECH_ET v3)."""

from __future__ import annotations

from typing import Any

from mechet.mech_et import MECH_ET_HEADER, verify_mech_et
from mechet.mech_graph import FlowERMechanismGraph
from mechet.sft import parse_mech_cot_output


def _is_mech_et_text(text: str) -> bool:
    body = (text or "").strip()
    if body.startswith(MECH_ET_HEADER) or body.startswith("MECH_ET"):
        return True
    if "<mechanism>" in body.lower() and "MECH_ET" in body:
        return True
    return False


def compute_mech_et_reward(
    text: str,
    product_smiles: str,
    *,
    expected_precursors: list[str] | None = None,
    expected_precursor: str | None = None,
    expected_graph: FlowERMechanismGraph | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reward verifier for MECH_ET v3 CoT (graph + BE_DELTA process rewards)."""
    cfg = config or {
        "format_reward": 1.0,
        "reachability_reward": 1.5,
        "be_delta_reward": 2.0,
        "electron_conserved_reward": 1.0,
        "edge_f1_reward": 1.5,
        "answer_reward": 3.0,
        "unsupported_penalty": -2.0,
        "hallucination_penalty": -3.0,
        "unreachable_penalty": -2.0,
    }
    details: dict[str, float] = {}
    parsed = (
        parse_mech_cot_output(text)
        if "<mechanism>" in (text or "").lower()
        else {
            "format_ok": bool((text or "").strip()),
            "mechanism": (text or "").strip(),
            "answer": ".".join(expected_precursors or []) if expected_precursors else "",
        }
    )
    mechanism = str(parsed.get("mechanism") or "")
    answer = str(parsed.get("answer") or "").strip()
    if expected_precursors and not answer:
        answer = ".".join(expected_precursors)
    expected = expected_precursor
    if expected is None and expected_precursors:
        expected = ".".join(expected_precursors)

    verified = verify_mech_et(
        mechanism_body=mechanism,
        answer=answer,
        main_product=product_smiles or None,
        expected_precursor=expected,
        expected_graph=expected_graph,
    )
    if not verified.get("format_ok"):
        details["format_reward"] = float(cfg["unsupported_penalty"])
        return {"total": details["format_reward"], "details": details, "hard_fail": True, "verified": verified}

    total = 0.0
    details["format_reward"] = float(cfg["format_reward"])
    total += details["format_reward"]

    if verified.get("reachability_ok"):
        details["reachability_reward"] = float(cfg["reachability_reward"])
        total += details["reachability_reward"]
    else:
        details["reachability_reward"] = float(cfg["unreachable_penalty"])
        total += details["reachability_reward"]
        return {"total": total, "details": details, "hard_fail": True, "verified": verified}

    if verified.get("be_delta_exact"):
        details["be_delta_reward"] = float(cfg["be_delta_reward"])
        total += details["be_delta_reward"]
    else:
        details["be_delta_reward"] = float(cfg["hallucination_penalty"])
        total += details["be_delta_reward"]
        if expected_graph is not None:
            return {"total": total, "details": details, "hard_fail": True, "verified": verified}

    if verified.get("electron_conserved"):
        details["electron_conserved_reward"] = float(cfg["electron_conserved_reward"])
        total += details["electron_conserved_reward"]

    if expected_graph is not None:
        edge_f1 = float(verified.get("edge_f1") or 0.0)
        details["edge_f1_reward"] = float(cfg["edge_f1_reward"]) * edge_f1
        total += details["edge_f1_reward"]
        if edge_f1 < 1.0 and verified.get("graph_exact") is False:
            return {
                "total": total + float(cfg["hallucination_penalty"]),
                "details": details,
                "hard_fail": True,
                "verified": verified,
            }

    if verified.get("answer_exact"):
        details["answer_reward"] = float(cfg["answer_reward"])
        total += details["answer_reward"]
    return {"total": total, "details": details, "hard_fail": False, "verified": verified}


def compute_reward(
    program_text: str,
    product_smiles: str,
    *,
    expected_precursors: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _is_mech_et_text(program_text):
        return compute_mech_et_reward(
            program_text,
            product_smiles,
            expected_precursors=expected_precursors,
            config=config,
        )
    return {
        "total": -2.0,
        "details": {"unsupported_penalty": -2.0},
        "hard_fail": True,
    }
