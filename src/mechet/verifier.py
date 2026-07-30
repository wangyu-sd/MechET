"""MechET process reward verifier (MECH_ET v3)."""

from __future__ import annotations

from typing import Any

from mechet.mech_et import (
    MECH_ET_HEADER,
    be_delta_exact,
    be_delta_from_mapped_smiles,
    electron_conserved,
    verify_mech_et,
    _maps_in_smiles,
)
from mechet.mech_graph import (
    FlowERMechanismGraph,
    _sides_equal,
    _split_frags,
    compact_mapped_smiles,
    official_state_key,
)
from mechet.sft import parse_mech_cot_output


def _is_mech_et_text(text: str) -> bool:
    body = (text or "").strip()
    if body.startswith(MECH_ET_HEADER) or body.startswith("MECH_ET"):
        return True
    if "<mechanism>" in body.lower() and "MECH_ET" in body:
        return True
    return False


def _declared_be_delta_edges(mechanism_body: str) -> set[tuple[str, str]]:
    """Return RETRO_EDGE pairs that explicitly contain a BE_DELTA block."""
    declared: set[tuple[str, str]] = set()
    current: tuple[str, str] | None = None
    for raw in (mechanism_body or "").splitlines():
        line = raw.strip()
        if line.startswith("RETRO_EDGE "):
            parts = line.split()
            current = (parts[1], parts[2]) if len(parts) == 3 else None
            continue
        if current is not None and line == "BE_DELTA":
            declared.add(current)
            continue
        if current is not None and line and not line.startswith(("BOND ", "LP ", "CHARGE ")):
            current = None
    return declared


def _target_state_contains_product(parsed: dict[str, Any], product_smiles: str) -> bool:
    """Require the product to occur in a TARGET_STATE, not only TARGET_SMILES."""
    product = (product_smiles or "").strip()
    if not product:
        return True
    product_key = official_state_key(product) or compact_mapped_smiles(product)
    states = parsed.get("states") or {}
    for target_id in parsed.get("target_state_ids") or []:
        for fragment in _split_frags(str(states.get(target_id) or "")):
            fragment_key = official_state_key(fragment) or compact_mapped_smiles(fragment)
            if fragment_key == product_key:
                return True
            if compact_mapped_smiles(fragment) == compact_mapped_smiles(product):
                return True
    return False


def verify_mech_et_strict(
    *,
    mechanism_body: str,
    answer: str,
    main_product: str | None = None,
    expected_precursor: str | None = None,
    expected_graph: FlowERMechanismGraph | None = None,
) -> dict[str, Any]:
    """Verify a generated mechanism using only its own states and edge program.

    For every generated edge ``a -> b``, the written ``BE_DELTA`` must equal
    ``BE(b) - BE(a)``. CHARGE lines are optional, but any written charge
    transitions must be correct.
    """
    verified = verify_mech_et(
        mechanism_body=mechanism_body,
        answer=answer,
        main_product=main_product,
        expected_precursor=expected_precursor,
        expected_graph=expected_graph,
    )
    parsed = verified.get("parsed") or {}
    verified["answer_gold_exact"] = bool(expected_precursor) and bool(verified.get("answer_exact"))
    verified["answer_state_agree"] = False
    verified["target_state_matches_product"] = False
    verified["be_delta_blocks_present"] = False
    verified["state_maps_consistent"] = False
    verified["local_transition_exact"] = False

    if not parsed.get("ok"):
        return verified

    states: dict[str, str] = dict(parsed.get("states") or {})
    edges: list[tuple[str, str]] = list(parsed.get("retro_edges") or [])
    edge_deltas = dict(parsed.get("edge_deltas") or {})
    precursor_id = str(parsed.get("precursor_state_id") or "")
    precursor_smiles = str(states.get(precursor_id) or "")

    verified["answer_state_agree"] = bool((answer or "").strip()) and bool(precursor_smiles) and _sides_equal(
        answer,
        precursor_smiles,
    )
    product = str(main_product or parsed.get("target_smiles") or "")
    verified["target_state_matches_product"] = _target_state_contains_product(parsed, product)
    verified["main_product_ok"] = verified["target_state_matches_product"]

    declared_blocks = _declared_be_delta_edges(mechanism_body)
    verified["be_delta_blocks_present"] = bool(edges) and all(edge in declared_blocks for edge in edges)

    exact_edges = 0
    conserved_edges = 0
    map_consistent_edges = 0
    for src, dst in edges:
        if src not in states or dst not in states:
            continue
        if _maps_in_smiles(states[src]) == _maps_in_smiles(states[dst]):
            map_consistent_edges += 1
        else:
            verified["diagnostics"].append(
                {"code": "STATE_MAP_MISMATCH", "message": f"{src}->{dst}"}
            )

        if (src, dst) not in declared_blocks:
            verified["diagnostics"].append(
                {"code": "MISSING_BE_DELTA_BLOCK", "message": f"{src}->{dst}"}
            )
            continue
        derived = be_delta_from_mapped_smiles(states[src], states[dst])
        if derived is None:
            verified["diagnostics"].append(
                {"code": "LOCAL_BE_PARSE_FAIL", "message": f"{src}->{dst}"}
            )
            continue
        written = edge_deltas.get((src, dst))
        if written is None:
            verified["diagnostics"].append(
                {"code": "MISSING_BE_DELTA", "message": f"{src}->{dst}"}
            )
            continue

        core_exact = be_delta_exact(written, derived, check_charge=False)
        charge_exact = not written.charges or sorted(written.charges) == sorted(derived.charges)
        if core_exact and charge_exact:
            exact_edges += 1
            if electron_conserved(written):
                conserved_edges += 1
            else:
                verified["diagnostics"].append(
                    {"code": "ELECTRON_NOT_CONSERVED", "message": f"{src}->{dst}"}
                )
        else:
            verified["diagnostics"].append(
                {"code": "LOCAL_BE_DELTA_MISMATCH", "message": f"{src}->{dst}"}
            )

    n_edges = len(edges)
    verified["state_maps_consistent"] = n_edges > 0 and map_consistent_edges == n_edges
    verified["local_transition_exact"] = (
        n_edges > 0
        and verified["be_delta_blocks_present"]
        and exact_edges == n_edges
        and verified["state_maps_consistent"]
    )
    verified["be_delta_exact"] = verified["local_transition_exact"]
    verified["electron_conserved"] = verified["local_transition_exact"] and conserved_edges == n_edges

    if expected_graph is not None:
        gold_checked = verify_mech_et(
            mechanism_body=mechanism_body,
            answer=answer,
            main_product=main_product,
            expected_precursor=expected_precursor,
            expected_graph=expected_graph,
        )
        verified["gold_be_delta_exact"] = bool(gold_checked.get("be_delta_exact"))
        verified["be_delta_exact"] = bool(verified["be_delta_exact"] and verified["gold_be_delta_exact"])
    else:
        verified["gold_be_delta_exact"] = None

    return verified


def _failure_payload(
    *,
    verified: dict[str, Any],
    details: dict[str, float],
    stage: str,
    reward: float,
) -> dict[str, Any]:
    details[f"{stage}_penalty"] = float(reward)
    return {
        "total": float(reward),
        "details": details,
        "hard_fail": True,
        "failure_stage": stage,
        "rlvr_failure_reward": float(reward),
        "verified": verified,
    }


def compute_mech_et_reward(
    text: str,
    product_smiles: str,
    *,
    expected_precursors: list[str] | None = None,
    expected_precursor: str | None = None,
    expected_graph: FlowERMechanismGraph | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reward a locally executable MECH_ET v3 mechanism and its endpoint."""
    defaults = {
        "format_reward": 0.5,
        "target_reward": 0.5,
        "reachability_reward": 1.0,
        "be_delta_reward": 2.0,
        "electron_conserved_reward": 1.0,
        "state_agree_reward": 1.0,
        "edge_f1_reward": 1.5,
        "answer_reward": 3.0,
        "unsupported_penalty": -4.0,
        "target_penalty": -3.5,
        "unreachable_penalty": -3.0,
        "hallucination_penalty": -2.0,
        "conservation_penalty": -1.5,
        "state_mismatch_penalty": -1.0,
    }
    cfg = {**defaults, **(config or {})}
    details: dict[str, float] = {}
    parsed = (
        parse_mech_cot_output(text)
        if "<mechanism>" in (text or "").lower()
        else {
            "format_ok": bool((text or "").strip()),
            "mechanism": (text or "").strip(),
            "answer": "",
        }
    )
    mechanism = str(parsed.get("mechanism") or "")
    answer = str(parsed.get("answer") or "").strip()
    expected = expected_precursor
    if expected is None and expected_precursors:
        expected = ".".join(expected_precursors)

    verified = verify_mech_et_strict(
        mechanism_body=mechanism,
        answer=answer,
        main_product=product_smiles or None,
        expected_precursor=expected,
        expected_graph=expected_graph,
    )
    if not verified.get("format_ok"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="format",
            reward=float(cfg["unsupported_penalty"]),
        )

    total = float(cfg["format_reward"])
    details["format_reward"] = float(cfg["format_reward"])

    if not verified.get("target_state_matches_product"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="target",
            reward=float(cfg["target_penalty"]),
        )
    details["target_reward"] = float(cfg["target_reward"])
    total += details["target_reward"]

    if not verified.get("reachability_ok"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="reachability",
            reward=float(cfg["unreachable_penalty"]),
        )
    details["reachability_reward"] = float(cfg["reachability_reward"])
    total += details["reachability_reward"]

    if not verified.get("local_transition_exact"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="be_delta",
            reward=float(cfg["hallucination_penalty"]),
        )
    details["be_delta_reward"] = float(cfg["be_delta_reward"])
    total += details["be_delta_reward"]

    if not verified.get("electron_conserved"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="conservation",
            reward=float(cfg["conservation_penalty"]),
        )
    details["electron_conserved_reward"] = float(cfg["electron_conserved_reward"])
    total += details["electron_conserved_reward"]

    if not verified.get("answer_state_agree"):
        return _failure_payload(
            verified=verified,
            details=details,
            stage="state_agree",
            reward=float(cfg["state_mismatch_penalty"]),
        )
    details["state_agree_reward"] = float(cfg["state_agree_reward"])
    total += details["state_agree_reward"]

    if expected_graph is not None:
        edge_f1 = float(verified.get("edge_f1") or 0.0)
        details["edge_f1_reward"] = float(cfg["edge_f1_reward"]) * edge_f1
        total += details["edge_f1_reward"]
        if edge_f1 < 1.0 and verified.get("graph_exact") is False:
            return _failure_payload(
                verified=verified,
                details=details,
                stage="gold_graph",
                reward=float(cfg["hallucination_penalty"]),
            )

    if verified.get("answer_gold_exact"):
        details["answer_reward"] = float(cfg["answer_reward"])
        total += details["answer_reward"]
    return {
        "total": total,
        "details": details,
        "hard_fail": False,
        "failure_stage": None,
        "verified": verified,
    }


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
        "total": -4.0,
        "details": {"unsupported_penalty": -4.0},
        "hard_fail": True,
        "failure_stage": "format",
        "rlvr_failure_reward": -4.0,
    }
