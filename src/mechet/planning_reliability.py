"""Execution-grounded metrics for Syntheseus AND/OR planning graphs.

An absent proof is *unknown*, not a chemically valid edge. In particular, the
hallucinated-edge and wasted-expansion rates are undefined unless every admitted
reaction edge has a checkable proof. Route certification is narrower than a
laboratory feasibility claim: it means only that the frozen proof executor
replays every selected reaction to the declared structural precursors.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Collection

from .proof_program import execute_proof, parse_proof_program, sides_equal
from .proof_routes import structural_precursors


def _trace_edge_audit(reaction: Any, certificate: Any) -> dict[str, Any]:
    """Rebuild a natural-language policy edge from product and accepted actions."""
    from .endpoints import split_precursor_endpoints
    from scripts.run_natural_language_value_search import (
        Action, Node, execute, mapped_atom_numbers, normal_smiles,
        product_only_private_state, visible,
    )

    if not isinstance(certificate, dict) or certificate.get("protocol") != "mechet_nl_reverse_et_v2":
        return {"status": "failed", "executable": False, "certified": False,
                "reason": "trace_protocol_mismatch"}
    if int(certificate.get("max_imports") or 0) != 32 or certificate.get(
        "reject_target_retained_finish"
    ) is not True:
        return {"status": "failed", "executable": False, "certified": False,
                "reason": "trace_budget_mismatch"}
    target = normal_smiles(str(reaction.product.smiles))
    if not sides_equal(str(certificate.get("product") or ""), target, ignore_maps=True):
        return {"status": "failed", "executable": False, "certified": False,
                "reason": "product_mismatch"}
    state = product_only_private_state(target)
    node = Node(
        target=visible(state), state=state,
        next_map=max(mapped_atom_numbers(state), default=0) + 1,
        visited={visible(state)},
    )
    actions = certificate.get("actions")
    if not isinstance(actions, list) or not actions or len(actions) > 40:
        return {"status": "failed", "executable": False, "certified": False,
                "reason": "trace_length_invalid"}
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or not isinstance(item.get("arguments"), dict):
            return {"status": "failed", "executable": False, "certified": False,
                    "reason": "trace_action_schema_invalid"}
        child, error = execute(
            node, Action(str(item.get("name") or ""), dict(item["arguments"]), "", 0.0, 1),
            max_imports=32, reject_target_retained_finish=True,
        )
        if child is None or (child.terminal and index != len(actions) - 1):
            return {"status": "failed", "executable": False, "certified": False,
                    "reason": error or "premature_terminal"}
        node = child
    if not node.terminal or not any(
        record["name"] == "apply_electron_flow" for record in node.actions
    ):
        return {"status": "failed", "executable": False, "certified": False,
                "reason": "nonterminal_or_no_electron_flow"}
    structural = split_precursor_endpoints(node.state, state).structural
    declared = ".".join(str(mol.smiles) for mol in reaction.reactants)
    if not structural or not declared or not sides_equal(structural, declared, ignore_maps=True):
        return {"status": "failed", "executable": True, "certified": False,
                "reason": "precursor_mismatch"}
    return {"status": "certified", "executable": True, "certified": True,
            "reason": None}


def edge_audit(reaction: Any) -> dict[str, Any]:
    """Replay a proposed edge independently of the planner's solution flag."""
    metadata = dict(getattr(reaction, "metadata", {}) or {})
    trace_certificate = metadata.get("trace_certificate")
    if trace_certificate is None:
        trace_certificate = dict(metadata.get("mechet") or {}).get("trace_certificate")
    if trace_certificate is not None:
        try:
            return _trace_edge_audit(reaction, trace_certificate)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return {"status": "failed", "executable": False, "certified": False,
                    "reason": type(exc).__name__}
    proof = str(metadata.get("proof") or "").strip()
    if not proof:
        return {"status": "unverified", "executable": None, "certified": None,
                "reason": "missing_proof"}
    try:
        result = execute_proof(proof)
        if not result.ok:
            return {"status": "failed", "executable": False, "certified": False,
                    "reason": "proof_execution_failed"}
        program = parse_proof_program(proof)
        product = str(reaction.product.smiles)
        if not sides_equal(program.target_smiles, product, ignore_maps=True):
            return {"status": "failed", "executable": True, "certified": False,
                    "reason": "product_mismatch"}
        expected = ".".join(structural_precursors(product, result.precursor_smiles))
        declared = ".".join(str(mol.smiles) for mol in reaction.reactants)
        if not expected or not declared or not sides_equal(expected, declared, ignore_maps=True):
            return {"status": "failed", "executable": True, "certified": False,
                    "reason": "precursor_mismatch"}
        return {"status": "certified", "executable": True, "certified": True,
                "reason": None}
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "failed", "executable": False, "certified": False,
                "reason": type(exc).__name__}


def _reachable(graph: Any, *, excluded: set[Any]) -> set[Any]:
    visited: set[Any] = set()
    queue = deque([graph.root_node])
    while queue:
        node = queue.popleft()
        if node in visited or node in excluded:
            continue
        visited.add(node)
        queue.extend(graph.successors(node))
    return visited


def _seconds_between(first: datetime, last: datetime) -> float:
    return max(0.0, (last - first).total_seconds())


def summarize_planning_graph(
    graph: Any, routes: Collection[Collection[Any]], *,
    model_calls: int, wall_seconds: float,
) -> dict[str, Any]:
    """Summarize admitted edges and solved routes without oracle precursors."""
    nodes = list(graph.nodes())
    audits = {
        node: edge_audit(node.reaction)
        for node in nodes if hasattr(node, "reaction")
    }
    failed = {node for node, audit in audits.items() if audit["status"] == "failed"}
    unverified = {node for node, audit in audits.items() if audit["status"] == "unverified"}
    executable_edges = sum(audit["executable"] is True for audit in audits.values())
    certified_edges = sum(audit["certified"] is True for audit in audits.values())
    route_nodes = [set(route) for route in routes]
    route_reaction_nodes = [route & audits.keys() for route in route_nodes]
    nontrivial_routes = sum(bool(reactions) for reactions in route_reaction_nodes)
    unverified_routes = sum(bool(route & unverified) for route in route_nodes)
    all_edge_executable = sum(
        bool(reactions) and all(audits[node]["executable"] is True for node in reactions)
        for reactions in route_reaction_nodes
    )
    certified_routes = sum(
        bool(reactions) and all(audits[node]["certified"] is True for node in reactions)
        for reactions in route_reaction_nodes
    )
    first_route = route_nodes[0] if route_nodes else set()
    first_calls = (
        max(int(getattr(node, "data", {}).get("num_calls_rxn_model", 0))
            for node in first_route)
        if first_route else None
    )
    first_time = (
        _seconds_between(
            graph.root_node.creation_time,
            max(node.creation_time for node in first_route),
        ) if first_route else None
    )
    expanded_molecules = {
        node for node in nodes if hasattr(node, "mol") and bool(node.is_expanded)
    }
    if unverified:
        wasted_expansions = None
        wasted_ratio = None
    else:
        reachable_all = _reachable(graph, excluded=set())
        reachable_clean = _reachable(graph, excluded=failed)
        wasted_expansions = len(expanded_molecules & (reachable_all - reachable_clean))
        wasted_ratio = (
            wasted_expansions / len(expanded_molecules)
            if expanded_molecules else 0.0
        )
    return {
        "n_graph_nodes": len(nodes),
        "n_admitted_edges": len(audits),
        "n_executable_edges": executable_edges,
        "n_certified_edges": certified_edges,
        "n_failed_edges": len(failed),
        "n_unverified_edges": len(unverified),
        "hallucinated_admitted_edge_rate": (
            len(failed) / len(audits) if audits and not unverified else None
        ),
        "n_routes": len(route_nodes),
        "solved": bool(route_nodes),
        "n_zero_step_routes": len(route_nodes) - nontrivial_routes,
        "n_nontrivial_routes": nontrivial_routes,
        "n_unverified_routes": unverified_routes,
        "n_all_edge_executable_routes": all_edge_executable,
        "n_certified_routes": certified_routes,
        "all_edge_executable_route_rate": (
            all_edge_executable / nontrivial_routes
            if nontrivial_routes and not unverified_routes else None
        ),
        "certified_route_rate": (
            certified_routes / nontrivial_routes
            if nontrivial_routes and not unverified_routes else None
        ),
        "n_expanded_molecules": len(expanded_molecules),
        "n_wasted_expansions_below_failed_edges": wasted_expansions,
        "wasted_expansion_ratio": wasted_ratio,
        "reaction_model_calls": model_calls,
        "wall_seconds": wall_seconds,
        "calls_to_first_solution": first_calls,
        "seconds_to_first_solution": first_time,
        "metric_scope": (
            "admitted_planner_edges_and_nontrivial_routes_only; "
            "zero_step_stock_solutions_not_certificates; proof_replay_not_chemical_truth"
        ),
    }
