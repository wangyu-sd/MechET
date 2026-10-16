"""Observed statistics with explicit pool, context and route denominators."""
from __future__ import annotations

from collections import Counter, defaultdict
import math


def candidate_metrics(candidates, *, has_reference):
    # The recorded native evaluator supplies stereochemical canonical strings.
    # Normalize component order only; do not silently neutralize or desalt.
    ordered = sorted(enumerate(candidates), key=lambda item: (item[1].get("mean_nll") if item[1].get("mean_nll") is not None else math.inf, item[0]))
    unique = {}
    for _, c in ordered:
        key = ".".join(sorted(c["structural_precursor"].split(".")))
        if key in unique and has_reference and unique[key].get("matches_reference") != c.get("matches_reference"):
            raise ValueError("same precursor has contradictory reference labels")
        unique.setdefault(key, c)
    hits = [i + 1 for i, c in enumerate(unique.values()) if c.get("matches_reference") is True]
    rank = min(hits) if hits else None
    if has_reference and any(c.get("matches_reference") is None for c in unique.values()):
        raise ValueError("reference flags missing within a labeled candidate pool")
    return {"unique_candidates": len(unique), "ordered_precursors": list(unique),
        "reference_rank": rank if has_reference else None,
        "pool_recall": bool(rank) if has_reference else None,
        "top1": rank == 1 if has_reference else None,
        "top5": bool(rank and rank <= 5) if has_reference else None,
        "top10": bool(rank and rank <= 10) if has_reference else None,
        "ranking_gap": int(bool(rank)) - int(rank == 1) if has_reference else None,
        "reciprocal_rank": (1 / rank if rank else 0.0) if has_reference else None,
        "independent_pass_at_k": None, "pass_at_k_status": "MCTS terminals are not independent trajectory episodes",
        "canonicalization": "source_native_stereo_canonical_components_sorted_no_neutralization"}


def morphology(data):
    edges = data["attempted_edges"]
    successors, flow_successors = defaultdict(set), defaultdict(set)
    expanded = {e["parent_node_id"] for e in edges if e["parent_node_id"] is not None}
    for e in edges:
        if e["execution_accepted"] and e["parent_node_id"] and e["child_node_id"]:
            successors[e["parent_node_id"]].add(e["child_node_id"])
            if e["action_family"] == "apply_electron_flow":
                flow_successors[e["parent_node_id"]].add(e["child_node_id"])
    status = Counter(e["status"] for e in edges)
    return {"native_contexts": len(data["nodes"]), "attempted_actions": len(edges),
        "expanded_contexts": len(expanded), "rejected_attempts": status["rejected"],
        "duplicate_attempts": status["duplicate_successor"],
        "unique_context_transitions": sum(map(len, successors.values())),
        "branching_mean_expanded": sum(map(len, successors.values())) / len(expanded) if expanded else None,
        "branching_max": max(map(len, successors.values()), default=0),
        "flow_fork_contexts": sum(len(v) > 1 for v in flow_successors.values()),
        "flow_branching_max": max(map(len, flow_successors.values()), default=0),
        "flow_depth_max": max((n["flow_depth"] for n in data["nodes"]), default=0),
        "terminal_contexts": len(data["terminals"]), "status_counts": dict(status),
        "action_counts": dict(Counter(e["action_family"] for e in edges)),
        "error_counts": dict(Counter(e["error"] for e in edges if not e["execution_accepted"]))}


def route_closure(leaves, stock, *, reaction_count):
    if stock is None:
        return {"all_leaves_in_stock": None, "stock_solved": None, "status": "stock_unavailable"}
    all_in_stock = bool(leaves) and all(leaf in stock for leaf in leaves)
    return {"all_leaves_in_stock": all_in_stock, "stock_solved": all_in_stock and reaction_count > 0,
        "already_stock_not_generated_route": all_in_stock and reaction_count == 0, "status": "frozen_stock_rechecked"}


def verify_reported_route(report, stock):
    """Check an actual selected AND-route witness, not just its asserted leaves."""
    if stock is None:
        return {"stock_solved": None, "status": "frozen_stock_unavailable"}
    if report.get("already_in_stock"):
        return {"stock_solved": False, "already_in_stock_verified": report.get("target") in stock,
            "status": "zero_step_not_generated_route"}
    by_product = defaultdict(list)
    for edge in report.get("complete_reaction_edges", []):
        by_product[edge["product"]].append(edge)
    unresolved = set()
    def solve(molecule, active):
        if molecule in stock:
            return [], {molecule}
        if molecule in active:
            return None
        for edge in sorted(by_product[molecule], key=lambda e: e["id"]):
            if not edge.get("precursors"):
                continue
            chosen, leaves = [edge["id"]], set()
            for precursor in edge["precursors"]:
                result = solve(precursor, active | {molecule})
                if result is None:
                    break
                chosen.extend(result[0]); leaves.update(result[1])
            else:
                return chosen, leaves
        unresolved.add(molecule)
        return None
    witness = solve(report.get("target"), set())
    claimed_leaves = set(report.get("stock_leaves", []))
    leaf_check = route_closure(list(claimed_leaves), stock, reaction_count=len(report.get("complete_reaction_edges", [])))
    valid = bool(witness and witness[0] and leaf_check["stock_solved"] and witness[1] == claimed_leaves)
    return {"stock_solved": valid, "status": "frozen_stock_AND_witness_rechecked",
        "selected_edge_ids": sorted(set(witness[0])) if witness else [],
        "verified_leaves": sorted(witness[1]) if witness else [], "reported_leaves_match_witness": bool(witness and witness[1] == claimed_leaves),
        "unresolved_precursors": sorted(x for x in unresolved if x is not None)}
