"""Source-ID preserving panel coordinates, separate from graphic layout."""
from collections import defaultdict


def target_reaction_program(report):
    reactions = [r for r in report.get("complete_reaction_edges", []) if r["product"] == report["target"]]
    if not reactions:
        raise ValueError("no recorded complete target reaction")
    return min(reactions, key=lambda r: r["id"])


def program_cells(reaction):
    cells = []
    for index, action in enumerate(reaction["program"]["actions"]):
        kind = action["name"]
        cell = {"reaction_id": reaction["id"], "action_index": index, "kind": kind}
        if kind == "import_fragments":
            cell["fragments"] = action["arguments"]["fragments"]
        elif kind == "apply_electron_flow":
            cell.update(state=action["state_before"], arguments=action["arguments"])
        elif kind == "finish_trace":
            cell["state"] = reaction["program"]["structural_terminal_mapped"]
        else:
            raise ValueError("unrecognized recorded reaction action")
        cells.append(cell)
    return cells


def ranking_gap_case(entries):
    """Frozen source-ID selection, not a chemical-error or baseline-rescue label."""
    eligible = [e for e in entries if e["summary"]["ranking"]["pool_recall"] and not e["summary"]["ranking"]["top1"]]
    if not eligible:
        return None
    row = min(eligible, key=lambda e: e["target_id"])
    ranked = row["summary"]["ranking"]
    by_key = {}
    for candidate in sorted(row["report"]["ranked_candidates"], key=lambda c: (float("inf") if c.get("mean_nll") is None else c["mean_nll"], c["structural_precursor"])):
        key = ".".join(sorted(candidate["structural_precursor"].split(".")))
        by_key.setdefault(key, candidate)
    return {"target_id": row["target_id"], "top1": by_key[ranked["ordered_precursors"][0]],
        "source_reference_match": by_key[ranked["ordered_precursors"][ranked["reference_rank"] - 1]],
        "reference_rank": ranked["reference_rank"], "chemical_invalidity_claim": False,
        "selection_rule": "lexicographically first recorded target with same-pool recall but NLL Top1 miss"}


def route_panel_data(entries, targets, *, run_id):
    target_by_id = {t["id"]: t for t in targets}
    if len(target_by_id) != len(targets):
        raise ValueError("duplicate frozen target IDs")
    observations = {}
    for e in entries:
        if e["run_id"] != run_id or not e.get("report"):
            continue
        target_id = e["target_id"]
        if target_id not in target_by_id:
            raise ValueError("observed target outside the frozen panel population")
        if target_id in observations and observations[target_id] != e["report"]:
            raise ValueError("conflicting same-run target observations")
        report = e["report"]
        if report.get("stock_solved") and (not report.get("complete_reaction_edges") or not report.get("stock_leaves")):
            raise ValueError("nontrivial stock solution requires reaction edges and leaves")
        observations[target_id] = e["report"]
    def group(subset):
        ids = [t["id"] for t in subset]
        observed = [i for i in ids if i in observations]
        zero = [i for i in observed if observations[i].get("already_in_stock")]
        solved = [i for i in observed if observations[i].get("stock_solved") and i not in zero]
        return {"denominator": len(ids), "target_ids": ids, "observed_target_ids": observed,
            "solved_target_ids": solved, "observed_nonzero_stock_connected": len(solved),
            "already_stock": len(zero), "pending": len(ids) - len(observed),
            "lower_bound_rate": len(solved) / len(ids) if ids else None,
            "full_result_complete": len(observed) == len(ids)}
    cohorts = defaultdict(list)
    depths = defaultdict(list)
    for t in targets:
        cohorts[t["cohort"]].append(t)
        if t.get("reference_depth") is not None:
            depths[str(t["reference_depth"])].append(t)
    return {"run_id": run_id, "metric": "observed_nonzero_stock_closure_lower_bound",
        "cohorts": {c: group(rows) for c, rows in sorted(cohorts.items())},
        "depth": {c: group(rows) for c, rows in sorted(depths.items())},
        "all": group(targets), "chemistry_validity_claim": False,
        "matched_compute_claim": False}
