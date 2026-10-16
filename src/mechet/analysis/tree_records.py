"""Adapters for recorded native electronic trees; never infer missing topology.

Node identity is a native *policy context*, not a canonical molecule. Chemical
projections are explicitly separate. All normalized IDs are source scoped.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def discover_inputs(roots):
    """Catalog report/tree pairs, including recorded cross-run resume references.

    Complete.json shards and per-case evaluations are alternative storage
    adapters. Unreferenced trees are catalogued separately, not discarded.
    Shared cache/WAL files are not mislabeled independent target observations.
    """
    records = []
    seen_reports = set()
    referenced_trees = set()
    resolved_roots = sorted({Path(p).resolve() for p in roots})
    basenames = Counter(p.name for p in resolved_roots)
    for root in resolved_roots:
        root_label = root.name if basenames[root.name] == 1 else root.name + "@" + digest(str(root))[:10]
        if not root.is_dir():
            raise FileNotFoundError(root)
        def absolute(path):
            # Do not perform repeated Ceph lstat/readlink walks for each row.
            # Preserve a normalized lexical source path; hashes authenticate the
            # opened contents even when a recorded artifact uses a symlink.
            return Path(os.path.abspath(path))
        def read_json(path):
            return json.loads(path.read_bytes())
        pairs = []
        evaluation_paths = sorted(root.glob("*/rank*/evaluations/*.json"))
        with ThreadPoolExecutor(max_workers=8) as pool:
            for p, data in zip(evaluation_paths, pool.map(read_json, evaluation_paths)):
                pairs.append((p, None, data, p.parent.parent))
        complete_manifests = {}
        for p in sorted(root.glob("*/rank*/complete.json")):
            data = read_json(p)
            complete_manifests[str(p)] = data.get("manifest", {})
            pairs.extend((p, i, row, p.parent) for i, row in enumerate(data.get("rows", [])))
        manifest_paths = sorted({shard / "manifest.json" for _, _, _, shard in pairs})
        manifests = {str(p): read_json(p) if p.exists() else {} for p in manifest_paths}
        manifest_presence = {str(p): p.exists() for p in manifest_paths}
        def make_record(pair):
            p, index, report, shard = pair
            artifact = report.get("tree_artifact")
            tree = absolute(shard / artifact) if artifact else None
            manifest_path = shard / "manifest.json"
            manifest = manifests[str(manifest_path)]
            if p.name == "complete.json":
                manifest = complete_manifests[str(p)]
            condition = shard.parent.name
            run_id = root_label if condition == "tree" else root_label + ":" + condition
            return {"run_id": run_id, "run_source_path": str(root), "target_id": report.get("id"),
                "report_path": str(absolute(p)), "report_index": index,
                "tree_path": str(tree) if tree else None,
                "tree_exists": tree.is_file() if tree else False,
                "manifest_path": str(manifest_path) if manifest_presence[str(manifest_path)] else None,
                "manifest": manifest, "report": report,
                "storage_adapter": "complete_shard" if index is not None else "per_target_evaluation"}
        with ThreadPoolExecutor(max_workers=8) as pool:
            for record in pool.map(make_record, pairs):
                key = record["report_path"], record["report_index"]
                if key in seen_reports:
                    continue
                seen_reports.add(key)
                if record["tree_path"]:
                    referenced_trees.add(record["tree_path"])
                records.append(record)
        for p in sorted(root.glob("*/rank*/trees/*.json")):
            if str(absolute(p)) in referenced_trees:
                continue
            data = json.loads(p.read_bytes())
            condition = p.parent.parent.parent.name
            records.append({"run_id": root_label if condition == "tree" else root_label + ":" + condition, "run_source_path": str(root), "target_id": data.get("case_id"),
                "report_path": None, "report_index": None, "tree_path": str(absolute(p)),
                "tree_exists": True, "manifest_path": None, "manifest": {}, "report": {},
                "storage_adapter": "unreferenced_tree", "tree_status": "report_unavailable"})
    return records


def canonical_entries(entries, *, run_id=None):
    """One authoritative report per (source-scoped run, target), not per file."""
    preference = {"per_target_evaluation": 2, "complete_shard": 1, "unreferenced_tree": 0}
    chosen = {}
    for e in entries:
        if (run_id is not None and e["run_id"] != run_id) or not e.get("summary"):
            continue
        key = e["run_id"], e["target_id"]
        old = chosen.get(key)
        if old is None or preference.get(e["storage_adapter"], 0) > preference.get(old["storage_adapter"], 0):
            chosen[key] = e
    return [chosen[k] for k in sorted(chosen)]


def _imported(actions):
    counts = {}
    for a in actions:
        if a.get("name") != "import_fragments":
            continue
        for f in a.get("arguments", {}).get("fragments", []):
            key = (f.get("smiles"), f.get("purpose"))
            counts[key] = counts.get(key, 0) + int(f.get("count", 1))
    return [{"smiles": k[0], "purpose": k[1], "count": n} for k, n in sorted(counts.items(), key=str)]


def normalize_native_tree(data, *, report, manifest, source, stock=None):
    if report.get("id") and data.get("case_id") != report["id"]:
        raise ValueError("report/tree target identity differs")
    if not isinstance(data.get("reaction_trees"), dict):
        raise ValueError("unsupported tree schema: recorded reaction_trees required")
    tree_id = digest([source["run_id"], data["case_id"], source["tree_sha256"]])
    nodes, edges, terminals, routes = [], [], [], []
    lineage_complete = True
    missing_references = []
    ranked = {c["context"]: (i + 1, c) for i, c in enumerate(report.get("ranked_candidates", []))}
    committed = {e["program"]["terminal_context"]: e for e in data.get("reaction_edges", []) if e.get("program", {}).get("terminal_context")}
    for inner_index, (product, inner) in enumerate(data["reaction_trees"].items()):
        scope = f"{tree_id}:reaction{inner_index}"
        contexts = {c["id"]: c for c in inner.get("contexts", [])}
        if len(contexts) != len(inner.get("contexts", [])):
            raise ValueError("duplicate native context ID within a reaction")
        chem = {c["id"]: c for c in inner.get("chemical_nodes", [])}
        parents = {k: [] for k in contexts}
        for edge_index, e in enumerate(inner.get("edges", [])):
            parent, child = e["parent_context"], e.get("child_context")
            rejected = e["status"] == "rejected"
            if rejected and child:
                raise ValueError("rejected action records a fictional child")
            for ref in (parent, child):
                if ref is not None and ref not in contexts:
                    lineage_complete = False
                    missing_references.append({"reaction": product, "edge": e["id"], "context": ref})
            if child in parents:
                parents[child].append(f"{scope}:{parent}")
            action = e.get("action", {})
            args = action.get("arguments", {})
            before = contexts.get(parent, {}).get("state")
            after = contexts.get(child, {}).get("state")
            edges.append({"attempt_id": f"{scope}:{e['id']}", "tree_id": tree_id,
                "target_id": data["case_id"], "reaction_product": product,
                "original_edge_id": e["id"], "parent_node_id": f"{scope}:{parent}" if parent in contexts else None,
                "child_node_id": f"{scope}:{child}" if child in contexts and not rejected else None,
                "recorded_child_context": child, "action_family": action.get("name", ""),
                "tool_payload": {"name": action.get("name"), "arguments": args},
                "component_moves": args.get("electron_flow", []),
                "component_move_count": len(args.get("electron_flow", [])),
                "be_delta": {k: args.get(k, []) for k in ("bond_order_changes", "charge_changes")},
                "status": e["status"], "execution_accepted": not rejected,
                "error": e.get("error"), "before_state": before, "after_state": after,
                "tokens": action.get("tokens"), "logprob": action.get("logprob"),
                "executor_seconds": e.get("executor_seconds"),
                "source_pointer": "/reaction_trees/" + product.replace("~", "~0").replace("/", "~1") + f"/edges/{edge_index}"})
        for context, c in contexts.items():
            actions = c.get("actions", [])
            mapped = c.get("state")
            node_id = f"{scope}:{context}"
            nodes.append({"node_id": node_id, "tree_id": tree_id, "target_id": data["case_id"],
                "native_context_id": context, "native_reaction_id": scope,
                "reaction_product": product, "is_root": context == inner.get("root_context"),
                "parent_node_ids": sorted(set(parents[context])), "private_state": mapped,
                "state_unmapped": chem.get(c.get("chemical_id"), {}).get("state_unmapped"),
                "chemical_projection_id": c.get("chemical_id"),
                "depth": c.get("depth"), "flow_depth": sum(a.get("name") == "apply_electron_flow" for a in actions),
                "history_sha256": digest(actions), "imported_fragments": _imported(actions),
                "atom_maps": sorted({int(x) for x in re.findall(r":(\d+)\]", mapped or "")}),
                "terminal": bool(c.get("terminal")), "mean_nll": c.get("mean_nll"),
                "resource_closed": c.get("resource_closed"), "sampling_draws": c.get("sampling_draws"),
                "visits": None, "visits_status": "planning-context statistics stored separately, not assignable by chemical projection",
                "field_presence": {k: k in c for k in ("state", "depth", "mean_nll", "sampling_draws", "resource_closed")}})
            if c.get("terminal"):
                rank, candidate = ranked.get(context, (None, {}))
                program = committed.get(context, {}).get("program", {})
                full = candidate.get("full_mixture", mapped)
                structural_mapped = candidate.get("structural_precursor_mapped", program.get("structural_terminal_mapped"))
                structural = candidate.get("structural_precursor")
                auxiliary = None
                if full and inner.get("root_context") in contexts:
                    # Reuse the executor's existing target-atom ownership rule.
                    # This is a display/evaluation projection, not a new state.
                    from mechet.endpoints import split_precursor_endpoints
                    from mechet.proof_program import _canonical_unmapped
                    parts = split_precursor_endpoints(full, contexts[inner["root_context"]]["state"])
                    structural_mapped = structural_mapped or parts.structural
                    auxiliary = list(parts.auxiliary)
                    structural = structural or ".".join(sorted(_canonical_unmapped(x) for x in parts.structural.split(".") if x))
                terminals.append({"terminal_id": node_id, "tree_id": tree_id,
                    "target_id": data["case_id"], "reaction_product": product,
                    "native_context_id": context, "full_mixture": full,
                    "structural_precursor": structural,
                    "structural_precursor_mapped": structural_mapped,
                    "auxiliary_fragments_mapped": auxiliary,
                    "ranking": rank, "mean_nll": candidate.get("mean_nll", c.get("mean_nll")),
                    "matches_reference": candidate.get("matches_reference"),
                    "reported_finish_accepted": True, "independent_replay": "not_performed",
                    "chemical_evidence_status": "unresolved"})
        # Resolved edge references alone do not prove the recorded topology is
        # complete: also detect disconnected contexts without inventing links.
        reached = set()
        frontier = [inner.get("root_context")]
        adjacency = {}
        for e in inner.get("edges", []):
            if e.get("child_context") and e["status"] != "rejected":
                adjacency.setdefault(e["parent_context"], []).append(e["child_context"])
        while frontier:
            key = frontier.pop()
            if key in reached:
                continue
            reached.add(key)
            frontier.extend(adjacency.get(key, []))
        for context in sorted(contexts.keys() - reached):
            lineage_complete = False
            missing_references.append({"reaction": product, "context": context, "reason": "unreachable_from_recorded_root"})
    if "stock_solved" in report:
        from .tree_metrics import verify_reported_route
        recheck = verify_reported_route(report, stock)
        if report["stock_solved"] and stock is not None and recheck["stock_solved"] is not True:
            raise ValueError("reported stock closure fails frozen AND-route recheck")
        routes.append({"tree_id": tree_id, "target_id": data["case_id"],
            "route_key": report.get("route_key"), "stock_solved_recorded": report["stock_solved"],
            "already_in_stock": report.get("already_in_stock"), "stock_leaves": report.get("stock_leaves", []),
            "reaction_edge_ids": [e["id"] for e in report.get("complete_reaction_edges", [])],
            "unique_route_count_lower_bound": report.get("unique_route_count_lower_bound"),
            "route_count_exhaustive": report.get("route_enumeration_exhaustive"),
            "stock_recheck": recheck,
            "stock_membership_verification": recheck["status"]})
    tree = {"tree_id": tree_id, "target_id": data["case_id"], "target": data.get("target"),
        "run_id": source["run_id"], "source": source,
        "lineage_kind": "recorded_full_tree" if lineage_complete else "recorded_partial_tree",
        "has_complete_node_lineage": lineage_complete, "missing_context_references": missing_references,
        "complete_report_available": bool(report), "search_stop_reason": data.get("stop_reason"),
        "benchmark_denominator": manifest.get("full_denominator"), "checkpoint_sha": manifest.get("adapter_sha"),
        "source_protocol": manifest.get("source_protocol"), "algorithm": manifest.get("search_algorithm"),
        "has_gold_endpoint": "expected_structural_precursor" in report,
        "has_complete_route": bool(report.get("stock_solved")) and bool(report.get("complete_reaction_edges")),
        "work": report.get("work", data.get("work", {})), "wall_seconds": report.get("wall_seconds"),
        "mcts_recorded": data.get("mcts", {}), "planning_state_count": len(data.get("mcts_graph", {})),
        "cohort": report.get("cohort"), "chemical_validation": "executor_acceptance_not_experimental_truth"}
    return {"tree": tree, "nodes": nodes, "attempted_edges": edges, "terminals": terminals, "routes": routes}


def load_frozen_stocks(manifest_path):
    """Read exact frozen stock identities once, rehashing their source files."""
    assets = json.loads(Path(manifest_path).read_text())
    stocks = {}
    hashes = {}
    for name, info in assets["stocks"].items():
        path = Path(info["canonical_path"])
        actual = file_digest(path)
        if actual != info["canonical_sha256"]:
            raise ValueError("frozen stock hash mismatch: " + name)
        stocks[name] = set(path.read_text().splitlines())
        hashes[name] = {"path": str(path), "sha256": actual, "rows": len(stocks[name])}
    by_cohort = {name: stocks[info["stock"]] for name, info in assets["cohorts"].items()}
    return by_cohort, {"manifest_path": str(Path(manifest_path).resolve()), "manifest_sha256": file_digest(manifest_path), "stocks": hashes,
        "cohort_stock_sha256": {name: assets["stocks"][info["stock"]]["canonical_sha256"] for name, info in assets["cohorts"].items()}}


def assert_stock_identity(manifest, cohort, stock_sources):
    expected = manifest.get("stock_canonical_sha_by_cohort", {}).get(cohort)
    if expected and stock_sources.get("cohort_stock_sha256", {}).get(cohort) != expected:
        raise ValueError("frozen stock identity differs from saved search contract")
