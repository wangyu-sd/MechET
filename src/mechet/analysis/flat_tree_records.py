"""Adapter for the historical parent-linked local electronic beam tree.

This does not convert an endpoint list into an invented tree. Rejected rows
are proposals, even when the serializer repeats their unchanged parent state.
"""
from __future__ import annotations

from .tree_records import digest, normalize_native_tree


def normalize_single_tree(data, *, report, manifest, source, stock=None):
    """The same native contexts/edges at top level, without an outer planner."""
    if not all(key in data for key in ["root_context", "contexts", "edges", "target"]):
        raise ValueError("unsupported single-reaction tree schema")
    native = {**data, "reaction_trees": {data["target"]: data}}
    value = normalize_native_tree(native, report=report, manifest=manifest, source=source, stock=stock)
    for edge in value["attempted_edges"]:
        # The wrapper is an adapter view, never a path in the original file.
        edge["source_pointer"] = "/edges/" + edge["source_pointer"].rsplit("/edges/", 1)[1]
    value["tree"]["adapter_note"] = "Native single-reaction contexts and edges wrapped without adding states, links or outer routes"
    return value


def normalize_flat_tree(data, *, source):
    from mechet.proof_program import _canonical_unmapped
    nodes = data["nodes"]
    roots = [n for n in nodes if n.get("parent") is None]
    if len(roots) != 1:
        raise ValueError("flat tree needs one recorded root")
    metadata = data["manifest"]
    target = _canonical_unmapped(roots[0]["state"])
    contexts, edges, chemicals, history = [], [], {}, {}
    indices = {}
    for index, node in enumerate(nodes):
        key = node["id"]
        if key in indices:
            raise ValueError("duplicate flat record ID")
        indices[key] = index
        rejected = node.get("status") == "rejected"
        if not rejected:
            prior = history.get(node.get("parent"), [])
            action = node.get("action")
            actions = prior + [{**action, "state_before": node.get("state_before")}] if action else []
            history[key] = actions
            chemical_id = digest(node["state"])
            chemicals[chemical_id] = {"id": chemical_id, "state_unmapped": _canonical_unmapped(node["state"])}
            contexts.append({"id": key, "state": node["state"], "chemical_id": chemical_id,
                "depth": node.get("depth"), "actions": actions, "terminal": node.get("status") == "terminal"})
        if node.get("parent") is not None:
            edges.append({"id": key, "parent_context": node["parent"],
                "child_context": None if rejected else key,
                "status": "rejected" if rejected else "terminal" if node.get("status") == "terminal" else "accepted",
                "error": node.get("error"), "action": node.get("action") or {"name": "unrecorded", "arguments": {}}})
    native = {"case_id": metadata["case_id"], "target": target, "reaction_trees": {target: {
        "root_context": roots[0]["id"], "contexts": contexts, "chemical_nodes": list(chemicals.values()),
        "edges": edges, "terminal_contexts": [c["id"] for c in contexts if c["terminal"]]}}}
    manifest = {"adapter_sha": metadata.get("adapter_model_sha256"),
        "source_protocol": "historical_single_case_not_full_benchmark", "search_algorithm": "recorded_local_beam"}
    report = {"id": metadata["case_id"], "wall_seconds": data.get("seconds")}
    value = normalize_native_tree(native, report=report, manifest=manifest, source=source)
    for edge in value["attempted_edges"]:
        index = indices[edge["original_edge_id"]]
        edge["source_pointer"] = f"/nodes/{index}/action"
        edge["component_moves"] = nodes[index].get("moves")
        edge["component_move_count"] = len(nodes[index].get("moves") or [])
    value["tree"]["adapter_note"] = "Recorded parent IDs retained; rejected proposal rows excluded from committed nodes; accepted ancestry supplies history only where recorded"
    value["tree"]["raw_row_count"] = len(nodes)
    return value
