"""Versioned logical schema for the lossless, typed JSONL analysis tables."""
from __future__ import annotations

COMMON = {"tree_id": ["string"], "target_id": ["string"]}
TABLE_SCHEMAS = {
    "trees": {"primary_key": ["tree_id"], "unit": "source-scoped target graph snapshot", "properties": {**COMMON, "run_id": ["string"], "has_complete_node_lineage": ["boolean"]}},
    "nodes": {"primary_key": ["node_id"], "unit": "native policy context, not globally unique molecule", "properties": {**COMMON, "node_id": ["string"], "parent_node_ids": ["array"], "private_state": ["string"], "state_unmapped": ["string", "null"], "depth": ["integer", "null"], "visits": ["integer", "number", "null"], "terminal": ["boolean"]}},
    "attempted_edges": {"primary_key": ["attempt_id"], "unit": "one recorded action attempt; coupled FLOW remains one edge", "properties": {**COMMON, "attempt_id": ["string"], "parent_node_id": ["string", "null"], "child_node_id": ["string", "null"], "execution_accepted": ["boolean"], "tool_payload": ["object"], "component_moves": ["array", "null"], "before_state": ["string", "null"], "after_state": ["string", "null"]}},
    "terminals": {"primary_key": ["terminal_id"], "unit": "recorded complete reaction context, not unique precursor set", "properties": {**COMMON, "terminal_id": ["string"], "structural_precursor": ["string", "null"], "full_mixture": ["string", "null"], "mean_nll": ["integer", "number", "null"], "matches_reference": ["boolean", "null"]}},
    "routes": {"primary_key": ["tree_id"], "unit": "one reported selected AND-route witness per target snapshot", "properties": {**COMMON, "stock_solved_recorded": ["boolean"], "stock_leaves": ["array"], "reaction_edge_ids": ["array"], "stock_recheck": ["object"]}},
    "edge_chemistry": {"primary_key": ["attempt_id"], "unit": "distinct accepted FLOW context transition", "properties": {**COMMON, "attempt_id": ["string"], "delta": ["object", "null"], "delta_status": ["string"]}},
    "programs": {"primary_key": ["terminal_id"], "unit": "complete terminal context with named exposure scope", "properties": {**COMMON, "terminal_id": ["string"], "actor_exposure_authenticated": ["boolean"], "complete_signature": ["string", "null"]}},
    "planning_states": {"primary_key": ["tree_id", "planning_id"], "unit": "per-target planning context including remaining AND obligations", "properties": {**COMMON, "planning_id": ["string"], "visits": ["integer", "number", "null"], "edges": ["array"]}}
}


def validate_row(table, row):
    schema = TABLE_SCHEMAS[table]
    for field in schema["primary_key"]:
        if field not in row:
            raise ValueError("missing primary-key field: " + field)
    for field, allowed in schema["properties"].items():
        if field not in row:
            continue
        value = row[field]
        kind = "null" if value is None else "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "string" if isinstance(value, str) else "array" if isinstance(value, list) else "object" if isinstance(value, dict) else "unsupported"
        if kind not in allowed:
            raise ValueError(f"{table}.{field}: {kind} not in declared JSON types {allowed}")


def logical_schema_manifest():
    return {"schema": "nmi_tree_tables_v1", "storage": "UTF-8 typed JSONL.gz; nested objects/arrays retained losslessly",
        "null_semantics": "null means unrecorded/not establishable, not zero or failure; absent optional keys differ from explicit null",
        "tables": TABLE_SCHEMAS, "additional_recorded_fields": "retained; this logical contract does not strip fields",
        "foreign_keys": {"nodes.tree_id": "trees.tree_id", "attempted_edges.parent_node_id": "nodes.node_id when available",
            "attempted_edges.child_node_id": "nodes.node_id or null for rejected/unrecorded successor", "terminals.terminal_id": "nodes.node_id",
            "planning_states.child_planning_id": "may refer to a shared context absent from this per-case export; explicit presence flag retained"},
        "validation_scope": "declared logical contract and unit-tested adapters; not an independent replay or row-by-row chemical validation"}
