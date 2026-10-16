#!/usr/bin/env python3
"""Read-only chemistry/visit enrichment of normalized existing-tree records."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.tree_records import canonical_entries, digest, file_digest
from mechet.analysis.tree_chemistry import molecular_delta, program_support
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.analyze_nmi_tree_evidence import _deterministic_gzip_text, _write_json, _dump_line


def enrich_tree(raw, normalized, exposure=None):
    tree = normalized["tree"]
    identity = {k: tree[k] for k in ["tree_id", "target_id", "run_id"]}
    chemistry, programs, planning = [], [], []
    unique = set()
    for edge in normalized["attempted_edges"]:
        if edge["action_family"] != "apply_electron_flow" or not edge["execution_accepted"] or not edge["child_node_id"]:
            continue
        key = edge["parent_node_id"], edge["child_node_id"]
        if key in unique:
            continue
        unique.add(key)
        record = {**identity, **{k: edge[k] for k in ["attempt_id", "parent_node_id", "child_node_id", "source_pointer"]}}
        try:
            record["delta"] = molecular_delta(edge["before_state"], edge["after_state"])
            record["delta_status"] = "available"
        except (ValueError, RuntimeError) as exc:
            record.update(delta=None, delta_status="not_establishable", delta_error=str(exc))
        try:
            moves = compile_event_arguments(edge["before_state"], edge["tool_payload"]["arguments"])
            record["ordinary_source_sink"] = not any(m.get("mode") == "BE_DELTA" for m in moves)
            record["source_sink_types"] = [m["source"]["kind"] + "→" + m["sink"]["kind"] for m in moves] if record["ordinary_source_sink"] else None
            record["recorded_component_move_count"] = len(moves) if record["ordinary_source_sink"] else None
            record["address_resolution_status"] = "available"
        except (ValueError, KeyError, TypeError) as exc:
            record.update(source_sink_types=None, address_resolution_status="not_establishable", address_error=str(exc))
        chemistry.append(record)
    node_lookup = defaultdict(list)
    for node in normalized["nodes"]:
        node_lookup[node["native_context_id"]].append(node["node_id"])
    authenticated = bool(exposure and exposure.get("model_sha256") and exposure["model_sha256"] == tree.get("checkpoint_sha"))
    for inner_index, (product, inner) in enumerate(raw.get("reaction_trees", {}).items()):
        for index, context in enumerate(inner.get("contexts", [])):
            if not context.get("terminal"):
                continue
            record = {**identity, "native_context_id": context["id"],
                "terminal_id": f'{tree["tree_id"]}:reaction{inner_index}:{context["id"]}',
                "source_pointer": "/reaction_trees/" + product.replace("~", "~0").replace("/", "~1") + f"/contexts/{index}",
                "actor_exposure_authenticated": authenticated, "route_C1_claim": False}
            if exposure:
                try:
                    record.update(program_support(context.get("actions", []), exposure["actual_train_exposure"]))
                    record["support_status"] = "actual_actor_exposure" if authenticated else "reference_corpus_only_not_actor_exposure"
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    record.update(support_status="not_establishable", support_error=str(exc), complete_signature=None)
            else:
                record.update(support_status="exposure_not_supplied", complete_signature=None)
            programs.append(record)
    graph = raw.get("mcts_graph", {})
    for state_id, snapshot in sorted(graph.items()):
        state = snapshot.get("state", {})
        edge_records = []
        for edge_id, edge in sorted(snapshot.get("edges", {}).items()):
            proposal = edge.get("proposal", {})
            child = edge.get("child")
            edge_records.append({"edge_id": edge_id, "child_planning_id": child,
                "child_recorded_in_this_snapshot": child in graph if child else False,
                "visits": edge.get("n"), "value_sum": edge.get("w"),
                "sample_count": edge.get("count"), "error": edge.get("error"),
                "proposal": {"name": proposal.get("name"), "arguments": proposal.get("arguments")}})
        planning.append({**identity, "planning_id": state_id,
            "native_node_ids": sorted(node_lookup[state["active"]]) if state.get("active") else [],
            "context_sha256": digest(state), "current_reaction_product": state.get("molecule"),
            "pending_AND_obligations": state.get("pending"), "ancestor_obligations": state.get("ancestors"),
            "solved_recorded": state.get("solved"), "visits": snapshot.get("visits"),
            "value_sum": snapshot.get("value_sum"), "selections": snapshot.get("selections"),
            "policy_batches": snapshot.get("policy_batches"), "edges": edge_records,
            "source_pointer": "/mcts_graph/" + state_id.replace("~", "~0").replace("/", "~1"),
            "counting_unit": "recorded planning-context snapshot; not globally unique across shared targets"})
    return {"edge_chemistry": chemistry, "programs": programs, "planning_states": planning}


def enrich(analysis_root, *, exposure_path=None, workers=8):
    root = Path(analysis_root)
    inventory_path = root / "input_inventory.json"
    inventory_bytes = inventory_path.read_bytes()
    entries = canonical_entries(json.loads(inventory_bytes)["inputs"])
    exposure = json.loads(Path(exposure_path).read_bytes()) if exposure_path else None
    exposure_sha = file_digest(exposure_path) if exposure_path else None
    producer_paths = [Path(__file__), ROOT / "src/mechet/analysis/tree_chemistry.py", ROOT / "src/mechet/natural_language_electron_flow.py", ROOT / "src/mechet/proof_to_trace.py"]
    producer = {str(p.relative_to(ROOT)): file_digest(p) for p in producer_paths}
    version = digest([producer, exposure_sha])
    cache_root = root / "chemistry_cache"
    cache_root.mkdir(exist_ok=True)
    def prepare(entry):
        path = cache_root / (digest([entry["tree_id"], version]) + ".json.gz")
        if not path.exists():
            raw_bytes = Path(entry["tree_path"]).read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != entry["tree_sha256"]:
                raise ValueError("source tree changed since the frozen inventory")
            with gzip.open(entry["cache_path"], "rt") as f:
                normalized = json.load(f)
            value = enrich_tree(json.loads(raw_bytes), normalized, exposure)
            temporary = path.with_name(path.name + ".tmp")
            with _deterministic_gzip_text(temporary) as f:
                json.dump(value, f, separators=(",", ":"), ensure_ascii=False)
            os.replace(temporary, path)
        return entry, path
    records = []
    _write_json(root / "enrichment_progress.json", {"phase": "enrich", "done": 0, "total": len(entries)})
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, pair in enumerate(pool.map(prepare, entries)):
            records.append(pair)
            if (i + 1) % 50 == 0:
                progress = {"phase": "enrich", "done": i + 1, "total": len(entries)}
                _write_json(root / "enrichment_progress.json", progress)
                print(json.dumps(progress), flush=True)
    tables = ["edge_chemistry", "programs", "planning_states"]
    counts = Counter()
    runs = defaultdict(lambda: Counter())
    with ExitStack() as stack:
        handles = {name: stack.enter_context(_deterministic_gzip_text(root / (name + ".jsonl.gz.tmp"))) for name in tables}
        for entry, path in records:
            with gzip.open(path, "rt") as f:
                value = json.load(f)
            stats = runs[entry["run_id"]]
            stats["targets"] += 1
            for name in tables:
                for row in value[name]:
                    _dump_line(handles[name], row)
                    counts[name] += 1
                    stats[name] += 1
                    if name == "edge_chemistry":
                        stats["delta_available"] += row["delta_status"] == "available"
                        if row["delta"]:
                            stats["changed_bonds"] += len(row["delta"]["changed_bonds"])
                            stats["ring_cycle_rank_changed_events"] += row["delta"]["cycle_rank_delta"] != 0
                    elif name == "programs":
                        eligible = row.get("actor_exposure_authenticated") and row.get("complete_signature")
                        stats["complete_ordinary_actor_programs"] += bool(eligible)
                        stats["training_unseen_generated_signature"] += bool(eligible and row.get("complete_signature_training_count") == 0)
    for name in tables:
        os.replace(root / (name + ".jsonl.gz.tmp"), root / (name + ".jsonl.gz"))
    if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != hashlib.sha256(inventory_bytes).hexdigest():
        raise ValueError("inventory changed while enriching; do not publish mixed snapshots")
    manifest = {"schema": "nmi_tree_chemistry_enrichment_v1", "producer_files": producer,
        "input_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(), "actor_exposure_manifest_sha256": exposure_sha,
        "actor_exposure_manifest_path": str(exposure_path) if exposure_path else None,
        "source_training_text_rehashed": False, "runs": {k: dict(v) for k, v in sorted(runs.items())},
        "tables": {name: {"rows": counts[name], "path": name + ".jsonl.gz", "sha256": file_digest(root / (name + ".jsonl.gz"))} for name in tables},
        "chemical_feasibility_claim": False, "route_C1_claim": False, "new_sampling": False}
    _write_json(root / "enrichment_manifest.json", manifest)
    _write_json(root / "enrichment_progress.json", {"phase": "complete", "done": len(entries), "total": len(entries)})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--exposure-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = enrich(args.analysis_root, exposure_path=args.exposure_manifest, workers=args.workers)
    print(json.dumps({"phase": "enrichment-complete", "tables": {k: v["rows"] for k, v in result["tables"].items()}}), flush=True)


if __name__ == "__main__":
    main()
