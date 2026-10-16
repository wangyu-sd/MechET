#!/usr/bin/env python3
"""Assemble a verified existing-artifact catalog and explicit availability ledger."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.tree_records import canonical_entries, file_digest
from mechet.analysis.flat_tree_records import normalize_flat_tree
from mechet.analysis.tree_metrics import morphology
from mechet.analysis.chemical_evidence import validate_evidence
from scripts.analyze_nmi_tree_evidence import _write_json, _dump_line, _deterministic_gzip_text


def report_pointer(entry, local_pointer):
    index = entry.get("report_index")
    return (f"/rows/{index}" if index is not None else "") + local_pointer


def verify_delivery_artifacts(root):
    root = Path(root)
    names = ["figure_manifest.json", "figure_qa.json", "enrichment_manifest.json", "chemistry_figure_manifest.json",
        "work_manifest.json", "captions.md", "case_registry.csv", "schema_manifest.json"]
    for name in names:
        if not (root / name).is_file():
            raise ValueError("missing delivery artifact: " + name)
    inventory_sha = file_digest(root / "input_inventory.json")
    main = json.loads((root / names[0]).read_text())
    qa = json.loads((root / names[1]).read_text())
    enrichment = json.loads((root / names[2]).read_text())
    chemistry = json.loads((root / names[3]).read_text())
    work = json.loads((root / names[4]).read_text())
    if qa["figure_manifest_sha256"] != file_digest(root / "figure_manifest.json"):
        raise ValueError("stale figure QA manifest binding")
    for manifest in [main, enrichment, work]:
        if manifest.get("input_inventory_sha256") != inventory_sha:
            raise ValueError("stale delivery inventory binding")
    if chemistry.get("enrichment_manifest_sha256") != file_digest(root / "enrichment_manifest.json"):
        raise ValueError("stale chemistry enrichment binding")
    for record in enrichment["tables"].values():
        if file_digest(root / record["path"]) != record["sha256"]:
            raise ValueError("enrichment table hash differs")
    for manifest, checks in [(main, qa["figures"]), (chemistry, chemistry["qa"]), (work, work["qa"])]:
        if not manifest["figures"] or set(manifest["figures"]) != set(checks):
            raise ValueError("missing figure QA records")
        for name, files in manifest["figures"].items():
            if not checks[name]["machine_checks_passed"]:
                raise ValueError("failed figure QA: " + name)
            for record in files.values():
                if not Path(record["path"]).is_file() or file_digest(record["path"]) != record["sha256"]:
                    raise ValueError("figure hash differs: " + name)
    return {name: file_digest(root / name) for name in names}


def verified_analysis(root):
    root = Path(root)
    if json.loads((root / "progress.json").read_text())["phase"] != "complete":
        raise ValueError("analysis has not completed")
    source = json.loads((root / "source_manifest.json").read_text())
    if source["input_inventory_sha256"] != file_digest(root / "input_inventory.json"):
        raise ValueError("source inventory hash differs")
    for name, table in source["tables"].items():
        if file_digest(root / table["path"]) != table["sha256"]:
            raise ValueError("table hash differs: " + name)
    return json.loads((root / "input_inventory.json").read_text())


def coverage_catalog(base, supplemental=None):
    def key(entry):
        return entry["run_id"], entry["target_id"], entry.get("tree_sha256")
    supported = {key(e) for inventory in [base, supplemental or {"inputs": []}]
        for e in inventory["inputs"] if e["analysis_status"] == "normalized"}
    handled = sum(key(e) in supported for e in base["inputs"])
    return {"original_input_records": base["input_count"], "accounted_for_records": handled,
        "canonical_run_target_graphs": len(supported), "unhandled_original_records": base["input_count"] - handled,
        "unhandled_statuses": dict(Counter(e.get("error", e["analysis_status"]) for e in base["inputs"] if key(e) not in supported)),
        "input_roots": base.get("input_roots"), "cross_run_metric_pooling": False,
        "storage_replicas_are_not_independent_targets": True}


def finalize(root, *, supplemental_root=None, flat_paths=(), evidence_path=None):
    root = Path(root)
    base = verified_analysis(root)
    extra = verified_analysis(supplemental_root) if supplemental_root else None
    coverage = coverage_catalog(base, extra)
    flat = []
    for path in flat_paths:
        path = Path(path)
        source = {"run_id": path.parent.name, "tree_sha256": file_digest(path), "tree_path": str(path.resolve())}
        value = normalize_flat_tree(json.loads(path.read_bytes()), source=source)
        value["summary"] = {"morphology": morphology(value)}
        flat.append(value)
    flat_root = root / "flat_tables"
    flat_root.mkdir(exist_ok=True)
    flat_tables = {}
    for table, field in [("trees", "tree"), ("nodes", "nodes"), ("attempted_edges", "attempted_edges"), ("terminals", "terminals")]:
        path = flat_root / (table + ".jsonl.gz")
        count = 0
        with _deterministic_gzip_text(path) as f:
            for value in flat:
                for row in [value[field]] if field == "tree" else value[field]:
                    _dump_line(f, row)
                    count += 1
        flat_tables[table] = {"path": str(path), "sha256": file_digest(path), "rows": count}
    coverage.update(historical_parent_linked_cases=len(flat), flat_tables=flat_tables,
        shared_cache_WAL="preserved source of shared transitions/planning statistics, not independent target trees; exported case snapshots analyzed",
        original_files_modified=False, new_sampling=False)
    _write_json(root / "artifact_coverage.json", coverage)
    entries = canonical_entries(base["inputs"] + (extra["inputs"] if extra else []))
    evidence = []
    if evidence_path:
        from mechet.proof_program import _canonical_unmapped
        def canonical(state):
            return ".".join(sorted(_canonical_unmapped(state).split(".")))
        registry = json.loads(Path(evidence_path).read_text())
        for record in registry["records"]:
            matches = [e for e in entries if e["run_id"] == record["run_id"] and e["target_id"] == record["target_id"]]
            if len(matches) != 1:
                raise ValueError("literature case must match exactly one actual tree")
            entry = matches[0]
            reactions = [(i, r) for i, r in enumerate(entry["report"].get("complete_reaction_edges", []))
                if canonical(r["product"]) == canonical(record["product"])
                and canonical(".".join(r["precursors"])) == canonical(".".join(record["structural_precursors"]))]
            if len(reactions) != 1:
                raise ValueError("literature precursor/product pair absent or nonunique in the recorded complete route")
            index, reaction = reactions[0]
            verified = validate_evidence(record, complete_reaction=True)
            evidence.append({**verified, "tree_id": entry["tree_id"], "tree_sha256": entry["tree_sha256"],
                "reaction_id": reaction["id"], "terminal_context": reaction["program"]["terminal_context"],
                "report_path": entry["report_path"], "report_sha256": entry.get("report_sha256"),
                "source_pointer": report_pointer(entry, f"/complete_reaction_edges/{index}"), "generated_pair_binding": "RDKit_stereo_canonical_exact",
                "whole_route_verified": False, "conditions_verified": False, "physical_mechanism_verified": False})
    with (root / "chemical_evidence.jsonl").open("w") as f:
        for row in evidence:
            _dump_line(f, row)
    availability = {"schema": "nmi_tree_evidence_availability_v1", "new_sampling": False,
        "observed_tree_analysis": "available; original and single-reaction adapters linked by source hashes",
        "FlowER_28971_endpoint_benchmark": "PENDING in this evidence collection; no extrapolation from route trees",
        "mech31k_3120_corrected_endpoint_benchmark": "PENDING; current saved MCTS endpoint scores use rxn_prod_min proxy diagnostic",
        "R1_route_cohorts": "460 frozen IDs retained; observed completed trees give a nonzero stock-AND closure lower bound",
        "equal_compute_route_comparison": "PENDING; current greedy and MCTS use unequal work and shared reuse",
        "paired_target_bootstrap_for_equal_compute_gain": "not_establishable_without_matched_work_outputs",
        "H2_27104_predictive_comparison": "PENDING; full-data 0.6B actor is not an authenticated H2-trained actor",
        "RetroCast_C0_C1_C2": "Unknown for all current 160 reference routes; depth is descriptive, not a composition label",
        "historical_signature_basis": "strict train 257167; local604/ordered6553; imports omitted and within-event order can matter",
        "current_actor_signature_basis": "actual 0.6B full-data exposure local603/ordered6491; nonordinary1377 kept separate",
        "independent_forward_round_trip": "not_available_in_this_saved_tree_collection",
        "literature_reviewed_complete_reaction_records": len(evidence),
        "literature_verified_structural_pairs": sum(r["verified_support"] for r in evidence),
        "literature_sampling_unit": "fixed illustrated complete reactions only; no population-wide support percentage",
        "training_source_overlap": "unknown unless a source audit is supplied; public patent support is not proof of holdout novelty",
        "runtime_decomposition": "only source-recorded fields; no inferred GPU utilization or missing timing decomposition",
        "remaining_editorial_decision": "six figures plus Table1 exceed six combined display items; not silently edited",
        "paper_main_modified": False, "historical_16_work_packages": "retained; no original work package is canceled or replaced by observed figures"}
    _write_json(root / "analysis_availability.json", availability)
    from mechet.analysis.table_schema import logical_schema_manifest
    _write_json(root / "schema_manifest.json", logical_schema_manifest())
    artifact_hashes = verify_delivery_artifacts(root)
    manifest = {"schema": "nmi_existing_tree_delivery_v1", "analysis_root": str(root),
        "original_source_manifest_sha256": file_digest(root / "source_manifest.json"),
        "supplemental_root": str(supplemental_root) if supplemental_root else None,
        "supplemental_source_manifest_sha256": file_digest(Path(supplemental_root) / "source_manifest.json") if supplemental_root else None,
        "flat_original_sources": [v["tree"]["source"] for v in flat],
        "coverage": coverage, "availability": availability,
        "evidence_registry_sha256": file_digest(evidence_path) if evidence_path else None,
        "chemical_evidence_sha256": file_digest(root / "chemical_evidence.jsonl"),
        "producer_sha256": file_digest(__file__), "unhandled_artifacts_are_pending_not_zero_results": True}
    manifest["verified_delivery_artifacts"] = artifact_hashes
    manifest["all_required_machine_checks_passed"] = True
    _write_json(root / "delivery_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--single-analysis-root", type=Path)
    parser.add_argument("--flat-tree", type=Path, action="append", default=[])
    parser.add_argument("--evidence-registry", type=Path)
    args = parser.parse_args()
    result = finalize(args.analysis_root, supplemental_root=args.single_analysis_root, flat_paths=args.flat_tree, evidence_path=args.evidence_registry)
    print(json.dumps({"phase": "delivery-catalog", "coverage": result["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
