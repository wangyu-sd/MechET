#!/usr/bin/env python3
"""Normalize existing tree artifacts, without inference, training or replay.

JSONL.gz tables preserve explicit types and original row references. Per-input
hash-addressed caches make interrupted analysis resumable. Scientific summaries
are per run; successive resume replicas never form an inflated pooled cohort.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from mechet.analysis.tree_records import discover_inputs, normalize_native_tree, file_digest, digest, canonical_entries, load_frozen_stocks, assert_stock_identity
from mechet.analysis.tree_metrics import morphology, candidate_metrics


def _write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def _dump_line(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


@contextmanager
def _deterministic_gzip_text(path):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=2, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as text:
                yield text


def _aggregate(entries):
    groups = defaultdict(list)
    for entry in entries:
        if entry.get("summary"):
            groups[entry["run_id"]].append(entry)
    runs = {}
    for run, rows in sorted(groups.items()):
        # Prefer authoritative completed report to a standalone graph for the
        # same run/target; still retain all raw catalog entries and tables.
        chosen = canonical_entries(rows, run_id=run)
        denoms = {r["manifest"].get("full_denominator") for r in chosen} - {None}
        if len(denoms) > 1:
            raise ValueError(f"conflicting benchmark denominators in {run}")
        sums = Counter()
        errors = Counter()
        rank_counts = Counter()
        cohort = defaultdict(lambda: Counter())
        for r in chosen:
            m = r["summary"]["morphology"]
            for k in ["native_contexts", "attempted_actions", "expanded_contexts", "rejected_attempts", "duplicate_attempts", "unique_context_transitions", "flow_fork_contexts", "terminal_contexts"]:
                sums[k] += m[k]
            errors.update(m["error_counts"])
            rank = r["summary"]["ranking"]
            if rank["pool_recall"] is not None:
                rank_counts["evaluated_targets"] += 1
                rank_counts["top1_hits"] += rank["top1"]
                rank_counts["top5_hits"] += rank["top5"]
                rank_counts["top10_hits"] += rank["top10"]
                rank_counts["pool_hits"] += rank["pool_recall"]
                rank_counts["ranking_only_losses"] += rank["ranking_gap"]
            name = r["report"].get("cohort", "unassigned")
            cohort[name]["observed_targets"] += 1
            cohort[name]["stock_solved_recorded"] += bool(r["report"].get("stock_solved"))
            cohort[name]["already_in_stock"] += bool(r["report"].get("already_in_stock"))
        n = len(chosen)
        runs[run] = {"observed_targets": n, "benchmark_denominator": next(iter(denoms), None),
            "target_ids": [r["target_id"] for r in chosen],
            "metric_scope": "observed exported trees; not finished full benchmark or matched compute",
            "counting_unit": "per-target native contexts/transitions; shared contexts across targets are NOT globally deduplicated",
            "complete_reports": sum(bool(r["report"]) for r in chosen),
            "source_protocols": sorted({r["manifest"].get("source_protocol", "unspecified") for r in chosen}),
            "model_shas": sorted({r["manifest"]["adapter_sha"] for r in chosen if r["manifest"].get("adapter_sha")}),
            "totals": dict(sums), "error_counts": dict(errors),
            "cases_with_FLOW_fork": sum(r["summary"]["morphology"]["flow_fork_contexts"] > 0 for r in chosen),
            "cases_with_multiple_routes_lower_bound": sum((r["report"].get("unique_route_count_lower_bound") or 0) > 1 for r in chosen),
            "candidate_ranking": dict(rank_counts), "cohorts": {k: dict(v) for k, v in sorted(cohort.items())},
            "timing_availability": {key: sum(key in r["report"].get("work", {}) for r in chosen)
                for key in ["cpu_executor_wall_seconds", "cpu_executor_thread_seconds", "sampled_tokens", "prompt_tokens", "model_forward_batches_participated"]},
            "unavailable": ["equal-compute control inference", "global GPU model-call count from per-target participation", "independent chemistry validity rate", "strict H2 predictive generalization for full-data actor"]}
    return {"schema": "nmi_observed_tree_metrics_v1", "runs": runs,
        "cross_run_pooling": False, "new_model_sampling": False}


def analyze(input_roots, output_root, *, resume=False, style=None, workers=4, assets_manifest=None, tree_schema="multi_reaction"):
    if tree_schema not in {"multi_reaction", "single_reaction"}:
        raise ValueError("unsupported tree schema adapter")
    adapter = normalize_native_tree
    if tree_schema == "single_reaction":
        from mechet.analysis.flat_tree_records import normalize_single_tree
        adapter = normalize_single_tree
    output = Path(output_root)
    if output.exists() and not resume:
        raise FileExistsError("preserve previous analysis; use a new output or --resume")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    _write_json(output / "progress.json", {"phase": "discover_inputs", "input_roots": [str(p) for p in input_roots]})
    catalog = discover_inputs(input_roots)
    stocks, stock_sources = load_frozen_stocks(assets_manifest) if assets_manifest else ({}, None)
    print(json.dumps({"phase": "catalog", "inputs": len(catalog)}), flush=True)
    _write_json(output / "progress.json", {"phase": "catalogued", "inputs": len(catalog), "done": 0})
    hashes = {}
    producer_files = [Path(__file__), ROOT / "src/mechet/analysis/tree_records.py", ROOT / "src/mechet/analysis/tree_metrics.py", ROOT / "src/mechet/analysis/chemical_evidence.py"]
    if tree_schema == "single_reaction":
        producer_files.append(ROOT / "src/mechet/analysis/flat_tree_records.py")
    producer = {str(p.relative_to(ROOT)): file_digest(p) for p in producer_files}
    transformation_sha = digest(producer)
    entries = []
    sources = []
    def prepare(record):
        entry = dict(record)
        source = {"run_id": record["run_id"], "target_id": record["target_id"],
            "tree_path": record["tree_path"], "report_path": record["report_path"],
            "report_index": record["report_index"]}
        for label in ("tree", "report", "manifest"):
            path = record.get(label + "_path")
            if path and Path(path).is_file():
                stat = Path(path).stat()
                fingerprint = (path, stat.st_size, stat.st_mtime_ns)
                if fingerprint not in hashes:
                    hashes[fingerprint] = file_digest(path)
                source[label + "_sha256"] = hashes[fingerprint]
        entry.update(source)
        if not record["tree_exists"]:
            entry["analysis_status"] = "tree_unavailable"
            return entry, source
        cache_key = digest([source, record["manifest"], record["report"], transformation_sha, stock_sources])
        cached = cache / (cache_key + ".json.gz")
        if not (resume and cached.exists()):
            data = json.loads(Path(source["tree_path"]).read_bytes())
            try:
                if stock_sources:
                    assert_stock_identity(record["manifest"], record["report"].get("cohort"), stock_sources)
                normalized = adapter(data, report=record["report"], manifest=record["manifest"], source=source,
                    stock=stocks.get(record["report"].get("cohort")))
            except ValueError as exc:
                entry["analysis_status"] = "unsupported_or_invalid"
                entry["error"] = str(exc)
                return entry, source
            m = morphology(normalized)
            ranking = candidate_metrics(record["report"].get("ranked_candidates", []), has_reference="expected_structural_precursor" in record["report"])
            normalized["summary"] = {"morphology": m, "ranking": ranking}
            temp = cached.with_name(cached.name + ".tmp")
            with _deterministic_gzip_text(temp) as f:
                json.dump(normalized, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(temp, cached)
        with gzip.open(cached, "rt", encoding="utf-8") as f:
            value = json.load(f)
        entry["analysis_status"] = "normalized"
        entry["cache_path"] = str(cached)
        entry["tree_id"] = value["tree"]["tree_id"]
        entry["summary"] = value["summary"]
        return entry, source
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (entry, source) in enumerate(pool.map(prepare, catalog)):
            sources.append(source)
            entries.append(entry)
            if (index + 1) % 25 == 0:
                progress = {"phase": "normalize", "done": index + 1, "total": len(catalog)}
                _write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    # Table rebuild is deterministic, including after an interrupted export.
    tables = {"trees": "tree", "nodes": "nodes", "attempted_edges": "attempted_edges", "terminals": "terminals", "routes": "routes"}
    row_counts = Counter()
    exported_tree_ids = set()
    _write_json(output / "progress.json", {"phase": "export_tables", "done": 0, "total": len(entries), "normalized_inputs": len(entries)})
    with ExitStack() as stack:
        handles = {name: stack.enter_context(_deterministic_gzip_text(output / (name + ".jsonl.gz.tmp"))) for name in tables}
        for table_index, entry in enumerate(entries):
            if entry["analysis_status"] != "normalized":
                continue
            if entry["tree_id"] in exported_tree_ids:
                entry["normalized_table_storage_alias"] = True
                continue
            exported_tree_ids.add(entry["tree_id"])
            entry["normalized_table_storage_alias"] = False
            with gzip.open(entry["cache_path"], "rt", encoding="utf-8") as f:
                data = json.load(f)
            for name, field in tables.items():
                rows = [data[field]] if field == "tree" else data[field]
                for row in rows:
                    _dump_line(handles[name], row)
                    row_counts[name] += 1
            if (table_index + 1) % 100 == 0:
                progress = {"phase": "export_tables", "done": table_index + 1, "total": len(entries), "rows": dict(row_counts)}
                _write_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    for name in tables:
        os.replace(output / (name + ".jsonl.gz.tmp"), output / (name + ".jsonl.gz"))
    metrics = _aggregate(entries)
    _write_json(output / "metrics.json", metrics)
    inventory = {"schema": "nmi_tree_input_inventory_v1", "tree_schema_adapter": tree_schema, "input_roots": sorted(str(Path(p).resolve()) for p in input_roots),
        "inputs": entries, "input_count": len(entries), "status_counts": dict(Counter(e["analysis_status"] for e in entries)),
        "unsupported_artifacts": "raw shared cache/WAL and pending controllers are not independent exported case trees",
        "all_discovered_inputs_accounted_for": len(entries) == len(catalog)}
    _write_json(output / "input_inventory.json", inventory)
    source_manifest = {"schema": "nmi_tree_source_manifest_v1", "sources": sources,
        "producer_files": producer, "transformation_sha256": transformation_sha,
        "frozen_stock_sources": stock_sources,
        "input_inventory_sha256": file_digest(output / "input_inventory.json"),
        "tables": {n: {"path": n + ".jsonl.gz", "rows": row_counts[n], "sha256": file_digest(output / (n + ".jsonl.gz"))} for n in tables},
        "style": {"path": str(style), "sha256": file_digest(style)} if style else None,
        "new_sampling": False, "original_files_modified": False}
    _write_json(output / "source_manifest.json", source_manifest)
    _write_json(output / "progress.json", {"phase": "complete", "done": len(entries), "total": len(catalog)})
    with (output / "case_registry.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "target_id", "tree_id", "tree_sha256", "tree_path", "flow_fork_contexts", "terminal_contexts", "selected", "selection_rule"])
        writer.writeheader()
        eligible = [e for e in entries if e.get("summary", {}).get("morphology", {}).get("flow_fork_contexts", 0) > 0]
        selected = {e["tree_id"] for e in sorted(eligible, key=lambda e: (e["run_id"], e["target_id"]))[:3]}
        for e in entries:
            if "summary" not in e:
                continue
            writer.writerow({**{k: e.get(k) for k in ["run_id", "target_id", "tree_id", "tree_sha256", "tree_path"]},
                **{k: e["summary"]["morphology"][k] for k in ["flow_fork_contexts", "terminal_contexts"]},
                "selected": e["tree_id"] in selected, "selection_rule": "first_three_lexicographic_actual_FLOW_fork_cases_not_baseline_rescue"})
    print(json.dumps({"phase": "complete", "input_count": len(entries), "normalized": inventory["status_counts"].get("normalized", 0), "output": str(output)}), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--route-assets-manifest", type=Path)
    parser.add_argument("--tree-schema", choices=["multi_reaction", "single_reaction"], default="multi_reaction")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    analyze(args.input_roots, args.output_root, resume=args.resume, style=args.style, workers=args.workers, assets_manifest=args.route_assets_manifest, tree_schema=args.tree_schema)


if __name__ == "__main__":
    main()
