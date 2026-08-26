#!/usr/bin/env python3
"""Build Qwen SFT JSONL from FlowER trajectories as MECH_ET v3 CoT.

Streams per trajectory. Annotates PERCEIVE / ET_SIGNATURE / BE_DELTA automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mechet.mech_et import format_mech_et_cot, verify_mech_et
from mechet.mech_graph import (
    FlowERMechanismGraph,
    build_mechanism_graph,
    endpoint_fallback_graph,
    executable_witness_graph,
    parse_flower_line,
)
from mechet.sft import convert_record_to_qwen_sft

DEFAULT_FLOWER_ROOT = Path("/aaa/fionafyang/buddy1/whaleywang/datasets/retro/data/flower_new_dataset")
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def iter_flower_groups(
    split_path: Path,
    *,
    limit: int | None = None,
) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    # FlowER trajectories are mostly, but not completely, contiguous.  The old
    # transition-based iterator split a trajectory every time another ID was
    # interleaved, creating 2,033 extra train groups, 19 extra validation
    # groups, and 266 extra test groups in the current upstream release.
    groups: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_flower_line(line)
            if parsed is None:
                continue
            reactants, products, tid = parsed
            if tid not in groups:
                if limit is not None and len(order) >= limit:
                    # Continue scanning: selected IDs can recur later in the
                    # file and their groups must remain complete.
                    continue
                groups[tid] = []
                order.append(tid)
            groups[tid].append((reactants, products))
    for tid in order:
        yield tid, groups[tid]


def _build_row(graph: FlowERMechanismGraph, *, source_split: str) -> dict[str, Any] | None:
    source_graph = graph
    graph = executable_witness_graph(graph)
    mechanism = format_mech_et_cot(graph)
    answer = graph.compact_precursor_smiles()
    main_product = graph.compact_main_product()
    # Formatter is the gold annotator; skip expensive expected_graph BE recompute.
    checked = verify_mech_et(
        mechanism_body=mechanism,
        answer=answer,
        main_product=main_product,
        expected_precursor=answer,
    )
    if not (
        checked.get("format_ok")
        and checked.get("reachability_ok")
        and checked.get("answer_exact")
        and checked.get("be_delta_exact")
        and checked.get("main_product_ok")
    ):
        return None
    raw = {
        "id": f"flower_mech_et_{source_split}_{graph.trajectory_id}",
        "task_type": "mech_et_cot_retro",
        "product_smiles": main_product,
        "main_product": main_product,
        "mechanism": mechanism,
        "reactants": answer,
        "initial_reactants": answer,
        "trajectory_id": graph.trajectory_id,
        "topology": graph.topology,
        "n_states": graph.n_states,
        "n_edges": graph.n_edges,
        "metadata": {
            "source": "flower_new_dataset",
            "source_path": graph.source_path,
            "source_split": source_split,
            "trajectory_id": graph.trajectory_id,
            "topology": graph.topology,
            "n_states": graph.n_states,
            "n_edges": graph.n_edges,
            "n_targets": len(graph.target_state_ids),
            "source_topology": source_graph.topology,
            "source_n_states": source_graph.n_states,
            "source_n_edges": source_graph.n_edges,
            "proof_projection": "map_aligned_executable_witness",
            "inferred_missing_edge": any(
                item.get("code") == "INFERRED_MISSING_EDGE"
                for item in graph.diagnostics
            ),
            "upstream_endpoint_fallback": any(
                item.get("code") == "UPSTREAM_ENDPOINT_FALLBACK"
                for item in graph.diagnostics
            ),
            "initial_reactants": answer,
            "task_type": "mech_et_cot_retro",
            "et_signature": checked.get("et_signature"),
            "be_delta_exact": True,
            "electron_conserved": bool(checked.get("electron_conserved")),
        },
    }
    return convert_record_to_qwen_sft(raw)


def _load_done_ids(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            rid = str(row.get("id") or "")
            if rid:
                done.add(rid)
            # Also index by trajectory_id for skip.
            tid = str((row.get("metadata") or {}).get("trajectory_id") or "")
            if tid:
                done.add(f"tid:{tid}")
    return done


def _build_task(
    task: tuple[str, list[tuple[str, str]], str, str, tuple[str, str] | None],
) -> tuple[str, str | None, dict[str, Any]]:
    tid, steps, source_path, source_split, expected = task
    graph = build_mechanism_graph(
        tid,
        steps,
        source_path=source_path,
        expected_reactants=expected[0] if expected else None,
        expected_products=expected[1] if expected else None,
    )
    if graph is None and expected and tid in {"RC", "PC", "PM", "RS"}:
        graph = endpoint_fallback_graph(
            tid,
            expected[0],
            expected[1],
            source_path=source_path,
        )
    if graph is None:
        return "invalid_official", None, {}
    row = _build_row(graph, source_split=source_split)
    if row is None:
        return "verify_failed", None, {}
    metadata = dict(row.get("metadata") or {})
    return "ok", json.dumps(row, ensure_ascii=False), {
        "topology": graph.topology,
        "et_signature": metadata.get("et_signature") or "unknown",
        "n_states": graph.n_states,
        "n_edges": graph.n_edges,
    }


def build_split_parallel(
    *,
    flower_root: Path,
    out_root: Path,
    split_name: str,
    limit: int | None,
    workers: int,
    resume: bool = False,
) -> dict[str, Any]:
    flower_split = SPLIT_MAP[split_name]
    src = flower_root / f"{flower_split}.txt"
    if not src.exists():
        raise FileNotFoundError(src)
    out_path = out_root / f"{split_name}.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    skipped = {"invalid_official": 0, "verify_failed": 0, "already_done": 0}
    topo: Counter[str] = Counter()
    sig_counts: Counter[str] = Counter()
    accepted = loaded = state_total = edge_total = 0
    retro_path = flower_root.parent / "flower_retro" / f"{flower_split}.txt"
    expected_rows: list[tuple[str, str]] = []
    if retro_path.is_file():
        with retro_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                text = line.strip()
                if ">>" in text:
                    expected_rows.append(tuple(text.split(">>", 1)))
    done_ids = _load_done_ids(out_path) if resume else set()
    skipped["already_done"] = sum(
        1 for value in done_ids if value.startswith("tid:")
    )

    def pending_tasks():
        for index, (tid, steps) in enumerate(
            iter_flower_groups(src, limit=limit)
        ):
            row_id = f"flower_mech_et_{flower_split}_{tid}"
            if row_id in done_ids or f"tid:{tid}" in done_ids:
                continue
            yield (
                tid,
                steps,
                str(src),
                flower_split,
                expected_rows[index] if index < len(expected_rows) else None,
            )

    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as handle, ProcessPoolExecutor(
        max_workers=workers
    ) as executor:
        for status, payload, stats in executor.map(
            _build_task, pending_tasks(), chunksize=8
        ):
            loaded += 1
            if status != "ok":
                skipped[status] += 1
                continue
            assert payload is not None
            handle.write(payload + "\n")
            accepted += 1
            topo[str(stats["topology"])] += 1
            sig_counts[str(stats["et_signature"])] += 1
            state_total += int(stats["n_states"])
            edge_total += int(stats["n_edges"])
            if loaded % 2000 == 0:
                print(
                    f"[{split_name}] loaded={loaded} accepted={accepted} "
                    f"failed={loaded - accepted}",
                    flush=True,
                )
    total_on_disk = sum(
        1 for line in out_path.open("r", encoding="utf-8") if line.strip()
    )
    return {
        "split": split_name,
        "source_file": str(src),
        "loaded_graphs": loaded,
        "accepted": accepted,
        "accepted_on_disk": total_on_disk,
        "skipped": skipped,
        "topology_counts": dict(topo),
        "et_signature_counts": dict(sig_counts.most_common(20)),
        "mean_states": state_total / accepted if accepted else 0.0,
        "mean_edges": edge_total / accepted if accepted else 0.0,
        "out_path": str(out_path),
        "resume": resume,
        "workers": workers,
    }


def build_split(
    *,
    flower_root: Path,
    out_root: Path,
    split_name: str,
    limit: int | None,
    resume: bool = False,
) -> dict[str, Any]:
    flower_split = SPLIT_MAP[split_name]
    src = flower_root / f"{flower_split}.txt"
    if not src.exists():
        raise FileNotFoundError(src)

    out_path = out_root / f"{split_name}.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)

    skipped = {"invalid_official": 0, "verify_failed": 0, "already_done": 0}
    topo = Counter()
    sig_counts = Counter()
    accepted = 0
    loaded = 0
    state_total = 0
    edge_total = 0

    done_ids = _load_done_ids(out_path) if resume else set()
    mode = "a" if resume and out_path.exists() else "w"
    if resume and done_ids:
        print(f"[{split_name}] resume: {len(done_ids)} existing id keys", flush=True)

    with out_path.open(mode, encoding="utf-8") as handle:
        for tid, steps in iter_flower_groups(src, limit=limit):
            loaded += 1
            row_id = f"flower_mech_et_{flower_split}_{tid}"
            if resume and (row_id in done_ids or f"tid:{tid}" in done_ids):
                skipped["already_done"] += 1
                continue
            graph = build_mechanism_graph(tid, steps, source_path=str(src))
            if graph is None:
                skipped["invalid_official"] += 1
                continue
            row = _build_row(graph, source_split=flower_split)
            if row is None:
                skipped["verify_failed"] += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            accepted += 1
            topo[graph.topology] += 1
            sig = (row.get("metadata") or {}).get("et_signature") or "unknown"
            sig_counts[str(sig)] += 1
            state_total += graph.n_states
            edge_total += graph.n_edges
            if accepted % 2000 == 0:
                # Bound memory: drop RDKit caches periodically.
                try:
                    from mechet import mech_et, mech_graph

                    mech_et._BE_CACHE.clear()
                    mech_et._CHARGE_CACHE.clear()
                    mech_graph._COMPACT_CACHE.clear()
                    mech_graph._STATE_KEY_CACHE.clear()
                except Exception:
                    pass
                print(
                    f"[{split_name}] accepted={accepted} loaded={loaded} "
                    f"invalid={skipped['invalid_official']} resume_skip={skipped['already_done']}",
                    flush=True,
                )

    # Count total lines on disk after resume.
    total_on_disk = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    total_on_disk += 1

    return {
        "split": split_name,
        "source_file": str(src),
        "loaded_graphs": loaded,
        "accepted": accepted,
        "accepted_on_disk": total_on_disk,
        "skipped": skipped,
        "topology_counts": dict(topo),
        "et_signature_counts": dict(sig_counts.most_common(20)),
        "mean_states": (state_total / accepted) if accepted else 0.0,
        "mean_edges": (edge_total / accepted) if accepted else 0.0,
        "out_path": str(out_path),
        "resume": resume,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=DEFAULT_FLOWER_ROOT)
    parser.add_argument("--out-dir", type=Path, default=REPO / "data/mechet_sft")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel reaction workers; 0 uses the sequential builder.",
    )
    parser.add_argument("--resume", action="store_true", help="Append, skipping existing trajectory ids")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write a partial artifact instead of failing when any FlowER reaction is rejected.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test", "val"],
    )
    args = parser.parse_args()

    reports = []
    for split in args.splits:
        name = "valid" if split == "val" else split
        print(f"Building split={name} ...", flush=True)
        builder = build_split_parallel if args.workers > 0 else build_split
        kwargs = dict(
                flower_root=args.flower_root,
                out_root=args.out_dir,
                split_name=name,
                limit=args.limit,
        )
        if args.workers > 0:
            kwargs["workers"] = min(args.workers, os.cpu_count() or args.workers)
            kwargs["resume"] = bool(args.resume)
        else:
            kwargs["resume"] = bool(args.resume)
        reports.append(builder(**kwargs))

    existing_splits: dict[str, Any] = {}
    manifest_path = args.out_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            existing_splits = dict(
                json.loads(manifest_path.read_text(encoding="utf-8")).get("splits")
                or {}
            )
        except Exception:
            existing_splits = {}
    merged_splits = dict(existing_splits)
    merged_splits.update({r["split"]: r for r in reports})
    accepted_total = sum(
        int(report.get("accepted_on_disk") or report.get("accepted") or 0)
        for report in merged_splits.values()
    )
    manifest = {
        "version": "mech_et_sft_v3",
        "source_root": str(args.flower_root),
        "format": "<mechanism>MECH_ET v3 (PERCEIVE + ET_SIGNATURE + BE_DELTA)</mechanism><answer>initial reactants</answer>",
        "task_type": "mech_et_cot_retro",
        "semantics": "FlowER DiGraph + FlowER BE-matrix edge deltas; strip-H STATE + SHARED",
        "limit": args.limit,
        "splits": merged_splits,
        "accepted_total": accepted_total,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    rejected_total = sum(
        int(report["skipped"].get("invalid_official", 0))
        + int(report["skipped"].get("verify_failed", 0))
        for report in reports
    )
    if rejected_total and not args.allow_incomplete:
        raise RuntimeError(
            "MECH_ET coverage is incomplete: "
            f"{rejected_total} reactions were rejected; "
            "partial artifacts are not accepted by default"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
