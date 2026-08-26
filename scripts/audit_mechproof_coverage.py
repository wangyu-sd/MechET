#!/usr/bin/env python3
"""Audit reaction-level MECH_ET -> MECH_PROOF coverage without filtering."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from mechet.mech_et import format_mech_et_cot
from mechet.mech_graph import (
    build_mechanism_graph,
    endpoint_fallback_graph,
    executable_witness_graph,
)
from mechet.proof_program import compile_mech_et_body
from scripts.build_mechet_sft import DEFAULT_FLOWER_ROOT, iter_flower_groups


def _audit_one(task) -> dict[str, object]:
    trajectory_id, steps, source_path, expected = task
    try:
        graph = build_mechanism_graph(
            trajectory_id,
            steps,
            source_path=source_path,
            expected_reactants=expected[0] if expected else None,
            expected_products=expected[1] if expected else None,
        )
        if (
            graph is None
            and expected
            and trajectory_id in {"RC", "PC", "PM", "RS"}
        ):
            graph = endpoint_fallback_graph(
                trajectory_id,
                expected[0],
                expected[1],
                source_path=source_path,
            )
        if graph is None:
            raise ValueError("build_mechanism_graph returned None")
        witness = executable_witness_graph(graph)
        compile_mech_et_body(format_mech_et_cot(witness))
        return {
            "ok": True,
            "trajectory_id": trajectory_id,
            "source_topology": graph.topology,
            "source_n_states": graph.n_states,
            "proof_n_states": witness.n_states,
            "inferred_missing_edge": any(
                item.get("code") == "INFERRED_MISSING_EDGE"
                for item in witness.diagnostics
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "trajectory_id": trajectory_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flower-root", type=Path, default=DEFAULT_FLOWER_ROOT)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16))
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "outputs" / "mechproof_coverage_audit.json",
    )
    args = parser.parse_args()

    source = args.flower_root / f"{args.split}.txt"
    retro_path = args.flower_root.parent / "flower_retro" / f"{args.split}.txt"
    expected_rows = []
    if retro_path.is_file():
        for line in retro_path.read_text(encoding="utf-8").splitlines():
            if ">>" in line:
                expected_rows.append(tuple(line.split(">>", 1)))
    tasks = [
        (
            trajectory_id,
            steps,
            str(source),
            expected_rows[index] if index < len(expected_rows) else None,
        )
        for index, (trajectory_id, steps) in enumerate(
            iter_flower_groups(source, limit=args.limit)
        )
    ]
    accepted = 0
    inferred = 0
    topology: Counter[str] = Counter()
    failures: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(_audit_one, tasks, chunksize=8),
            start=1,
        ):
            if result["ok"]:
                accepted += 1
                inferred += int(bool(result["inferred_missing_edge"]))
                topology[str(result["source_topology"])] += 1
            else:
                failures.append(result)
                if len(failures) <= 50:
                    print("FAILURE " + json.dumps(result, ensure_ascii=False), flush=True)
            if index % 1000 == 0 or index == len(tasks):
                print(
                    f"audited={index}/{len(tasks)} accepted={accepted} "
                    f"failed={len(failures)} inferred_edges={inferred}",
                    flush=True,
                )

    report = {
        "protocol": "reaction-level MECH_ET -> MECH_PROOF -> strict executor replay",
        "source": str(source),
        "split": args.split,
        "total": len(tasks),
        "accepted": accepted,
        "failed": len(failures),
        "coverage": accepted / len(tasks) if tasks else 0.0,
        "inferred_missing_edge": inferred,
        "source_topology_counts": dict(topology),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
