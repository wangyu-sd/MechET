#!/usr/bin/env python3
"""Run the frozen H1 normal/intervention rollout suite and paired evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
INTERVENTIONS = [
    "remove_tool_observations",
    "stale_tool_observations",
    "shuffle_tool_observations",
    "disable_inspect_state",
    "disable_intermediate_execution",
]


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(cmd))
    else:
        subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/agent/inverse_trace_grpo.yaml",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO / "data/benchmarks/h1/test.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/h1")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-target", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    args.out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        py,
        str(REPO / "scripts/infer_mechet.py"),
        "--config",
        str(args.config),
        "--data",
        str(args.data),
        "--mode",
        "trace",
        "--condition-name",
        "trace_no_knowledge",
        "--samples-per-target",
        str(args.samples_per_target),
        "--seed",
        str(args.seed),
        "--max-iterations",
        str(args.max_iterations),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    if args.adapter:
        common += ["--adapter", args.adapter]
    if args.model_revision:
        common += ["--model-revision", args.model_revision]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.resume:
        common += ["--resume"]

    normal = args.out_dir / "normal.jsonl"
    _run(common + ["--output", str(normal), "--intervention", "none"], dry_run=args.dry_run)

    outputs: dict[str, Path] = {}
    for intervention in INTERVENTIONS:
        path = args.out_dir / f"{intervention}.jsonl"
        cmd = common + [
            "--output",
            str(path),
            "--intervention",
            intervention,
        ]
        if intervention == "shuffle_tool_observations":
            cmd += ["--intervention-source", str(normal)]
        _run(cmd, dry_run=args.dry_run)
        outputs[intervention] = path

    summary = args.out_dir / "summary.json"
    eval_cmd = [
        py,
        str(REPO / "scripts/evaluate_faithfulness.py"),
        "--reference",
        str(args.data),
        "--normal",
        str(normal),
    ]
    for name, path in outputs.items():
        eval_cmd += ["--intervention", f"{name}={path}"]
    eval_cmd += ["--output", str(summary)]
    _run(eval_cmd, dry_run=args.dry_run)

    manifest = {
        "artifact_type": "h1_runner_manifest",
        "data": str(args.data),
        "config": str(args.config),
        "normal": str(normal),
        "interventions": {name: str(path) for name, path in outputs.items()},
        "summary": str(summary),
        "seed": args.seed,
        "samples_per_target": args.samples_per_target,
        "max_iterations": args.max_iterations,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        (args.out_dir / "runner_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    else:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
