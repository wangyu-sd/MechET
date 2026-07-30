#!/usr/bin/env python3
"""Run infer -> eval -> TSV export in one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data/mechet_sft/valid.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/mechet_eval/run")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--method", default="MechET v3 (MECH_ET CoT)")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    py = sys.executable
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    generations = out_dir / "generations.jsonl"
    summary = out_dir / "model_eval_summary.json"
    table = out_dir / "main_table.tsv"

    if not args.skip_infer:
        infer_script = "infer_mechet_vllm.py" if args.use_vllm else "infer_mechet.py"
        cmd = [
            py,
            str(REPO / "scripts" / infer_script),
            "--data",
            str(args.data),
            "--out",
            str(generations),
            "--num-beams",
            str(args.num_beams),
            "--num-return-sequences",
            str(args.num_return_sequences),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.adapter:
            cmd.extend(["--adapter", str(args.adapter)])
        if args.model_path:
            cmd.extend(["--model-path", args.model_path])
        if args.use_vllm:
            cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
        subprocess.check_call(cmd)

    subprocess.check_call(
        [
            py,
            str(REPO / "scripts/eval_mechet_generations.py"),
            "--data",
            str(args.data),
            "--predictions",
            str(generations),
            "--out",
            str(summary),
        ]
        + (["--limit", str(args.limit)] if args.limit else [])
    )
    subprocess.check_call(
        [
            py,
            str(REPO / "scripts/collect_mechet_results.py"),
            "--summary",
            str(summary),
            "--out",
            str(table),
            "--method",
            args.method,
        ]
    )
    print(f"wrote {generations}\n{summary}\n{table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
