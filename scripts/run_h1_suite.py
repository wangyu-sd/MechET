#!/usr/bin/env python3
"""Run the frozen H1 normal/intervention rollout suite and paired evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.model_revision import is_immutable_revision

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


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected config object: {path}")
    return dict(value)


def _load_manifest(adapter: Path) -> dict:
    path = adapter / "adapter_manifest.json"
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid adapter manifest: {path}")
    return dict(value)


def _resolve_adapter_and_revision(args) -> tuple[Path | None, str, dict]:
    cfg = _load_yaml(args.config)
    adapter_text = str(args.adapter or cfg.get("initial_adapter_path") or "").strip()
    adapter = Path(adapter_text) if adapter_text else None
    manifest = _load_manifest(adapter) if adapter and adapter.exists() else {}
    requested = str(args.model_revision or "").strip()
    if requested:
        if not is_immutable_revision(requested):
            raise ValueError(
                "run_h1_suite --model-revision must be the immutable 40-hex commit SHA"
            )
        revision = requested.lower()
        observed = str(manifest.get("base_model_revision") or "").strip()
        if observed and observed.lower() != revision:
            raise ValueError(
                f"H1_ADAPTER_MODEL_REVISION_MISMATCH:{observed}!={revision}"
            )
    else:
        revision = str(manifest.get("base_model_revision") or "").strip().lower()
        if not args.dry_run and not is_immutable_revision(revision):
            raise ValueError(
                "H1 requires an immutable model revision from --model-revision or "
                "the selected adapter_manifest.json"
            )
    if not args.dry_run:
        if adapter is None or not adapter.exists():
            raise FileNotFoundError(
                "H1 adapter does not exist; pass --adapter or train the config's "
                f"initial_adapter_path: {adapter}"
            )
    return adapter, revision, manifest


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
    parser.add_argument(
        "--adapter",
        default="",
        help="checkpoint/adapter to evaluate; defaults to config.initial_adapter_path",
    )
    parser.add_argument(
        "--model-revision",
        default="",
        help="optional immutable 40-hex revision; otherwise read from adapter manifest",
    )
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

    adapter, immutable_revision, adapter_manifest = _resolve_adapter_and_revision(args)
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
    if adapter is not None:
        common += ["--adapter", str(adapter)]
    if immutable_revision:
        common += ["--model-revision", immutable_revision]
    if args.limit:
        common += ["--limit", str(args.limit)]
    if args.resume:
        common += ["--resume"]

    normal = args.out_dir / "normal.jsonl"
    _run(
        common + ["--output", str(normal), "--intervention", "none"],
        dry_run=args.dry_run,
    )

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
        "adapter": str(adapter) if adapter else None,
        "adapter_sha256": adapter_manifest.get("adapter_sha256"),
        "inference_model_revision": immutable_revision or None,
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
