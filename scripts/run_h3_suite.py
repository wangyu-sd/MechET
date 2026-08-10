#!/usr/bin/env python3
"""Run the six matched H3 conditions on the frozen held-out test suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "trace_no_knowledge": {
        "mode": "trace",
        "config": REPO / "configs/knowledge/tool_sft_trace_no_knowledge.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_trace_no_knowledge_qwen3_0_6b",
    },
    "trace_length_matched_irrelevant": {
        "mode": "irrelevant",
        "config": REPO / "configs/knowledge/tool_sft_irrelevant.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_irrelevant_qwen3_0_6b",
    },
    "trace_textbook_rag": {
        "mode": "textbook",
        "config": REPO / "configs/knowledge/tool_sft_textbook.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_textbook_qwen3_0_6b",
    },
    "trace_structured_anchors": {
        "mode": "anchors",
        "config": REPO / "configs/knowledge/tool_sft_anchors.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_anchors_qwen3_0_6b",
    },
    "trace_text_plus_anchors": {
        "mode": "combined",
        "config": REPO / "configs/knowledge/tool_sft_combined.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_text_plus_anchors_qwen3_0_6b",
    },
    "direct_textbook_rag": {
        "mode": "direct",
        "config": REPO / "configs/knowledge/tool_sft_direct_textbook.yaml",
        "adapter": REPO / "outputs/agent/tool_sft_direct_textbook_qwen3_0_6b",
    },
}


def _mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("override must be CONDITION=PATH")
    name, path = value.split("=", 1)
    if name not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"unknown H3 condition: {name}")
    return name, Path(path)


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_adapter(condition: str, adapter: Path, train_file: Path) -> dict[str, Any]:
    manifest_path = adapter / "adapter_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{condition}: adapter manifest missing: {manifest_path}")
    manifest = _load_object(manifest_path)
    observed = str(manifest.get("train_file_sha256") or "")
    expected = _sha256(train_file)
    if observed != expected:
        raise ValueError(
            f"H3_ADAPTER_TRAIN_SPLIT_MISMATCH:{condition}:{observed}!={expected}"
        )
    revision = str(manifest.get("base_model_revision") or "")
    if len(revision) != 40:
        raise ValueError(f"{condition}: adapter lacks immutable base-model revision")
    return manifest


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(cmd))
    else:
        subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root",
        type=Path,
        default=REPO / "data/knowledge_ablation/v2/test",
        help="held-out test condition directory produced by build_knowledge_ablation_suite.py",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/h3")
    parser.add_argument("--adapter", action="append", type=_mapping, default=[])
    parser.add_argument("--config", action="append", type=_mapping, default=[])
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

    adapters = {name: Path(spec["adapter"]) for name, spec in CONDITIONS.items()}
    configs = {name: Path(spec["config"]) for name, spec in CONDITIONS.items()}
    adapters.update(dict(args.adapter))
    configs.update(dict(args.config))

    train_root = args.suite_root.parent / "train"
    test_manifest = args.suite_root / "manifest.json"
    if not args.dry_run:
        if not test_manifest.exists():
            raise FileNotFoundError(test_manifest)
        test_meta = _load_object(test_manifest)
        if test_meta.get("split") != "test":
            raise ValueError(
                f"H3 suite root is not a frozen test split: {test_meta.get('split')}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, Path] = {}
    adapter_meta: dict[str, Any] = {}
    for condition, spec in CONDITIONS.items():
        data = args.suite_root / f"{condition}.jsonl"
        train_file = train_root / f"{condition}.jsonl"
        adapter = adapters[condition]
        config = configs[condition]
        if not args.dry_run:
            for path in (data, train_file, config):
                if not path.exists():
                    raise FileNotFoundError(path)
            adapter_meta[condition] = _check_adapter(condition, adapter, train_file)
        output = args.out_dir / f"{condition}.jsonl"
        predictions[condition] = output
        cmd = [
            sys.executable,
            str(REPO / "scripts/infer_mechet.py"),
            "--config",
            str(config),
            "--data",
            str(data),
            "--output",
            str(output),
            "--mode",
            str(spec["mode"]),
            "--condition-name",
            condition,
            "--adapter",
            str(adapter),
            "--seed",
            str(args.seed),
            "--samples-per-target",
            str(args.samples_per_target),
            "--max-iterations",
            str(args.max_iterations),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.resume:
            cmd += ["--resume"]
        _run(cmd, dry_run=args.dry_run)

    reference = args.suite_root / "trace_textbook_rag.jsonl"
    summary = args.out_dir / "summary.json"
    eval_cmd = [
        sys.executable,
        str(REPO / "scripts/evaluate_knowledge_ablation.py"),
        "--reference",
        str(reference),
    ]
    for condition, path in predictions.items():
        eval_cmd += ["--condition", f"{condition}={path}"]
    eval_cmd += ["--output", str(summary)]
    _run(eval_cmd, dry_run=args.dry_run)

    runner_manifest = {
        "artifact_type": "h3_runner_manifest",
        "suite_root": str(args.suite_root),
        "training_suite_root": str(train_root),
        "outputs": {name: str(path) for name, path in predictions.items()},
        "summary": str(summary),
        "adapter_base_model_revisions": {
            name: value.get("base_model_revision")
            for name, value in adapter_meta.items()
        },
        "seed": args.seed,
        "samples_per_target": args.samples_per_target,
        "dry_run": args.dry_run,
        "scientific_boundary": (
            "Evidence-content interventions with quarantined donors must be evaluated "
            "against the paired reference written by build_evidence_interventions.py."
        ),
    }
    if args.dry_run:
        print(json.dumps(runner_manifest, indent=2))
    else:
        (args.out_dir / "runner_manifest.json").write_text(
            json.dumps(runner_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
