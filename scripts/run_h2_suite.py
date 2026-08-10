#!/usr/bin/env python3
"""Run H2 trace-owned inference on a frozen composition-OOD split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import align_prediction_artifact, file_sha256, read_jsonl, row_id
from mechet.prediction_metrics import prediction_set_metrics
from mechet.strict_prediction_evaluation import condition_metrics, endpoint_evaluation


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _validate_adapter_train_split(adapter: Path, train_file: Path) -> dict[str, Any]:
    manifest_path = adapter / "adapter_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"H2 adapter manifest missing: {manifest_path}")
    manifest = _load_object(manifest_path)
    expected = file_sha256(train_file)
    observed = str(manifest.get("train_file_sha256") or "")
    if observed != expected:
        raise ValueError(
            "H2_ADAPTER_TRAIN_SPLIT_MISMATCH: adapter was not trained on the "
            f"frozen H2 train split: {observed} != {expected}"
        )
    if not manifest.get("base_model_revision"):
        raise ValueError("H2 adapter has no frozen base_model_revision")
    return manifest


def _stratified_metrics(reference_rows, aligned_rows) -> dict[str, Any]:
    reference = {row_id(row): row for row in reference_rows}
    buckets: dict[str, list[bool]] = {}
    for row in aligned_rows:
        identifier = row_id(row)
        overlap = dict(
            (reference[identifier].get("metadata") or {}).get(
                "mechcomp_structural_overlap"
            )
            or {}
        )
        exact = bool(endpoint_evaluation(row)["structural_exact"])
        labels = {
            "scaffold_seen": overlap.get("murcko_scaffold_seen_in_train") is True,
            "scaffold_unseen": overlap.get("murcko_scaffold_seen_in_train") is False,
            "reaction_center_seen": overlap.get("reaction_center_context_seen_in_train") is True,
            "reaction_center_unseen": overlap.get("reaction_center_context_seen_in_train") is False,
            "family_seen": overlap.get("family_seen_in_train") is True,
            "family_unseen": overlap.get("family_seen_in_train") is False,
        }
        for name, selected in labels.items():
            if selected:
                buckets.setdefault(name, []).append(exact)
    return {
        name: {
            "n": len(values),
            "structural_exact": sum(values) / max(len(values), 1),
        }
        for name, values in buckets.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO / "data/ood/mechcomp_source_sink",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/agent/tool_sft_mechcomp_trace.yaml",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO / "outputs/h2/tool_sft_trace_qwen3_0_6b",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/h2")
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

    train_file = args.split_dir / "train.jsonl"
    test_file = args.split_dir / "test.jsonl"
    split_manifest_path = args.split_dir / "manifest.json"
    for path in (train_file, test_file, split_manifest_path):
        if not path.exists() and not args.dry_run:
            raise FileNotFoundError(path)

    adapter_manifest = None
    if not args.dry_run:
        adapter_manifest = _validate_adapter_train_split(args.adapter, train_file)
        split_manifest = _load_object(split_manifest_path)
        gate = dict(split_manifest.get("claim_gate") or {})
        if gate.get("all_test_primitives_seen_in_train") is False:
            raise ValueError("H2 split contains unseen test primitives")
        if gate.get("zero_train_test_composition_overlap") is False:
            raise ValueError("H2 split has train/test composition overlap")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / "trace_owned.jsonl"
    cmd = [
        sys.executable,
        str(REPO / "scripts/infer_mechet.py"),
        "--config",
        str(args.config),
        "--data",
        str(test_file),
        "--output",
        str(prediction_path),
        "--mode",
        "trace",
        "--condition-name",
        "h2_trace_owned",
        "--adapter",
        str(args.adapter),
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
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    subprocess.check_call(cmd)

    references = read_jsonl(test_file)
    if args.limit:
        references = references[: args.limit]
    predictions = read_jsonl(prediction_path)
    aligned = align_prediction_artifact(
        references, predictions, condition_name="h2_trace_owned"
    )
    summary = {
        "artifact_type": "h2_trace_owned_evaluation",
        "split_manifest": str(split_manifest_path),
        "train_sha256": file_sha256(train_file),
        "test_sha256": file_sha256(test_file),
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_manifest.get("adapter_sha256") if adapter_manifest else None,
        "adapter_base_model_revision": adapter_manifest.get("base_model_revision") if adapter_manifest else None,
        "headline": {
            **condition_metrics(aligned),
            **prediction_set_metrics(aligned),
        },
        "composition_ood_strata": _stratified_metrics(references, aligned),
        "scientific_boundary": (
            "This runner evaluates the trace-owned H2 condition only. Complete-proof, "
            "direct, CoT, and net-edit baselines must be trained on the same H2/train IDs."
        ),
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
