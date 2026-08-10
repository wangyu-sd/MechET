#!/usr/bin/env python3
"""Run one H3 evidence-content intervention on its frozen paired ID universe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.knowledge_ablation import (
    align_prediction_artifact,
    read_jsonl,
    row_id,
)
from mechet.model_revision import is_immutable_revision
from mechet.prediction_metrics import prediction_runtime_contract, prediction_set_metrics
from mechet.statistical_evaluation import paired_binary_contrast
from mechet.strict_prediction_evaluation import condition_metrics, endpoint_evaluation

INTERVENTIONS = {
    "passage_shuffle",
    "same_topic_wrong",
    "remove_warnings",
    "remove_competing_pathways",
}


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(cmd))
    else:
        subprocess.check_call(cmd)


def _outcomes(rows):
    return {
        row_id(row): bool(endpoint_evaluation(row)["structural_exact"])
        for row in rows
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_manifest(adapter: Path, train_file: Path) -> dict:
    path = adapter / "adapter_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("H3_INTERVENTION_ADAPTER_MANIFEST_INVALID")
    observed = str(value.get("train_file_sha256") or "")
    expected = _sha256(train_file)
    if observed != expected:
        raise ValueError(
            "H3_INTERVENTION_ADAPTER_TRAIN_SPLIT_MISMATCH:"
            f"{observed}!={expected}"
        )
    revision = str(value.get("base_model_revision") or "")
    if not is_immutable_revision(revision):
        raise ValueError("H3_INTERVENTION_ADAPTER_REVISION_NOT_IMMUTABLE")
    return dict(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(INTERVENTIONS), required=True)
    parser.add_argument(
        "--intervention-dir",
        type=Path,
        default=REPO / "data/evidence_interventions/v2",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs/knowledge/tool_sft_combined.yaml",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPO / "outputs/agent/tool_sft_text_plus_anchors_qwen3_0_6b",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=REPO / "data/knowledge_ablation/v2/train/trace_text_plus_anchors.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs/h3/interventions")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples-per-target", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    intervention_data = args.intervention_dir / f"{args.name}.jsonl"
    reference = args.intervention_dir / f"{args.name}.reference.jsonl"
    eligible = args.intervention_dir / f"{args.name}.eligible_ids.json"
    immutable_revision = ""
    adapter_meta = None
    if not args.dry_run:
        for path in (
            intervention_data,
            reference,
            eligible,
            args.config,
            args.train_file,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        eligible_payload = json.loads(eligible.read_text(encoding="utf-8"))
        reference_ids = [row_id(row) for row in read_jsonl(reference)]
        if reference_ids != list(eligible_payload.get("stable_ids") or []):
            raise ValueError("H3_INTERVENTION_ELIGIBLE_ID_MISMATCH")
        adapter_meta = _adapter_manifest(args.adapter, args.train_file)
        immutable_revision = str(adapter_meta["base_model_revision"])

    condition_dir = args.out_dir / args.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    baseline_prediction = condition_dir / "baseline.jsonl"
    intervention_prediction = condition_dir / "intervention.jsonl"
    common = [
        sys.executable,
        str(REPO / "scripts/infer_mechet.py"),
        "--config",
        str(args.config),
        "--mode",
        "combined",
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
    if immutable_revision:
        common += ["--model-revision", immutable_revision]
    _run(
        common
        + [
            "--data",
            str(reference),
            "--output",
            str(baseline_prediction),
            "--condition-name",
            "trace_text_plus_anchors_paired_baseline",
        ],
        dry_run=args.dry_run,
    )
    _run(
        common
        + [
            "--data",
            str(intervention_data),
            "--output",
            str(intervention_prediction),
            "--condition-name",
            args.name,
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return 0

    reference_rows = read_jsonl(reference)
    baseline_raw = read_jsonl(baseline_prediction)
    intervention_raw = read_jsonl(intervention_prediction)
    baseline = align_prediction_artifact(
        reference_rows,
        baseline_raw,
        condition_name="trace_text_plus_anchors_paired_baseline",
    )
    intervention = align_prediction_artifact(
        reference_rows,
        intervention_raw,
        condition_name=args.name,
    )
    baseline_outcomes = _outcomes(baseline)
    intervention_outcomes = _outcomes(intervention)
    if set(baseline_outcomes) != set(intervention_outcomes):
        raise ValueError("H3_INTERVENTION_PAIRED_PREDICTION_ID_MISMATCH")
    identifiers = list(baseline_outcomes)
    # Positive delta means the intact evidence baseline outperforms the intervention.
    contrast = paired_binary_contrast(
        [baseline_outcomes[item] for item in identifiers],
        [intervention_outcomes[item] for item in identifiers],
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    baseline_runtime = prediction_runtime_contract(baseline_raw, include_adapter=False)
    intervention_runtime = prediction_runtime_contract(
        intervention_raw, include_adapter=False
    )
    if baseline_runtime["runtime_contract_sha256"] != intervention_runtime[
        "runtime_contract_sha256"
    ]:
        raise ValueError("H3_INTERVENTION_RUNTIME_CONTRACT_MISMATCH")

    summary = {
        "artifact_type": "h3_paired_evidence_intervention_evaluation",
        "intervention": args.name,
        "reference": str(reference),
        "eligible_ids": str(eligible),
        "n_paired_ids": len(identifiers),
        "adapter": str(args.adapter),
        "adapter_sha256": adapter_meta.get("adapter_sha256") if adapter_meta else None,
        "adapter_train_file": str(args.train_file),
        "adapter_train_file_sha256": _sha256(args.train_file),
        "inference_model_revision": immutable_revision,
        "baseline": {
            **condition_metrics(baseline),
            **prediction_set_metrics(baseline),
        },
        "intervention_metrics": {
            **condition_metrics(intervention),
            **prediction_set_metrics(intervention),
        },
        "paired_structural_exact_contrast": contrast,
        "same_runtime_contract": True,
        "interpretation": (
            "positive paired delta means intact combined evidence outperformed the "
            "content intervention on exactly the intervention-eligible IDs"
        ),
    }
    summary_path = condition_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
