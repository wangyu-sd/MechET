#!/usr/bin/env python3
"""Run one H3 evidence-content intervention on its frozen paired ID universe."""
from __future__ import annotations

import argparse
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
    if not args.dry_run:
        for path in (intervention_data, reference, eligible, args.config):
            if not path.exists():
                raise FileNotFoundError(path)
        eligible_payload = json.loads(eligible.read_text(encoding="utf-8"))
        reference_ids = [row_id(row) for row in read_jsonl(reference)]
        if reference_ids != list(eligible_payload.get("stable_ids") or []):
            raise ValueError("H3_INTERVENTION_ELIGIBLE_ID_MISMATCH")

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
