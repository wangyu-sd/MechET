#!/usr/bin/env python3
"""Score paired Base/Mech R2 proposals without dropping failed executions.

This consumes complete, independently produced model scores. It does not run a
model, infer chemical truth, or convert non-reference products into negatives.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.autoresearch.freeze_r2_plausibility import NEGATIVE_CLASSES
from scripts.autoresearch.stratified_manifest import digest, verify_evaluation_source


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _auroc(labels: list[int], probabilities: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires positive and negative proposals")
    pairs = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in pairs[start:end])
        start = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(labels: list[int], probabilities: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        raise ValueError("AUPRC requires positive proposals")
    pairs = sorted(zip(probabilities, labels), key=lambda item: -item[0])
    true_positive = 0
    processed = 0
    area = 0.0
    start = 0
    while start < len(pairs):
        end = start + 1
        while end < len(pairs) and pairs[end][0] == pairs[start][0]:
            end += 1
        added = sum(label for _, label in pairs[start:end])
        true_positive += added
        processed += end - start
        area += added / positives * true_positive / processed
        start = end
    return area


def _calibration(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    bins: list[list[tuple[int, float]]] = [[] for _ in range(10)]
    for label, probability in zip(labels, probabilities):
        bins[min(9, int(probability * 10))].append((label, probability))
    reliability = []
    ece = 0.0
    for index, group in enumerate(bins):
        count = len(group)
        mean_probability = sum(score for _, score in group) / count if count else None
        observed_fraction = sum(label for label, _ in group) / count if count else None
        if count:
            ece += count / len(labels) * abs(mean_probability - observed_fraction)
        reliability.append({
            "bin": index, "lower_inclusive": index / 10,
            "upper_inclusive": index == 9, "count": count,
            "mean_probability": mean_probability,
            "observed_positive_fraction": observed_fraction,
        })
    return {
        "brier_score": sum((probability - label) ** 2
                           for label, probability in zip(labels, probabilities)) / len(labels),
        "expected_calibration_error_10bin": ece,
        "reliability_10bin": reliability,
    }


def _read_cohort(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    verify_evaluation_source(path, name="r2_plausibility")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (manifest.get("artifact_type") != "r2_plausibility_manifest_v1"
            or manifest.get("positive_proposals") != 400
            or manifest.get("negative_proposals") != 400):
        raise ValueError("R2 source is not the audited 400/400 freeze artifact")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["proposal_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("R2 cohort has duplicated proposal IDs")
    labels = Counter("positive" if row["private_label"].get("known_recorded_positive") is True
                     else "negative" for row in rows)
    if labels != {"positive": 400, "negative": 400}:
        raise ValueError(f"R2 cohort is not 400/400: {dict(labels)}")
    classes = Counter(row["private_label"].get("negative_class") for row in rows
                      if not row["private_label"].get("known_recorded_positive"))
    if set(classes) != set(NEGATIVE_CLASSES) or any(count != 50 for count in classes.values()):
        raise ValueError("R2 cohort lacks eight balanced negative classes")
    return rows, dict(classes)


def _read_scores(path: Path, *, condition: str, cohort_hash: str,
                 proposal_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"R2 {condition} scores and sidecar manifest are required: {path}")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if (manifest.get("scores_sha256") != digest(path)
            or manifest.get("cohort_sha256") != cohort_hash
            or manifest.get("condition") != condition
            or manifest.get("input_fields") != ["product_smiles", "proposed_precursors"]
            or manifest.get("score_semantics") != "probability_known_valid_proposal"
            or not manifest.get("checkpoint_identifier")
            or not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64
            or any(char not in "0123456789abcdef" for char in checkpoint_sha)):
        raise ValueError(f"R2 {condition} score provenance/hash mismatch")
    scores: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        proposal_id = row.get("proposal_id")
        if not isinstance(proposal_id, str) or proposal_id in scores:
            raise ValueError(f"R2 {condition} duplicated/invalid score ID: {proposal_id}")
        probability = row.get("support_probability")
        if (not isinstance(probability, (int, float)) or isinstance(probability, bool)
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError(f"R2 {condition} invalid probability: {proposal_id}")
        compile_status = row.get("compile_status")
        execute_status = row.get("execute_status")
        if (compile_status not in {"success", "failed"}
                or execute_status not in {"success", "failed", "not_run"}
                or (compile_status == "failed" and execute_status != "not_run")
                or (compile_status == "success" and execute_status == "not_run")):
            raise ValueError(f"R2 {condition} invalid compile/execute outcome: {proposal_id}")
        scores[proposal_id] = row
    if set(scores) != proposal_ids:
        raise ValueError(f"R2 {condition} scores do not cover all 800 frozen proposals")
    return scores, {"scores_sha256": digest(path), "manifest_sha256": digest(manifest_path),
                    "checkpoint_identifier": manifest["checkpoint_identifier"],
                    "checkpoint_sha256": manifest["checkpoint_sha256"]}


def _condition_metrics(rows: list[dict[str, Any]],
                       scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["private_label"].get("known_recorded_positive") is True) for row in rows]
    probabilities = [float(scores[row["proposal_id"]]["support_probability"]) for row in rows]
    compiled = sum(scores[row["proposal_id"]]["compile_status"] == "success" for row in rows)
    executed = sum(scores[row["proposal_id"]]["execute_status"] == "success" for row in rows)
    hard = [index for index, row in enumerate(rows) if row["private_label"].get(
        "negative_class") == "executor_valid_wrong_successor"]
    if len(hard) != 50:
        raise ValueError("R2 mandatory executor-valid hard-negative denominator is not 50")
    return {
        "rows": len(rows), "positive_rows": sum(labels), "negative_rows": len(rows) - sum(labels),
        "auroc": _auroc(labels, probabilities),
        "auprc_average_precision": _average_precision(labels, probabilities),
        "compile_fraction": _rate(compiled, len(rows)),
        "execute_fraction": _rate(executed, len(rows)),
        "executor_valid_hard_negatives": {
            "rows": len(hard),
            "false_positive_fraction_at_0_5": _rate(sum(probabilities[i] >= 0.5 for i in hard), len(hard)),
            "mean_support_probability": sum(probabilities[i] for i in hard) / len(hard),
            "execute_fraction": _rate(sum(scores[rows[i]["proposal_id"]]["execute_status"] == "success"
                                         for i in hard), len(hard)),
        },
        "calibration": _calibration(labels, probabilities),
    }


def score(cohort: Path, base_scores: Path, mech_scores: Path, output: Path,
          scientific_freeze: Path,
          *, bootstrap_replicates: int = 1000, seed: int = 17) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"R2 score package already exists: {output}")
    if bootstrap_replicates <= 0:
        raise ValueError("R2 paired bootstrap requires a positive replicate count")
    rows, classes = _read_cohort(cohort)
    cohort_hash = digest(cohort)
    if not scientific_freeze.is_file():
        raise FileNotFoundError("R2 scoring requires the matched scientific freeze")
    frozen = json.loads(scientific_freeze.read_text())
    if (frozen.get("engineering_only") is not False
            or frozen.get("evaluation_hashes", {}).get("r2_plausibility") != cohort_hash
            or not frozen.get("mech_comparison_identifiable")):
        raise ValueError("R2 scoring is not bound to an identifiable scientific freeze")
    ids = {row["proposal_id"] for row in rows}
    base, base_provenance = _read_scores(base_scores, condition="base",
                                         cohort_hash=cohort_hash, proposal_ids=ids)
    mech, mech_provenance = _read_scores(mech_scores, condition="mech",
                                         cohort_hash=cohort_hash, proposal_ids=ids)
    base_metrics = _condition_metrics(rows, base)
    mech_metrics = _condition_metrics(rows, mech)
    by_product: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_product[row["product_smiles"]].append(index)
    products_with_negative = sorted(product for product, indices in by_product.items()
                                    if any(rows[index]["private_label"].get("negative_class")
                                           for index in indices))
    products_without_negative = sorted(set(by_product) - set(products_with_negative))
    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap_replicates):
        sampled = [rng.choice(products_with_negative) for _ in products_with_negative]
        sampled += [rng.choice(products_without_negative) for _ in products_without_negative]
        indices = [index for product in sampled for index in by_product[product]]
        labels = [int(rows[index]["private_label"].get("known_recorded_positive") is True)
                  for index in indices]
        deltas.append(_auroc(labels, [float(mech[rows[index]["proposal_id"]]["support_probability"])
                                      for index in indices])
                      - _auroc(labels, [float(base[rows[index]["proposal_id"]]["support_probability"])
                                        for index in indices]))
    deltas.sort()
    result = {
        "artifact_type": "r2_paired_plausibility_result_v1", "package": "r2", "status": "complete",
        "cohort_sha256": cohort_hash, "cohort_manifest_sha256": digest(cohort.parent / "manifest.json"),
        "scientific_freeze_sha256": digest(scientific_freeze),
        "evaluation_source_hashes": {"r2_plausibility": cohort_hash},
        "model_checkpoint_sha256": {
            "base": base_provenance["checkpoint_sha256"],
            "mech": mech_provenance["checkpoint_sha256"],
        },
        "denominators": {"proposals": 800, "positives": 400, "negatives": 400,
                         "executor_valid_hard_negatives": 50},
        "metrics": {"base_auroc": base_metrics["auroc"],
                    "mech_auroc": mech_metrics["auroc"],
                    "base_auprc": base_metrics["auprc_average_precision"],
                    "mech_auprc": mech_metrics["auprc_average_precision"]},
        "source_provenance": {"base": base_provenance, "mech": mech_provenance},
        "negative_strata": classes,
        "base": base_metrics, "mech": mech_metrics,
        "paired_delta": {
            "auroc_mech_minus_base": mech_metrics["auroc"] - base_metrics["auroc"],
            "auroc_product_cluster_bootstrap_95ci": [
                deltas[int(0.025 * (len(deltas) - 1))],
                deltas[int(0.975 * (len(deltas) - 1))]],
            "bootstrap_replicates": bootstrap_replicates, "seed": seed,
        },
        "data_contract_errors": 0,
        "claim_boundary": "Recorded-positive versus evidence-audited-negative discrimination. Executor success alone is not chemical correctness; positive records are not exhaustive chemistry truth.",
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--mech-scores", type=Path, required=True)
    parser.add_argument("--scientific-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(score(args.cohort, args.base_scores, args.mech_scores,
                           args.output, args.scientific_freeze,
                           bootstrap_replicates=args.bootstrap_replicates,
                           seed=args.seed), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
