#!/usr/bin/env python3
"""Paired, reaction-level H2 analysis on three frozen K-matched evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


CONDITIONS = ("direct", "open_flow", "closed_loop")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _row_results(
    condition: str, evaluation: Path, matched_dir: Path,
    *, expected_k: int,
) -> tuple[dict[str, dict[str, bool]], dict[str, Any]]:
    report = json.loads(evaluation.read_text())
    manifest = json.loads((matched_dir / condition / "manifest.json").read_text())
    test_path = matched_dir / condition / "test.jsonl"
    if _sha(test_path) != manifest["output_sha256"]["test"]:
        raise ValueError(f"{condition} frozen test file SHA mismatch")
    if report.get("reference_sha256") != manifest["output_sha256"]["test"]:
        raise ValueError(f"{condition} evaluation used a different test reference")
    if int(report.get("n_reference_rows") or 0) != int(manifest["rows"]["test"]):
        raise ValueError(f"{condition} evaluation test denominator mismatch")
    if int(report.get("candidates_per_target") or report.get("candidate_count_min") or 0) != expected_k:
        raise ValueError(f"{condition} evaluation candidate budget mismatch")
    if report.get("candidate_count_max") is not None and int(report["candidate_count_max"]) != expected_k:
        raise ValueError(f"{condition} evaluation candidate count varies")
    row_path = Path(report.get("row_evaluation") or evaluation.with_suffix(".rows.jsonl"))
    if report.get("row_evaluation_sha256") and _sha(row_path) != report["row_evaluation_sha256"]:
        raise ValueError(f"{condition} row-evaluation SHA mismatch")
    references = {str(row["id"]): str(row["source_id"]) for row in _jsonl(test_path)}
    if len(references) != manifest["rows"]["test"]:
        raise ValueError(f"{condition} duplicate reference IDs")
    outcomes: dict[str, dict[str, bool]] = {}
    for row in _jsonl(row_path):
        identifier = str(row.get("id") or "")
        source_id = str(row.get("source_id") or references.get(identifier) or "")
        if not identifier or references.get(identifier) != source_id or source_id in outcomes:
            raise ValueError(f"{condition} row/source ID mismatch: {identifier}/{source_id}")
        candidates = list(row.get("candidates") or [])
        if len(candidates) != expected_k:
            raise ValueError(f"{condition} row has wrong K: {source_id}")
        order = list(row.get("generation_order") or range(expected_k))
        if order != list(range(expected_k)):
            raise ValueError(f"{condition} generation order not canonical: {source_id}")
        outcomes[source_id] = {
            "endpoint_at_1": bool(candidates[0]["structural_exact"]),
            "endpoint_at_k": any(bool(item["structural_exact"]) for item in candidates),
            "execute_at_1": bool(candidates[0].get("execute_ok")) if condition != "direct" else False,
            "execute_at_k": any(bool(item.get("execute_ok")) for item in candidates) if condition != "direct" else False,
        }
    if len(outcomes) != len(references):
        raise ValueError(f"{condition} per-reaction outcome coverage mismatch")
    return outcomes, {
        "evaluation": str(evaluation.resolve()),
        "evaluation_sha256": _sha(evaluation),
        "row_evaluation_sha256": _sha(row_path),
        "test_sha256": manifest["output_sha256"]["test"],
        "n": len(outcomes),
    }


def _bootstrap_mean(value: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, Any]:
    n = len(value)
    estimates = np.empty(draws, dtype=float)
    for index in range(draws):
        estimates[index] = float(np.mean(value[rng.integers(0, n, size=n)]))
    return {"estimate": float(np.mean(value)),
            "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
            "unit": "reaction", "bootstrap_draws": draws}


def _adjusted_primitive_association(
    features: np.ndarray, outcome: np.ndarray, rng: np.random.Generator,
    *, draws: int,
) -> dict[str, Any]:
    if len(np.unique(outcome)) < 2:
        return {"status": "unidentifiable_single_outcome_class"}

    def fit(x, y):
        model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=500)
        model.fit(x, y)
        return float(model.coef_[0, 0])

    coefficient = fit(features, outcome)
    estimates = []
    for _ in range(draws):
        indices = rng.integers(0, len(outcome), size=len(outcome))
        if len(np.unique(outcome[indices])) < 2:
            continue
        estimates.append(fit(features[indices], outcome[indices]))
    if len(estimates) < max(20, draws // 2):
        return {"status": "bootstrap_outcome_too_sparse", "fit_log_odds_coefficient": coefficient}
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {
        "status": "estimated",
        "feature": "log2(1+minimum_primitive_train_frequency)",
        "adjusted_for": ["scaffold_seen", "local_center_fraction_seen", "near_duplicate",
                         "log2(1+trajectory_steps)", "fragment_imports"],
        "complete_program_novelty_control": "all H2 test program_train_frequency == 0",
        "regularization": "L2 C=10.0; descriptive adjusted association, not causal effect",
        "log_odds_coefficient": coefficient,
        "odds_ratio_per_unit": float(np.exp(coefficient)),
        "odds_ratio_ci95": [float(np.exp(lo)), float(np.exp(hi))],
        "reaction_bootstrap_draws_used": len(estimates),
    }


def analyze(
    covariates: Path, matched_dir: Path, evaluations: dict[str, Path], output: Path,
    *, k: int = 10, bootstrap_draws: int = 2000,
    regression_draws: int = 250, seed: int = 42,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing existing result: {output}")
    if k < 2 or bootstrap_draws < 100 or regression_draws < 20:
        raise ValueError("insufficient candidate/bootstrap budget")
    cov = {str(row["source_id"]): row for row in _jsonl(covariates)
           if row.get("split") == "test"}
    if not cov or len(cov) != sum(row.get("split") == "test" for row in _jsonl(covariates)):
        raise ValueError("duplicate/missing H2 test covariates")
    if any(int(row["program_train_frequency"]) != 0 or
           float(row["fraction_primitives_seen_in_train"]) != 1 for row in cov.values()):
        raise ValueError("H2 test no longer represents program-unseen/primitive-seen")
    outcomes = {}
    lineage = {}
    for condition in CONDITIONS:
        outcomes[condition], lineage[condition] = _row_results(
            condition, evaluations[condition], matched_dir, expected_k=k,
        )
        if set(outcomes[condition]) != set(cov):
            raise ValueError(f"{condition} differs from H2 test covariate IDs")
    ids = sorted(cov)
    rng = np.random.default_rng(seed)
    success = {
        condition: {
            metric: np.array([int(outcomes[condition][identifier][metric]) for identifier in ids], dtype=float)
            for metric in ("endpoint_at_1", "endpoint_at_k", "execute_at_1", "execute_at_k")
        } for condition in CONDITIONS
    }
    methods = {
        condition: {metric: _bootstrap_mean(values, rng, bootstrap_draws)
                    for metric, values in success[condition].items()
                    if condition != "direct" or not metric.startswith("execute")}
        for condition in CONDITIONS
    }
    contrasts = {}
    for other in ("open_flow", "direct"):
        contrasts[f"closed_minus_{other}"] = {
            metric: _bootstrap_mean(success["closed_loop"][metric] - success[other][metric],
                                    rng, bootstrap_draws)
            for metric in ("endpoint_at_1", "endpoint_at_k")
        }
    features = np.array([[
        np.log2(1 + float(cov[identifier]["minimum_primitive_train_frequency"])),
        float(bool(cov[identifier]["structural_overlap"]["murcko_scaffold_seen_in_train"])),
        float(cov[identifier]["structural_overlap"]["local_center_fraction_seen_in_train"]),
        float(bool(cov[identifier]["structural_overlap"]["near_duplicate_at_threshold"])),
        np.log2(1 + float(cov[identifier]["trajectory_steps"])),
        float(cov[identifier]["fragment_imports"]),
    ] for identifier in ids], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("nonfinite H2 regression covariate")
    adjusted = {
        condition: _adjusted_primitive_association(
            features, success[condition]["endpoint_at_1"], rng, draws=regression_draws,
        ) for condition in CONDITIONS
    }
    strata = {}
    for name, selector in (
        ("scaffold_seen", features[:, 1] == 1),
        ("scaffold_unseen", features[:, 1] == 0),
        ("local_center_any_seen", features[:, 2] > 0),
        ("local_center_none_seen", features[:, 2] == 0),
        ("near_duplicate", features[:, 3] == 1),
        ("not_near_duplicate", features[:, 3] == 0),
    ):
        strata[name] = {
            "n": int(selector.sum()),
            "endpoint_at_1": {
                condition: float(success[condition]["endpoint_at_1"][selector].mean())
                if selector.any() else None for condition in CONDITIONS
            },
        }
    report = {
        "artifact_type": "nmi_h2_matched_reaction_analysis_v1",
        "scope": "frozen composition-unseen primitive-seen H2 test; no model selection",
        "covariates_sha256": _sha(covariates),
        "lineage": lineage,
        "n_reactions": len(ids),
        "candidate_budget_k": k,
        "bootstrap_seed": seed,
        "methods": methods,
        "paired_contrasts": contrasts,
        "structural_strata": strata,
        "adjusted_primitive_familiarity_association": adjusted,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--matched-dir", type=Path, required=True)
    for condition in CONDITIONS:
        parser.add_argument(f"--{condition.replace('_', '-')}-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--regression-draws", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluations = {condition: getattr(args, f"{condition}_evaluation") for condition in CONDITIONS}
    print(json.dumps(analyze(
        args.covariates, args.matched_dir, evaluations, args.output,
        k=args.k, bootstrap_draws=args.bootstrap_draws,
        regression_draws=args.regression_draws, seed=args.seed,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
