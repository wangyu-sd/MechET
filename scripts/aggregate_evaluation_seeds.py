#!/usr/bin/env python3
"""Aggregate independent-seed H1 or H3 evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.statistical_evaluation import aggregate_seed_effects


def _seed_from_runtime(value: dict[str, Any]) -> str:
    runtime = dict(value.get("runtime_contracts") or {})
    seeds: set[str] = set()
    for contract_summary in runtime.values():
        for contract in contract_summary.get("runtime_contracts") or []:
            seed = contract.get("seed")
            if seed not in (None, ""):
                seeds.add(str(seed))
    if len(seeds) != 1:
        raise ValueError(
            f"evaluation must contain exactly one global seed, observed {sorted(seeds)}"
        )
    return next(iter(seeds))


def _effects(value: dict[str, Any]) -> dict[str, float]:
    hypothesis = str(value.get("scientific_hypothesis") or "")
    if hypothesis == "H1_causal_faithfulness":
        output = {}
        for name, row in dict(value.get("paired_effects") or {}).items():
            effect = row.get("structural_exact_delta_normal_minus_intervention")
            if effect is not None:
                output[str(name)] = float(effect)
        return output
    if hypothesis == "H3_empirical_evidence_separation":
        output = {}
        for name, row in dict(value.get("paired_contrasts") or {}).items():
            if row is None:
                continue
            effect = row.get("delta_left_minus_right")
            if effect is not None:
                output[str(name)] = float(effect)
        return output
    raise ValueError(f"unsupported scientific_hypothesis: {hypothesis!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--minimum-effect", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=43)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    artifacts: list[dict[str, Any]] = []
    hypotheses: set[str] = set()
    effects_by_name: dict[str, dict[str, float]] = {}
    source_rows: list[dict[str, Any]] = []
    seen_seeds: set[str] = set()
    for path in args.evaluation:
        if not path.exists():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"evaluation is not a JSON object: {path}")
        seed = _seed_from_runtime(value)
        if seed in seen_seeds:
            raise ValueError(f"duplicate evaluation seed: {seed}")
        seen_seeds.add(seed)
        hypothesis = str(value.get("scientific_hypothesis") or "")
        hypotheses.add(hypothesis)
        extracted = _effects(value)
        if not extracted:
            raise ValueError(f"evaluation contains no paired effects: {path}")
        for name, effect in extracted.items():
            effects_by_name.setdefault(name, {})[seed] = effect
        artifacts.append(value)
        source_rows.append(
            {
                "path": str(path),
                "seed": seed,
                "scientific_hypothesis": hypothesis,
                "artifact_type": value.get("artifact_type"),
            }
        )
    if len(hypotheses) != 1:
        raise ValueError(f"cannot mix hypotheses: {sorted(hypotheses)}")

    expected_effects = set(effects_by_name)
    for name, values in effects_by_name.items():
        if set(values) != seen_seeds:
            raise ValueError(
                f"effect {name!r} is missing seeds: {sorted(seen_seeds - set(values))}"
            )
    summaries = {
        name: aggregate_seed_effects(
            values,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed + index * 17,
            confidence=args.confidence,
        )
        for index, (name, values) in enumerate(sorted(effects_by_name.items()))
    }
    gates = {
        name: {
            "minimum_seed_count_met": int(summary["n_seeds"]) >= args.minimum_seeds,
            "bootstrap_ci_lower_exceeds_minimum_effect": float(
                summary["seed_bootstrap_ci"][0]
            )
            >= args.minimum_effect,
            "effect_direction_consistent_across_seeds": float(
                summary["positive_seed_fraction"]
            )
            == 1.0,
        }
        for name, summary in summaries.items()
    }
    result = {
        "artifact_type": "multi_seed_evaluation_aggregate",
        "scientific_hypothesis": next(iter(hypotheses)),
        "sources": source_rows,
        "n_seeds": len(seen_seeds),
        "seeds": sorted(seen_seeds),
        "effect_names": sorted(expected_effects),
        "effects": summaries,
        "claim_gates": gates,
        "minimum_seeds": args.minimum_seeds,
        "minimum_effect": args.minimum_effect,
        "confidence": args.confidence,
        "interpretation": (
            "Seed-level confidence intervals quantify variation across independent "
            "training runs. They complement, rather than replace, paired target-level "
            "bootstrap intervals and McNemar tests inside each evaluation artifact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
