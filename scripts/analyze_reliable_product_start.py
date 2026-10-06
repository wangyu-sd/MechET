#!/usr/bin/env python3
"""Analyze matched product-start rollouts without treating GT-path mismatch as chemistry truth."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.run_natural_language_value_search import normal_smiles, read_selected


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _gold_successor(row: Mapping[str, Any]) -> str:
    messages = list(row["messages"])
    result = json.loads(str(messages[3]["content"]))
    state = str(result.get("current_state") or result.get("derived_precursor") or "")
    if not state:
        raise ValueError(f"missing reference successor: {row['id']}")
    return normal_smiles(state)


def _rejection_category(name: str, error: str) -> str:
    if name == "import_fragments":
        return "fragment_proposal"
    if name == "finish_trace":
        return "termination"
    if name != "apply_electron_flow":
        return "generation_or_budget"
    grounding_markers = (
        "unrecognized natural-language", "electron-source bond is absent",
        "destination bond is absent", "invalid atom container",
        "invalid lone-pair container", "event direction must be",
    )
    if any(marker in error.lower() for marker in grounding_markers):
        return "source_sink_grounding"
    return "formal_execution"


def first_reference_divergence(
    result: Mapping[str, Any], gold: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Find the first *reference-relative* divergence, not a chemical verdict."""

    attempts = list(result.get("attempts") or [])
    if not attempts:
        raise ValueError(f"{result['id']}: no recorded product-start attempts")
    if any(int(item["depth"]) != index for index, item in enumerate(attempts)):
        raise ValueError(f"{result['id']}: expected one ordered K=1 attempt per depth")
    for depth, attempt in enumerate(attempts):
        name = str(attempt["name"])
        error = str(attempt.get("error") or "")
        if not bool(attempt["accepted"]):
            return {
                "depth": depth, "category": _rejection_category(name, error),
                "kind": "rejected_or_unparseable", "error": error,
            }
        if depth >= len(gold):
            return {
                "depth": depth, "category": "termination",
                "kind": "continued_after_reference_finish", "error": "",
            }
        gold_name = str(gold[depth]["messages"][2]["tool_calls"][0]["function"]["name"])
        expected_state = _gold_successor(gold[depth])
        observed_state = normal_smiles(str(attempt.get("state_after") or ""))
        expected_terminal = gold_name == "finish_trace"
        if observed_state != expected_state or bool(attempt["terminal"]) != expected_terminal:
            category = (
                "fragment_proposal" if name == "import_fragments"
                else "termination" if name == "finish_trace" or expected_terminal
                else "executable_wrong_successor"
            )
            return {
                "depth": depth, "category": category,
                "kind": "reference_successor_mismatch",
                "error": "", "expected_action": gold_name,
                "observed_action": name,
            }
    if len(attempts) < len(gold):
        return {
            "depth": len(attempts), "category": "termination",
            "kind": "stopped_before_reference_finish", "error": "",
        }
    return {"depth": None, "category": None, "kind": "reference_aligned", "error": ""}


def analyze(
    *, source: Path, decisions: Path, results: Path,
    sample_reactions: int, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = read_selected(source, sample_reactions, seed)
    by_source = {str(row["source_id"]): row for row in selected}
    if len(by_source) != sample_reactions:
        raise ValueError("selected source reactions are not unique")
    gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(decisions):
        source_id = str(row["source_id"])
        if source_id in by_source:
            gold[source_id].append(row)
    if set(gold) != set(by_source):
        raise ValueError("reference decisions do not cover selected reactions")
    for source_id, rows in gold.items():
        rows.sort(key=lambda row: int(row["metadata"]["decision_index"]))
        indices = [int(row["metadata"]["decision_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"{source_id}: noncontiguous reference decisions")

    result_rows = list(_rows(results))
    if len(result_rows) != sample_reactions:
        raise ValueError("incomplete product-start result denominator")
    observed = [str(row["source_id"]) for row in result_rows]
    if len(set(observed)) != len(observed) or set(observed) != set(by_source):
        raise ValueError("product-start result IDs disagree with frozen selection")

    cases: list[dict[str, Any]] = []
    for row in result_rows:
        source_id = str(row["source_id"])
        divergence = first_reference_divergence(row, gold[source_id])
        exact = bool(row["top1_exact"])
        terminal = bool(row["top_terminal"])
        raw_score = float(row["top_policy_score"])
        policy_score = (
            raw_score if terminal and int(row["n_actions"]) > 0
            and math.isfinite(raw_score) else None
        )
        cases.append({
            "id": str(row["id"]), "source_id": source_id,
            "endpoint_exact": exact, "terminal": terminal,
            "policy_score": policy_score,
            "accepted_actions": int(row["n_actions"]),
            "reference_decisions": len(gold[source_id]),
            "first_reference_divergence": divergence,
            "failure_category": divergence["category"] if not exact else None,
            "recovered_after_reference_divergence": bool(
                exact and divergence["depth"] is not None
            ),
        })
    failure_categories = Counter(
        case["failure_category"] for case in cases if case["failure_category"]
    )
    sorted_cases = sorted(
        cases,
        key=lambda case: (
            case["policy_score"] is not None,
            case["policy_score"] if case["policy_score"] is not None else 0.0,
        ),
        reverse=True,
    )
    risk_coverage = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        n = max(1, int(len(sorted_cases) * fraction))
        subset = sorted_cases[:n]
        risk_coverage.append({
            "coverage": n / len(sorted_cases), "n": n,
            "endpoint_miss_rate": sum(not case["endpoint_exact"] for case in subset) / n,
            "min_policy_score": subset[-1]["policy_score"],
        })
    report = {
        "artifact_type": "reliable_mechet_product_start_diagnostic_v1",
        "denominator": sample_reactions, "seed": seed,
        "endpoint_exact": sum(case["endpoint_exact"] for case in cases),
        "terminal": sum(case["terminal"] for case in cases),
        "terminal_endpoint_wrong": sum(
            case["terminal"] and not case["endpoint_exact"] for case in cases
        ),
        "nonterminal": sum(not case["terminal"] for case in cases),
        "recovered_after_reference_divergence": sum(
            case["recovered_after_reference_divergence"] for case in cases
        ),
        "first_failure_category_counts": dict(failure_categories),
        "confidence_policy": "terminal_finite_policy_score_only; nonterminal_abstains_first",
        "risk_coverage_by_policy_score": risk_coverage,
        "interpretation": (
            "Reference-path divergence is not a chemical falsehood. "
            "Only terminal endpoint match and executor acceptance are direct observations; "
            "non-reference precursors require independent chemical adjudication."
        ),
    }
    return report, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-reactions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report, cases = analyze(
        source=args.source, decisions=args.decisions, results=args.results,
        sample_reactions=args.sample_reactions, seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "failure_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "failure_analysis.cases.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
