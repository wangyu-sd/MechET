#!/usr/bin/env python3
"""Audit and aggregate all distributed CRN/Qwen product-start result shards.

The strict executable FlowER denominator is 28,967. The official reaction
denominator is 28,971; the four upstream-corrupt endpoints are explicitly
unsupported by this graph-FLOW condition, not silently removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechet.crn_graph_bridge import file_sha256
from scripts.run_natural_language_value_search import read_selected


def aggregate(*, source: Path, output: Path, sample_reactions: int, seed: int,
              world: int, source_sha256: str, official_denominator: int | None = None,
              baseline: Path | None = None) -> dict:
    if world < 1 or sample_reactions < 1:
        raise ValueError("invalid aggregation size")
    if file_sha256(source) != source_sha256:
        raise ValueError("evaluation source SHA-256 mismatch")
    selected = read_selected(source, sample_reactions, seed)
    if len(selected) != sample_reactions:
        raise ValueError("selected source count differs from requested denominator")
    ids = {str(row["id"]) for row in selected}
    if len(ids) != sample_reactions:
        raise ValueError("duplicate selected source IDs")
    expected_rank = {str(row["id"]): index % world for index, row in enumerate(selected)}

    seen: set[str] = set()
    hybrid_hits: dict[str, bool] = {}
    exact = 0
    terminal = 0
    rejected: dict[str, int] = {}
    for rank in range(world):
        contract_path = output / f"contract.shard-{rank:02d}-of-{world:02d}.json"
        result_path = output / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (contract["source"] != str(source.resolve()) or
                contract["sample_reactions"] != sample_reactions or
                contract["seed"] != seed or contract["rank"] != rank or
                contract["world"] != world or
                contract["source_sha256"] != source_sha256):
            raise ValueError(f"shard {rank} violates the frozen source/selection contract")
        with result_path.open(encoding="utf-8") as handle:
            for line in handle:
                result = json.loads(line)
                identifier = str(result["id"])
                if identifier not in ids or identifier in seen or expected_rank[identifier] != rank:
                    raise ValueError(f"duplicate, wrong-shard or non-source result ID: {identifier}")
                seen.add(identifier)
                hit = bool(result["top1_structural_exact"])
                hybrid_hits[identifier] = hit
                exact += hit
                terminal += bool(result["top_terminal"])
                for code, count in result.get("rejected", {}).items():
                    rejected[code] = rejected.get(code, 0) + int(count)
    missing = sorted(ids - seen)
    if missing:
        raise ValueError(f"incomplete evaluation: {len(missing)} source IDs missing; first={missing[0]}")
    report = {
        "evaluated_strict_executable": sample_reactions,
        "top1_structural_exact": exact,
        "top1_structural_rate_strict": exact / sample_reactions,
        "terminal": terminal,
        "terminal_rate_strict": terminal / sample_reactions,
        "rejected": rejected,
        "source_sha256": source_sha256,
        "seed": seed,
        "world": world,
    }
    if baseline is not None:
        baseline_hits: dict[str, bool] = {}
        for rank in range(world):
            path = baseline / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    identifier = str(row["id"])
                    if identifier not in ids or identifier in baseline_hits or expected_rank[identifier] != rank:
                        raise ValueError(f"baseline has duplicate, wrong-shard or non-source ID: {identifier}")
                    baseline_hits[identifier] = bool(row["top1_structural_exact"])
        if set(baseline_hits) != ids:
            raise ValueError("baseline and hybrid do not cover identical frozen IDs")
        report["paired_baseline"] = {
            "top1_structural_exact": sum(baseline_hits.values()),
            "top1_structural_rate": sum(baseline_hits.values()) / sample_reactions,
            "hybrid_only_correct": sum(hybrid_hits[key] and not baseline_hits[key] for key in ids),
            "baseline_only_correct": sum(baseline_hits[key] and not hybrid_hits[key] for key in ids),
            "both_correct": sum(hybrid_hits[key] and baseline_hits[key] for key in ids),
        }
    if official_denominator is not None:
        if official_denominator < sample_reactions:
            raise ValueError("official denominator cannot be smaller than strict denominator")
        report.update({
            "official_reaction_denominator": official_denominator,
            "upstream_corrupt_graph_unsupported": official_denominator - sample_reactions,
            "top1_structural_rate_official_with_unsupported_failures": exact / official_denominator,
            "note": "The unsupported upstream-corrupt rows are retained as failures, not graph predictions.",
        })
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-reactions", type=int, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--official-denominator", type=int)
    parser.add_argument("--baseline", type=Path,
                        help="matched pure-Qwen sharded results for a paired comparison")
    args = parser.parse_args()
    report = aggregate(
        source=args.source, output=args.output,
        sample_reactions=args.sample_reactions, seed=args.seed,
        world=args.world, source_sha256=args.source_sha256,
        official_denominator=args.official_denominator,
        baseline=args.baseline,
    )
    path = args.output / "summary.all-shards.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
