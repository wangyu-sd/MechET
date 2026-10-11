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
              baseline: Path | None = None,
              official_source: Path | None = None,
              official_source_sha256: str | None = None) -> dict:
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
    hybrid_first_calls: dict[str, str] = {}
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
                if result.get("attempts"):
                    first = result["attempts"][0]
                    hybrid_first_calls[identifier] = json.dumps(
                        [first["name"], first["arguments"]], sort_keys=True,
                    )
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
        baseline_first_calls: dict[str, str] = {}
        for rank in range(world):
            path = baseline / f"results.shard-{rank:02d}-of-{world:02d}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    identifier = str(row["id"])
                    if identifier not in ids or identifier in baseline_hits or expected_rank[identifier] != rank:
                        raise ValueError(f"baseline has duplicate, wrong-shard or non-source ID: {identifier}")
                    baseline_hits[identifier] = bool(row["top1_structural_exact"])
                    if row.get("attempts"):
                        first = row["attempts"][0]
                        baseline_first_calls[identifier] = json.dumps(
                            [first["name"], first["arguments"]], sort_keys=True,
                        )
        if set(baseline_hits) != ids:
            raise ValueError("baseline and hybrid do not cover identical frozen IDs")
        report["paired_baseline"] = {
            "top1_structural_exact": sum(baseline_hits.values()),
            "top1_structural_rate": sum(baseline_hits.values()) / sample_reactions,
            "hybrid_only_correct": sum(hybrid_hits[key] and not baseline_hits[key] for key in ids),
            "baseline_only_correct": sum(baseline_hits[key] and not hybrid_hits[key] for key in ids),
            "both_correct": sum(hybrid_hits[key] and baseline_hits[key] for key in ids),
        }
        if len(hybrid_first_calls) == len(baseline_first_calls) == len(ids):
            different = sorted(
                key for key in ids
                if hybrid_first_calls[key] != baseline_first_calls[key]
            )
            report["paired_baseline"]["first_call_identical"] = len(ids) - len(different)
            report["paired_baseline"]["first_call_different_ids"] = different
    if official_denominator is not None:
        if official_denominator < sample_reactions:
            raise ValueError("official denominator cannot be smaller than strict denominator")
        report.update({
            "official_reaction_denominator": official_denominator,
            "upstream_corrupt_graph_unsupported": official_denominator - sample_reactions,
            "top1_structural_rate_official_with_unsupported_failures": exact / official_denominator,
            "note": "The unsupported upstream-corrupt rows are retained as failures, not graph predictions.",
        })
    if official_source is not None:
        if official_denominator is None or not official_source_sha256:
            raise ValueError("official source requires its complete denominator and SHA-256")
        if file_sha256(official_source) != official_source_sha256:
            raise ValueError("official source SHA-256 mismatch")
        official_rows: dict[str, dict] = {}
        with official_source.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                source_id = str(row["source_id"])
                if source_id in official_rows:
                    raise ValueError(f"duplicate official source ID: {source_id}")
                official_rows[source_id] = row
        if len(official_rows) != official_denominator:
            raise ValueError("official source row count differs from frozen denominator")
        strict_source_ids: set[str] = set()
        prefix = "flower_mech_proof_test_"
        for row in selected:
            source_id = str(row["source_id"])
            if not source_id.startswith(prefix):
                raise ValueError(f"strict test source ID has unexpected lineage: {source_id}")
            strict_source_ids.add(source_id[len(prefix):])
        if len(strict_source_ids) != sample_reactions or not strict_source_ids <= official_rows.keys():
            raise ValueError("strict test IDs do not map one-to-one into official test IDs")
        unsupported = sorted(set(official_rows) - strict_source_ids)
        if unsupported != ["PC", "PM", "RC", "RS"]:
            raise ValueError(f"unexpected official rows outside strict graph coverage: {unsupported}")
        report["official_source_sha256"] = official_source_sha256
        report["official_unsupported_source_ids"] = unsupported
        report["official_unsupported_records"] = [
            {"id": str(official_rows[source_id]["id"]), "source_id": source_id,
             "top1_structural_exact": False, "top_terminal": False,
             "status": "UPSTREAM_NONCONSERVING_ENDPOINT_GRAPH_UNSUPPORTED"}
            for source_id in unsupported
        ]
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
    parser.add_argument("--official-source", type=Path,
                        help="official 28,971-row endpoint test for explicit four-row coverage")
    parser.add_argument("--official-source-sha256")
    args = parser.parse_args()
    report = aggregate(
        source=args.source, output=args.output,
        sample_reactions=args.sample_reactions, seed=args.seed,
        world=args.world, source_sha256=args.source_sha256,
        official_denominator=args.official_denominator,
        baseline=args.baseline,
        official_source=args.official_source,
        official_source_sha256=args.official_source_sha256,
    )
    unsupported_records = report.pop("official_unsupported_records", [])
    if args.official_source is not None:
        (args.output / "official_unsupported_rows.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unsupported_records),
            encoding="utf-8",
        )
    path = args.output / "summary.all-shards.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
