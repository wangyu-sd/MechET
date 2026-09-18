#!/usr/bin/env python3
"""Aggregate completed shards from a grounded constrained-search audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(records: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    values = [dict(item[method]) for item in records]
    n = len(values)
    return {
        "n": n,
        "endpoint_pass_at_1": sum(item["endpoint_pass_at_1"] for item in values) / n,
        "endpoint_pass_oracle_at_budget": sum(item["endpoint_pass_oracle_at_budget"] for item in values) / n,
        "explicit_finish_rate": sum(item["explicit_finish"] for item in values) / n,
        "mean_generated_tokens": sum(item["generated_tokens"] for item in values) / n,
        "mean_responses": sum(item["responses"] for item in values) / n,
        "mean_latency_seconds": sum(item["latency_seconds"] for item in values) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = sorted(args.directory.glob("summary.shard-*.json"))
    predictions = sorted(args.directory.glob("predictions.shard-*.jsonl"))
    if not summaries or len(summaries) != len(predictions):
        raise ValueError("shard summary/prediction files are missing or unaligned")
    metadata = [json.loads(path.read_text()) for path in summaries]
    expected_count = int(metadata[0]["shard_count"])
    if len(metadata) != expected_count:
        raise ValueError(f"expected {expected_count} completed shards, found {len(metadata)}")
    if {int(item["shard_index"]) for item in metadata} != set(range(expected_count)):
        raise ValueError("shard indices are incomplete")
    for key in (
        "global_subset_id_sha256",
        "global_subset_size",
        "checkpoint",
        "adapter_model_sha256",
        "base_model_revision",
        "data_sha256",
        "gold_source_data_sha256",
        "budget_contract",
    ):
        values = {json.dumps(item[key], sort_keys=True) for item in metadata}
        if len(values) != 1:
            raise ValueError(f"shard metadata mismatch: {key}")
    records = [row for path in predictions for row in read_jsonl(path)]
    identifiers = [str(item["id"]) for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate IDs across shards")
    if len(records) != int(metadata[0]["global_subset_size"]):
        raise ValueError("aggregated row count does not match frozen subset")
    summary = {
        "artifact_type": "grounded_executor_constrained_search_audit_v1_aggregate",
        "checkpoint": metadata[0]["checkpoint"],
        "adapter_model_sha256": metadata[0]["adapter_model_sha256"],
        "base_model_revision": metadata[0]["base_model_revision"],
        "data_sha256": metadata[0]["data_sha256"],
        "gold_source_data_sha256": metadata[0]["gold_source_data_sha256"],
        "global_subset_id_sha256": metadata[0]["global_subset_id_sha256"],
        "global_subset_size": metadata[0]["global_subset_size"],
        "budget_contract": metadata[0]["budget_contract"],
        "product_only_model_input": True,
        "gold_used_for_generation_pruning_or_ranking": False,
        "independent": aggregate(records, "independent"),
        "search": aggregate(records, "search"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
