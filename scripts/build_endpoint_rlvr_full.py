#!/usr/bin/env python3
"""Build the full strict-executable Endpoint Process RLVR train artifact.

The source is intentionally streamed and the converted rows are written into
rank-local shards.  This avoids materialising either the 5+ GiB source or the
full private-reward artifact in Python memory on every training rank.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from build_endpoint_rlvr_pilot import build_pilot_record, file_sha256
from mechet.chemical_runtime import require_endpoint_process_rdkit


EXPECTED_TRAIN_ROWS = 257_167


def _convert_raw(raw: bytes) -> tuple[str, dict[str, Any]]:
    source = json.loads(raw)
    identifier = str(source.get("id") or "")
    try:
        record = build_pilot_record(source)
        record["artifact_type"] = "endpoint_process_rlvr_full_strict_v1"
        return identifier, record
    except Exception as exc:
        raise ValueError(f"{identifier}: endpoint RL record build failed") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-source",
        type=Path,
        default=REPO / "data/flower_inverse_tool_sft_action_delta_v1/train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data/endpoint_process_rlvr_full_strict_v1",
    )
    parser.add_argument(
        "--parent-adapter",
        type=Path,
        default=REPO
        / "outputs/agent/in_place_grounded_flow_qwen3_8b_a100_seed17_20260911/checkpoint-8037",
    )
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_TRAIN_ROWS)
    args = parser.parse_args()
    if args.num_shards < 1 or args.workers < 1 or args.chunksize < 1:
        raise ValueError("num-shards, workers and chunksize must be positive")

    runtime_version = require_endpoint_process_rdkit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = args.output_dir / "_SUCCESS"
    if complete.is_file():
        manifest = json.loads((args.output_dir / "manifest.json").read_text())
        if int(manifest["train"]["rows"]) != args.expected_rows:
            raise ValueError("existing full artifact has the wrong denominator")
        print(
            json.dumps(
                {"type": "full_build_reuse", "rows": args.expected_rows, "path": str(args.output_dir)}
            ),
            flush=True,
        )
        return 0

    paths = [args.output_dir / f"train.rank{rank:02d}.jsonl" for rank in range(args.num_shards)]
    temporary = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    handles = [path.open("w", encoding="utf-8") for path in temporary]
    shard_rows = [0] * args.num_shards
    shard_ids = [hashlib.sha256() for _ in range(args.num_shards)]
    strata: Counter[str] = Counter()
    seen_ids: set[str] = set()
    rows = 0
    try:
        # Fork preserves the already-imported RDKit runtime and is materially
        # faster than spawning it afresh for every worker on Linux build hosts.
        context = mp.get_context("fork")
        with args.train_source.open("rb") as source, context.Pool(args.workers) as pool:
            raw_rows = (raw for raw in source if raw.strip())
            for index, (identifier, record) in enumerate(
                pool.imap(_convert_raw, raw_rows, chunksize=args.chunksize)
            ):
                if not identifier or identifier in seen_ids:
                    raise ValueError(f"empty or duplicate source id at row {index + 1}: {identifier!r}")
                seen_ids.add(identifier)
                rank = index % args.num_shards
                handles[rank].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                shard_rows[rank] += 1
                shard_ids[rank].update((identifier + "\n").encode())
                strata[str(record["metadata"]["event_length_stratum"])] += 1
                rows += 1
                if rows % 1_000 == 0:
                    print(
                        json.dumps(
                            {
                                "type": "full_build_progress",
                                "rows": rows,
                                "expected": args.expected_rows,
                                "percent": round(100.0 * rows / args.expected_rows, 3),
                            }
                        ),
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.close()

    if rows != args.expected_rows:
        raise ValueError(f"source row count mismatch: {rows} != {args.expected_rows}")
    for source, target in zip(temporary, paths, strict=True):
        source.replace(target)

    adapter_weights = args.parent_adapter / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(f"missing parent adapter: {adapter_weights}")
    manifest = {
        "artifact_type": "endpoint_process_rlvr_full_strict_manifest_v1",
        "dataset_name": "FlowER strict-executable train universe",
        "official_flowER_full_train_rows": 257_171,
        "strict_executable_train_rows": rows,
        "excluded_corrupt_endpoint_rows": 4,
        "no_additional_filtering": True,
        "train": {
            "rows": rows,
            "source": str(args.train_source),
            "source_sha256": file_sha256(args.train_source),
            "num_shards": args.num_shards,
            "shards": [
                {
                    "rank": rank,
                    "path": str(path),
                    "rows": shard_rows[rank],
                    "ids_sha256": shard_ids[rank].hexdigest(),
                    "file_sha256": file_sha256(path),
                }
                for rank, path in enumerate(paths)
            ],
            "strata": dict(sorted(strata.items())),
        },
        "parent": {
            "adapter": str(args.parent_adapter),
            "adapter_model_sha256": file_sha256(adapter_weights),
        },
        "reward_contract": "endpoint_grounded_process_rlvr_v1",
        "chemical_runtime": {"rdkit": runtime_version, "minimum": "2026.03.4"},
        "gold_model_visible": False,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete.write_text(file_sha256(manifest_path) + "\n")
    print(json.dumps({"type": "full_build_complete", **manifest}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
