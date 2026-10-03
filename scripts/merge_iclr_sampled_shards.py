#!/usr/bin/env python3
"""Losslessly merge completed, modulo-sharded K-sample prediction files.

The reference is used only to verify the frozen ID/order contract. Its answer
fields are never read by this script, and no candidate is selected here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--generation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--shards", required=True, type=int)
    parser.add_argument("--evaluation-scope", required=True)
    parser.add_argument("--prediction-mode", choices=("direct", "trace"), default="direct")
    parser.add_argument("--shard-layout", choices=("modulo", "prepartitioned"), default="modulo")
    parser.add_argument("--manifest-data-sha256", default="")
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--direct-sample-batch-size", type=int, default=1)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if min(args.expected_rows, args.k, args.shards) < 1:
        parser.error("expected rows, K and shards must be positive")
    if args.task_shard_count < 1 or not 0 <= args.task_shard_index < args.task_shard_count:
        parser.error("task shard index must be in [0, task shard count)")
    reference_sha = sha256_file(args.reference)
    manifest_data_sha = args.manifest_data_sha256 or reference_sha
    shard_paths = [
        args.generation_dir / f"predictions.shard-{index:03d}.jsonl"
        for index in range(args.shards)
    ]
    if sorted(args.generation_dir.glob("predictions.shard-*.jsonl")) != shard_paths:
        raise ValueError("prediction shard names/count differ from frozen shard count")
    manifests = []
    for index, path in enumerate(shard_paths):
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_shard_index = index if args.shard_layout == "modulo" else 0
        expected_shard_count = args.shards if args.shard_layout == "modulo" else 1
        if (
            int(manifest["shard_index"]) != expected_shard_index
            or int(manifest["shard_count"]) != expected_shard_count
        ):
            raise ValueError(f"invalid shard assignment: {manifest_path}")
        if manifest["data_sha256"] != manifest_data_sha:
            raise ValueError(f"reference hash mismatch: {manifest_path}")
        completed = int(manifest["n_predictions_written"]) + int(
            manifest.get("n_predictions_skipped_by_resume") or 0
        )
        if completed != len(range(index, args.expected_rows, args.shards)):
            raise ValueError(f"incomplete shard: {manifest_path}")
        if manifest["mode"] != args.prediction_mode:
            raise ValueError(f"unexpected inference mode: {manifest_path}")
        manifests.append(manifest)
    for key in ("condition_name", "model_revision", "tokenizer_revision", "adapter_sha256", "seed", "backend"):
        if len({json.dumps(manifest.get(key), sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"inconsistent shard {key}")

    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "predictions.jsonl"
    temporary = args.output_dir / "predictions.jsonl.merge-in-progress"
    if not args.validate_only and (output.exists() or temporary.exists()):
        raise FileExistsError("merged prediction output already exists; refusing to overwrite")
    sources = [path.open(encoding="utf-8") for path in shard_paths]
    digest = hashlib.sha256()
    count = 0
    try:
        sink_context = open(os.devnull, "wb") if args.validate_only else temporary.open("wb")
        with args.reference.open(encoding="utf-8") as reference, sink_context as sink:
            for index, reference_line in enumerate(reference):
                if not reference_line.strip():
                    raise ValueError(f"blank reference row at {index}")
                identifier = json.loads(reference_line)["id"]
                source = sources[index % args.shards]
                prediction_line = source.readline()
                if not prediction_line:
                    raise ValueError(f"missing shard row at reference index {index}")
                row = json.loads(prediction_line)
                if row.get("id") != identifier:
                    raise ValueError(f"shard/reference ID mismatch at {index}")
                if row.get("prediction_mode") != args.prediction_mode:
                    raise ValueError(f"{identifier}: prediction mode mismatch")
                model = row.get("model") or {}
                shard_manifest = manifests[index % args.shards]
                if (
                    model.get("adapter_sha256") != shard_manifest["adapter_sha256"]
                    or model.get("model_revision") != shard_manifest["model_revision"]
                    or model.get("seed") != shard_manifest["seed"]
                    or int(model.get("samples_per_target") or 0) != args.k
                ):
                    raise ValueError(f"{identifier}: row/model lineage differs from shard manifest")
                candidates = row.get("candidates") or []
                if len(candidates) != args.k:
                    raise ValueError(f"{identifier}: {len(candidates)} candidates, expected {args.k}")
                if sorted(int(candidate["sample_index"]) for candidate in candidates) != list(range(args.k)):
                    raise ValueError(f"{identifier}: incomplete sample indices")
                encoded = prediction_line.encode("utf-8")
                sink.write(encoded)
                digest.update(encoded)
                count += 1
            for index, source in enumerate(sources):
                if source.readline():
                    raise ValueError(f"extra predictions in shard {index}")
        if count != args.expected_rows:
            raise ValueError(f"merged {count} rows, expected {args.expected_rows}")
        if not args.validate_only:
            os.replace(temporary, output)
    except Exception:
        if not args.validate_only:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        for source in sources:
            source.close()

    report = {
        "n_targets": count,
        "samples_per_target": args.k,
        "n_candidates": count * args.k,
        "predictions": str(output.resolve()),
        "predictions_sha256": digest.hexdigest(),
        "reference_sha256": reference_sha,
        "shard_data_sha256": manifest_data_sha,
        "shard_adapter_sha256": manifests[0]["adapter_sha256"],
        "shard_model_revision": manifests[0]["model_revision"],
    }
    if args.prediction_mode == "trace":
        report.update({
            "artifact_type": (
                "flower_a7_compact_full_state_sampled_test_manifest"
                if args.evaluation_scope == "full_official_test"
                else "nmi_h2_closed_loop_sampled_test_manifest"
            ),
            "paper_condition": "A7",
            "headline_eligible": True,
            "task_shard_count": args.task_shard_count,
            "task_shard_index": args.task_shard_index,
            "candidate_selection": "formal-execution/reward rank; no ground truth used",
        })
    else:
        report.update({
            "artifact_type": (
                "iclr_full_baseline_sampled_test_manifest"
                if args.evaluation_scope == "full_official_test"
                else "nmi_h2_sampled_test_manifest"
            ),
            "paper_status": (
                "full-test evaluation; compare methods on the shared 28,967-ID universe"
                if args.evaluation_scope == "full_official_test"
                else "frozen Issue #79 H2 composition-heldout evaluation"
            ),
            "baseline": args.baseline,
            "candidate_semantics": "stochastic samples; generation-order Success@K and separately frozen ranking",
            "direct_sample_batch_size": args.direct_sample_batch_size,
        })
    if not args.validate_only:
        (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        report["validation_only"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
