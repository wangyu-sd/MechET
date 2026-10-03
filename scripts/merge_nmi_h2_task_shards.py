#!/usr/bin/env python3
"""Merge complete H2 Closed-Loop task shards without choosing candidates.

The frozen reference supplies only row IDs and order. Endpoint labels are not
read, and all K generated candidates are copied byte-for-byte from the task
outputs. This is optional runtime glue for multiple independent Taiji tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_task_shards(
    reference: Path,
    task_outputs: list[Path],
    output_dir: Path,
    *,
    expected_rows: int,
    k: int,
    task_shard_count: int,
    validate_only: bool = False,
) -> dict[str, Any]:
    if expected_rows < 1 or k < 1 or task_shard_count < 2:
        raise ValueError("expected rows and K must be positive; require at least two task shards")
    if len(task_outputs) != task_shard_count:
        raise ValueError("require exactly one output directory per task shard")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    full_reference_sha = _sha(reference)

    shards: dict[int, tuple[Path, dict[str, Any]]] = {}
    for directory in task_outputs:
        manifest_path = directory / "manifest.json"
        prediction_path = directory / "predictions.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_type") != "nmi_h2_closed_loop_sampled_test_manifest":
            raise ValueError(f"not a frozen H2 Closed-Loop task shard: {manifest_path}")
        if int(manifest.get("task_shard_count") or 0) != task_shard_count:
            raise ValueError(f"task shard count mismatch: {manifest_path}")
        index = int(manifest.get("task_shard_index", -1))
        if not 0 <= index < task_shard_count or index in shards:
            raise ValueError(f"duplicate/invalid task shard index: {manifest_path}")
        expected_shard_rows = len(range(index, expected_rows, task_shard_count))
        if (int(manifest.get("n_targets") or 0) != expected_shard_rows
                or int(manifest.get("samples_per_target") or 0) != k
                or int(manifest.get("n_candidates") or 0) != expected_shard_rows * k
                or manifest.get("shard_data_sha256") != full_reference_sha):
            raise ValueError(f"incomplete or wrong-source task shard: {manifest_path}")
        if _sha(prediction_path) != manifest.get("predictions_sha256"):
            raise ValueError(f"task prediction SHA mismatch: {prediction_path}")
        shards[index] = (prediction_path, manifest)
    if set(shards) != set(range(task_shard_count)):
        raise ValueError("missing task shard index")
    for field in ("shard_adapter_sha256", "shard_model_revision", "samples_per_target"):
        if len({json.dumps(shards[index][1].get(field), sort_keys=True)
                for index in shards}) != 1:
            raise ValueError(f"task shards differ in {field}")

    output = output_dir / "predictions.jsonl"
    temporary = output_dir / "predictions.jsonl.merge-in-progress"
    manifest_output = output_dir / "manifest.json"
    if not validate_only:
        if any(path.exists() for path in (output, temporary, manifest_output)):
            raise FileExistsError("refusing to overwrite an existing merged artifact")
        output_dir.mkdir(parents=True, exist_ok=True)
    handles = {index: shards[index][0].open("rb") for index in shards}
    reference_digests = {index: hashlib.sha256() for index in shards}
    digest = hashlib.sha256()
    count = 0
    first_model: str | None = None
    try:
        sink = open(os.devnull, "wb") if validate_only else temporary.open("wb")
        with reference.open("rb") as source, sink:
            for row_index, reference_line in enumerate(source):
                if not reference_line.strip():
                    raise ValueError(f"blank reference row at {row_index}")
                if row_index >= expected_rows:
                    raise ValueError("reference has extra rows")
                shard_index = row_index % task_shard_count
                reference_digests[shard_index].update(reference_line)
                reference_row = json.loads(reference_line)
                prediction_line = handles[shard_index].readline()
                if not prediction_line:
                    raise ValueError(f"missing prediction at reference row {row_index}")
                row = json.loads(prediction_line)
                if row.get("id") != reference_row.get("id") or row.get("prediction_mode") != "trace":
                    raise ValueError(f"prediction/reference ID or mode mismatch at row {row_index}")
                if (row.get("source_id") is not None and reference_row.get("source_id") is not None
                        and row["source_id"] != reference_row["source_id"]):
                    raise ValueError(f"source ID mismatch at row {row_index}")
                candidates = row.get("candidates")
                if not isinstance(candidates, list) or len(candidates) != k:
                    raise ValueError(f"wrong K at row {row_index}")
                if sorted(candidate.get("sample_index") for candidate in candidates) != list(range(k)):
                    raise ValueError(f"incomplete candidate indices at row {row_index}")
                model = row.get("model") or {}
                manifest = shards[shard_index][1]
                if (model.get("adapter_sha256") != manifest["shard_adapter_sha256"]
                        or model.get("model_revision") != manifest["shard_model_revision"]
                        or int(model.get("samples_per_target") or 0) != k):
                    raise ValueError(f"row/model lineage mismatch at row {row_index}")
                serialized_model = json.dumps(model, sort_keys=True, separators=(",", ":"))
                if first_model is None:
                    first_model = serialized_model
                elif serialized_model != first_model:
                    raise ValueError(f"inference runtime differs at row {row_index}")
                sink.write(prediction_line)
                digest.update(prediction_line)
                count += 1
            if count != expected_rows:
                raise ValueError(f"reference has {count} rows, expected {expected_rows}")
            for index, handle in handles.items():
                if handle.readline():
                    raise ValueError(f"extra prediction in task shard {index}")
                if reference_digests[index].hexdigest() != shards[index][1].get("reference_sha256"):
                    raise ValueError(f"selected-reference SHA mismatch in task shard {index}")
        if not validate_only:
            os.replace(temporary, output)
    except Exception:
        if not validate_only:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        for handle in handles.values():
            handle.close()

    report = {
        "artifact_type": "nmi_h2_closed_loop_cross_task_merge_v1",
        "n_targets": count,
        "samples_per_target": k,
        "n_candidates": count * k,
        "task_shard_count": task_shard_count,
        "task_outputs": [str(shards[index][0].parent.resolve()) for index in shards],
        "predictions": str(output.resolve()),
        "predictions_sha256": digest.hexdigest(),
        "reference_sha256": full_reference_sha,
        "shard_adapter_sha256": shards[0][1]["shard_adapter_sha256"],
        "shard_model_revision": shards[0][1]["shard_model_revision"],
        "candidate_policy": "all generated candidates preserved in original reaction/candidate order",
        "validation_only": validate_only,
    }
    if not validate_only:
        manifest_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--task-output", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--task-shard-count", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(merge_task_shards(
        args.reference, args.task_output, args.output_dir,
        expected_rows=args.expected_rows, k=args.k,
        task_shard_count=args.task_shard_count, validate_only=args.validate_only,
    ), indent=2))


if __name__ == "__main__":
    main()
