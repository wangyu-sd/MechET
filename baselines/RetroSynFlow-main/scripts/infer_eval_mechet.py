#!/usr/bin/env python3
"""Run resumable RetroSynFlow inference on a frozen MechET test export."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from retflow.datasets import TorchDrugRetroDataset
from retflow.methods import GraphDiscreteFM
from retflow.models import GraphTransformer
from retflow.problems import SynthonRetrosynthesis


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} at line {line_number}: {error}"
                ) from error
    return rows


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id") or row.get("stable_id") or "")
        if not identifier or identifier in indexed:
            raise ValueError(f"{label} has a missing or duplicate ID: {identifier!r}")
        indexed[identifier] = row
    return indexed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
    with path.open("ab", buffering=0) as handle:
        handle.write(encoded)
        os.fsync(handle.fileno())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ranked_candidates(
    predictions: list[str], expected_samples: int, expected_candidates: int
) -> list[dict[str, Any]]:
    if len(predictions) != expected_samples:
        raise ValueError(
            f"model returned {len(predictions)} samples, expected {expected_samples}"
        )
    normalized = [str(value or "") for value in predictions]
    counts = Counter(normalized)
    first_seen: dict[str, int] = {}
    for sample_index, value in enumerate(normalized):
        first_seen.setdefault(value, sample_index)
    ranked = sorted(counts, key=lambda value: (-counts[value], first_seen[value]))

    candidates: list[dict[str, Any]] = []
    for value in ranked[:expected_candidates]:
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "sample_index": first_seen[value],
                "precursors": value,
                "prediction": value,
                "score": counts[value] / expected_samples,
                "sample_count": counts[value],
                "generation_error": "",
            }
        )
    while len(candidates) < expected_candidates:
        candidates.append(
            {
                "rank": len(candidates) + 1,
                "sample_index": len(normalized),
                "precursors": "",
                "prediction": "",
                "score": 0.0,
                "sample_count": 0,
                "generation_error": "fewer_than_10_unique_native_samples",
            }
        )
    return candidates


def error_candidates(message: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "sample_index": rank - 1,
            "precursors": "",
            "prediction": "",
            "score": 0.0,
            "sample_count": 0,
            "generation_error": message,
        }
        for rank in range(1, count + 1)
    ]


def prediction_row(
    reference: dict[str, Any],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    source_row_index: int,
    runtime_ms: float,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    identifier = str(reference.get("stable_id") or reference.get("id"))
    return {
        "id": identifier,
        "stable_id": identifier,
        "product": reference.get("product_mapped") or reference.get("product") or "",
        "reference_precursors": reference.get("precursor_mapped")
        or reference.get("reference_precursors")
        or "",
        "candidates": candidates,
        "runtime_ms": runtime_ms,
        "source_method": "RetroSynFlow",
        "checkpoint": str(args.flow_checkpoint.resolve()),
        "center_checkpoint": str(args.center_checkpoint.resolve()),
        "ranking": "native_sample_frequency",
        "native_samples": args.samples,
        "flow_steps": args.steps,
        "synthon_topk": args.synthon_topk,
        "samples_per_synthon": args.samples_per_synthon,
        "source_row_index": source_row_index,
        "evaluation_status": status,
        "generation_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--center-checkpoint", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--top-candidates", type=int, default=10)
    parser.add_argument("--synthon-topk", type=int, default=2)
    parser.add_argument("--samples-per-synthon", type=int, nargs="+", default=[70, 30])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-processed-rows",
        type=int,
        help="generate at most this many new supported rows; do not finalize",
    )
    args = parser.parse_args()

    args.dataset_root = args.dataset_root.resolve()
    args.reference = args.reference.resolve()
    args.flow_checkpoint = args.flow_checkpoint.resolve()
    args.center_checkpoint = args.center_checkpoint.resolve()
    args.work_dir = args.work_dir.resolve()
    args.predictions = args.predictions.resolve()
    if args.steps < 1 or args.samples < 1 or args.top_candidates < 1:
        raise ValueError("steps, samples, and top-candidates must be positive")
    if args.synthon_topk < 1:
        raise ValueError("synthon-topk must be positive")
    if len(args.samples_per_synthon) != args.synthon_topk:
        raise ValueError("samples-per-synthon must have synthon-topk entries")
    if sum(args.samples_per_synthon) != args.samples:
        raise ValueError("samples-per-synthon must sum to samples")
    if args.limit_processed_rows is not None and args.limit_processed_rows < 1:
        raise ValueError("limit-processed-rows must be positive")
    for required in (
        args.dataset_root,
        args.reference,
        args.flow_checkpoint,
        args.center_checkpoint,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if not torch.cuda.is_available():
        raise RuntimeError("RetroSynFlow full inference requires a CUDA GPU")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.work_dir / "processed_predictions.jsonl"
    config_path = args.work_dir / "inference_config.json"
    references_list = read_jsonl(args.reference)
    references = index_rows(references_list, "reference")
    run_config = {
        "artifact_type": "retrosynflow_resumable_inference_config",
        "dataset_root": str(args.dataset_root),
        "reference": str(args.reference),
        "reference_sha256": sha256_file(args.reference),
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_checkpoint_sha256": sha256_file(args.flow_checkpoint),
        "center_checkpoint": str(args.center_checkpoint),
        "center_checkpoint_sha256": sha256_file(args.center_checkpoint),
        "steps": args.steps,
        "samples": args.samples,
        "top_candidates": args.top_candidates,
        "synthon_topk": args.synthon_topk,
        "samples_per_synthon": args.samples_per_synthon,
        "seed": args.seed,
        "row_seed_policy": "seed_plus_frozen_source_row_index",
    }
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise RuntimeError(
                f"resume configuration differs from {config_path}; use a new work-dir"
            )
    else:
        atomic_write_json(config_path, run_config)

    completed_list = read_jsonl(partial_path) if partial_path.is_file() else []
    completed = index_rows(completed_list, "partial predictions")
    unknown_completed = sorted(set(completed) - set(references))
    if unknown_completed:
        raise RuntimeError(f"partial predictions contain unknown IDs: {unknown_completed[:5]}")
    print(
        json.dumps(
            {
                "event": "setup_start",
                "dataset": args.dataset_root.name,
                "reference_rows": len(references_list),
                "resumed_rows": len(completed),
                "cuda_device": torch.cuda.get_device_name(0),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    dataset = TorchDrugRetroDataset(
        name=args.dataset_root.name,
        batch_size=1,
        product_context=True,
        dataset_root=str(args.dataset_root),
    )
    problem = SynthonRetrosynthesis(
        GraphTransformer(),
        dataset,
        GraphDiscreteFM(steps=args.steps),
        synthon_topk=args.synthon_topk,
        samples_per_synthon=args.samples_per_synthon,
        product_context=True,
        center_checkpoint=str(args.center_checkpoint),
    )
    problem.setup_problem_eval(str(args.flow_checkpoint))
    original_loader = problem.test_loader
    supported_ids = [str(value) for value in problem.dataset.test_dataset.ids]
    source_indices = [
        int(value) for value in problem.dataset.test_dataset.source_row_indices
    ]
    if len(supported_ids) != len(set(supported_ids)):
        raise RuntimeError("reaction-center test set contains duplicate stable IDs")
    failure_rows = list(problem.dataset.eval_preprocessing_failures)
    failures = index_rows(failure_rows, "preprocessing failures")
    if set(supported_ids) & set(failures):
        raise RuntimeError("supported rows and preprocessing failures overlap")
    if set(supported_ids) | set(failures) != set(references):
        missing = sorted(set(references) - set(supported_ids) - set(failures))
        extra = sorted((set(supported_ids) | set(failures)) - set(references))
        raise RuntimeError(
            "preprocessing does not partition the frozen reference IDs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    print(
        json.dumps(
            {
                "event": "setup_complete",
                "supported_rows": len(supported_ids),
                "preprocessing_failures": len(failures),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    new_rows = 0
    total_supported = len(supported_ids)
    for processed_index, data in enumerate(original_loader):
        identifier = supported_ids[processed_index]
        if identifier in completed:
            continue
        if (
            args.limit_processed_rows is not None
            and new_rows >= args.limit_processed_rows
        ):
            break
        row_started = time.monotonic()
        row_seed = args.seed + source_indices[processed_index]
        seed_everything(row_seed)
        reference = references[identifier]
        try:
            problem.test_loader = [data]
            result = problem.sample_generation_eval(args.samples)
            generated = list(result["predicted_reactants"][0])
            candidates = ranked_candidates(
                generated, args.samples, args.top_candidates
            )
            status = "ok"
            error = ""
        except Exception as caught:
            error = f"{type(caught).__name__}: {caught}"
            candidates = error_candidates(error, args.top_candidates)
            status = "generation_error"
            print(
                json.dumps(
                    {"event": "row_error", "id": identifier, "error": error},
                    sort_keys=True,
                ),
                flush=True,
            )
            torch.cuda.empty_cache()
        finally:
            problem.test_loader = original_loader
        runtime_ms = (time.monotonic() - row_started) * 1000
        row = prediction_row(
            reference,
            candidates,
            args,
            source_row_index=source_indices[processed_index],
            runtime_ms=runtime_ms,
            status=status,
            error=error,
        )
        append_jsonl(partial_path, row)
        completed[identifier] = row
        new_rows += 1
        print(
            json.dumps(
                {
                    "event": "row_complete",
                    "id": identifier,
                    "processed_progress": f"{len(completed)}/{total_supported}",
                    "runtime_seconds": round(runtime_ms / 1000, 3),
                    "status": status,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.limit_processed_rows is not None:
        print(
            json.dumps(
                {
                    "event": "limited_run_complete",
                    "new_rows": new_rows,
                    "total_resumable_rows": len(completed),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    missing_supported = sorted(set(supported_ids) - set(completed))
    if missing_supported:
        raise RuntimeError(f"inference ended with missing supported IDs: {missing_supported[:5]}")

    final_rows: list[dict[str, Any]] = []
    preprocessing_error = (
        "preprocessing_error: unsupported_by_published_single_center_construction"
    )
    for source_row_index, reference in enumerate(references_list):
        identifier = str(reference.get("stable_id") or reference.get("id"))
        if identifier in completed:
            final_rows.append(completed[identifier])
        else:
            failure = failures[identifier]
            detail = f"{preprocessing_error}: {failure.get('error', '')}".rstrip()
            final_rows.append(
                prediction_row(
                    reference,
                    error_candidates(detail, args.top_candidates),
                    args,
                    source_row_index=source_row_index,
                    runtime_ms=0.0,
                    status="preprocessing_error",
                    error=detail,
                )
            )
    atomic_write_jsonl(args.predictions, final_rows)
    report = {
        "artifact_type": "retrosynflow_inference_report",
        "dataset": args.dataset_root.name,
        "reference_rows": len(references_list),
        "generated_rows": len(completed),
        "preprocessing_failure_rows": len(failures),
        "generation_error_rows": sum(
            row["evaluation_status"] == "generation_error" for row in completed.values()
        ),
        "candidates_per_row": args.top_candidates,
        "predictions": str(args.predictions),
        "predictions_sha256": sha256_file(args.predictions),
        "resume_file": str(partial_path),
        "config": run_config,
    }
    atomic_write_json(args.predictions.with_suffix(".report.json"), report)
    print(json.dumps({"event": "inference_complete", **report}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
