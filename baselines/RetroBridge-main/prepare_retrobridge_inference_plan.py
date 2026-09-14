#!/usr/bin/env python3
"""Create size-bucketed, balanced worker assignments and an inference contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from rdkit import Chem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def identifier(row: dict) -> str:
    return str(row.get("id") or row.get("stable_id") or "")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_path = args.data_root / "metadata.json"
    raw_path = args.data_root / "raw" / "uspto50k_test.csv"
    processed_path = (
        args.data_root / "processed_retrobridge_extra_nodes" / "test.pt"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    references = read_jsonl(args.reference)
    reference_ids = [identifier(row) for row in references]
    if not all(reference_ids) or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference contains missing or duplicate stable IDs")

    records: list[tuple[int, str, int]] = []
    with raw_path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            stable_id = str(row["id"])
            reaction = row["reactants>reagents>production"]
            product = reaction.rsplit(">", 1)[-1]
            molecule = Chem.MolFromSmiles(product)
            if molecule is None:
                raise ValueError(f"RDKit failed to parse product at converted row {index}")
            node_count = molecule.GetNumAtoms() + int(metadata["max_n_dummy_nodes"])
            records.append((index, stable_id, node_count))

    converted_ids = [stable_id for _, stable_id, _ in records]
    if len(converted_ids) != len(set(converted_ids)):
        raise ValueError("Converted test CSV contains duplicate stable IDs")
    converted_id_set = set(converted_ids)
    missing_from_reference = sorted(converted_id_set - set(reference_ids))
    if missing_from_reference:
        raise ValueError(f"Converted IDs missing from reference: {missing_from_reference[:10]}")
    excluded_ids = [stable_id for stable_id in reference_ids if stable_id not in converted_id_set]
    declared_excluded = [
        str(row["stable_id"])
        for row in metadata.get("source_filtering", {})
        .get("excluded_by_split", {})
        .get("test", [])
    ]
    if excluded_ids != declared_excluded:
        raise ValueError(
            f"Reference/converted excluded IDs {excluded_ids} do not match metadata "
            f"declaration {declared_excluded}"
        )

    # Keep each actual DataLoader batch internally size-homogeneous, then use
    # longest-processing-time assignment to balance the three GPU workers.
    sorted_records = sorted(records, key=lambda item: (item[2], item[0]))
    batches = [
        sorted_records[offset : offset + args.batch_size]
        for offset in range(0, len(sorted_records), args.batch_size)
    ]
    weighted_batches = [
        (max(row[2] for row in batch) ** 2 * len(batch), batch) for batch in batches
    ]
    worker_batches: list[list[list[tuple[int, str, int]]]] = [[] for _ in args.gpus]
    worker_costs = [0 for _ in args.gpus]
    for cost, batch in sorted(weighted_batches, key=lambda item: item[0], reverse=True):
        worker_index = min(range(len(args.gpus)), key=lambda index: worker_costs[index])
        worker_batches[worker_index].append(batch)
        worker_costs[worker_index] += cost

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    assigned_ids: set[str] = set()
    for worker_index, (gpu, batches_for_worker, estimated_cost) in enumerate(
        zip(args.gpus, worker_batches, worker_costs)
    ):
        # Every batch remains contiguous in this flattened order.
        assigned = [row for batch in batches_for_worker for row in batch]
        indices = [row[0] for row in assigned]
        ids = [row[1] for row in assigned]
        overlap = assigned_ids.intersection(ids)
        if overlap:
            raise ValueError(f"Worker assignments overlap: {sorted(overlap)[:10]}")
        assigned_ids.update(ids)
        indices_path = parts_dir / f"worker-{worker_index:02d}.indices.json"
        indices_path.write_text(json.dumps(indices) + "\n", encoding="utf-8")
        workers.append(
            {
                "worker_index": worker_index,
                "gpu": gpu,
                "rows": len(indices),
                "batches": len(batches_for_worker),
                "estimated_dense_cost": estimated_cost,
                "indices": str(indices_path.resolve()),
                "predictions": str(
                    (parts_dir / f"worker-{worker_index:02d}.predictions.jsonl").resolve()
                ),
                "traces": str(
                    (parts_dir / f"worker-{worker_index:02d}.traces.jsonl").resolve()
                ),
                "log": str((parts_dir / f"worker-{worker_index:02d}.log").resolve()),
            }
        )
    if assigned_ids != set(converted_ids):
        raise ValueError("Worker assignments do not exactly cover converted test IDs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    filtering = metadata.get("source_filtering", {})
    plan = {
        "artifact_type": "retrobridge_inference_plan_v1",
        "dataset_name": args.dataset_name,
        "endpoint_only": True,
        "data_root": str(args.data_root.resolve()),
        "converted_test_csv": str(raw_path.resolve()),
        "converted_test_csv_sha256": sha256(raw_path),
        "processed_test_graphs": str(processed_path.resolve()),
        "processed_test_graphs_sha256": sha256(processed_path),
        "metadata": str(metadata_path.resolve()),
        "metadata_sha256": sha256(metadata_path),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256(args.reference),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "reference_rows": len(reference_ids),
        "retrobridge_compatible_rows": len(converted_ids),
        "excluded_incompatible_rows": len(excluded_ids),
        "excluded_stable_ids": excluded_ids,
        "source_filtering_mode": filtering.get("mode"),
        "proof_or_replay_filtering": False,
        "overlap_filtering": False,
        "token_length_filtering": False,
        "sampling_steps": args.n_steps,
        "candidates_per_target": args.n_samples,
        "candidate_semantics": "independent_stochastic_samples",
        "ranking": "generation_order",
        "sampling_seed_per_worker": args.sampling_seed,
        "batch_size": args.batch_size,
        "batching": "product-node-count sorted batches, LPT-balanced across workers",
        "workers": workers,
    }
    plan_path = args.output_dir / "inference_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
