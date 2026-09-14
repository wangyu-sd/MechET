#!/usr/bin/env python3
"""Resumable RetroBridge sampling over an explicit ordered list of test indices.

This entry point imports the exact source tree used by the completed MechET
training runs, while keeping orchestration artifacts in this external-baseline
workspace.  It writes complete JSONL records incrementally and repairs only an
incomplete trailing record when resuming after interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

TRAINING_SOURCE = Path(
    "/home/estar/pxy/mechet/MechET/baselines/RetroBridge-main"
)
sys.path.insert(0, str(TRAINING_SOURCE))

import torch
from rdkit import Chem
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.analysis.rdkit_functions import build_molecule
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.frameworks.markov_bridge import MarkovBridge
from src.utils import disable_rdkit_logging, set_deterministic


def unmapped_canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return ""
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True)


def decode_prediction(prediction, atom_decoder):
    try:
        molecule = build_molecule(prediction[0], prediction[1], atom_decoder)
        smiles = Chem.MolToSmiles(molecule, canonical=True)
        valid = bool(smiles) and Chem.MolFromSmiles(smiles) is not None
        return smiles, valid, None
    except Exception as error:
        return "", False, f"{type(error).__name__}: {error}"


def scalar(value):
    value = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    return value if math.isfinite(value) else None


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary_path.replace(path)


def read_complete_jsonl(path: Path) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    repaired = False
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number != len(lines):
                raise ValueError(f"Malformed non-final JSONL record in {path}:{line_number}")
            repaired = True
    return rows, repaired


def append_rows(handle, rows: list[dict]) -> None:
    handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    handle.flush()
    os.fsync(handle.fileno())


def deterministic_batch_seed(base_seed: int, stable_ids: list[str]) -> int:
    material = f"{base_seed}:" + "\n".join(stable_ids)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--indices-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-one-hot", action="store_true")
    return parser.parse_args()


def load_resume_rows(args) -> tuple[list[dict], list[dict]]:
    output_exists = args.output.exists()
    trace_exists = args.trace_output.exists()
    if not args.resume:
        if output_exists or trace_exists:
            raise FileExistsError("Output exists; pass --resume to continue")
        return [], []
    if output_exists != trace_exists:
        raise ValueError("Resume requires both output files or neither output file")
    if not output_exists:
        return [], []

    output_rows, output_repaired = read_complete_jsonl(args.output)
    trace_rows, trace_repaired = read_complete_jsonl(args.trace_output)
    common_length = min(len(output_rows), len(trace_rows))
    for index in range(common_length):
        if output_rows[index].get("stable_id") != trace_rows[index].get("stable_id"):
            raise ValueError(f"Resume stable-ID mismatch at record {index}")
    needs_rewrite = (
        output_repaired
        or trace_repaired
        or len(output_rows) != common_length
        or len(trace_rows) != common_length
    )
    output_rows = output_rows[:common_length]
    trace_rows = trace_rows[:common_length]
    if needs_rewrite:
        write_jsonl_atomic(args.output, output_rows)
        write_jsonl_atomic(args.trace_output, trace_rows)
    return output_rows, trace_rows


def validate_resume(
    args,
    indices: list[int],
    dataset,
    output_rows: list[dict],
    trace_rows: list[dict],
) -> None:
    if len(output_rows) > len(indices):
        raise ValueError("Resume output is longer than the assigned index list")
    expected_checkpoint = str(args.checkpoint.resolve())
    for position, (output_row, trace_row) in enumerate(zip(output_rows, trace_rows)):
        expected_index = indices[position]
        expected_id = str(dataset[expected_index].stable_id)
        if output_row.get("stable_id") != expected_id:
            raise ValueError(
                f"Resume ID mismatch at {position}: expected {expected_id}, "
                f"got {output_row.get('stable_id')}"
            )
        if output_row.get("source_index") != expected_index:
            raise ValueError(f"Resume source_index mismatch for {expected_id}")
        if output_row.get("checkpoint") != expected_checkpoint:
            raise ValueError(f"Resume checkpoint mismatch for {expected_id}")
        if len(output_row.get("candidates", [])) != args.n_samples:
            raise ValueError(f"Resume candidate budget mismatch for {expected_id}")
        if trace_row.get("sampling_steps") != args.n_steps:
            raise ValueError(f"Resume sampling-step mismatch for {expected_id}")
        if trace_row.get("sampling_seed") != args.sampling_seed:
            raise ValueError(f"Resume sampling-seed mismatch for {expected_id}")


def main():
    args = parse_args()
    if args.n_steps < 2:
        raise ValueError("--n-steps must be at least 2")
    if args.batch_size < 1 or args.n_samples < 1:
        raise ValueError("--batch-size and --n-samples must be positive")
    if args.torch_threads < 0:
        raise ValueError("--torch-threads must be non-negative")

    indices = json.loads(args.indices_json.read_text(encoding="utf-8"))
    if not isinstance(indices, list) or not all(isinstance(value, int) for value in indices):
        raise ValueError("--indices-json must contain a JSON list of integer indices")
    if len(indices) != len(set(indices)):
        raise ValueError("--indices-json contains duplicate indices")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    output_rows, trace_rows = load_resume_rows(args)

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
    set_deterministic(args.sampling_seed)
    datamodule = RetroBridgeDataModule(
        data_root=str(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        extra_nodes=True,
        evaluation=True,
        swap=False,
    )
    dataset_infos = RetroBridgeDatasetInfos(datamodule)
    base_dataloader = (
        datamodule.test_dataloader() if args.mode == "test" else datamodule.val_dataloader()
    )
    dataset = base_dataloader.dataset
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"Assigned index is outside dataset of size {len(dataset)}")
    validate_resume(args, indices, dataset, output_rows, trace_rows)

    model = MarkovBridge.load_from_checkpoint(str(args.checkpoint), map_location=args.device)
    if model.dataset_info.atom_decoder != dataset_infos.atom_decoder:
        raise ValueError("Checkpoint and dataset atom vocabularies differ")
    if model.dataset_info.max_n_dummy_nodes != dataset_infos.max_n_dummy_nodes:
        raise ValueError("Checkpoint and dataset dummy-node capacities differ")
    model.visualization_tools = None
    model.T = args.n_steps
    model.eval().to(args.device)

    processed = len(output_rows)
    remaining_indices = indices[processed:]
    dataloader = DataLoader(
        Subset(dataset, remaining_indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    valid_candidates = sum(
        int(row.get("valid_candidate_count", 0)) for row in output_rows
    )
    expected_checkpoint = str(args.checkpoint.resolve())

    with args.output.open("a", encoding="utf-8") as output_handle, args.trace_output.open(
        "a", encoding="utf-8"
    ) as trace_handle:
        for data in tqdm(dataloader, desc="RetroBridge batches"):
            data = data.to(args.device)
            batch_size = data.num_graphs
            batch_indices = remaining_indices[:batch_size]
            remaining_indices = remaining_indices[batch_size:]
            stable_ids = [str(value) for value in data.stable_id]
            batch_sampling_seed = deterministic_batch_seed(args.sampling_seed, stable_ids)
            set_deterministic(batch_sampling_seed)
            started = time.perf_counter()
            grouped_predictions = [[] for _ in range(batch_size)]

            for sample_index in range(args.n_samples):
                predictions, _, _, scores, nlls, ells = model.sample_batch(
                    data=data,
                    batch_id=processed,
                    batch_size=batch_size,
                    save_final=0,
                    keep_chain=0,
                    number_chain_steps_to_save=1,
                    sample_idx=sample_index,
                    save_true_reactants=True,
                    use_one_hot=args.use_one_hot,
                )
                for row_index, prediction in enumerate(predictions):
                    smiles, valid, error = decode_prediction(
                        prediction, dataset_infos.atom_decoder
                    )
                    grouped_predictions[row_index].append(
                        {
                            "rank": sample_index + 1,
                            "sample_index": sample_index,
                            "precursors": smiles,
                            "prediction": smiles,
                            "score": scalar(scores[row_index]),
                            "valid": valid,
                            "nll": scalar(nlls[row_index]),
                            "ell": scalar(ells[row_index]),
                            "decode_error": error,
                        }
                    )

            runtime_ms = (time.perf_counter() - started) * 1000.0 / batch_size
            new_output_rows: list[dict] = []
            new_trace_rows: list[dict] = []
            for row_index, (stable_id, source_index) in enumerate(
                zip(stable_ids, batch_indices)
            ):
                graph = data.get_example(row_index)
                detailed_candidates = grouped_predictions[row_index]
                valid_count = sum(candidate["valid"] for candidate in detailed_candidates)
                new_output_rows.append(
                    {
                        "id": stable_id,
                        "stable_id": stable_id,
                        "source_index": source_index,
                        "product": unmapped_canonical(graph.p_smiles),
                        "reference_precursors": unmapped_canonical(graph.r_smiles),
                        "candidates": [
                            {
                                "rank": candidate["rank"],
                                "sample_index": candidate["sample_index"],
                                "precursors": candidate["precursors"],
                                "prediction": candidate["prediction"],
                                "score": candidate["score"],
                            }
                            for candidate in detailed_candidates
                        ],
                        "runtime_ms": runtime_ms,
                        "source_method": "RetroBridge",
                        "checkpoint": expected_checkpoint,
                        "candidate_budget": args.n_samples,
                        "generated_candidate_count": args.n_samples,
                        "valid_candidate_count": valid_count,
                        "candidate_semantics": "independent_stochastic_samples",
                        "ranking": "generation_order",
                        "score_type": "retrobridge_native_score",
                    }
                )
                new_trace_rows.append(
                    {
                        "stable_id": stable_id,
                        "source_index": source_index,
                        "sampling_steps": args.n_steps,
                        "candidate_budget": args.n_samples,
                        "sampling_seed": args.sampling_seed,
                        "batch_sampling_seed": batch_sampling_seed,
                        "independent_stochastic_samples": True,
                        "runtime_ms": runtime_ms,
                        "candidates": detailed_candidates,
                    }
                )
                valid_candidates += valid_count

            # If interrupted between these writes, --resume truncates both files
            # to their last common stable-ID prefix.
            append_rows(output_handle, new_output_rows)
            append_rows(trace_handle, new_trace_rows)
            processed += batch_size

    total_candidates = processed * args.n_samples
    print(
        json.dumps(
            {
                "rows": processed,
                "total_candidates": total_candidates,
                "valid_candidates": valid_candidates,
                "invalid_candidates": total_candidates - valid_candidates,
                "sampling_steps": args.n_steps,
                "candidate_budget": args.n_samples,
                "output": str(args.output),
                "trace_output": str(args.trace_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    disable_rdkit_logging()
    main()
