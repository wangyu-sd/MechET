#!/usr/bin/env python3
"""Sample RetroBridge while preserving MechET IDs and the common JSONL contract."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from rdkit import Chem
from tqdm import tqdm

from src.analysis.rdkit_functions import build_molecule
from src.data.retrobridge_dataset import RetroBridgeDataModule, RetroBridgeDatasetInfos
from src.frameworks.markov_bridge import MarkovBridge
from src.utils import disable_rdkit_logging, set_deterministic


def unmapped_canonical(smiles):
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
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu())
    else:
        value = float(value)
    return value if math.isfinite(value) else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--mode", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-one-hot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_steps < 2:
        raise ValueError("--n-steps must be at least 2")
    if args.limit is not None and args.limit % args.batch_size:
        raise ValueError("For deterministic batching, --limit must be divisible by --batch-size")

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
    model = MarkovBridge.load_from_checkpoint(
        str(args.checkpoint),
        map_location=args.device,
    )
    if model.dataset_info.atom_decoder != dataset_infos.atom_decoder:
        raise ValueError("Checkpoint and dataset atom vocabularies differ")
    if model.dataset_info.max_n_dummy_nodes != dataset_infos.max_n_dummy_nodes:
        raise ValueError("Checkpoint and dataset dummy-node capacities differ")

    model.visualization_tools = None
    model.T = args.n_steps
    model.eval().to(args.device)
    dataloader = (
        datamodule.test_dataloader() if args.mode == "test" else datamodule.val_dataloader()
    )

    common_rows = []
    trace_rows = []
    processed = 0
    for batch_index, data in enumerate(tqdm(dataloader, desc="RetroBridge batches")):
        if args.limit is not None and processed >= args.limit:
            break
        data = data.to(args.device)
        batch_size = data.num_graphs
        stable_ids = [str(value) for value in data.stable_id]
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
                smiles, valid, error = decode_prediction(prediction, dataset_infos.atom_decoder)
                grouped_predictions[row_index].append({
                    "rank": sample_index + 1,
                    "precursors": smiles,
                    "score": scalar(scores[row_index]),
                    "valid": valid,
                    "nll": scalar(nlls[row_index]),
                    "ell": scalar(ells[row_index]),
                    "decode_error": error,
                })

        runtime_ms = (time.perf_counter() - started) * 1000.0 / batch_size
        for row_index, stable_id in enumerate(stable_ids):
            graph = data.get_example(row_index)
            detailed_candidates = grouped_predictions[row_index]
            common_rows.append({
                "stable_id": stable_id,
                "product": unmapped_canonical(graph.p_smiles),
                "reference_precursors": unmapped_canonical(graph.r_smiles),
                "candidates": [
                    {
                        "rank": candidate["rank"],
                        "precursors": candidate["precursors"],
                        "score": candidate["score"],
                    }
                    for candidate in detailed_candidates
                ],
                "runtime_ms": runtime_ms,
                "source_method": "RetroBridge",
                "checkpoint": str(args.checkpoint.resolve()),
            })
            trace_rows.append({
                "stable_id": stable_id,
                "batch_index": batch_index,
                "sampling_steps": args.n_steps,
                "candidate_budget": args.n_samples,
                "sampling_seed": args.sampling_seed,
                "independent_stochastic_samples": True,
                "runtime_ms": runtime_ms,
                "candidates": detailed_candidates,
            })
        processed += batch_size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in common_rows:
            handle.write(json.dumps(row) + "\n")
    with args.trace_output.open("w") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row) + "\n")

    valid_candidates = sum(
        candidate["valid"]
        for row in trace_rows
        for candidate in row["candidates"]
    )
    total_candidates = len(trace_rows) * args.n_samples
    print(json.dumps({
        "rows": len(common_rows),
        "total_candidates": total_candidates,
        "valid_candidates": valid_candidates,
        "invalid_candidates": total_candidates - valid_candidates,
        "sampling_steps": args.n_steps,
        "candidate_budget": args.n_samples,
        "output": str(args.output),
        "trace_output": str(args.trace_output),
    }, indent=2))


if __name__ == "__main__":
    disable_rdkit_logging()
    main()
