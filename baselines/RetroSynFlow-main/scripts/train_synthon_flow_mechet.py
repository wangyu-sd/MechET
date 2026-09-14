#!/usr/bin/env python3
"""Train and resumably checkpoint the published synthon-to-reactant flow."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from retflow.datasets import SynthonDataset
from retflow.methods import GraphDiscreteFM
from retflow.models import GraphTransformer
from retflow.optimizers.optimizer import AdamW
from retflow.optimizers.schedulers import ConsLR
from retflow.problems import SynthonCompletion


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def apply_dataset_config(root: Path, args) -> None:
    config_path = root / "training_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    requested = {
        "flow_epochs": args.epochs,
        "flow_batch_size": args.batch_size,
        "flow_gradient_accumulation": args.gradient_accumulation,
    }
    configured = {
        key: int(config.get(key, value)) for key, value in requested.items()
    }
    if any(value < 1 for value in configured.values()):
        raise ValueError(f"flow training values must be positive in {config_path}")
    args.epochs = configured["flow_epochs"]
    args.batch_size = configured["flow_batch_size"]
    args.gradient_accumulation = configured["flow_gradient_accumulation"]
    if configured != requested:
        print(
            f"Using dataset flow configuration from {config_path}: "
            f"{configured} (CLI/default requested {requested}).",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument(
        "--limit-rows",
        type=int,
        help="limit train/validation rows for a startup smoke test only",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    root = args.dataset_root.resolve()
    apply_dataset_config(root, args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SynthonDataset(
        name=root.name,
        batch_size=args.batch_size,
        dataset_root=str(root),
    )
    problem = SynthonCompletion(
        GraphTransformer(),
        dataset,
        GraphDiscreteFM(),
        use_product_context=True,
        gradient_accumulation_steps=args.gradient_accumulation,
    )
    problem.setup_problem()
    if args.limit_rows is not None:
        if args.limit_rows < 1:
            raise ValueError("--limit-rows must be positive")
        train_dataset = problem.train_loader.dataset
        val_dataset = problem.val_loader.dataset
        problem.train_loader = DataLoader(
            Subset(train_dataset, range(min(args.limit_rows, len(train_dataset)))),
            batch_size=min(args.batch_size, args.limit_rows),
            shuffle=False,
        )
        problem.val_loader = DataLoader(
            Subset(val_dataset, range(min(args.limit_rows, len(val_dataset)))),
            batch_size=min(args.batch_size, args.limit_rows),
            shuffle=False,
        )
    optimizer_definition = AdamW(lr=2e-4, lr_sched=ConsLR())
    optimizer = problem.get_optimizer(optimizer_definition)
    scheduler = optimizer_definition.lr_sched.get_scheduler(optimizer)

    latest_path = output_dir / "flow_latest.pt"
    start_epoch = 0
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        problem.torch_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])

    history_path = output_dir / "flow_metrics.jsonl"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_result = problem.one_epoch(optimizer, scheduler, dist_helper=None)
        valid_result = problem.validation(dist_helper=None)
        record = {
            "epoch": epoch,
            **train_result["metrics"],
            **valid_result["metrics"],
            "train_rows": train_result["num_points"],
            "valid_rows": valid_result["num_points"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        checkpoint = {
            "epoch": epoch,
            "model": problem.torch_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dataset": root.name,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "seed": args.seed,
        }
        atomic_torch_save(checkpoint, latest_path)
        if epoch % args.checkpoint_every == 0:
            atomic_torch_save(checkpoint, output_dir / f"flow_epoch_{epoch}.pt")

    atomic_torch_save(problem.torch_model.state_dict(), output_dir / "final_model.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
