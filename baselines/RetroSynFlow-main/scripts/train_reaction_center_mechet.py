#!/usr/bin/env python3
"""Train the published RetroSynFlow reaction-center component with audit logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchdrug import core, models, tasks

from retflow.datasets.data.uspto_drug import _TorchDrugUSPTO


def json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def full_denominator_metrics(metrics, valid_fraction: float) -> dict:
    """Charge preprocessing failures as wrong without scaling loss values."""
    adjusted = {}
    for key, value in metrics.items():
        scalar = json_value(value)
        adjusted[key] = (
            scalar * valid_fraction if "accuracy" in key.lower() else scalar
        )
    return adjusted


def configured_epochs(root: Path, requested_epochs: int) -> int:
    config_path = root / "training_config.json"
    if not config_path.is_file():
        return requested_epochs
    config = json.loads(config_path.read_text(encoding="utf-8"))
    epochs = int(config.get("center_epochs", requested_epochs))
    if epochs < 1:
        raise ValueError(f"center_epochs must be positive in {config_path}")
    if epochs != requested_epochs:
        print(
            f"Using dataset center_epochs={epochs} from {config_path} "
            f"(CLI requested {requested_epochs}).",
            flush=True,
        )
    return epochs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    args.epochs = configured_epochs(root, args.epochs)
    raw_dir = root / "raw"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: _TorchDrugUSPTO(
            split,
            str(raw_dir),
            verbose=1,
            kekulize=False,
            atom_feature="center_identification",
        )
        for split in ("train", "val", "test")
    }
    preprocessing = {
        split: dataset.preprocessing_report for split, dataset in datasets.items()
    }
    save_json(output_dir / "center_preprocessing.json", preprocessing)

    reaction_model = models.RGCN(
        input_dim=datasets["train"].node_feature_dim,
        hidden_dims=[256, 256, 256, 256],
        num_relation=4,
        short_cut=True,
        concat_hidden=True,
    )
    reaction_task = tasks.CenterIdentification(
        reaction_model, feature=("graph", "atom", "bond")
    )
    optimizer = torch.optim.AdamW(reaction_task.parameters(), lr=1e-4)
    solver = core.Engine(
        reaction_task,
        datasets["train"],
        datasets["val"],
        datasets["test"],
        optimizer,
        gpus=[0] if args.device == "cuda" else None,
        batch_size=args.batch_size,
        num_worker=args.num_workers,
        logger="logging",
        log_interval=100,
    )

    latest_path = output_dir / "center_latest.pth"
    progress_path = output_dir / "center_progress.json"
    start_epoch = 0
    if latest_path.is_file() and progress_path.is_file():
        solver.load(str(latest_path))
        start_epoch = int(
            json.loads(progress_path.read_text(encoding="utf-8"))["epoch"]
        )

    history_path = output_dir / "center_metrics.jsonl"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        solver.train(num_epoch=1)
        record = {"epoch": epoch}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            valid_metrics = solver.evaluate("valid", log=True)
            record["valid_processed"] = {
                key: json_value(value) for key, value in valid_metrics.items()
            }
            valid_fraction = (
                len(datasets["val"])
                / datasets["val"].preprocessing_report["input_rows"]
            )
            record["valid_full_denominator"] = full_denominator_metrics(
                valid_metrics, valid_fraction
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        temporary_checkpoint = output_dir / "center_latest.pth.tmp"
        solver.save(str(temporary_checkpoint))
        temporary_checkpoint.replace(latest_path)
        save_json(progress_path, {"epoch": epoch, "target_epochs": args.epochs})
        if epoch % 10 == 0:
            solver.save(str(output_dir / f"center_epoch_{epoch}.pth"))

    test_metrics = solver.evaluate("test", log=True)
    test_fraction = (
        len(datasets["test"])
        / datasets["test"].preprocessing_report["input_rows"]
    )
    final_metrics = {
        "processed_rows": len(datasets["test"]),
        "full_denominator_rows": datasets["test"].preprocessing_report["input_rows"],
        "preprocessing_errors": datasets["test"].preprocessing_report["failed_rows"],
        "processed_metrics": {
            key: json_value(value) for key, value in test_metrics.items()
        },
        "full_denominator_metrics": full_denominator_metrics(
            test_metrics, test_fraction
        ),
    }
    save_json(output_dir / "center_test_metrics.json", final_metrics)
    solver.save(str(output_dir / "g2g_center.pth"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
