#!/usr/bin/env python3
"""Train the leakage-free Retro-MTGR adapter on prepared native targets."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any

import torch
from torch import nn

from mechet_retro_mtgr import graph_tensors, read_jsonl
from retro_mtgr_adapter_model import RetroMTGRAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--alignment-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def prepare_records(
    data_dir: Path,
    split: str,
    label_to_id: dict[str, int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = read_jsonl(data_dir / f"{split}_targets.jsonl")
    records: list[dict[str, Any]] = []
    skipped = {"unsupported": 0, "unseen_label": 0, "missing_center": 0}
    for row in rows:
        if row["status"] != "supported":
            skipped["unsupported"] += 1
            continue
        if row["left_label_key"] not in label_to_id or row["right_label_key"] not in label_to_id:
            skipped["unseen_label"] += 1
            continue
        product_graph = graph_tensors(row["product_mapped"], device)
        center = tuple(row["center_atom_indices"])
        try:
            center_index = product_graph["bonds"].index(center)
        except ValueError:
            skipped["missing_center"] += 1
            continue
        records.append({
            "stable_id": row["stable_id"],
            "product": product_graph,
            "precursor": graph_tensors(row["reference_precursors"], device),
            "center_index": center_index,
            "left_label": label_to_id[row["left_label_key"]],
            "right_label": label_to_id[row["right_label_key"]],
        })
    return records, skipped


def collate_graphs(graphs: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = len(graphs)
    max_nodes = max(graph["x"].shape[0] for graph in graphs)
    max_bonds = max(len(graph["bonds"]) for graph in graphs)
    feature_dim = graphs[0]["x"].shape[1]
    x = torch.zeros(batch_size, max_nodes, feature_dim, device=device)
    adjacency = torch.zeros(batch_size, max_nodes, max_nodes, device=device)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    bonds = torch.zeros(batch_size, max_bonds, 2, dtype=torch.long, device=device)
    bond_features = torch.zeros(batch_size, max_bonds, 6, device=device)
    bond_mask = torch.zeros(batch_size, max_bonds, dtype=torch.bool, device=device)
    for index, graph in enumerate(graphs):
        node_count = graph["x"].shape[0]
        bond_count = len(graph["bonds"])
        x[index, :node_count] = graph["x"]
        adjacency[index, :node_count, :node_count] = graph["adj"]
        node_mask[index, :node_count] = True
        if bond_count:
            bonds[index, :bond_count] = torch.tensor(graph["bonds"], device=device)
            bond_features[index, :bond_count] = graph["bond_features"]
            bond_mask[index, :bond_count] = True
    return {
        "x": x,
        "adj": adjacency,
        "node_mask": node_mask,
        "bonds": bonds,
        "bond_features": bond_features,
        "bond_mask": bond_mask,
    }


def collate_records(records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    return {
        "product": collate_graphs([record["product"] for record in records], device),
        "precursor": collate_graphs([record["precursor"] for record in records], device),
        "center": torch.tensor([record["center_index"] for record in records], device=device),
        "left": torch.tensor([record["left_label"] for record in records], device=device),
        "right": torch.tensor([record["right_label"] for record in records], device=device),
    }


def evaluate(
    model: RetroMTGRAdapter,
    records: list[dict[str, Any]],
    loss_fn: nn.Module,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    center_correct = 0
    left_correct = 0
    right_correct = 0
    joint_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            batch = collate_records(records[offset:offset + batch_size], device)
            hidden, pooled = model.encode_batch(batch["product"])
            center_logits = model.score_bonds_batch(batch["product"], hidden, pooled)
            left_logits, right_logits = model.score_leaving_groups_batch(
                batch["product"], hidden, pooled, batch["center"]
            )
            loss = (
                loss_fn(center_logits, batch["center"])
                + loss_fn(left_logits, batch["left"])
                + loss_fn(right_logits, batch["right"])
            )
            current_size = batch["center"].shape[0]
            total_loss += float(loss) * current_size
            center_hit = center_logits.argmax(dim=1) == batch["center"]
            left_hit = left_logits.argmax(dim=1) == batch["left"]
            right_hit = right_logits.argmax(dim=1) == batch["right"]
            center_correct += int(center_hit.sum())
            left_correct += int(left_hit.sum())
            right_correct += int(right_hit.sum())
            joint_correct += int((center_hit & left_hit & right_hit).sum())
    denominator = max(len(records), 1)
    return {
        "loss": total_loss / denominator,
        "center_accuracy": center_correct / denominator,
        "left_label_accuracy": left_correct / denominator,
        "right_label_accuracy": right_correct / denominator,
        "joint_target_accuracy": joint_correct / denominator,
    }


def save_checkpoint(
    path: Path,
    model: RetroMTGRAdapter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
    metrics: dict[str, float],
) -> None:
    torch.save({
        "format": "mechet-retro-mtgr-adapter-v1",
        "model_configuration": model.configuration(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "labels": labels,
        "training_arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "metrics": metrics,
    }, path)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("epochs/batch size must be positive and patience nonnegative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = json.loads((args.data_dir / "label_vocabulary.json").read_text())
    labels = vocabulary["labels"]
    if vocabulary.get("source_splits") != ["train"]:
        raise ValueError("label vocabulary must be derived from train only")
    label_to_id = {item["key"]: int(item["id"]) for item in labels}
    train_records, train_skipped = prepare_records(args.data_dir, "train", label_to_id, device)
    valid_records, valid_skipped = prepare_records(args.data_dir, "valid", label_to_id, device)
    if not train_records or not valid_records:
        raise RuntimeError("prepared train/valid data has no usable records")

    atom_feature_dim = train_records[0]["product"]["x"].shape[1]
    model = RetroMTGRAdapter(atom_feature_dim, args.hidden_dim, len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-5)
    loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0
    best_loss = math.inf
    best_epoch = 0
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume.get("format") != "mechet-retro-mtgr-adapter-v1":
            raise ValueError("unsupported resume checkpoint format")
        if resume.get("model_configuration") != model.configuration():
            raise ValueError("resume checkpoint model configuration does not match arguments")
        if [item["key"] for item in resume.get("labels", [])] != [item["key"] for item in labels]:
            raise ValueError("resume checkpoint label vocabulary does not match data")
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"])
        best_epoch = start_epoch
        best_loss = float(resume.get("metrics", {}).get("loss", math.inf))
        print(json.dumps({
            "event": "resumed",
            "checkpoint": str(args.resume.resolve()),
            "start_epoch": start_epoch,
            "best_loss": best_loss,
        }, sort_keys=True), flush=True)
    initial_train = evaluate(model, train_records, loss_fn, args.batch_size, device)
    initial_valid = evaluate(model, valid_records, loss_fn, args.batch_size, device)
    history: list[dict[str, Any]] = [{
        "epoch": start_epoch,
        "train": initial_train,
        "valid": initial_valid,
    }]
    no_improvement = 0
    last_epoch = start_epoch
    stopped_early = False

    for epoch in range(start_epoch + 1, args.epochs + 1):
        last_epoch = epoch
        model.train()
        order = list(range(len(train_records)))
        random.shuffle(order)
        for offset in range(0, len(order), args.batch_size):
            records = [train_records[index] for index in order[offset:offset + args.batch_size]]
            batch = collate_records(records, device)
            optimizer.zero_grad(set_to_none=True)
            hidden, product_embeddings = model.encode_batch(batch["product"])
            _, precursor_embeddings = model.encode_batch(batch["precursor"])
            center_logits = model.score_bonds_batch(batch["product"], hidden, product_embeddings)
            left_logits, right_logits = model.score_leaving_groups_batch(
                batch["product"], hidden, product_embeddings, batch["center"]
            )
            positive_distance = ((product_embeddings - precursor_embeddings) ** 2).mean(dim=1)
            if len(records) > 1:
                negative_distance = ((product_embeddings - precursor_embeddings.roll(1, 0)) ** 2).mean(dim=1)
                alignment_loss = torch.relu(0.5 + positive_distance - negative_distance).mean()
            else:
                alignment_loss = positive_distance.mean()
            loss = (
                loss_fn(center_logits, batch["center"])
                + 0.5 * (loss_fn(left_logits, batch["left"]) + loss_fn(right_logits, batch["right"]))
                + args.alignment_weight * alignment_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, train_records, loss_fn, args.batch_size, device)
            valid_metrics = evaluate(model, valid_records, loss_fn, args.batch_size, device)
            entry = {"epoch": epoch, "train": train_metrics, "valid": valid_metrics}
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)
            if valid_metrics["loss"] < best_loss - args.min_delta:
                best_loss = valid_metrics["loss"]
                best_epoch = epoch
                no_improvement = 0
                save_checkpoint(
                    args.output_dir / "best.pt", model, optimizer, epoch, labels, args, valid_metrics
                )
            else:
                no_improvement += 1
            if args.patience and no_improvement >= args.patience:
                stopped_early = True
                print(json.dumps({
                    "event": "early_stopping",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_valid_loss": best_loss,
                    "patience": args.patience,
                }, sort_keys=True), flush=True)
                break

    final_train = evaluate(model, train_records, loss_fn, args.batch_size, device)
    final_valid = evaluate(model, valid_records, loss_fn, args.batch_size, device)
    save_checkpoint(
        args.output_dir / "last.pt", model, optimizer, last_epoch, labels, args, final_valid
    )
    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "train_loss", "valid_loss", "train_center_accuracy",
            "train_left_label_accuracy", "train_right_label_accuracy",
            "train_joint_target_accuracy", "valid_joint_target_accuracy",
        ))
        writer.writeheader()
        for entry in history:
            writer.writerow({
                "epoch": entry["epoch"],
                "train_loss": entry["train"]["loss"],
                "valid_loss": entry["valid"]["loss"],
                "train_center_accuracy": entry["train"]["center_accuracy"],
                "train_left_label_accuracy": entry["train"]["left_label_accuracy"],
                "train_right_label_accuracy": entry["train"]["right_label_accuracy"],
                "train_joint_target_accuracy": entry["train"]["joint_target_accuracy"],
                "valid_joint_target_accuracy": entry["valid"]["joint_target_accuracy"],
            })
    data_manifest_path = args.data_dir / "manifest.json"
    data_manifest = (
        json.loads(data_manifest_path.read_text(encoding="utf-8"))
        if data_manifest_path.is_file()
        else {}
    )
    diagnostic = bool(data_manifest.get("diagnostic_duplicate_splits", False))
    report = {
        "artifact_type": (
            "retro_mtgr_overfit_training_report"
            if diagnostic
            else "retro_mtgr_full_training_report"
        ),
        "diagnostic_duplicate_splits": diagnostic,
        "device": str(device),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "train_skipped": train_skipped,
        "valid_skipped": valid_skipped,
        "label_vocabulary_source_splits": ["train"],
        "label_count": len(labels),
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_valid": initial_valid,
        "final_valid": final_valid,
        "loss_reduction_fraction": 1.0 - final_train["loss"] / initial_train["loss"],
        "resumed_from": str(args.resume.resolve()) if args.resume else None,
        "start_epoch": start_epoch,
        "epochs_completed": last_epoch,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "best_epoch": best_epoch,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "last_checkpoint": str((args.output_dir / "last.pt").resolve()),
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
