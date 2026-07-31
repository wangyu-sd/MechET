#!/usr/bin/env python3
"""Train the compact graph-pointer Forward Electron-Flow Expert."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from mechet.forward_expert import (
    ElectronMove,
    ForwardElectronExpert,
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("install mechet[forward] for YAML configs") from exc
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:limit] if limit else rows


def container_index(values, target):
    for index, value in enumerate(values):
        if value == target:
            return index
        # ATOM and LP identify the same atom when used as a receiving LP slot.
        if value.atoms == target.atoms and {value.kind, target.kind} <= {
            "ATOM",
            "LP",
        }:
            return index
    raise ValueError(f"container label is not available in this state: {target.id}")


def choose_negative(
    row: dict[str, Any],
    product_pool: list[str],
    rng: random.Random,
) -> str | None:
    competitors = [
        value
        for value in row.get("competitor_products") or []
        if value and value != row.get("products")
    ]
    if competitors:
        return rng.choice(competitors)
    candidates = [value for value in product_pool if value != row.get("products")]
    return rng.choice(candidates) if candidates else None


def compute_loss(
    model: ForwardElectronExpert,
    row: dict[str, Any],
    *,
    negative_product: str | None,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    total = torch.zeros((), device=model.device)
    stats: dict[str, float] = {}
    moves = [ElectronMove.parse(value) for value in row.get("moves") or []]
    if moves:
        sources, sinks, source_logits, sink_logits = model.move_logits(
            row["state_smiles"],
            conditions=row.get("conditions"),
        )
        source_losses = []
        sink_losses = []
        for move in moves:
            source_y = container_index(sources, move.source)
            sink_y = container_index(sinks, move.sink)
            source_losses.append(
                F.cross_entropy(
                    source_logits.unsqueeze(0),
                    torch.tensor([source_y], device=model.device),
                )
            )
            sink_losses.append(
                F.cross_entropy(
                    sink_logits[source_y].unsqueeze(0),
                    torch.tensor([sink_y], device=model.device),
                )
            )
        source_loss = torch.stack(source_losses).mean()
        sink_loss = torch.stack(sink_losses).mean()
        total = (
            total
            + weights.get("source", 1.0) * source_loss
            + weights.get("sink", 1.0) * sink_loss
        )
        stats.update(
            source_loss=float(source_loss.detach()),
            sink_loss=float(sink_loss.detach()),
        )

    reactants = row.get("reactants") or row.get("state_smiles")
    product = row.get("products") or row.get("target_product")
    positive_logit = model.reaction_score(
        reactants,
        product,
        conditions=row.get("conditions"),
    )
    reaction_loss = F.binary_cross_entropy_with_logits(
        positive_logit,
        torch.ones_like(positive_logit),
    )
    stats["positive_score"] = float(torch.sigmoid(positive_logit).detach())
    if negative_product:
        negative_logit = model.reaction_score(
            reactants,
            negative_product,
            conditions=row.get("conditions"),
        )
        reaction_loss = reaction_loss + F.binary_cross_entropy_with_logits(
            negative_logit,
            torch.zeros_like(negative_logit),
        )
        margin = float(weights.get("margin_value", 0.2))
        ranking_loss = F.relu(
            torch.tensor(margin, device=model.device)
            - positive_logit
            + negative_logit
        )
        total = total + weights.get("rank", 0.5) * ranking_loss
        stats["negative_score"] = float(torch.sigmoid(negative_logit).detach())
        stats["ranking_loss"] = float(ranking_loss.detach())
    total = total + weights.get("reaction", 1.0) * reaction_loss
    stats["reaction_loss"] = float(reaction_loss.detach())
    stats["loss"] = float(total.detach())
    return total, stats


@torch.no_grad()
def evaluate(
    model,
    rows,
    *,
    product_pool,
    weights,
    seed=0,
    max_examples=0,
):
    model.eval()
    rng = random.Random(seed)
    totals = []
    move_correct = move_total = 0
    target_wins = target_pairs = 0
    for row in rows[: max_examples or None]:
        try:
            negative = choose_negative(row, product_pool, rng)
            loss, _ = compute_loss(
                model,
                row,
                negative_product=negative,
                weights=weights,
            )
            totals.append(float(loss))
            moves = [ElectronMove.parse(value) for value in row.get("moves") or []]
            if moves:
                sources, sinks, source_logits, sink_logits = model.move_logits(
                    row["state_smiles"],
                    conditions=row.get("conditions"),
                )
                source_pred = int(source_logits.argmax())
                for move in moves:
                    source_y = container_index(sources, move.source)
                    sink_y = container_index(sinks, move.sink)
                    move_correct += int(
                        source_pred == source_y
                        and int(sink_logits[source_y].argmax()) == sink_y
                    )
                    move_total += 1
            if negative:
                reactants = row.get("reactants") or row["state_smiles"]
                conditions = row.get("conditions")
                positive = float(
                    model.reaction_score(
                        reactants,
                        row.get("products") or row["target_product"],
                        conditions=conditions,
                    )
                )
                negative_score = float(
                    model.reaction_score(
                        reactants,
                        negative,
                        conditions=conditions,
                    )
                )
                target_wins += int(positive > negative_score)
                target_pairs += 1
        except Exception:
            continue
    return {
        "loss": sum(totals) / max(len(totals), 1),
        "move_top1": move_correct / max(move_total, 1),
        "selectivity_pair_accuracy": target_wins / max(target_pairs, 1),
        "n_examples": len(totals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 17))
    random.seed(seed)
    torch.manual_seed(seed)
    train_rows = load_jsonl(
        Path(cfg["train_file"]),
        int(cfg.get("limit_train", 0) or 0),
    )
    valid_rows = load_jsonl(
        Path(cfg["validation_file"]),
        int(cfg.get("limit_validation", 0) or 0),
    )
    if not train_rows:
        raise ValueError("empty training set")
    product_pool = sorted(
        {
            str(row.get("products") or row.get("target_product") or "")
            for row in train_rows + valid_rows
            if row.get("products") or row.get("target_product")
        }
    )
    model_cfg = dict(cfg.get("model") or {})
    model = ForwardElectronExpert(**model_cfg).to(args.device)
    if args.resume:
        model = ForwardElectronExpert.load(args.resume, device=args.device)
    output = Path(cfg.get("output_dir", "outputs/forward_expert"))
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "train": len(train_rows),
                    "valid": len(valid_rows),
                    "products": len(product_pool),
                    "model": model.config,
                    "device": args.device,
                },
                indent=2,
            )
        )
        return 0

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg.get("epochs", 10))
    accumulation = int(cfg.get("gradient_accumulation_steps", 8))
    clip = float(cfg.get("max_grad_norm", 1.0))
    weights = dict(
        cfg.get("loss_weights")
        or {
            "source": 1.0,
            "sink": 1.0,
            "reaction": 1.0,
            "rank": 0.5,
            "margin_value": 0.2,
        }
    )
    rng = random.Random(seed)
    best = math.inf
    global_step = 0
    log_path = output / "train_log.jsonl"
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        optimizer.zero_grad(set_to_none=True)
        running = []
        accepted = 0
        for index, row in enumerate(train_rows):
            try:
                negative = choose_negative(row, product_pool, rng)
                loss, stats = compute_loss(
                    model,
                    row,
                    negative_product=negative,
                    weights=weights,
                )
                (loss / accumulation).backward()
                running.append(stats["loss"])
                accepted += 1
            except (ValueError, RuntimeError) as exc:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "row": row.get("id"),
                                "skipped": str(exc),
                            }
                        )
                        + "\n"
                    )
                continue
            if accepted % accumulation == 0 or index == len(train_rows) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        metrics = evaluate(
            model,
            valid_rows,
            product_pool=product_pool,
            weights=weights,
            seed=seed + epoch,
            max_examples=int(cfg.get("max_validation_examples", 0) or 0),
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": sum(running) / max(len(running), 1),
            "accepted_train_rows": accepted,
            **{f"valid_{key}": value for key, value in metrics.items()},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        model.save(output / "last", metadata={"config": cfg, "metrics": record})
        if metrics["loss"] < best:
            best = metrics["loss"]
            model.save(
                output / "best",
                metadata={"config": cfg, "metrics": record},
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
