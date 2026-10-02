#!/usr/bin/env python3
"""Eight-GPU synchronized training for full graph-electron decision shards."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import torch
import torch.distributed as dist

from mechet.graph_electron_policy import ACTION_FAMILIES, GraphElectronPolicy
from mechet.graph_fragment_actions import ReactiveFragmentProgram
from train_graph_electron_pilot import decision_loss


def decisions(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            value = json.loads(line)
            if value["kind"] == "IMPORT_REACTIVE":
                value["program"] = ReactiveFragmentProgram.from_dict(value["program"])
            yield value


def synchronize_gradients(model: torch.nn.Module, world_size: int) -> None:
    parameters = [item for item in model.parameters() if item.requires_grad]
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        gradients.append(parameter.grad.reshape(-1))
    flat = torch.cat(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
        offset += count


def latest_checkpoint(output: Path) -> tuple[int, Path | None]:
    candidates = []
    for path in output.glob("checkpoint-epoch*.pt"):
        try:
            candidates.append((int(path.stem.split("epoch")[-1]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--import-negatives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-updates", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise SystemExit(f"full run requires exactly 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest.get("reaction_denominator") != {
        "train": 257167,
        "valid": 2890,
        "test": 28967,
    }:
        raise SystemExit("full decision manifest denominator mismatch")
    train_meta = manifest["splits"]["train"]
    if int(manifest["shard_count"]) != world_size:
        raise SystemExit("decision shard count must equal world size")
    shard_meta = train_meta["shards"][rank]
    shard = args.data / shard_meta["file"]
    local_rows = int(shard_meta["rows"])
    row_tensor = torch.tensor([local_rows], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(row_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, row_tensor)
    rank_rows = [int(value.item()) for value in gathered]
    max_rows = max(rank_rows)
    if max_rows - min(rank_rows) > 1:
        raise SystemExit(f"unbalanced full decision shards: {rank_rows}")
    bank = json.loads((args.data / manifest["environment_fragment_bank"]["file"]).read_text())
    if not bank:
        raise SystemExit("empty environment fragment bank")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = GraphElectronPolicy(
        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, checkpoint = latest_checkpoint(args.output)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
    if rank == 0:
        print(
            f"[graph-full] world_size={world_size} train_reactions=257167 "
            f"train_decisions={train_meta['decisions']} rank_rows={rank_rows} "
            f"parameters={sum(p.numel() for p in model.parameters())} resume_epoch={start_epoch}",
            flush=True,
        )
    dist.barrier()
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        rng = random.Random(args.seed + epoch * 1000 + rank)
        iterator = decisions(shard)
        first: dict[str, Any] | None = None
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        real_rows = 0
        updates = 0
        started = time.time()
        for offset in range(max_rows):
            if offset < local_rows:
                decision = next(iterator)
                if first is None:
                    first = decision
                weight = 1.0
                real_rows += 1
            else:
                if first is None:
                    raise RuntimeError("cannot pad an empty decision shard")
                decision = first
                weight = 0.0
            loss = decision_loss(
                model,
                decision,
                bank=bank,
                negatives=args.import_negatives,
                rng=rng,
            )
            (loss * weight / args.accumulate).backward()
            loss_sum += float(loss.detach()) * weight
            boundary = (offset + 1) % args.accumulate == 0 or offset + 1 == max_rows
            if boundary:
                synchronize_gradients(model, world_size)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if rank == 0 and updates % args.log_updates == 0:
                    elapsed = time.time() - started
                    print(
                        f"[graph-full] epoch={epoch + 1}/{args.epochs} update={updates}/"
                        f"{math.ceil(max_rows / args.accumulate)} decisions_rank0={real_rows}/{local_rows} "
                        f"mean_loss_rank0={loss_sum / max(1, real_rows):.5f} "
                        f"rate_rank0={real_rows / max(1e-6, elapsed):.2f}decision/s",
                        flush=True,
                    )
        totals = torch.tensor([loss_sum, real_rows], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        epoch_result = {
            "epoch": epoch + 1,
            "mean_loss": float(totals[0].item() / totals[1].item()),
            "decisions": int(totals[1].item()),
            "updates": updates,
            "wall_seconds": time.time() - started,
        }
        history.append(epoch_result)
        if rank == 0:
            temporary = args.output / f"checkpoint-epoch{epoch + 1}.pt.partial"
            final = args.output / f"checkpoint-epoch{epoch + 1}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": vars(args),
                    "epoch": epoch + 1,
                    "source_manifest": manifest,
                },
                temporary,
            )
            os.replace(temporary, final)
            (args.output / "train_report.json").write_text(
                json.dumps(
                    {
                        "status": "training" if epoch + 1 < args.epochs else "completed",
                        "action_families": list(ACTION_FAMILIES),
                        "reaction_denominator": manifest["reaction_denominator"],
                        "train_decisions": train_meta["decisions"],
                        "history": history,
                        "latest_checkpoint": str(final),
                    },
                    indent=2,
                    default=str,
                )
                + "\n"
            )
            print(f"[graph-full] epoch_complete={epoch_result} checkpoint={final}", flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
