#!/usr/bin/env python3
"""DDP training that expands frozen reaction traces only inside DataLoader workers."""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from build_graph_electron_full import EXPECTED, compile_row, sha256_file
from mechet.electron_policy_protocol import STAGE_STATE_BC, STAGE_TRAJECTORY_BC
from mechet.graph_electron_policy import ACTION_FAMILIES, GraphElectronPolicy
from train_graph_electron_full_batched import PreparedDecision, prepare_value


EXPECTED_DECISIONS = {"train": 2644501, "valid": 29625, "test": 297541}


def build_offset_index(source: Path, output: Path, expected_rows: int) -> None:
    """Write one uint64 per reaction; this is an index, never expanded supervision."""

    temporary = output.with_suffix(output.suffix + ".partial")
    offsets = array("Q")
    offset = 0
    started = time.time()
    with source.open("rb") as handle:
        for line in handle:
            offsets.append(offset)
            offset += len(line)
            if len(offsets) % 50000 == 0:
                print(
                    f"[graph-online-index] reactions={len(offsets)}/{expected_rows} "
                    f"rate={len(offsets) / max(1e-6, time.time() - started):.1f}/s",
                    flush=True,
                )
    if len(offsets) != expected_rows:
        raise ValueError(f"source rows changed: {len(offsets)} != {expected_rows}")
    with temporary.open("wb") as handle:
        offsets.tofile(handle)
    os.replace(temporary, output)
    print(
        f"[graph-online-index] complete reactions={len(offsets)} bytes={output.stat().st_size}",
        flush=True,
    )


class ReactionOnlineDataset(Dataset[list[PreparedDecision]]):
    """Random-access reactions, expanded into transient state/action supervision."""

    def __init__(
        self,
        source: Path,
        offset_index: Path,
        *,
        rank: int,
        world_size: int,
        seed: int,
        skip_reactions: int = 0,
        reaction_limit_per_rank: int | None = None,
    ) -> None:
        offsets = array("Q")
        with offset_index.open("rb") as handle:
            offsets.fromfile(handle, offset_index.stat().st_size // offsets.itemsize)
        selected = list(range(rank, len(offsets), world_size))
        random.Random(seed).shuffle(selected)
        if skip_reactions < 0 or skip_reactions > len(selected):
            raise ValueError("skip_reactions lies outside the rank-local shard")
        selected = selected[skip_reactions:]
        if reaction_limit_per_rank is not None:
            if reaction_limit_per_rank < 1:
                raise ValueError("reaction limit must be positive")
            selected = selected[:reaction_limit_per_rank]
        self.source = source
        self.entries = tuple((index, int(offsets[index])) for index in selected)
        self._handle = None

    def __len__(self) -> int:
        return len(self.entries)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def __getitem__(self, index: int) -> list[PreparedDecision]:
        if self._handle is None:
            self._handle = self.source.open("rb")
        reaction_index, offset = self.entries[index]
        self._handle.seek(offset)
        line = self._handle.readline().decode("utf-8")
        compiled_index, decisions = compile_row((reaction_index, line))
        if compiled_index != reaction_index:
            raise RuntimeError("reaction compiler index drift")
        return [prepare_value(decision) for decision in decisions]


def collate_reactions(
    reactions: list[list[PreparedDecision]],
) -> list[PreparedDecision]:
    return [decision for reaction in reactions for decision in reaction]


def save_checkpoint(
    output: Path,
    *,
    model: GraphElectronPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    source_manifest: dict[str, Any],
    epoch: int,
    update: int,
    processed_reactions: int,
    processed_decisions: int,
) -> Path:
    temporary = output / f"checkpoint-epoch{epoch}-update{update}.pt.partial"
    final = output / f"checkpoint-epoch{epoch}-update{update}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "epoch": epoch,
            "update": update,
            "processed_reactions": processed_reactions,
            "processed_decisions": processed_decisions,
            "source_manifest": source_manifest,
            "data_contract": "reaction_level_online_expansion_v1",
        },
        temporary,
    )
    os.replace(temporary, final)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--verified-offset-index",
        type=Path,
        help=(
            "Reuse an offset index produced only after this trainer verified the "
            "frozen source SHA256; avoids re-hashing the same 5 GB source."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--reactions-per-batch", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--prefetch", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument(
        "--reaction-limit-per-rank",
        type=int,
        default=None,
        help="Fixed shuffled reaction subset used by intentional overfit smokes.",
    )
    parser.add_argument(
        "--amp-dtype", choices=("auto", "bf16", "fp16"), default="auto"
    )
    parser.add_argument("--stage", choices=(STAGE_STATE_BC, STAGE_TRAJECTORY_BC), default=STAGE_STATE_BC)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Exact same-stage checkpoint including optimizer state.",
    )
    parser.add_argument(
        "--resume-update", type=int, default=0,
        help="Completed same-stage updates to skip deterministically.",
    )
    parser.add_argument("--log-updates", type=int, default=10)
    parser.add_argument("--checkpoint-updates", type=int, default=1000)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Optional smoke-test cap on rank-synchronous optimizer updates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.reactions_per_batch, args.cpu_workers, args.prefetch) < 1:
        raise SystemExit("epochs, batch size, workers, and prefetch must be positive")
    if args.max_updates is not None and args.max_updates < 1:
        raise SystemExit("max-updates must be positive when provided")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != args.expected_world_size:
        raise SystemExit(
            f"world-size mismatch: expected {args.expected_world_size}, got {world_size}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    manifest = json.loads(args.source_manifest.read_text())
    if not manifest.get("strict_trace_universe_complete"):
        raise SystemExit("source is not the frozen strict trace universe")
    train_meta = manifest["splits"]["train"]
    source = args.source_root / "train.jsonl"
    if int(train_meta["rows"]) != EXPECTED["train"]:
        raise SystemExit("reaction denominator mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    offset_index = args.verified_offset_index or args.output / "train.reaction_offsets.u64"
    if rank == 0:
        if args.verified_offset_index is not None:
            expected_bytes = EXPECTED["train"] * array("Q").itemsize
            if not offset_index.is_file() or offset_index.stat().st_size != expected_bytes:
                raise SystemExit("verified offset index is missing or has the wrong row count")
            print(
                f"[graph-online] source_verified provenance_index={offset_index} "
                f"reactions={EXPECTED['train']} manifest_sha256={train_meta['sha256']} "
                "expanded_decisions_on_disk=0",
                flush=True,
            )
        else:
            actual_sha = sha256_file(source)
            if actual_sha != train_meta["sha256"]:
                raise SystemExit("frozen train source hash mismatch")
            print(
                f"[graph-online] source_verified reactions={EXPECTED['train']} sha256={actual_sha} "
                "expanded_decisions_on_disk=0",
                flush=True,
            )
            if not offset_index.exists():
                build_offset_index(source, offset_index, EXPECTED["train"])
    dist.barrier()

    if args.initialize_from and args.resume_from:
        raise SystemExit("initialize-from and resume-from are mutually exclusive")
    skipped_reactions = args.resume_update * args.reactions_per_batch
    dataset = ReactionOnlineDataset(
        source,
        offset_index,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        skip_reactions=skipped_reactions,
        reaction_limit_per_rank=args.reaction_limit_per_rank,
    )
    local_reactions = len(dataset)
    batch_counts = torch.tensor(
        [math.ceil(local_reactions / args.reactions_per_batch)],
        dtype=torch.long,
        device=device,
    )
    minimum = batch_counts.clone()
    maximum = batch_counts.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if int(minimum) != int(maximum):
        raise SystemExit("reaction sharding gives unequal DDP update counts")
    available_updates = int(maximum)
    total_updates = (
        min(available_updates, args.max_updates)
        if args.max_updates is not None
        else available_updates
    )
    loader = DataLoader(
        dataset,
        batch_size=args.reactions_per_batch,
        shuffle=False,
        num_workers=args.cpu_workers,
        collate_fn=collate_reactions,
        multiprocessing_context="spawn",
        persistent_workers=True,
        prefetch_factor=args.prefetch,
        pin_memory=False,
        drop_last=False,
    )

    model = GraphElectronPolicy(
        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=0.1
    )
    model.freeze_enumerated_action_heads()
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if int(resume_checkpoint.get("update") or -1) != args.resume_update:
            raise SystemExit("resume checkpoint/update mismatch")
        if str((resume_checkpoint.get("config") or {}).get("stage")) != args.stage:
            raise SystemExit("resume checkpoint stage mismatch")
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    elif args.initialize_from:
        checkpoint = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in distributed_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
    amp_dtype = (
        torch.bfloat16
        if args.amp_dtype == "bf16"
        else torch.float16
        if args.amp_dtype == "fp16"
        else torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)
    family_counts = Counter(train_meta.get("family_counts") or {})
    if not family_counts:
        family_counts = Counter(
            {
                "IMPORT_ENV": 743579,
                "IMPORT_REACTIVE": 600403,
                "FLOW": 1042187,
                "FINISH": 257167,
                "BE_DELTA": 1165,
            }
        )
    mean_family = sum(family_counts.values()) / len(ACTION_FAMILIES)
    family_weights = {
        family: min(8.0, math.sqrt(mean_family / max(1, family_counts[family])))
        for family in ACTION_FAMILIES
    }

    if rank == 0:
        print(
            f"[graph-online] world_size={world_size} reactions={EXPECTED['train']} "
            f"conceptual_decisions={EXPECTED_DECISIONS['train']} expanded_decisions_on_disk=0 "
            f"reactions_per_batch_rank={args.reactions_per_batch} updates_remaining={total_updates} "
            f"resume_update={args.resume_update} "
            f"cpu_workers_total={args.cpu_workers * world_size} stage={args.stage} "
            f"amp_dtype={str(amp_dtype).replace('torch.', '')}",
            flush=True,
        )
    try:
        for epoch_index in range(args.epochs):
            distributed_model.train()
            loss_numerator = torch.zeros((), dtype=torch.float64, device=device)
            weight_denominator = torch.zeros((), dtype=torch.float64, device=device)
            processed_reactions = processed_decisions = 0
            started = time.time()
            epoch_update_base = args.resume_update + epoch_index * total_updates
            for relative_update, prepared in enumerate(loader, 1):
                if relative_update > total_updates:
                    break
                update = epoch_update_base + relative_update
                reaction_ids = {str(item.value["reaction_id"]) for item in prepared}
                real_reactions = len(reaction_ids)
                weights = torch.tensor(
                    [
                        family_weights[str(item.value["kind"])]
                        / max(1, int(item.value["reaction_decision_count"]))
                        for item in prepared
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    losses = distributed_model(
                        [item.value for item in prepared],
                        [item.current_graph for item in prepared],
                        [item.target_graph for item in prepared],
                        (
                            [item.history for item in prepared]
                            if args.stage == STAGE_TRAJECTORY_BC
                            else None
                        ),
                    )
                    weighted = losses * weights
                    loss = weighted.sum() / weights.sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                processed_reactions += real_reactions
                processed_decisions += len(prepared)
                loss_numerator += weighted.detach().sum().to(torch.float64)
                weight_denominator += weights.detach().sum().to(torch.float64)
                if rank == 0 and update % args.log_updates == 0:
                    elapsed = time.time() - started
                    print(
                        f"[graph-online] epoch={epoch_index + 1}/{args.epochs} "
                        f"update={update}/{args.resume_update + args.epochs * total_updates} "
                        f"reactions_rank0={processed_reactions}/{local_reactions} "
                        f"decisions_transient_rank0={processed_decisions} "
                        f"mean_loss_rank0={float(loss_numerator / weight_denominator):.5f} "
                        f"reaction_rate_rank0={processed_reactions / max(1e-6, elapsed):.2f}/s",
                        flush=True,
                    )
                if rank == 0 and update % args.checkpoint_updates == 0:
                    checkpoint_path = save_checkpoint(
                        args.output,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        source_manifest=manifest,
                        epoch=epoch_index + 1,
                        update=update,
                        processed_reactions=processed_reactions,
                        processed_decisions=processed_decisions,
                    )
                    print(f"[graph-online] checkpoint={checkpoint_path}", flush=True)
            totals = torch.stack(
                (
                    loss_numerator,
                    weight_denominator,
                    torch.tensor(processed_reactions, dtype=torch.float64, device=device),
                    torch.tensor(processed_decisions, dtype=torch.float64, device=device),
                )
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if rank == 0:
                checkpoint_path = save_checkpoint(
                    args.output,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    source_manifest=manifest,
                    epoch=epoch_index + 1,
                    update=epoch_update_base + total_updates,
                    processed_reactions=processed_reactions,
                    processed_decisions=processed_decisions,
                )
                report = {
                    "status": "completed" if epoch_index + 1 == args.epochs else "training",
                    "trainer": "reaction_level_online_expansion_ddp_bf16_v1",
                    "stage": args.stage,
                    "reaction_denominator": EXPECTED,
                    "expanded_decisions_on_disk": 0,
                    "epoch": epoch_index + 1,
                    "global_reactions": int(totals[2]),
                    "global_transient_decisions": int(totals[3]),
                    "mean_loss": float(totals[0] / totals[1]),
                    "wall_seconds": time.time() - started,
                    "latest_checkpoint": str(checkpoint_path),
                }
                (args.output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"[graph-online] epoch_complete={json.dumps(report, sort_keys=True)}", flush=True)
            dist.barrier()
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            try:
                iterator._shutdown_workers()
            except RuntimeError as exc:
                # A capped smoke run deliberately exits while workers still
                # hold prefetched reactions.  Some RDKit/PyTorch combinations
                # report their expected termination as SIGABRT.  The optimizer
                # updates and final checkpoint are already synchronized above;
                # never hide this error during an uncapped/full run.
                if args.max_updates is None:
                    raise
                if rank == 0:
                    print(
                        f"[graph-online] capped_worker_shutdown_warning={exc}",
                        flush=True,
                    )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
