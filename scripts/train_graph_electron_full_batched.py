#!/usr/bin/env python3
"""Full 8-GPU training for the no-enumeration direct graph policy."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from mechet.electron_policy_protocol import (
    CompressedTrajectory,
    STAGE_STATE_BC,
    STAGE_TRAJECTORY_BC,
)
from mechet.graph_electron_policy import (
    ACTION_FAMILIES,
    GraphElectronPolicy,
    GraphTensor,
    smiles_to_graph,
)
from mechet.graph_fragment_actions import ReactiveFragmentProgram


@dataclass(frozen=True)
class PreparedDecision:
    value: dict[str, Any]
    current_graph: GraphTensor
    target_graph: GraphTensor
    history: CompressedTrajectory


def prepare_line(line: str) -> PreparedDecision:
    value = json.loads(line)
    if value["kind"] in {"IMPORT_ENV", "IMPORT_REACTIVE"}:
        value["program"] = ReactiveFragmentProgram.from_dict(value["program"])
    target_graph = smiles_to_graph(value["target"])
    current_graph = smiles_to_graph(
        value["current"],
        target_smiles=value["target"],
        target_maps=target_graph.maps,
    )
    return PreparedDecision(
        value,
        current_graph,
        target_graph,
        CompressedTrajectory.from_dict(value.get("history")),
    )


class JsonlDecisionDataset(Dataset[PreparedDecision]):
    """Indexed JSONL dataset whose workers own parsing and RDKit conversion.

    JSONL remains the auditable source contract.  ``DataLoader`` supplies the
    standard worker lifecycle, deterministic batching, exception propagation,
    and persistent prefetching.  Production runs use
    :class:`TensorChunkDecisionDataset`; this backend remains an auditable
    fallback and the one-time tensor-cache compiler's source.
    """

    def __init__(self, path: Path):
        self.path = path
        self.offsets: list[int] = []
        offset = 0
        with path.open("rb") as handle:
            for line in handle:
                self.offsets.append(offset)
                offset += len(line)
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["_handle"] = None
        return value

    def __getitem__(self, index: int) -> PreparedDecision:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return prepare_line(self._handle.readline().decode("utf-8"))


def collate_prepared(items: list[PreparedDecision]) -> list[PreparedDecision]:
    return items


def write_tensor_chunk(items: list[PreparedDecision], path: Path) -> None:
    """Write a trusted tensor chunk with graph de-duplication inside the chunk."""

    graph_ids: dict[tuple[str, ...], int] = {}
    graphs: list[GraphTensor] = []

    def add(key: tuple[str, ...], graph: GraphTensor) -> int:
        found = graph_ids.get(key)
        if found is not None:
            return found
        index = len(graphs)
        graph_ids[key] = index
        graphs.append(graph)
        return index

    records = []
    for item in items:
        value = item.value
        target = str(value["target"])
        current = str(value["current"])
        records.append(
            {
                "value": value,
                "target_graph": add(("target", target), item.target_graph),
                "current_graph": add(("current", target, current), item.current_graph),
                "history": item.history.to_dict(),
            }
        )
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save({"graphs": graphs, "records": records}, temporary)
    os.replace(temporary, path)


class TensorChunkDecisionDataset(Dataset[PreparedDecision]):
    """Memory-bounded map dataset over precompiled graph/action tensor chunks."""

    def __init__(self, root: Path, *, cache_chunks: int = 2):
        manifest = json.loads((root / "manifest.json").read_text())
        self.root = root
        self.chunks = tuple(manifest["chunks"])
        self.starts: list[int] = []
        total = 0
        for item in self.chunks:
            self.starts.append(total)
            total += int(item["rows"])
        self.rows = total
        if self.rows != int(manifest["rows"]):
            raise ValueError("tensor-cache row count mismatch")
        self.cache_chunks = max(1, int(cache_chunks))
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return self.rows

    def __getstate__(self) -> dict[str, Any]:
        value = dict(self.__dict__)
        value["_cache"] = OrderedDict()
        return value

    def _chunk(self, index: int) -> dict[str, Any]:
        value = self._cache.get(index)
        if value is not None:
            self._cache.move_to_end(index)
            return value
        value = torch.load(
            self.root / self.chunks[index]["file"],
            map_location="cpu",
            weights_only=False,
        )
        self._cache[index] = value
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index: int) -> PreparedDecision:
        if index < 0:
            index += self.rows
        if index < 0 or index >= self.rows:
            raise IndexError(index)
        chunk_index = bisect_right(self.starts, index) - 1
        chunk = self._chunk(chunk_index)
        record = chunk["records"][index - self.starts[chunk_index]]
        graphs = chunk["graphs"]
        return PreparedDecision(
            record["value"],
            graphs[int(record["current_graph"])],
            graphs[int(record["target_graph"])],
            CompressedTrajectory.from_dict(record["history"]),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--prefetch", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-updates", type=int, default=20)
    parser.add_argument("--checkpoint-updates", type=int, default=2000)
    parser.add_argument(
        "--tensor-cache",
        type=Path,
        help="Root containing train.rankXX tensor-cache directories.",
    )
    parser.add_argument(
        "--stage",
        choices=(STAGE_STATE_BC, STAGE_TRAJECTORY_BC),
        default=STAGE_STATE_BC,
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Stage-1 checkpoint used to initialize trajectory behavior cloning.",
    )
    return parser.parse_args()


def save_checkpoint(
    output: Path,
    *,
    model: GraphElectronPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    epoch: int,
    update: int,
    processed_rows: int,
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
            "processed_rows": processed_rows,
            "source_manifest": manifest,
        },
        temporary,
    )
    os.replace(temporary, final)
    return final


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.cpu_workers < 1 or args.prefetch < 1:
        raise SystemExit("batch-size, cpu-workers and prefetch must be positive")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8:
        raise SystemExit(f"full run requires exactly 8 ranks, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = json.loads((args.data / "manifest.json").read_text())
    expected = {"train": 257167, "valid": 2890, "test": 28967}
    if manifest.get("reaction_denominator") != expected:
        raise SystemExit("full decision manifest denominator mismatch")
    train_meta = manifest["splits"]["train"]
    family_counts = {
        str(key): int(value) for key, value in train_meta["family_counts"].items()
    }
    family_reference = max(family_counts.values())
    family_weights = {
        key: min(12.0, math.sqrt(family_reference / max(1, value)))
        for key, value in family_counts.items()
    }
    mean_reaction_decisions = float(train_meta["decisions"]) / float(
        train_meta["reactions"]
    )
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
    if manifest.get("artifact_type") != "graph_electron_direct_pointer_decisions":
        raise SystemExit("trainer requires direct-pointer decision artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = GraphElectronPolicy(
        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=0.1
    )
    model.freeze_enumerated_action_heads()
    model = model.to(device)
    if args.initialize_from is not None:
        if args.stage != STAGE_TRAJECTORY_BC:
            raise SystemExit("--initialize-from is reserved for trajectory_bc")
        parent = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
        incompatible = model.load_state_dict(parent["model"], strict=False)
        allowed_missing = {
            name
            for name in model.state_dict()
            if name.startswith("history_encoder.")
            or name.startswith("history_residual.")
            or name == "history_gate"
        }
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise SystemExit(
                "stage-1 checkpoint is incompatible beyond the new history adapter: "
                f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
            )
    distributed_model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
        gradient_as_bucket_view=True,
    )
    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in distributed_model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
    )
    # v2 starts clean; its checkpoints are deliberately incompatible with the
    # interrupted per-decision optimizer trajectory even though shapes match.
    if any(args.output.glob("checkpoint-*.pt")):
        raise SystemExit("batched output directory already contains a checkpoint")

    dataset: Dataset[PreparedDecision]
    if args.tensor_cache is not None:
        dataset = TensorChunkDecisionDataset(
            args.tensor_cache / f"train.rank{rank:02d}"
        )
        loader_backend = "tensor_cache"
    else:
        dataset = JsonlDecisionDataset(shard)
        loader_backend = "jsonl_audit_fallback"
    if len(dataset) != local_rows:
        raise SystemExit(f"rank {rank}: shard rows changed: {len(dataset)} != {local_rows}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpu_workers,
        collate_fn=collate_prepared,
        multiprocessing_context="spawn",
        persistent_workers=True,
        prefetch_factor=args.prefetch,
        pin_memory=False,
        drop_last=False,
    )
    if int(manifest.get("schema_version", 0)) < 4:
        raise SystemExit("repaired graph training requires schema-v4 first-use decisions")
    try:
        if rank == 0:
            print(
                f"[graph-batched] world_size={world_size} train_decisions={train_meta['decisions']} "
                f"rank_rows={rank_rows} batch_per_rank={args.batch_size} "
                f"global_batch={args.batch_size * world_size} cpu_workers_total="
                f"{args.cpu_workers * world_size} process_prefetch={args.prefetch} "
                f"loader=torch.DataLoader/{loader_backend} stage={args.stage} "
                "action_contract=factorized_pointer_with_legality_masks "
                f"distributed=torch.DDP parameters={sum(p.numel() for p in model.parameters())}",
                flush=True,
            )
        dist.barrier()
        for epoch_index in range(args.epochs):
            distributed_model.train()
            loss_sum = 0.0
            real_rows = 0
            updates = 0
            started = time.time()
            batches = iter(loader)
            total_updates = math.ceil(max_rows / args.batch_size)
            first: PreparedDecision | None = None
            loss_sum_tensor = torch.zeros((), dtype=torch.float64, device=device)
            weight_sum_tensor = torch.zeros((), dtype=torch.float64, device=device)
            for update_index in range(total_updates):
                try:
                    prepared = next(batches)
                except StopIteration:
                    prepared = []
                if prepared and first is None:
                    first = prepared[0]
                real_count = len(prepared)
                if first is None:
                    raise RuntimeError("cannot pad an empty decision shard")
                weights = [
                    family_weights[str(item.value["kind"])]
                    * mean_reaction_decisions
                    / max(1, int(item.value.get("reaction_decision_count") or 1))
                    for item in prepared
                ]
                while len(prepared) < args.batch_size:
                    prepared.append(first)
                weights.extend([0.0] * (args.batch_size - real_count))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                    weight_tensor = torch.tensor(weights, device=device)
                    weighted_losses = losses * weight_tensor
                    batch_loss = weighted_losses.sum() / weight_tensor.sum().clamp_min(1.0)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(distributed_model.parameters(), 1.0)
                optimizer.step()
                updates += 1
                real_rows += real_count
                loss_sum_tensor += weighted_losses.detach().sum().to(torch.float64)
                weight_sum_tensor += weight_tensor.detach().sum().to(torch.float64)
                if rank == 0 and updates % args.log_updates == 0:
                    elapsed = time.time() - started
                    loss_sum = float(loss_sum_tensor)
                    print(
                        f"[graph-batched] epoch={epoch_index + 1}/{args.epochs} "
                        f"update={updates}/{total_updates} decisions_rank0={real_rows}/{local_rows} "
                        f"mean_loss_rank0={loss_sum / max(1e-9, float(weight_sum_tensor)):.5f} "
                        f"rate_rank0={real_rows / max(1e-6, elapsed):.2f}decision/s",
                        flush=True,
                    )
                if rank == 0 and updates % args.checkpoint_updates == 0:
                    checkpoint = save_checkpoint(
                        args.output,
                        model=model,
                        optimizer=optimizer,
                        args=args,
                        manifest=manifest,
                        epoch=epoch_index + 1,
                        update=updates,
                        processed_rows=real_rows,
                    )
                    print(f"[graph-batched] checkpoint={checkpoint}", flush=True)
            totals = torch.stack(
                (
                    loss_sum_tensor,
                    weight_sum_tensor,
                    torch.tensor(float(real_rows), dtype=torch.float64, device=device),
                )
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            result = {
                "epoch": epoch_index + 1,
                "mean_loss": float(totals[0].item() / totals[1].item()),
                "decisions": int(totals[2].item()),
                "updates": updates,
                "wall_seconds": time.time() - started,
            }
            if rank == 0:
                checkpoint = save_checkpoint(
                    args.output,
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    manifest=manifest,
                    epoch=epoch_index + 1,
                    update=updates,
                    processed_rows=real_rows,
                )
                (args.output / "train_report.json").write_text(
                    json.dumps(
                        {
                            "status": (
                                "completed"
                                if epoch_index + 1 == args.epochs
                                else "training"
                            ),
                            "trainer": "two_track_direct_pointer_ddp_tensor_bf16_v1",
                            "stage": args.stage,
                            "action_families": list(ACTION_FAMILIES),
                            "reaction_denominator": expected,
                            "train_decisions": train_meta["decisions"],
                            "family_weights": family_weights,
                            "latest_epoch": result,
                            "latest_checkpoint": str(checkpoint),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    f"[graph-batched] epoch_complete={result} checkpoint={checkpoint}",
                    flush=True,
                )
            dist.barrier()
    finally:
        # Explicitly release persistent workers before distributed shutdown.
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
