#!/usr/bin/env python3
"""CPU-prefetched, packed-graph 8-GPU training for the full graph policy."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from typing import Any, Iterator

import torch
import torch.distributed as dist

from mechet.graph_electron_policy import (
    ACTION_FAMILIES,
    GraphElectronPolicy,
    GraphTensor,
    smiles_to_graph,
)
from mechet.graph_fragment_actions import ReactiveFragmentProgram
from mechet.transactional_event_space import MoveInventory
from train_graph_electron_full import synchronize_gradients
from train_graph_electron_pilot import import_candidates


@dataclass(frozen=True)
class PreparedDecision:
    value: dict[str, Any]
    current_graph: GraphTensor
    target_graph: GraphTensor
    inventory: MoveInventory | None


def prepare_line(line: str) -> PreparedDecision:
    value = json.loads(line)
    if value["kind"] == "IMPORT_REACTIVE":
        value["program"] = ReactiveFragmentProgram.from_dict(value["program"])
    target_graph = smiles_to_graph(value["target"])
    current_graph = smiles_to_graph(
        value["current"], target_maps=target_graph.maps
    )
    inventory = (
        MoveInventory.from_state(value["current"])
        if value["kind"] == "FLOW"
        else None
    )
    return PreparedDecision(value, current_graph, target_graph, inventory)


def prepare_lines(lines: list[str]) -> list[PreparedDecision]:
    """Process one complete mini-batch in a worker to amortize IPC overhead."""

    return [prepare_line(line) for line in lines]


def raw_batches(path: Path, batch_size: int, skip: int = 0) -> Iterator[list[str]]:
    with path.open() as handle:
        for _ in range(skip):
            if not handle.readline():
                return
        batch: list[str] = []
        for line in handle:
            batch.append(line)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def prefetched_batches(
    path: Path,
    *,
    batch_size: int,
    skip: int,
    executor: ProcessPoolExecutor,
    prefetch: int,
) -> Iterator[list[PreparedDecision]]:
    source = iter(raw_batches(path, batch_size, skip))
    queue: deque[Future[list[PreparedDecision]]] = deque()

    def schedule(lines: list[str]) -> Future[list[PreparedDecision]]:
        return executor.submit(prepare_lines, lines)

    for _ in range(prefetch):
        try:
            queue.append(schedule(next(source)))
        except StopIteration:
            break
    while queue:
        future = queue.popleft()
        try:
            queue.append(schedule(next(source)))
        except StopIteration:
            pass
        yield future.result()


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
    parser.add_argument("--import-negatives", type=int, default=31)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-updates", type=int, default=20)
    parser.add_argument("--checkpoint-updates", type=int, default=2000)
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
    bank = json.loads(
        (args.data / manifest["environment_fragment_bank"]["file"]).read_text()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = GraphElectronPolicy(
        hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=0.1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # v2 starts clean; its checkpoints are deliberately incompatible with the
    # interrupted per-decision optimizer trajectory even though shapes match.
    if any(args.output.glob("checkpoint-*.pt")):
        raise SystemExit("batched output directory already contains a checkpoint")

    # Parsing SMILES and constructing the legal FLOW inventory are CPU-heavy.
    # Threads do not scale because the Python/RDKit path retains the GIL, so
    # each GPU rank owns independent spawn workers.  Batches (rather than
    # individual rows) cross IPC to keep serialization overhead bounded.
    process_context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.cpu_workers, mp_context=process_context
    ) as executor, ThreadPoolExecutor(max_workers=args.cpu_workers) as bank_executor:
        fragment_graphs = list(
            bank_executor.map(
                lambda fragment: smiles_to_graph(fragment, require_maps=False), bank
            )
        )
        bank_graph = dict(zip(bank, fragment_graphs))
        if rank == 0:
            print(
                f"[graph-batched] world_size={world_size} train_decisions={train_meta['decisions']} "
                f"rank_rows={rank_rows} batch_per_rank={args.batch_size} "
                f"global_batch={args.batch_size * world_size} cpu_workers_total="
                f"{args.cpu_workers * world_size} process_prefetch={args.prefetch} "
                f"env_bank={len(bank)} "
                f"parameters={sum(p.numel() for p in model.parameters())}",
                flush=True,
            )
        dist.barrier()
        for epoch_index in range(args.epochs):
            model.train()
            rng = random.Random(args.seed + epoch_index * 1000 + rank)
            loss_sum = 0.0
            real_rows = 0
            updates = 0
            started = time.time()
            batches = prefetched_batches(
                shard,
                batch_size=args.batch_size,
                skip=0,
                executor=executor,
                prefetch=args.prefetch,
            )
            total_updates = math.ceil(max_rows / args.batch_size)
            first: PreparedDecision | None = None
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
                while len(prepared) < args.batch_size:
                    prepared.append(first)
                weights = [1.0] * real_count + [0.0] * (args.batch_size - real_count)
                env_payload: dict[int, tuple[list[str], int, int, int]] = {}
                flat_candidate_graphs: list[GraphTensor] = []
                for index, item in enumerate(prepared):
                    if item.value["kind"] != "IMPORT_ENV" or weights[index] == 0:
                        continue
                    candidates, gold = import_candidates(
                        item.value["fragment"],
                        bank,
                        negatives=args.import_negatives,
                        rng=rng,
                    )
                    start = len(flat_candidate_graphs)
                    for fragment in candidates:
                        graph = bank_graph.get(fragment)
                        if graph is None:
                            graph = smiles_to_graph(fragment, require_maps=False)
                        flat_candidate_graphs.append(graph)
                    env_payload[index] = (
                        candidates,
                        gold,
                        start,
                        len(flat_candidate_graphs),
                    )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    contexts = model.encode_context_batch(
                        [item.current_graph for item in prepared],
                        [item.target_graph for item in prepared],
                    )
                    candidate_pools = (
                        model.encode_fragment_batch(flat_candidate_graphs)
                        if flat_candidate_graphs
                        else None
                    )
                    losses: list[torch.Tensor] = []
                    for index, (item, context, weight) in enumerate(
                        zip(prepared, contexts, weights)
                    ):
                        value = item.value
                        kind = value["kind"]
                        if kind == "FLOW":
                            loss = model.flow_nll(
                                value["current"],
                                value["target"],
                                value["moves"],
                                context=context,
                                inventory=item.inventory,
                            )[0]
                        elif kind == "BE_DELTA":
                            loss = model.be_delta_nll(
                                value["current"],
                                value["target"],
                                value["moves"][0],
                                context=context,
                            )[0]
                        elif kind == "IMPORT_ENV":
                            if weight == 0:
                                # Padding still traverses the family head so all
                                # ranks materialize identical gradient tensors.
                                loss = model.finish_nll(
                                    value["current"], value["target"], context=context
                                )
                            else:
                                candidates, gold, start, end = env_payload[index]
                                assert candidate_pools is not None
                                loss = model.import_nll(
                                    value["current"],
                                    value["target"],
                                    candidates,
                                    gold,
                                    context=context,
                                    candidate_pools=candidate_pools[start:end],
                                )
                        elif kind == "IMPORT_REACTIVE":
                            loss = model.reactive_fragment_nll(
                                value["current"],
                                value["target"],
                                value["program"],
                                context=context,
                            )[0]
                        elif kind == "FINISH":
                            loss = model.finish_nll(
                                value["current"], value["target"], context=context
                            )
                        else:
                            raise ValueError(f"unknown decision kind: {kind}")
                        losses.append(loss * weight)
                    batch_loss = torch.stack(losses).sum() / args.batch_size
                batch_loss.backward()
                synchronize_gradients(model, world_size)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                updates += 1
                real_rows += real_count
                loss_sum += float(torch.stack(losses).detach().sum())
                if rank == 0 and updates % args.log_updates == 0:
                    elapsed = time.time() - started
                    print(
                        f"[graph-batched] epoch={epoch_index + 1}/{args.epochs} "
                        f"update={updates}/{total_updates} decisions_rank0={real_rows}/{local_rows} "
                        f"mean_loss_rank0={loss_sum / max(1, real_rows):.5f} "
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
            totals = torch.tensor(
                [loss_sum, real_rows], dtype=torch.float64, device=device
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            result = {
                "epoch": epoch_index + 1,
                "mean_loss": float(totals[0].item() / totals[1].item()),
                "decisions": int(totals[1].item()),
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
                            "trainer": "packed_graph_multiprocess_prefetch_bf16_v3",
                            "action_families": list(ACTION_FAMILIES),
                            "reaction_denominator": expected,
                            "train_decisions": train_meta["decisions"],
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
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
