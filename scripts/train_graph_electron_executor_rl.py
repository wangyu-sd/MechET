#!/usr/bin/env python3
"""Executor-feedback actor/value training after reaction-online trajectory SFT."""
from __future__ import annotations

import argparse
from array import array
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from build_graph_electron_full import EXPECTED, compile_row, sha256_file
from mechet.direct_graph_rl_env import DirectGraphElectronEnv
from mechet.graph_electron_policy import ACTION_FAMILIES, GraphElectronPolicy
from train_graph_electron_reaction_online import build_offset_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--verified-offset-index",
        type=Path,
        help="Offset index created after frozen-source SHA256 verification.",
    )
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--invalid-penalty", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--bc-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--log-updates", type=int, default=5)
    parser.add_argument("--checkpoint-updates", type=int, default=25)
    return parser.parse_args()


def fallback_family_nll(
    model: GraphElectronPolicy, current: str, target: str, history: Any, kind: str
) -> torch.Tensor:
    context = model.encode_context(current, target, history=history)
    label = torch.tensor([ACTION_FAMILIES.index(kind)], device=model.device)
    return F.cross_entropy(model.family_logits(context)[None, :], label)


def save_checkpoint(
    output: Path,
    *,
    model: GraphElectronPolicy,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    update: int,
    episodes: int,
) -> Path:
    temporary = output / f"checkpoint-update{update}.pt.partial"
    final = output / f"checkpoint-update{update}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "update": update,
            "episodes": episodes,
            "stage": "online_executor_rl",
        },
        temporary,
    )
    os.replace(temporary, final)
    return final


def main() -> None:
    args = parse_args()
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8 or args.episodes % (world_size * args.episodes_per_update):
        raise SystemExit("episodes must be divisible by 8 * episodes-per-update")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    source = args.source_root / "train.jsonl"
    manifest = json.loads(args.source_manifest.read_text())
    if not manifest.get("strict_trace_universe_complete"):
        raise SystemExit("source is not the frozen strict trace universe")
    args.output.mkdir(parents=True, exist_ok=True)
    offsets_path = args.verified_offset_index or args.output / "train.reaction_offsets.u64"
    if rank == 0:
        if args.verified_offset_index is not None:
            expected_bytes = EXPECTED["train"] * array("Q").itemsize
            if not offsets_path.is_file() or offsets_path.stat().st_size != expected_bytes:
                raise SystemExit("verified offset index is missing or has the wrong row count")
            print(
                f"[graph-executor-rl] source_verified provenance_index={offsets_path} "
                f"manifest_sha256={manifest['splits']['train']['sha256']}",
                flush=True,
            )
        else:
            if sha256_file(source) != manifest["splits"]["train"]["sha256"]:
                raise SystemExit("frozen train source hash mismatch")
            if not offsets_path.exists():
                build_offset_index(source, offsets_path, EXPECTED["train"])
    dist.barrier()
    offsets = array("Q")
    with offsets_path.open("rb") as handle:
        offsets.fromfile(handle, offsets_path.stat().st_size // offsets.itemsize)

    checkpoint = torch.load(args.initialize_from, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or {}
    model = GraphElectronPolicy(
        hidden_dim=int(config.get("hidden_dim", 192)),
        num_layers=int(config.get("layers", 6)),
        dropout=0.0,
    )
    model.freeze_enumerated_action_heads()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    rank_episodes = args.episodes // world_size
    updates = rank_episodes // args.episodes_per_update
    source_order = list(range(rank, len(offsets), world_size))
    random.Random(args.seed).shuffle(source_order)
    handle = source.open("rb")
    totals = {
        "episodes": 0,
        "steps": 0,
        "accepted": 0,
        "invalid": 0,
        "penalized_invalid": 0,
        "invalid_penalty_reward": 0.0,
        "endpoint_exact": 0,
        "reward": 0.0,
    }
    started = time.time()
    if rank == 0:
        print(
            f"[graph-executor-rl] episodes={args.episodes} updates={updates} "
            f"max_steps={args.max_steps} invalid_penalty={args.invalid_penalty} "
            f"checkpoint={args.initialize_from}",
            flush=True,
        )

    for update in range(1, updates + 1):
        policy_terms: list[torch.Tensor] = []
        value_terms: list[torch.Tensor] = []
        bc_terms: list[torch.Tensor] = []
        for within in range(args.episodes_per_update):
            episode_index = (update - 1) * args.episodes_per_update + within
            source_index = source_order[episode_index]
            handle.seek(int(offsets[source_index]))
            line = handle.readline().decode("utf-8")
            row = json.loads(line)
            target = str(row["target_smiles"])
            expected = str(row["expected_precursor"])
            env = DirectGraphElectronEnv(max_steps=args.max_steps)
            observation = env.reset(target=target, expected_precursor=expected)
            transitions: list[dict[str, Any]] = []
            while True:
                with torch.no_grad():
                    sampled = model.sample_action(
                        observation.current,
                        observation.target,
                        trajectory=observation.history,
                        greedy=False,
                        temperature=args.temperature,
                    )
                action = dict(sampled.get("action") or {"kind": sampled.get("family")})
                transition = env.step(action)
                reward = (
                    float(transition.reward)
                    if transition.done or transition.accepted
                    else -float(args.invalid_penalty)
                )
                transitions.append(
                    {
                        "current": observation.current,
                        "target": observation.target,
                        "history": observation.history,
                        "action": action,
                        "reward": reward,
                        "done": transition.done,
                    }
                )
                totals["steps"] += 1
                totals["accepted"] += int(transition.accepted)
                totals["invalid"] += int(not transition.accepted)
                explicitly_penalized = not transition.done and not transition.accepted
                totals["penalized_invalid"] += int(explicitly_penalized)
                if explicitly_penalized:
                    totals["invalid_penalty_reward"] += reward
                totals["endpoint_exact"] += int(
                    bool(transition.result.get("endpoint_exact"))
                )
                observation = transition.next_observation
                if transition.done:
                    break
            returns: list[float] = []
            running = 0.0
            for item in reversed(transitions):
                running = float(item["reward"]) + args.discount * running
                returns.append(running)
            returns.reverse()
            for item, return_value in zip(transitions, returns):
                action = {**item["action"], "current": item["current"], "target": item["target"]}
                try:
                    nll = model.canonical_action_nll(action, history=item["history"])
                except (KeyError, ValueError):
                    nll = fallback_family_nll(
                        model,
                        item["current"],
                        item["target"],
                        item["history"],
                        str(item["action"].get("kind")),
                    )
                context = model.encode_context(
                    item["current"], item["target"], history=item["history"]
                )
                value = model.value(context)
                target_return = value.new_tensor(return_value)
                advantage = (target_return - value.detach()).clamp(-5.0, 5.0)
                policy_terms.append(advantage * nll)
                value_terms.append(F.mse_loss(value, target_return))
            _, gold = compile_row((source_index, line))
            for decision in gold[: min(4, len(gold))]:
                from mechet.electron_policy_protocol import CompressedTrajectory

                bc_terms.append(
                    model.canonical_action_nll(
                        decision,
                        history=CompressedTrajectory.from_dict(decision.get("history")),
                    )
                )
            totals["episodes"] += 1
            totals["reward"] += sum(float(item["reward"]) for item in transitions)

        actor = torch.stack(policy_terms).mean()
        value_loss = torch.stack(value_terms).mean()
        bc_loss = torch.stack(bc_terms).mean()
        loss = actor + args.value_weight * value_loss + args.bc_weight * bc_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if rank == 0 and update % args.log_updates == 0:
            print(
                f"[graph-executor-rl] update={update}/{updates} episodes_rank0={totals['episodes']} "
                f"accept={totals['accepted']}/{totals['steps']} endpoint_exact={totals['endpoint_exact']} "
                f"mean_reward={totals['reward']/max(1,totals['episodes']):.4f} "
                f"loss={float(loss.detach()):.5f} actor={float(actor.detach()):.5f} "
                f"value={float(value_loss.detach()):.5f} bc={float(bc_loss.detach()):.5f} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
        if rank == 0 and update % args.checkpoint_updates == 0:
            path = save_checkpoint(
                args.output,
                model=model,
                optimizer=optimizer,
                config={**config, **vars(args)},
                update=update,
                episodes=totals["episodes"] * world_size,
            )
            print(f"[graph-executor-rl] checkpoint={path}", flush=True)

    summary = torch.tensor(
        [
            totals["episodes"],
            totals["steps"],
            totals["accepted"],
            totals["invalid"],
            totals["penalized_invalid"],
            totals["invalid_penalty_reward"],
            totals["endpoint_exact"],
            totals["reward"],
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(summary, op=dist.ReduceOp.SUM)
    if rank == 0:
        final = save_checkpoint(
            args.output,
            model=model,
            optimizer=optimizer,
            config={**config, **vars(args)},
            update=updates,
            episodes=int(summary[0]),
        )
        report = {
            "status": "completed",
            "stage": "online_executor_rl",
            "episodes": int(summary[0]),
            "steps": int(summary[1]),
            "accepted": int(summary[2]),
            "invalid": int(summary[3]),
            "penalized_invalid": int(summary[4]),
            "invalid_penalty_reward": float(summary[5]),
            "endpoint_exact": int(summary[6]),
            "mean_reward": float(summary[7] / summary[0]),
            "invalid_penalty": args.invalid_penalty,
            "latest_checkpoint": str(final),
            "wall_seconds": time.time() - started,
        }
        (args.output / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[graph-executor-rl] completed={json.dumps(report, sort_keys=True)}", flush=True)
    dist.barrier()
    handle.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
