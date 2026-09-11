#!/usr/bin/env python3
"""Train Endpoint-Grounded Process RLVR with event-level RLOO credit."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from mechet.endpoint_process_rl_env import EndpointProcessRLEnv, ProcessRewardConfig
from mechet.chemical_runtime import require_endpoint_process_rdkit
from mechet.endpoint_process_rollout import EpisodeRollout, run_rollout_group
from mechet.mixed_horizon import CurriculumController, MixedHorizonSampler
from mechet.model import resolve_qwen_model_path
from mechet.rlvr import (
    EventPolicySample,
    event_level_policy_loss,
    event_rloo_advantages,
    event_token_log_probs_batch,
)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def read_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def rank_shard_path(pattern: str, *, rank: int, world_size: int) -> Path:
    """Resolve an explicit rank-local shard without silently sharing rows."""

    if "{rank" not in pattern:
        raise ValueError("train_file_pattern must contain a {rank} placeholder")
    rendered = pattern.format(rank=rank, world_size=world_size)
    value = Path(rendered)
    return value if value.is_absolute() else REPO / value


def unwrap(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def activate_adapter(model: Any, name: str, *, trainable: bool) -> None:
    actor = unwrap(model)
    actor.set_adapter(name)
    for parameter_name, parameter in actor.named_parameters():
        if "lora_" not in parameter_name:
            parameter.requires_grad = False
        elif f".{name}." in parameter_name:
            parameter.requires_grad = trainable
        else:
            parameter.requires_grad = False


def load_actor(
    *,
    base_model: str,
    parent_adapter: Path,
    actor_adapter: Path | None = None,
    local_rank: int,
    use_4bit: bool,
) -> tuple[Any, Any, str]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    manifest_path = next(
        (
            parent / "adapter_manifest.json"
            for parent in (parent_adapter, *parent_adapter.parents)
            if (parent / "adapter_manifest.json").is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"no adapter_manifest.json above {parent_adapter}")
    revision = str(json.loads(manifest_path.read_text())["base_model_revision"])
    active_adapter = actor_adapter or parent_adapter
    tokenizer = AutoTokenizer.from_pretrained(
        str(active_adapter), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.get_device_capability(local_rank)[0] >= 8:
        dtype = torch.bfloat16
    quantization = None
    if use_4bit and torch.cuda.is_available():
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=None if Path(base_model).exists() else revision,
        trust_remote_code=True,
        local_files_only=Path(base_model).exists(),
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )
    if quantization is not None:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
    elif hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = PeftModel.from_pretrained(
        base,
        str(active_adapter),
        adapter_name="default",
        is_trainable=True,
    )
    model.load_adapter(
        str(parent_adapter), adapter_name="reference", is_trainable=False
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    activate_adapter(model, "default", trainable=True)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(".reference." in name for name in trainable):
        raise RuntimeError("actor/reference adapter trainability contract failed")
    return model, tokenizer, revision


def distributed_context(*, timeout_seconds: int = 10_800) -> tuple[int, int, int]:
    import torch

    if timeout_seconds < 1:
        raise ValueError("distributed timeout must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        # Executor-driven rollouts have data-dependent latency. Fast ranks can
        # reach a metric reduction long before the rank with the longest trace,
        # so the default ten-minute watchdog is not a correctness bound.
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(seconds=timeout_seconds)
        )
    return rank, world_size, local_rank


def all_reduce_values(values: Mapping[str, float], *, world_size: int) -> dict[str, float]:
    import torch

    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )
    if world_size > 1:
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.cpu().tolist(), strict=True)}


def rollout_metrics(rollouts: Sequence[EpisodeRollout]) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for rollout in rollouts:
        summary = rollout.env.summary()
        values["episodes"] += 1
        values["endpoint_exact"] += int(summary.endpoint_exact)
        values["explicit_finish"] += int(summary.explicit_finish)
        values["formal_terminal"] += int(summary.formal_terminal)
        values["committed_events"] += summary.generated_committed_events
        values["rejected_proposals"] += summary.rejected_proposals
        values["retry_exhausted"] += int(summary.retry_exhausted)
        values["cycles"] += int(summary.cycle)
        values["start_distance"] += summary.start_distance
        values["terminal_distance"] += summary.terminal_distance
        values["progress"] += summary.cumulative_progress
        values["reward"] += summary.total_reward
    return dict(values)


def build_event_samples(
    rollouts: Sequence[EpisodeRollout], *, gamma: float
) -> tuple[list[tuple[EventPolicySample, EpisodeRollout]], bool]:
    rewards = [[credit.total_reward for credit in rollout.env.credits] for rollout in rollouts]
    _, advantages = event_rloo_advantages(rewards, gamma=gamma)
    output: list[tuple[EventPolicySample, EpisodeRollout]] = []
    nonzero = False
    for rollout, rollout_advantages in zip(rollouts, advantages, strict=True):
        for span, credit, advantage in zip(
            rollout.spans, rollout.env.credits, rollout_advantages, strict=True
        ):
            nonzero = nonzero or abs(advantage) >= 1e-12
            output.append(
                (
                    EventPolicySample(
                        prompt_token_ids=span.prompt_token_ids,
                        completion_token_ids=span.completion_token_ids,
                        advantage=advantage,
                        rejected=not credit.accepted,
                    ),
                    rollout,
                )
            )
    return output, nonzero


def attach_reference_logprobs(
    model: Any,
    samples: Sequence[EventPolicySample],
    *,
    max_length: int,
    microbatch_size: int,
    pad_token_id: int,
) -> list[EventPolicySample]:
    import torch

    activate_adapter(model, "reference", trainable=False)
    output: list[EventPolicySample] = []
    with torch.no_grad():
        for start in range(0, len(samples), microbatch_size):
            batch = samples[start : start + microbatch_size]
            logps, _ = event_token_log_probs_batch(
                model,
                [item.prompt_token_ids for item in batch],
                [item.completion_token_ids for item in batch],
                max_length=max_length,
                pad_token_id=pad_token_id,
            )
            for item, logp in zip(batch, logps.tolist(), strict=True):
                output.append(
                    EventPolicySample(
                        prompt_token_ids=item.prompt_token_ids,
                        completion_token_ids=item.completion_token_ids,
                        advantage=item.advantage,
                        reference_logprob=float(logp),
                        rejected=item.rejected,
                    )
                )
    activate_adapter(model, "default", trainable=True)
    return output


def evaluate_monitor(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    rank: int,
    world_size: int,
    config: ProcessRewardConfig,
    max_proposals: int,
    max_input_tokens: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, float]:
    local: list[EpisodeRollout] = []
    assigned = sum(index % world_size == rank for index in range(len(rows)))
    completed = 0
    for index, row in enumerate(rows):
        if index % world_size != rank:
            continue
        env = EndpointProcessRLEnv(
            row, prefix_events=0, start_horizon="product", reward_config=config
        )
        local.extend(
            run_rollout_group(
                model,
                tokenizer,
                [env],
                max_proposals=max_proposals,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=seed + index,
            )
        )
        completed += 1
        print(
            json.dumps(
                {
                    "type": "monitor_progress",
                    "rank": rank,
                    "completed": completed,
                    "assigned": assigned,
                    "id": str(row.get("id", index)),
                }
            ),
            flush=True,
        )
    totals = all_reduce_values(rollout_metrics(local), world_size=world_size)
    n = max(totals.get("episodes", 0.0), 1.0)
    return {
        "n": totals.get("episodes", 0.0),
        "endpoint_pass_at_1": totals.get("endpoint_exact", 0.0) / n,
        "formal_terminal_rate": totals.get("formal_terminal", 0.0) / n,
        "explicit_finish_rate": totals.get("explicit_finish", 0.0) / n,
        "invalid_actions_per_episode": totals.get("rejected_proposals", 0.0) / n,
        "mean_committed_events": totals.get("committed_events", 0.0) / n,
    }


def save_actor(model: Any, tokenizer: Any, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    actor = unwrap(model)
    actor.save_pretrained(
        directory,
        safe_serialization=True,
        selected_adapters=["default"],
    )
    tokenizer.save_pretrained(directory)


def latest_resumable_checkpoint(output_dir: Path) -> tuple[int, Path] | None:
    candidates: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return None
    for path in output_dir.glob("checkpoint-*"):
        try:
            update = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if (path / "adapter_model.safetensors").is_file() and (
            path / "trainer_state.pt"
        ).is_file():
            candidates.append((update, path))
    return max(candidates, default=None)


def save_training_checkpoint(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    directory: Path,
    *,
    update: int,
    curriculum_phase: str,
) -> None:
    import torch

    save_actor(model, tokenizer, directory)
    torch.save(
        {
            "update": int(update),
            "curriculum_phase": str(curriculum_phase),
            "optimizer": optimizer.state_dict(),
        },
        directory / "trainer_state.pt",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--limit-reactions", type=int, default=0)
    parser.add_argument("--groups-per-update", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--event-microbatch-size", type=int, default=None)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--train-file-pattern", default=None)
    parser.add_argument("--monitor-file", type=Path, default=None)
    parser.add_argument("--parent-adapter", type=Path, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-monitor", action="store_true")
    args = parser.parse_args()
    runtime_version = require_endpoint_process_rdkit()
    cfg = load_yaml(args.config)
    distributed_timeout_seconds = int(
        cfg.get("distributed_timeout_seconds", 10_800)
    )
    rank, world_size, local_rank = distributed_context(
        timeout_seconds=distributed_timeout_seconds
    )
    if rank == 0:
        print(
            json.dumps(
                {
                    "type": "runtime",
                    "rdkit": runtime_version,
                    "distributed_timeout_seconds": distributed_timeout_seconds,
                }
            ),
            flush=True,
        )

    def path_value(key: str) -> Path:
        value = Path(str(cfg[key]))
        return value if value.is_absolute() else REPO / value

    train_pattern = args.train_file_pattern or cfg.get("train_file_pattern")
    if args.train_file is not None and train_pattern:
        raise ValueError("use only one of train-file and train-file-pattern")
    train_file = (
        rank_shard_path(str(train_pattern), rank=rank, world_size=world_size)
        if train_pattern
        else (args.train_file or path_value("train_file"))
    )
    monitor_file = args.monitor_file or path_value("monitor_file")
    train_rows = read_jsonl(train_file, limit=args.limit_reactions)
    monitor_rows = read_jsonl(monitor_file)
    if not train_rows or not monitor_rows:
        raise ValueError("train and monitor pilot files must be nonempty")
    seed = int(cfg.get("seed", 17))
    random.Random(seed + rank).shuffle(train_rows)
    parent_adapter = args.parent_adapter or path_value("parent_adapter")
    output_dir = args.output_dir or path_value("output_dir")
    if not output_dir.is_absolute():
        output_dir = REPO / output_dir
    resume = (
        latest_resumable_checkpoint(output_dir)
        if bool(cfg.get("auto_resume", False))
        else None
    )
    base_model = str(
        args.base_model or cfg.get("base_model") or resolve_qwen_model_path() or ""
    )
    if not base_model:
        raise ValueError("base_model is required")
    model, tokenizer, base_revision = load_actor(
        base_model=base_model,
        parent_adapter=parent_adapter,
        actor_adapter=resume[1] if resume is not None else None,
        local_rank=local_rank,
        use_4bit=bool(cfg.get("use_4bit", True)),
    )
    if world_size > 1:
        import torch

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    import torch

    optimizer = torch.optim.AdamW(
        [value for value in model.parameters() if value.requires_grad],
        lr=float(cfg.get("learning_rate", 5e-6)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    start_update = 0
    resumed_phase: str | None = None
    if resume is not None:
        state = torch.load(
            resume[1] / "trainer_state.pt",
            map_location=torch.device("cuda", local_rank)
            if torch.cuda.is_available()
            else torch.device("cpu"),
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"])
        resumed_phase = str(state.get("curriculum_phase") or "A")
        if start_update != resume[0]:
            raise ValueError("checkpoint directory and trainer update disagree")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "type": "resume",
                        "checkpoint": str(resume[1]),
                        "start_update": start_update,
                        "curriculum_phase": resumed_phase,
                    }
                ),
                flush=True,
            )
    reward_config = ProcessRewardConfig(
        exact_endpoint=float(cfg.get("exact_endpoint_reward", 4.0)),
        wrong_finish=float(cfg.get("wrong_finish_reward", -1.0)),
        invalid_proposal=float(cfg.get("invalid_action_reward", -0.25)),
        state_cycle=float(cfg.get("cycle_reward", -0.5)),
        retry_exhausted=float(cfg.get("retry_exhausted_reward", -0.75)),
        unfinished=float(cfg.get("unfinished_reward", -1.0)),
        progress_event_cap=float(cfg.get("progress_event_cap", 0.25)),
        progress_total_cap=float(cfg.get("progress_total_cap", 1.0)),
        same_state_retry_limit=int(cfg.get("same_state_retry_limit", 2)),
        max_committed_events=int(cfg.get("max_committed_events", 8)),
        max_import_fragments=int(cfg.get("max_import_fragments", 12)),
        max_import_atoms=int(cfg.get("max_import_atoms", 96)),
        max_fragment_heavy_atoms=int(cfg.get("max_fragment_heavy_atoms", 48)),
    )
    max_updates = int(args.max_updates or cfg.get("max_updates", 64))
    group_size = int(args.group_size or cfg.get("group_size", 8))
    groups_per_update = int(
        args.groups_per_update or cfg.get("groups_per_update_per_rank", 1)
    )
    gamma = float(cfg.get("gamma", 0.95))
    beta_kl = float(cfg.get("beta_kl", 0.01))
    max_proposals = int(args.max_proposals or cfg.get("max_proposals", 24))
    max_input_tokens = int(
        args.max_input_tokens or cfg.get("max_input_tokens", 8192)
    )
    max_new_tokens = int(args.max_new_tokens or cfg.get("max_new_tokens", 384))
    max_seq_length = int(args.max_seq_length or cfg.get("max_seq_length", 12288))
    temperature = float(cfg.get("temperature", 0.7))
    top_p = float(cfg.get("top_p", 0.95))
    event_microbatch = int(
        args.event_microbatch_size or cfg.get("event_microbatch_size", 1)
    )
    monitor_interval = int(cfg.get("monitor_interval", 16))
    save_interval = int(cfg.get("save_interval", 16))
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        torch.distributed.barrier()
    log_path = output_dir / "updates.jsonl"
    controller = CurriculumController(
        near_end_threshold=float(cfg.get("near_end_promotion_threshold", 0.60)),
        stable_product_checks=int(cfg.get("stable_product_monitor_checks", 2)),
    )
    if resumed_phase is not None:
        controller.phase = resumed_phase
    sampler = MixedHorizonSampler(seed=seed, phase=controller.phase)
    near_end_window: deque[float] = deque(maxlen=int(cfg.get("near_end_window", 64)))
    monitor_history: list[dict[str, Any]] = []

    if (
        start_update == 0
        and not args.skip_monitor
        and bool(cfg.get("monitor_at_start", True))
    ):
        activate_adapter(model, "reference", trainable=False)
        parent_monitor = evaluate_monitor(
            model,
            tokenizer,
            monitor_rows,
            rank=rank,
            world_size=world_size,
            config=reward_config,
            max_proposals=max_proposals,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        activate_adapter(model, "default", trainable=True)
        if rank == 0:
            monitor_history.append({"update": 0, "adapter": "parent", **parent_monitor})
            print(json.dumps({"type": "monitor", **monitor_history[-1]}), flush=True)

    local_train_rows = len(train_rows)
    count_values = all_reduce_values(
        {"train_rows": float(local_train_rows)}, world_size=world_size
    )
    global_train_rows = int(count_values["train_rows"])
    expected_train_rows = int(cfg.get("expected_train_rows", 0) or 0)
    if expected_train_rows and global_train_rows != expected_train_rows:
        raise ValueError(
            f"distributed train denominator mismatch: {global_train_rows} != {expected_train_rows}"
        )
    exact_single_coverage = bool(cfg.get("exact_single_coverage", False))
    required_updates = max(
        (local_train_rows + groups_per_update - 1) // groups_per_update,
        1,
    )
    if world_size > 1:
        required_tensor = torch.tensor(
            [required_updates], dtype=torch.int64, device=torch.device("cuda", local_rank)
        )
        torch.distributed.all_reduce(required_tensor, op=torch.distributed.ReduceOp.MAX)
        required_updates = int(required_tensor.item())
    if exact_single_coverage and max_updates != required_updates:
        raise ValueError(
            f"exact coverage requires max_updates={required_updates}, got {max_updates}"
        )
    if rank == 0:
        print(
            json.dumps(
                {
                    "type": "train_contract",
                    "global_train_rows": global_train_rows,
                    "rank_local_rows": local_train_rows,
                    "exact_single_coverage": exact_single_coverage,
                    "max_updates": max_updates,
                }
            ),
            flush=True,
        )

    for update in range(start_update, max_updates):
        sampled_rollouts: list[EpisodeRollout] = []
        group_nonzero = 0
        raw_samples: list[EventPolicySample] = []
        local_groups = 0
        for group_index in range(groups_per_update):
            if train_pattern:
                row_index = update * groups_per_update + group_index
                if row_index >= len(train_rows):
                    continue
            else:
                row_index = (
                    update * world_size * groups_per_update
                    + rank * groups_per_update
                    + group_index
                ) % len(train_rows)
            row = train_rows[row_index]
            local_groups += 1
            n_events = int((row.get("metadata") or {}).get("n_events") or 0)
            horizon = sampler.sample(
                identifier=str(row["id"]),
                n_events=n_events,
                draw_index=update * groups_per_update + group_index,
            )
            envs = [
                EndpointProcessRLEnv(
                    row,
                    prefix_events=horizon.prefix_events,
                    start_horizon=horizon.horizon,
                    reward_config=reward_config,
                )
                for _ in range(group_size)
            ]
            rollouts = run_rollout_group(
                model,
                tokenizer,
                envs,
                max_proposals=max_proposals,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed + update * 100_003 + rank * 997 + group_index,
            )
            pairs, nonzero = build_event_samples(rollouts, gamma=gamma)
            group_nonzero += int(nonzero)
            raw_samples.extend(item for item, _ in pairs if item.completion_token_ids)
            sampled_rollouts.extend(rollouts)

        reference_samples = attach_reference_logprobs(
            unwrap(model),
            raw_samples,
            max_length=max_seq_length,
            microbatch_size=event_microbatch,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        activate_adapter(model, "default", trainable=True)
        unwrap(model).train()
        unwrap(model).config.use_cache = False
        local_count = len(reference_samples)
        local_rejected_count = sum(int(item.rejected) for item in reference_samples)
        count_tensor = torch.tensor(
            [local_count], dtype=torch.int64, device=torch.device("cuda", local_rank)
        )
        if world_size > 1:
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.MAX)
        padded_count = int(count_tensor.item())
        if padded_count == 0:
            raise RuntimeError("rollout update produced no trainable event tokens")
        if reference_samples:
            template = reference_samples[0]
        else:
            prompt_token = int(tokenizer.bos_token_id or tokenizer.eos_token_id)
            completion_token = int(tokenizer.eos_token_id or tokenizer.pad_token_id)
            template = EventPolicySample((prompt_token,), (completion_token,), 0.0)
        while len(reference_samples) < padded_count:
            reference_samples.append(
                EventPolicySample(
                    prompt_token_ids=template.prompt_token_ids,
                    completion_token_ids=template.completion_token_ids,
                    advantage=0.0,
                    reference_logprob=None,
                    rejected=False,
                )
            )
        chunks = [
            reference_samples[start : start + event_microbatch]
            for start in range(0, padded_count, event_microbatch)
        ]
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        loss_stats = {
            "mean_kl": 0.0,
            "event_terms": 0,
            "rejected_event_terms": 0,
        }
        for chunk_index, chunk in enumerate(chunks):
            sync_context = (
                model.no_sync()
                if world_size > 1 and chunk_index + 1 < len(chunks)
                else nullcontext()
            )
            with sync_context:
                chunk_loss, chunk_stats = event_level_policy_loss(
                    model,
                    chunk,
                    beta_kl=beta_kl,
                    max_length=max_seq_length,
                    microbatch_size=len(chunk),
                    pad_token_id=int(tokenizer.pad_token_id),
                    retain_zero_advantage=True,
                )
                (chunk_loss / len(chunks)).backward()
            loss_total += float(chunk_loss.detach().cpu())
            loss_stats["mean_kl"] += float(chunk_stats["mean_kl"])
            loss_stats["event_terms"] += int(chunk_stats["event_terms"])
            loss_stats["rejected_event_terms"] += int(
                chunk_stats["rejected_event_terms"]
            )
        loss_value = loss_total / len(chunks)
        loss_stats["mean_kl"] /= len(chunks)
        trainable_parameters = [value for value in model.parameters() if value.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, float(cfg.get("max_grad_norm", 1.0))
        )
        optimizer.step()

        local_metrics = rollout_metrics(sampled_rollouts)
        local_metrics.update(
            {
                "loss_sum": loss_value,
                "kl_sum": float(loss_stats["mean_kl"]),
                "gradient_norm_sum": float(gradient_norm.detach().cpu()),
                # Exclude DDP-only zero-advantage padding from rollout metrics.
                "event_terms": float(local_count),
                "rejected_event_terms": float(local_rejected_count),
                "groups": float(local_groups),
                "effective_groups": float(group_nonzero),
            }
        )
        totals = all_reduce_values(local_metrics, world_size=world_size)
        n = max(totals.get("episodes", 0.0), 1.0)
        groups = max(totals.get("groups", 0.0), 1.0)
        for rollout in sampled_rollouts:
            summary = rollout.env.summary()
            if summary.start_horizon == "near_end":
                near_end_window.append(float(summary.endpoint_exact))
        near_counts = all_reduce_values(
            {
                "success": sum(near_end_window),
                "count": len(near_end_window),
            },
            world_size=world_size,
        )
        near_rate = near_counts["success"] / max(near_counts["count"], 1.0)
        if rank == 0:
            controller.observe(near_end_exact_rate=near_rate)
        if world_size > 1:
            phase_tensor = torch.tensor(
                [ord(controller.phase) if rank == 0 else 0],
                dtype=torch.int64,
                device=torch.device("cuda", local_rank),
            )
            torch.distributed.broadcast(phase_tensor, src=0)
            controller.phase = chr(int(phase_tensor.item()))
        sampler.set_phase(controller.phase)
        record = {
            "type": "update",
            "update": update + 1,
            "phase": controller.phase,
            "loss": totals["loss_sum"] / world_size,
            "mean_kl": totals["kl_sum"] / world_size,
            "gradient_norm": totals["gradient_norm_sum"] / world_size,
            "endpoint_rate": totals.get("endpoint_exact", 0.0) / n,
            "explicit_finish_rate": totals.get("explicit_finish", 0.0) / n,
            "formal_terminal_rate": totals.get("formal_terminal", 0.0) / n,
            "mean_committed_events": totals.get("committed_events", 0.0) / n,
            "mean_rejected_proposals": totals.get("rejected_proposals", 0.0) / n,
            "retry_exhaustion_rate": totals.get("retry_exhausted", 0.0) / n,
            "cycle_rate": totals.get("cycles", 0.0) / n,
            "mean_start_distance": totals.get("start_distance", 0.0) / n,
            "mean_terminal_distance": totals.get("terminal_distance", 0.0) / n,
            "mean_progress": totals.get("progress", 0.0) / n,
            "mean_reward": totals.get("reward", 0.0) / n,
            "effective_group_rate": totals.get("effective_groups", 0.0) / groups,
            "event_terms": int(totals.get("event_terms", 0.0)),
            "rejected_event_terms": int(totals.get("rejected_event_terms", 0.0)),
            "near_end_window_exact_rate": near_rate,
        }
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        should_monitor = (
            not args.skip_monitor
            and monitor_interval > 0
            and ((update + 1) % monitor_interval == 0 or update + 1 == max_updates)
        )
        if should_monitor:
            activate_adapter(model, "default", trainable=False)
            current_monitor = evaluate_monitor(
                model,
                tokenizer,
                monitor_rows,
                rank=rank,
                world_size=world_size,
                config=reward_config,
                max_proposals=max_proposals,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            activate_adapter(model, "default", trainable=True)
            if rank == 0:
                monitor_record = {
                    "update": update + 1,
                    "adapter": "actor",
                    **current_monitor,
                }
                monitor_history.append(monitor_record)
                controller.observe(
                    product_monitor_exact_rate=current_monitor["endpoint_pass_at_1"]
                )
                print(json.dumps({"type": "monitor", **monitor_record}), flush=True)
            if world_size > 1:
                phase_tensor = torch.tensor(
                    [ord(controller.phase) if rank == 0 else 0],
                    dtype=torch.int64,
                    device=torch.device("cuda", local_rank),
                )
                torch.distributed.broadcast(phase_tensor, src=0)
                controller.phase = chr(int(phase_tensor.item()))
            sampler.set_phase(controller.phase)

        if save_interval and (update + 1) % save_interval == 0:
            if world_size > 1:
                torch.distributed.barrier()
            if rank == 0:
                save_training_checkpoint(
                    model,
                    tokenizer,
                    optimizer,
                    output_dir / f"checkpoint-{update + 1}",
                    update=update + 1,
                    curriculum_phase=controller.phase,
                )
            if world_size > 1:
                torch.distributed.barrier()

    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        save_actor(model, tokenizer, output_dir / "adapter")
        summary = {
            "artifact_type": "endpoint_grounded_process_rlvr_v1",
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "parent_adapter": str(parent_adapter),
            "base_model_revision": base_revision,
            "train_rows": global_train_rows,
            "rank_local_train_rows": local_train_rows,
            "monitor_rows": len(monitor_rows),
            "world_size": world_size,
            "max_updates": max_updates,
            "group_size": group_size,
            "curriculum_final_phase": controller.phase,
            "monitor_history": monitor_history,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary), flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
