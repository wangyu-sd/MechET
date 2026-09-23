#!/usr/bin/env python3
"""Collect and optimize verified same-state electron-program branches."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from python_continual_stage import log, read_rows, train
from mechet.anchor_branch_rl import (
    anchor_messages,
    anchor_supervised_record,
    assign_anchor_advantages,
    build_anchor_task,
    choose_horizon,
    first_step_token_mask,
    reference_states,
    score_completion,
    stable_row_rng,
    task_record,
)
from mechet.assistant_masking import render_chat


def _prompt_messages(row, task):
    # Full-horizon rehearsal and validation retain the deployed product-only
    # prompt byte-for-byte. Intermediate resets use the explicit anchor prompt.
    return list(row["messages"][:-1]) if task.is_full_episode else anchor_messages(task)


def _completion(tokenizer, value, eos_ids):
    ids = list(value.token_ids)
    if value.logprobs is None or len(value.logprobs) != len(ids):
        raise ValueError("Missing behavior token likelihoods")
    logps = [float(item[token].logprob) for token, item in zip(ids, value.logprobs)]
    if not all(math.isfinite(value) for value in logps):
        raise ValueError("Nonfinite behavior token likelihoods")
    terminated = bool(ids and ids[-1] in eos_ids and value.finish_reason == "stop")
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return ids, logps, terminated, text


def collect(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm

    if vllm.__version__ != "0.8.5":
        raise ValueError(f"Expected tested vLLM 0.8.5, got {vllm.__version__}")
    if args.k < 1 or args.frontier < 1:
        raise ValueError("k and frontier must be positive")
    if args.max_context > 12288 or args.max_new_tokens < 1:
        raise ValueError("Unsupported generation budget")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data)[args.rank :: args.world_size]
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_context,
        max_num_seqs=32,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        enforce_eager=True,
        seed=(args.seed + args.rank) % (2**32),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    lora = LoRARequest("anchor_branch_actor", 1, str(Path(args.adapter).resolve()))
    eos_ids = sorted(
        {
            value
            for value in (
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
            )
            if isinstance(value, int) and value >= 0
        }
    )
    parameters = SamplingParams(
        n=1 if args.evaluation else args.k,
        temperature=0.0 if args.evaluation else args.temperature,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=eos_ids,
        logprobs=0,
    )

    with output.open("x", encoding="utf-8") as handle:
        for offset in range(0, len(rows), args.batch_products):
            batch = rows[offset : offset + args.batch_products]
            tasks = []
            prompts = []
            for row in batch:
                program, _ = reference_states(row)
                rng = stable_row_rng(args.seed, args.round_index, str(row["id"]))
                horizon = (
                    len(program.steps)
                    if args.evaluation or args.full_only
                    else choose_horizon(
                        len(program.steps),
                        args.frontier,
                        rng,
                        full_episode_fraction=args.full_episode_fraction,
                    )
                )
                task = build_anchor_task(row, horizon)
                messages = _prompt_messages(row, task)
                prompt = tokenizer.encode(
                    render_chat(tokenizer, messages, add_generation_prompt=True),
                    add_special_tokens=False,
                )
                if len(prompt) + args.max_new_tokens > args.max_context:
                    raise ValueError(f"{row['id']}: anchor prompt exceeds context budget")
                tasks.append(task)
                prompts.append(prompt)
            generations = llm.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                parameters,
                lora_request=lora,
                use_tqdm=False,
            )
            for row, task, prompt, generated in zip(
                batch, tasks, prompts, generations, strict=True
            ):
                records = []
                for candidate_index, value in enumerate(generated.outputs):
                    ids, logps, terminated, text = _completion(
                        tokenizer, value, eos_ids
                    )
                    score = score_completion(
                        task,
                        text,
                        terminated=terminated,
                        invalid_penalty=args.invalid_penalty,
                    )
                    action_mask = (
                        first_step_token_mask(tokenizer, ids, text)
                        if terminated
                        else [0] * len(ids)
                    )
                    record = {
                        "id": task.reaction_id,
                        "kind": "rl",
                        "input_ids": prompt + ids,
                        "loss_mask": [0] * len(prompt) + action_mask,
                        "old_logps": [0.0] * len(prompt) + logps,
                        "advantage": 0.0,
                        "reward": float(score["reward"]),
                        "candidate_index": candidate_index,
                        "terminated": terminated,
                        "prediction": text,
                        "score": score,
                        "action_fingerprint": score["action_fingerprint"],
                        "anchor": task_record(task),
                    }
                    if not (
                        len(record["input_ids"])
                        == len(record["loss_mask"])
                        == len(record["old_logps"])
                    ):
                        raise ValueError("Anchor rollout token/mask/logprob misalignment")
                    records.append(record)
                summary = assign_anchor_advantages(records)
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                replay_rng = stable_row_rng(
                    args.seed + 7919, args.round_index, task.reaction_id
                )
                if (
                    not args.evaluation
                    and args.replay_fraction > 0
                    and replay_rng.random() < args.replay_fraction
                ):
                    handle.write(
                        json.dumps(
                            anchor_supervised_record(tokenizer, prompt, task),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                log(
                    stage="anchor-group",
                    rank=args.rank,
                    id=task.reaction_id,
                    horizon=task.horizon,
                    total_steps=task.total_steps,
                    full_episode=task.is_full_episode,
                    **summary,
                )
            log(
                stage="anchor-batch",
                rank=args.rank,
                products=min(offset + len(batch), len(rows)),
                total=len(rows),
                frontier=args.frontier,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "train"])
    for key in ("data", "output", "model", "adapter"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--reference")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--frontier", type=int, default=1)
    parser.add_argument("--full-episode-fraction", type=float, default=0.2)
    parser.add_argument("--replay-fraction", type=float, default=0.25)
    parser.add_argument("--invalid-penalty", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-products", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-context", type=int, default=12288)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--eligible-policy-only", action="store_true")
    parser.add_argument("--replay-epochs", type=int, default=1)
    args = parser.parse_args()
    args.memory_efficient_logps = True
    (collect if args.mode == "collect" else train)(args)


if __name__ == "__main__":
    main()
