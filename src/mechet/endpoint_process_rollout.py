"""Interactive model rollouts for endpoint-grounded process RLVR."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from typing import Any, Sequence

import torch

from .agent_inference import parse_tool_calls
from .endpoint_process_rl_env import EndpointProcessRLEnv
from .grounded_event_search import GroundedProposal


@dataclass(frozen=True)
class SampledEventSpan:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]


@dataclass
class EpisodeRollout:
    env: EndpointProcessRLEnv
    spans: list[SampledEventSpan]

    def validate(self) -> None:
        if len(self.spans) != len(self.env.credits):
            raise RuntimeError("sampled span/reward ledger length mismatch")


def _unwrap(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _model_device(model: Any) -> torch.device:
    return next(_unwrap(model).parameters()).device


def _trim_at_eos(values: list[int], eos_ids: set[int]) -> list[int]:
    stop = next(
        (index + 1 for index, token in enumerate(values) if token in eos_ids),
        len(values),
    )
    return values[:stop]


def _generate_active(
    model: Any,
    tokenizer: Any,
    rollouts: Sequence[EpisodeRollout],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[tuple[list[int], list[int], str]]:
    actor = _unwrap(model)
    previous_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompts = [
            tokenizer.apply_chat_template(
                rollout.env.public_messages(),
                tools=rollout.env.public_tools(),
                tokenize=False,
                add_generation_prompt=True,
            )
            for rollout in rollouts
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
    finally:
        tokenizer.padding_side = previous_padding
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(inputs["input_ids"].shape[1])
    generation_config = copy.deepcopy(actor.generation_config)
    generation_config.do_sample = temperature > 0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "generation_config": generation_config,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
    else:
        # Qwen's saved defaults contain sampling-only parameters. Clear them
        # for the greedy monitor without changing stochastic train rollouts.
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    with torch.inference_mode():
        output = actor.generate(**inputs, **generation_kwargs)
    generated = output[:, input_width:]
    eos_ids = (
        {int(tokenizer.eos_token_id)}
        if tokenizer.eos_token_id is not None
        else set()
    )
    configured = getattr(actor.generation_config, "eos_token_id", None)
    if isinstance(configured, int):
        eos_ids.add(configured)
    elif configured is not None:
        eos_ids.update(int(value) for value in configured)
    records: list[tuple[list[int], list[int], str]] = []
    for row, values_raw in enumerate(generated):
        values = _trim_at_eos([int(value) for value in values_raw.tolist()], eos_ids)
        attention = inputs.get("attention_mask")
        prompt_values = (
            inputs["input_ids"][row][attention[row].bool()].tolist()
            if attention is not None
            else inputs["input_ids"][row].tolist()
        )
        raw = tokenizer.decode(values, skip_special_tokens=False)
        records.append(([int(value) for value in prompt_values], values, raw))
    return records


def run_rollout_group(
    model: Any,
    tokenizer: Any,
    envs: Sequence[EndpointProcessRLEnv],
    *,
    max_proposals: int,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> list[EpisodeRollout]:
    """Sample a group of same-start-state episodes with live executor feedback."""

    if not envs or max_proposals < 1:
        raise ValueError("rollout group and proposal budget must be nonempty")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    actor = _unwrap(model)
    was_training = actor.training
    old_use_cache = getattr(actor.config, "use_cache", None)
    actor.eval()
    if old_use_cache is not None:
        actor.config.use_cache = True
    rollouts = [EpisodeRollout(env=env, spans=[]) for env in envs]
    try:
        for turn in range(max_proposals):
            active = [rollout for rollout in rollouts if not rollout.env.done]
            if not active:
                break
            for rollout in active:
                rollout.env.assert_no_reward_leakage()
            generated = _generate_active(
                model,
                tokenizer,
                active,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for candidate_index, (rollout, (prompt_ids, completion_ids, raw)) in enumerate(
                zip(active, generated, strict=True)
            ):
                rollout.spans.append(
                    SampledEventSpan(tuple(prompt_ids), tuple(completion_ids))
                )
                prefix = torch.tensor(
                    prompt_ids, dtype=torch.long, device=_model_device(model)
                )
                try:
                    calls = parse_tool_calls(raw, tokenizer=tokenizer, prefix=prefix)
                    if len(calls) != 1:
                        raise ValueError(f"expected one tool call, parsed {len(calls)}")
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    rollout.env.reject_unparsed(
                        raw_response=raw,
                        token_count=max(len(completion_ids), 1),
                        code="PARSE_FAILED",
                        message=str(exc),
                    )
                    continue
                call = calls[0]
                proposal = GroundedProposal(
                    name=call.name,
                    arguments=call.arguments,
                    raw_response=raw,
                    logprob_sum=0.0,
                    token_count=max(len(completion_ids), 1),
                    seed=seed + turn * len(active) + candidate_index,
                    call_id=call.call_id,
                )
                # Executor failures are represented by the environment itself.
                # Unexpected implementation errors must remain visible instead
                # of being mislabeled as model parse failures.
                rollout.env.step(proposal)
        for rollout in rollouts:
            if not rollout.env.done:
                rollout.env.terminate_unfinished()
            rollout.validate()
        return rollouts
    finally:
        if old_use_cache is not None:
            actor.config.use_cache = old_use_cache
        if was_training:
            actor.train()
