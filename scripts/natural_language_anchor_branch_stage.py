#!/usr/bin/env python3
"""Collect natural-language anchor branches and run the shared PPO learner."""

from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from anchor_branch_stage import train
from python_continual_stage import log, read_rows
from mechet.assistant_masking import (
    encode_assistant_only_conversation, render_chat, render_qwen_sft_tool_prefix,
)
from mechet.endpoints import split_precursor_endpoints, structural_exact
from mechet.in_place_grounded_flow import mapped_atom_numbers
from mechet.natural_language_anchor_branch_rl import (
    assign_local_advantages,
    contains_unchanged_target,
    endpoint_shaped_reward,
    state_value_margin,
    stable_rng,
    successor_fingerprint,
    task_from_episode,
    task_record,
)
from mechet.vnext_tree_credit import assign_sibling_advantages, search_teacher_distribution
from mechet.successor_value import (
    SUCCESSOR_VALUE_SYSTEM,
    successor_value_margin,
    successor_value_prompt,
)
from mechet.anchor_branch_rl import choose_horizon
from scripts.build_natural_language_event_sft import (
    SYSTEM,
    TOOLS,
    _import_arguments,
    _prompt,
)
from scripts.build_natural_language_state_value import VALUE_SYSTEM, value_prompt
from scripts.eval_natural_language_event_local import prediction_call
from scripts.eval_natural_language_event_suffix import reference_episode
from scripts.run_natural_language_value_search import Action, Node, execute, policy_prompt, visible
from scripts.earho_v2_protocol import (
    V2AnchorTask,
    anchor_task as v2_anchor_task,
    locate_first_divergence,
    replay_reference,
)


PROMPT_MODES = ("action", "event")


class AsyncVLLMBridge:
    """Expose the vLLM async request queue to independent reaction workers.

    Each reaction owns its own executor state.  Only model requests enter the
    shared async engine; token IDs and logprobs are never decoded/re-encoded.
    """

    def __init__(self, engine, loop, *, seed: int, rank: int):
        self.engine = engine
        self.loop = loop
        self.seed = int(seed)
        self.rank = int(rank)
        self.local = threading.local()

    def begin_reaction(self, reaction_id: str) -> None:
        self.local.reaction_id = str(reaction_id)
        self.local.call_index = 0
        self.local.stats = {
            "model_calls": 0, "model_requests": 0,
            "model_prompt_tokens": 0, "model_output_tokens": 0,
            "model_wait_wall_s": 0.0, "model_queue_s": 0.0,
            "model_ttft_s": 0.0, "model_decode_s": 0.0,
        }

    def end_reaction(self) -> dict[str, float | int]:
        stats = dict(self.local.stats)
        del self.local.stats
        del self.local.reaction_id
        return stats

    def generate(self, prompts, parameters, *, lora_request=None, use_tqdm=False):
        if not hasattr(self.local, "reaction_id"):
            raise RuntimeError("async model request has no reaction identity")
        call_index = self.local.call_index
        self.local.call_index += 1
        reaction_id = self.local.reaction_id
        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(
            self._generate(prompts, parameters, lora_request, reaction_id, call_index),
            self.loop,
        )
        outputs = future.result()
        stats = self.local.stats
        stats["model_calls"] += 1
        stats["model_requests"] += len(outputs)
        stats["model_wait_wall_s"] += time.monotonic() - started
        for prompt, output in zip(prompts, outputs, strict=True):
            if isinstance(prompt, Mapping) and "prompt_token_ids" in prompt:
                stats["model_prompt_tokens"] += len(prompt["prompt_token_ids"])
            elif getattr(output, "prompt_token_ids", None) is not None:
                stats["model_prompt_tokens"] += len(output.prompt_token_ids)
            stats["model_output_tokens"] += sum(
                len(completion.token_ids) for completion in output.outputs
            )
            metrics = getattr(output, "metrics", None)
            if metrics is not None:
                stats["model_queue_s"] += float(metrics.time_in_queue or 0.0)
                first = metrics.first_token_time
                finished = metrics.finished_time
                if first is not None:
                    stats["model_ttft_s"] += max(float(first - metrics.arrival_time), 0.0)
                if first is not None and finished is not None:
                    stats["model_decode_s"] += max(float(finished - first), 0.0)
        return outputs

    async def _generate(self, prompts, parameters, lora_request, reaction_id, call_index):
        async def one(prompt, prompt_index):
            identity = f"{self.seed}:{self.rank}:{reaction_id}:{call_index}:{prompt_index}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            request_id = f"earho-{digest[:24]}"
            sampling = copy.deepcopy(parameters)
            sampling.seed = int(digest[:8], 16)
            final = None
            async for output in self.engine.generate(
                prompt, sampling, request_id=request_id, lora_request=lora_request,
            ):
                final = output
            if final is None or not final.finished:
                raise RuntimeError(f"vLLM request did not finish: {request_id}")
            return final

        return await asyncio.gather(*(one(prompt, index) for index, prompt in enumerate(prompts)))


def _bounded_results(pool, fn, rows, limit):
    """Keep only ``limit`` reactions live and emit completed groups promptly."""

    source = iter(rows)
    pending = set()

    def fill():
        while len(pending) < limit:
            try:
                row = next(source)
            except StopIteration:
                break
            pending.add(pool.submit(fn, row))

    fill()
    while pending:
        future = next(as_completed(pending))
        pending.remove(future)
        result = future.result()
        fill()
        yield result


def _prompt_modes(args) -> tuple[str, ...]:
    return PROMPT_MODES if getattr(args, "legacy_dual_prompt", False) else ("unified",)


def _first_action_sampling_plan(args) -> tuple[int, float]:
    """Keep K candidates legal in vLLM for training and validation alike."""

    modes = _prompt_modes(args)
    if args.k < 2 or args.k % len(modes):
        raise ValueError("k must be at least 2 and divisible by the prompt-mode count")
    per_mode = args.k // len(modes)
    temperature = float(args.temperature)
    if temperature < 0 or (per_mode > 1 and temperature < 1e-5):
        raise ValueError("multiple candidates per prompt require non-greedy sampling")
    return per_mode, temperature


def _messages(task, mode: str, *, state_only: bool = False) -> list[dict[str, Any]]:
    if mode not in (*PROMPT_MODES, "unified"):
        raise ValueError(f"unsupported prompt mode: {mode}")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                policy_prompt(
                    task.target, task.anchor_state, include_inventory=True,
                    actions=task.anchor_actions, compact_history=not state_only,
                )
                if isinstance(task, V2AnchorTask)
                else _prompt(
                    task.target, task.anchor_state,
                    include_inventory=mode in {"event", "unified"},
                )
            ),
        },
    ]


def _render_prompt(tokenizer, task, state: str, mode: str, *, actions=None, state_only: bool = False) -> list[int]:
    if isinstance(task, V2AnchorTask):
        if mode != "unified":
            raise ValueError("v2 trajectory policy requires the unified prompt")
        content = policy_prompt(
            task.target, state, include_inventory=True,
            actions=task.anchor_actions if actions is None else actions,
            compact_history=not state_only,
        )
    else:
        content = _prompt(
            task.target, state, include_inventory=mode in {"event", "unified"},
        )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": content},
    ]
    rendered = (
        render_qwen_sft_tool_prefix(tokenizer, messages, tools=TOOLS)
        if isinstance(task, V2AnchorTask)
        else render_chat(tokenizer, messages, tools=TOOLS, add_generation_prompt=True)
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _completion(tokenizer, value, eos_ids):
    ids = list(value.token_ids)
    if value.logprobs is None or len(value.logprobs) != len(ids):
        raise ValueError("missing behavior token likelihoods")
    logps = [float(item[token].logprob) for token, item in zip(ids, value.logprobs)]
    if not all(math.isfinite(item) for item in logps):
        raise ValueError("nonfinite behavior token likelihood")
    terminated = bool(ids and ids[-1] in eos_ids and value.finish_reason == "stop")
    return ids, logps, terminated, tokenizer.decode(ids, skip_special_tokens=False)


def _allowed(mode: str, name: str) -> bool:
    if mode == "unified":
        return name in {"import_fragments", "apply_electron_flow", "finish_trace"}
    return (
        name == "apply_electron_flow"
        if mode == "event"
        else name in {"import_fragments", "finish_trace"}
    )


def _decode_action(tokenizer, value, eos_ids, mode: str):
    ids, logps, terminated, text = _completion(tokenizer, value, eos_ids)
    name, arguments, error = prediction_call(text, tokenizer)
    if not terminated:
        error = error or "TRUNCATED_TOOL_CALL"
    if not error and not _allowed(mode, name):
        error = f"WRONG_TOOL_FOR_PROMPT_MODE:{mode}:{name or 'NONE'}"
    return {
        "ids": ids,
        "logps": logps,
        "terminated": terminated,
        "text": text,
        "name": name,
        "arguments": arguments,
        "error": error,
        "mode": mode,
        "mean_logprob": sum(logps) / max(len(logps), 1),
    }


def _node(task) -> Node:
    state = task.anchor_state
    return Node(
        target=task.target,
        state=state,
        next_map=max(mapped_atom_numbers(state), default=0) + 1,
        actions=list(task.anchor_actions) if isinstance(task, V2AnchorTask) else [],
        visited=(
            set(task.prefix_states)
            if isinstance(task, V2AnchorTask)
            else {visible(state)}
        ),
    )


def _advance(
    node: Node,
    decoded: Mapping[str, Any],
    max_imports: int,
    *,
    reject_target_retained_finish: bool = False,
):
    if decoded["error"]:
        return None, str(decoded["error"])
    name = str(decoded["name"])
    arguments = dict(decoded["arguments"])
    action = Action(
        name=name,
        arguments=arguments,
        raw=str(decoded["text"]),
        logprob=sum(float(value) for value in decoded["logps"]),
        tokens=max(len(decoded["ids"]), 1),
    )
    child, error = execute(
        node,
        action,
        max_imports=max_imports,
        reject_target_retained_finish=reject_target_retained_finish,
    )
    if error == "ValueError:TARGET_RETAINED_NO_TRANSFORM":
        error = "TARGET_RETAINED_NO_TRANSFORM"
    return child, error


def _critic_scores(
    llm, tokenizer, value_lora, parameters, task, nodes, *, value_kind="state_abc"
):
    if value_lora is None:
        return [0.0] * len(nodes)
    if value_kind == "successor_pn":
        prompts = []
        for node in nodes:
            current = (
                str(node.actions[-1]["state_before"])
                if node.actions
                else task.anchor_state
            )
            prompts.append(
                render_chat(
                    tokenizer,
                    [
                        {"role": "system", "content": SUCCESSOR_VALUE_SYSTEM},
                        {
                            "role": "user",
                            "content": successor_value_prompt(
                                task.target,
                                current,
                                node.state,
                                terminal=node.terminal,
                            ),
                        },
                    ],
                    tools=[],
                    add_generation_prompt=True,
                )
            )
        labels = "PN"
    elif value_kind == "state_abc":
        prompts = [
            render_chat(
                tokenizer,
                [
                    {"role": "system", "content": VALUE_SYSTEM},
                    {"role": "user", "content": value_prompt(task.target, node.state)},
                ],
                tools=[],
                add_generation_prompt=True,
            )
            for node in nodes
        ]
        labels = "ABC"
    else:
        raise ValueError(f"unsupported value critic kind: {value_kind}")
    generated = llm.generate(
        prompts, parameters, lora_request=value_lora, use_tqdm=False
    )
    label_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"][0]
        for label in labels
    }
    output = []
    for node, generation in zip(nodes, generated, strict=True):
        distribution = generation.outputs[0].logprobs[0]
        label_logps = {}
        for label, token_id in label_ids.items():
            value = distribution.get(token_id)
            if value is None:
                raise ValueError(f"critic omitted allowed label {label}")
            label_logps[label] = float(
                value.logprob if hasattr(value, "logprob") else value
            )
        output.append(
            successor_value_margin(label_logps)
            if value_kind == "successor_pn"
            else state_value_margin(label_logps, terminal=node.terminal)
        )
    return output


def _greedy_continue(
    llm,
    tokenizer,
    lora,
    parameters,
    value_lora,
    value_parameters,
    eos_ids,
    task,
    node,
    args,
):
    """Gold-free continuation ranked by a frozen state-value critic."""

    candidates = []
    prompts = []
    modes = []
    for mode in _prompt_modes(args):
        prompt = _render_prompt(tokenizer, task, node.state, mode, actions=node.actions,
                                state_only=getattr(args, "state_only_observation", False))
        if len(prompt) + args.max_new_tokens > args.max_context:
            return None, "CONTEXT_BUDGET"
        prompts.append({"prompt_token_ids": prompt})
        modes.append(mode)
    generated = llm.generate(
        prompts, parameters, lora_request=lora, use_tqdm=False
    )
    for mode, output in zip(modes, generated, strict=True):
        for generated_value in output.outputs:
            decoded = _decode_action(tokenizer, generated_value, eos_ids, mode)
            child, error = _advance(
                node,
                decoded,
                args.max_imports,
                reject_target_retained_finish=args.reject_target_retained_finish,
            )
            if child is not None:
                candidates.append((float(decoded["mean_logprob"]), child))
    if not candidates:
        return None, "NO_EXECUTABLE_CONTINUATION"
    # Different surface actions may execute to the same chemical successor.
    # Keep the strongest policy realization before spending critic compute.
    unique = {}
    for policy_score, child in candidates:
        key = (visible(child.state), bool(child.terminal))
        if key not in unique or policy_score > unique[key][0]:
            unique[key] = (policy_score, child)
    candidates = list(unique.values())
    critic_scores = _critic_scores(
        llm,
        tokenizer,
        value_lora,
        value_parameters,
        task,
        [child for _, child in candidates],
        value_kind=args.value_kind,
    )
    ranked = [
        (
            float(args.value_score_weight) * critic_score
            + float(args.policy_score_weight) * policy_score,
            child,
        )
        for (policy_score, child), critic_score in zip(
            candidates, critic_scores, strict=True
        )
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], ""


def _beam_continue(
    llm,
    tokenizer,
    lora,
    parameters,
    value_lora,
    value_parameters,
    eos_ids,
    task,
    start,
    args,
    *,
    remaining_decisions: int,
):
    return _beam_continue_many(
        llm, tokenizer, lora, parameters, value_lora, value_parameters,
        eos_ids, task, [start], args,
        remaining_decisions=[remaining_decisions],
    )[0]


def _beam_continue_many(
    llm,
    tokenizer,
    lora,
    parameters,
    value_lora,
    value_parameters,
    eos_ids,
    task,
    starts,
    args,
    *,
    remaining_decisions,
):
    """Replan after every executed event while retaining fallback branches.

    Batch independent first-action continuations into one generation per depth.
    Beam ranking, deduplication, and termination remain private to each start.
    """

    if len(starts) != len(remaining_decisions):
        raise ValueError("starts and remaining_decisions must have equal length")
    width = max(int(args.continuation_beam_width), 1)
    frontiers = [[start] if not start.terminal else [] for start in starts]
    terminals = [[(0.0, start)] if start.terminal else [] for start in starts]
    last_errors: list[list[str]] = [[] for _ in starts]
    active = [not start.terminal and limit > 0 for start, limit in zip(starts, remaining_decisions, strict=True)]
    for depth in range(max(remaining_decisions, default=0)):
        jobs = []
        prompts = []
        for group, frontier in enumerate(frontiers):
            if not active[group] or depth >= remaining_decisions[group]:
                continue
            for node in frontier:
                for mode in _prompt_modes(args):
                    prompt = _render_prompt(tokenizer, task, node.state, mode, actions=node.actions,
                                            state_only=getattr(args, "state_only_observation", False))
                    if len(prompt) + args.max_new_tokens > args.max_context:
                        last_errors[group].append("CONTEXT_BUDGET")
                        continue
                    jobs.append((group, node, mode))
                    prompts.append({"prompt_token_ids": prompt})
        if not jobs:
            break
        generated = llm.generate(
            prompts, parameters, lora_request=lora, use_tqdm=False
        )
        candidates_by_group = [[] for _ in starts]
        for (group, node, mode), output in zip(jobs, generated, strict=True):
            for generated_value in output.outputs:
                decoded = _decode_action(tokenizer, generated_value, eos_ids, mode)
                child, error = _advance(
                    node,
                    decoded,
                    args.max_imports,
                    reject_target_retained_finish=args.reject_target_retained_finish,
                )
                if child is None:
                    if error:
                        last_errors[group].append(error)
                    continue
                candidates_by_group[group].append(child)

        # Pool convergent paths only within the same first-action candidate.
        unique_by_group = []
        for candidates in candidates_by_group:
            unique = {}
            for child in candidates:
                key = (visible(child.state), bool(child.terminal))
                incumbent = unique.get(key)
                if incumbent is None or child.policy_score > incumbent.policy_score:
                    unique[key] = child
            unique_by_group.append(list(unique.values()))
        all_candidates = [child for candidates in unique_by_group for child in candidates]
        if not all_candidates:
            break
        critic_scores = _critic_scores(
            llm,
            tokenizer,
            value_lora,
            value_parameters,
            task,
            all_candidates,
            value_kind=args.value_kind,
        )
        offset = 0
        for group, candidates in enumerate(unique_by_group):
            if not candidates:
                active[group] = False
                continue
            scores = critic_scores[offset : offset + len(candidates)]
            offset += len(candidates)
            ranked = []
            for child, critic_score in zip(candidates, scores, strict=True):
                child.value = float(critic_score)
                score = (
                    float(args.value_score_weight) * float(critic_score)
                    + float(args.policy_score_weight) * float(child.policy_score)
                )
                ranked.append((score, child))
            ranked.sort(key=lambda item: item[0], reverse=True)
            terminals[group].extend(item for item in ranked if item[1].terminal)
            terminals[group] = sorted(terminals[group], key=lambda item: item[0], reverse=True)[:width]
            frontiers[group] = [child for _, child in ranked if not child.terminal][:width]
            active[group] = bool(frontiers[group])

    results = []
    for group, frontier in enumerate(frontiers):
        if terminals[group]:
            results.append((terminals[group][0][1], ""))
        elif frontier:
            best = max(
                frontier,
                key=lambda child: (
                    float(args.value_score_weight) * float(child.value)
                    + float(args.policy_score_weight) * float(child.policy_score)
                ),
            )
            results.append((best, "DECISION_BUDGET"))
        else:
            results.append((None, last_errors[group][-1] if last_errors[group] else "NO_EXECUTABLE_CONTINUATION"))
    return results


def _score_rollout(
    task,
    node,
    error: str,
    steps: int,
    *,
    first_successor_state: str,
    invalid_penalty: float,
    wrong_terminal_penalty: float,
    endpoint_similarity_weight: float,
    first_successor_progress_weight: float,
    nonexact_reward_ceiling: float,
    target_retained_penalty: float,
    reference_first_successor_state: str,
    reference_first_successor_weight: float,
    endpoint_metric: str = "full",
):
    terminal = bool(node is not None and node.terminal)
    precursor = visible(node.state) if node is not None else ""
    full_correct = bool(terminal and precursor == task.expected_precursor)
    if endpoint_metric == "structural":
        if not isinstance(task, V2AnchorTask):
            raise ValueError("structural EARHO reward requires a mapped v2 reference task")
        projected = lambda state: split_precursor_endpoints(
            state, task.target_mapped,
        ).structural if state else ""
        structural_precursor = projected(node.state) if node is not None else ""
        correct = bool(
            terminal and structural_exact(
                structural_precursor, task.expected_structural_precursor,
            )
        )
        reward_anchor = projected(task.anchor_state)
        reward_first = projected(first_successor_state)
        reward_final = structural_precursor
        reward_reference = task.expected_structural_precursor
    elif endpoint_metric == "full":
        structural_precursor = ""
        correct = full_correct
        reward_anchor = task.anchor_state
        reward_first = first_successor_state
        reward_final = node.state if node is not None else ""
        reward_reference = task.expected_precursor
    else:
        raise ValueError(f"unsupported EARHO endpoint metric: {endpoint_metric}")
    target_retained = bool(
        terminal
        and not correct
        and contains_unchanged_target(node.state, task.target)
        and not contains_unchanged_target(task.expected_precursor, task.target)
    )
    reference_first_successor_exact = bool(
        first_successor_state
        and reference_first_successor_state
        and visible(first_successor_state) == visible(reference_first_successor_state)
    )
    shaped = endpoint_shaped_reward(
        correct=correct,
        terminal=terminal,
        anchor_state=reward_anchor,
        first_successor_state=reward_first,
        final_state=reward_final,
        expected_precursor=reward_reference,
        invalid_penalty=invalid_penalty,
        wrong_terminal_penalty=wrong_terminal_penalty,
        endpoint_similarity_weight=endpoint_similarity_weight,
        first_successor_progress_weight=first_successor_progress_weight,
        nonexact_reward_ceiling=nonexact_reward_ceiling,
        target_retained=target_retained,
        target_retained_penalty=target_retained_penalty,
        reference_first_successor_exact=reference_first_successor_exact,
        reference_first_successor_weight=reference_first_successor_weight,
    )
    return {
        "formal_execute": terminal,
        "productive_execute": bool(terminal and not target_retained),
        "target_retained": target_retained,
        "correct": correct,
        "endpoint_metric": endpoint_metric,
        "full_endpoint_exact": full_correct,
        "structural_endpoint_exact": correct if endpoint_metric == "structural" else None,
        "precursor_smiles": precursor,
        "structural_precursor_smiles": visible(structural_precursor) if structural_precursor else "",
        "reward": float(shaped["reward"]),
        "reward_terms": shaped,
        "first_successor_state": (
            visible(first_successor_state) if first_successor_state else ""
        ),
        # Private training label retained in rollout artifacts for auditable
        # successor-value mining.  It is never rendered into an actor prompt.
        "reference_first_successor_exact": reference_first_successor_exact,
        "failure": error,
        "decisions": steps,
        "trajectory": list(node.actions) if node is not None else [],
    }


def _reference_first_decision(row: Mapping[str, Any], episode: Mapping[str, Any]):
    """Return private training-only supervision for the first anchor decision."""

    if isinstance(episode, V2AnchorTask):
        return (
            "unified",
            episode.reference_name,
            dict(episode.reference_arguments),
            episode.reference_next_state,
        )

    event = dict(episode["events"][0])
    imports = [str(value) for value in event.get("imports") or []]
    if imports:
        step = dict(
            ((row.get("metadata") or {}).get("trace_plan") or {})["steps"][
                int(event["event_index"])
            ]
        )
        visible_imports = [visible(value) for value in imports]
        arguments = _import_arguments(imports, visible_imports, step.get("moves") or [])
        return "action", "import_fragments", arguments, str(event["event_state"])
    return (
        "event",
        "apply_electron_flow",
        dict(event["gold_arguments"]),
        str(event["reference_successor"]),
    )


def _verified_replay_record(tokenizer, task, row, episode, max_context: int, args):
    mode, name, arguments, _ = _reference_first_decision(row, episode)
    if not getattr(args, "legacy_dual_prompt", False):
        mode = "unified"
    messages = _messages(task, mode, state_only=getattr(args, "state_only_observation", False)) + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "verified_anchor_replay",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    ]
    encoded, metadata = encode_assistant_only_conversation(
        tokenizer,
        {"messages": messages, "tools": TOOLS},
        max_length=max_context,
    )
    if metadata["exceeds_max_length"]:
        raise ValueError("VERIFIED_REPLAY_EXCEEDS_CONTEXT")
    input_ids = list(encoded["input_ids"])
    loss_mask = [int(value != -100) for value in encoded["labels"]]
    return {
        "id": task.reaction_id,
        "kind": "verified_replay",
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "old_logps": [0.0] * len(input_ids),
        "advantage": 0.0,
        "reward": 1.0,
        "prompt_mode": mode,
        "reference_action": name,
        "anchor": _task_record(task, state_only=getattr(args, "state_only_observation", False)),
    }


def _task_record(task, *, state_only: bool = False):
    record = task_record(task)
    if isinstance(task, V2AnchorTask):
        record.update(
            version="earho_first_divergence_v2",
            divergence_reason=task.divergence_reason,
            decision_index=task.prefix_events,
            compact_history=not state_only,
        )
    return record


def _v2_successor_fingerprint(state: str, terminal: bool, invalid_text: str) -> str:
    """Pool equivalent executed successors across action names and surface text."""

    if not state:
        return "INVALID:" + hashlib.sha256(invalid_text.encode()).hexdigest()[:16]
    payload = f"{int(terminal)}:{visible(state)}"
    return "CHEM:" + hashlib.sha256(payload.encode()).hexdigest()


def _v2_probe(llm, tokenizer, lora, parameters, eos_ids, reference, args):
    """Locate the first divergence using only product and public executor feedback."""

    probe_task = v2_anchor_task(reference, 0, divergence_reason="PRODUCT_PROBE")

    def step(node):
        prompt = _render_prompt(
            tokenizer, probe_task, node.state, "unified", actions=node.actions,
            state_only=getattr(args, "state_only_observation", False),
        )
        if len(prompt) + args.max_new_tokens > args.max_context:
            return None, "PRODUCT_PROBE_CONTEXT_BUDGET"
        generated = llm.generate(
            [{"prompt_token_ids": prompt}], parameters,
            lora_request=lora, use_tqdm=False,
        )
        decoded = _decode_action(tokenizer, generated[0].outputs[0], eos_ids, "unified")
        return _advance(
            node, decoded, args.max_imports,
            reject_target_retained_finish=args.reject_target_retained_finish,
        )

    return locate_first_divergence(reference, step)


def _collector_error_records(row, args, total: int, exc: Exception):
    reaction_id = str(row.get("id") or row.get("source_id") or "UNKNOWN")
    digest = hashlib.sha256(reaction_id.encode("utf-8")).hexdigest()
    failure = f"COLLECTOR_EXCEPTION:{type(exc).__name__}:{exc}"
    anchor = {
        "version": "collector_error_v1",
        "state_hash": digest,
        "horizon": -1,
        "total_steps": total,
        "prefix_steps": -1,
        "is_full_episode": bool(args.evaluation or args.full_only),
    }
    return [
        {
            "id": reaction_id,
            "kind": "collector_error",
            "input_ids": [],
            "loss_mask": [],
            "old_logps": [],
            "advantage": 0.0,
            "reward": -abs(float(args.invalid_penalty)),
            "candidate_index": index,
            "terminated": False,
            "prediction": "",
            "score": {
                "formal_execute": False,
                "productive_execute": False,
                "target_retained": False,
                "correct": False,
                "precursor_smiles": "",
                "reward": -abs(float(args.invalid_penalty)),
                "reward_terms": {"outcome": "collector_error"},
                "failure": failure,
                "decisions": 0,
                "trajectory": [],
            },
            "prompt_mode": "collector_error",
            "action_fingerprint": f"collector_error:{digest}",
            "anchor": anchor,
        }
        for index in range(int(args.k))
    ]


def collect(args):
    if getattr(args, "async_reactions", 1) > 1:
        return asyncio.run(_collect_async(args))
    return _collect_sync(args)


def _engine_kwargs(args):
    return dict(
        model=args.model,
        tokenizer=args.model,
        dtype=getattr(args, "dtype", "bfloat16"),
        tensor_parallel_size=1,
        gpu_memory_utilization=(
            0.7 if os.environ.get("MECHET_V100_TORCH_LORA") == "1" else 0.85
        ),
        max_model_len=args.max_context,
        max_num_seqs=32,
        enable_prefix_caching=True,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=2 if args.value_adapter else 1,
        max_cpu_loras=2 if args.value_adapter else 1,
        enforce_eager=args.engine_mode == "eager",
        seed=(args.seed + args.rank) % (2**32),
        trust_remote_code=True,
    )


def _configure_v100_torch_lora() -> None:
    """Avoid Triton LoRA kernels that fail LLVM layout lowering on V100."""

    if os.environ.get("MECHET_V100_TORCH_LORA") != "1":
        return
    import torch
    from vllm.platforms import current_platform

    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 7:
        raise RuntimeError("the V100 PyTorch LoRA fallback requires compute capability 7.x")
    current_platform.get_punica_wrapper = lambda: (
        "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"
    )
    print("[earho] V100 PyTorch LoRA kernels selected", flush=True)


def _collect_sync(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm

    if vllm.__version__ != "0.8.5":
        raise ValueError(f"expected vLLM 0.8.5, got {vllm.__version__}")
    _configure_v100_torch_lora()
    llm = LLM(**_engine_kwargs(args))
    return _collect_initialized(args, llm, llm.get_tokenizer(), SamplingParams, LoRARequest)


async def _collect_async(args):
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.lora.request import LoRARequest
    import vllm

    if vllm.__version__ != "0.8.5":
        raise ValueError(f"expected vLLM 0.8.5, got {vllm.__version__}")
    _configure_v100_torch_lora()
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**_engine_kwargs(args)))
    try:
        tokenizer = await engine.get_tokenizer()
        bridge = AsyncVLLMBridge(engine, asyncio.get_running_loop(), seed=args.seed, rank=args.rank)
        await asyncio.to_thread(
            _collect_initialized, args, bridge, tokenizer, SamplingParams, LoRARequest,
        )
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        else:
            engine.shutdown_background_loop()


def _collect_initialized(args, llm, tokenizer, SamplingParams, LoRARequest):
    prompt_modes = _prompt_modes(args)
    first_n, first_temperature = _first_action_sampling_plan(args)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.data)[args.rank :: args.world_size]
    lora = LoRARequest("nl_anchor_actor", 1, str(Path(args.adapter).resolve()))
    value_lora = (
        LoRARequest("nl_anchor_value", 2, str(Path(args.value_adapter).resolve()))
        if args.value_adapter
        else None
    )
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
    first_parameters = SamplingParams(
        n=first_n,
        temperature=first_temperature,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=eos_ids,
        logprobs=0,
    )
    probe_parameters = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=eos_ids,
        logprobs=0,
    )
    continuation_parameters = SamplingParams(
        n=args.continuation_candidates_per_mode,
        temperature=(
            args.continuation_temperature
            if args.continuation_candidates_per_mode > 1
            else 0.0
        ),
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        max_tokens=args.max_new_tokens,
        stop_token_ids=eos_ids,
        logprobs=0,
    )
    value_labels = "PN" if args.value_kind == "successor_pn" else "ABC"
    label_ids = []
    for label in value_labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"critic label is not one token: {label}={ids}")
        label_ids.append(ids[0])
    value_parameters = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        logprobs=len(value_labels),
        allowed_token_ids=label_ids,
    )

    def process_row(indexed_row):
        row_index, row = indexed_row
        if hasattr(llm, "begin_reaction"):
            llm.begin_reaction(
                f"{row_index}:{row.get('id') or row.get('source_id') or 'UNKNOWN'}"
            )
        detail = None
        group_started = time.monotonic()
        cpu_started = time.thread_time()
        probe_wall = 0.0
        first_generate_wall = 0.0
        continuation_wall = 0.0
        total = (
            len(row.get("earho_v2_reference_decisions") or ())
            if args.protocol_v2
            else len(((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
        )
        try:
            if total < 1:
                raise ValueError(f"{row.get('id')}: empty trace")
            rng = stable_rng(args.seed, args.round_index, str(row["id"]))
            no_correction_frontier = False
            first_divergence_index = None
            if args.protocol_v2:
                reference = replay_reference(
                    row, row["earho_v2_reference_decisions"],
                    max_imports=args.max_imports,
                    compact_history=not getattr(args, "state_only_observation", False),
                )
                if args.evaluation or args.full_only:
                    anchor_index, reason = 0, "PRODUCT_ONLY_EVALUATION"
                else:
                    probe_started = time.monotonic()
                    divergence = _v2_probe(
                        llm, tokenizer, lora, probe_parameters, eos_ids,
                        reference, args,
                    )
                    probe_wall = time.monotonic() - probe_started
                    first_divergence_index = divergence.decision_index
                    no_correction_frontier = divergence.decision_index is None
                    if no_correction_frontier:
                        anchor_index, reason = 0, divergence.reason
                    elif rng.random() < args.full_episode_fraction:
                        anchor_index, reason = 0, "FULL_EPISODE_REHEARSAL"
                    else:
                        anchor_index = int(divergence.decision_index)
                        reason = divergence.reason
                task = v2_anchor_task(
                    reference, anchor_index, divergence_reason=reason
                )
                episode = task
                reference_first_successor = task.reference_next_state
            else:
                horizon = (
                    total
                    if args.evaluation or args.full_only
                    else choose_horizon(
                        total, args.frontier, rng,
                        full_episode_fraction=args.full_episode_fraction,
                    )
                )
                episode = reference_episode(row, horizon)
                task = task_from_episode(episode)
                _, _, _, reference_first_successor = _reference_first_decision(
                    row, episode
                )
            prompts = {
                mode: _render_prompt(
                    tokenizer, task, task.anchor_state, mode,
                    actions=(task.anchor_actions if isinstance(task, V2AnchorTask) else None),
                    state_only=getattr(args, "state_only_observation", False),
                )
                for mode in prompt_modes
            }
            if any(
                len(prompt) + args.max_new_tokens > args.max_context
                for prompt in prompts.values()
            ):
                raise ValueError(
                    f"{task.reaction_id}: first-action prompt exceeds context"
                )
            first_started = time.monotonic()
            generations = llm.generate(
                [{"prompt_token_ids": prompts[mode]} for mode in prompt_modes],
                first_parameters,
                lora_request=lora,
                use_tqdm=False,
            )
            first_generate_wall = time.monotonic() - first_started
            prepared = []
            continuation_starts = []
            continuation_limits = []
            for mode, generated in zip(prompt_modes, generations, strict=True):
                for value in generated.outputs:
                    decoded = _decode_action(tokenizer, value, eos_ids, mode)
                    node, error = _advance(
                        _node(task),
                        decoded,
                        args.max_imports,
                        reject_target_retained_finish=args.reject_target_retained_finish,
                    )
                    decisions = int(node is not None)
                    first_state = node.state if node is not None else ""
                    first_terminal = bool(node is not None and node.terminal)
                    fingerprint = (
                        _v2_successor_fingerprint(
                            first_state, first_terminal, str(decoded["text"])
                        )
                        if args.protocol_v2
                        else successor_fingerprint(
                            prompt_mode=mode,
                            action_name=str(decoded["name"]),
                            successor_state=first_state,
                            terminal=first_terminal,
                            invalid_text=str(decoded["text"]),
                        )
                    )
                    limit = (
                        args.max_decisions
                        if args.protocol_v2 and (
                            args.evaluation or task.divergence_reason == "FULL_EPISODE_REHEARSAL"
                        )
                        else min(args.max_decisions, max(1, args.frontier))
                        if args.protocol_v2
                        else min(args.max_decisions, 2 * task.horizon + 2)
                    )
                    item = {
                        "mode": mode,
                        "decoded": decoded,
                        "node": node,
                        "error": error,
                        "decisions": decisions,
                        "first_state": first_state,
                        "first_terminal": first_terminal,
                        "fingerprint": fingerprint,
                    }
                    prepared.append(item)
                    if node is not None and not node.terminal and decisions < limit:
                        item["before_actions"] = len(node.actions)
                        item["continuation_index"] = len(continuation_starts)
                        continuation_starts.append(node)
                        continuation_limits.append(limit - decisions)
            if continuation_starts:
                continuation_started = time.monotonic()
                continuations = _beam_continue_many(
                    llm, tokenizer, lora, continuation_parameters, value_lora,
                    value_parameters, eos_ids, task, continuation_starts, args,
                    remaining_decisions=continuation_limits,
                )
                continuation_wall = time.monotonic() - continuation_started
                for item in prepared:
                    if "continuation_index" not in item:
                        continue
                    node, continuation_error = continuations[item["continuation_index"]]
                    item["node"] = node
                    item["decisions"] += (
                        max(len(node.actions) - item["before_actions"], 0)
                        if node is not None else 0
                    )
                    if continuation_error:
                        item["error"] = continuation_error
            records = []
            for candidate_index, item in enumerate(prepared):
                mode = item["mode"]
                decoded = item["decoded"]
                node = item["node"]
                error = item["error"]
                decisions = item["decisions"]
                first_state = item["first_state"]
                first_terminal = item["first_terminal"]
                fingerprint = item["fingerprint"]
                if node is not None and not node.terminal and not error:
                    error = "DECISION_BUDGET"
                score = _score_rollout(
                    task,
                    node,
                    error,
                    decisions,
                    first_successor_state=first_state,
                    invalid_penalty=args.invalid_penalty,
                    wrong_terminal_penalty=args.wrong_terminal_penalty,
                    endpoint_similarity_weight=args.endpoint_similarity_weight,
                    first_successor_progress_weight=args.first_successor_progress_weight,
                    nonexact_reward_ceiling=args.nonexact_reward_ceiling,
                    target_retained_penalty=args.target_retained_penalty,
                    reference_first_successor_state=reference_first_successor,
                    reference_first_successor_weight=args.reference_first_successor_weight,
                    endpoint_metric=args.endpoint_metric,
                )
                score["first_successor_terminal"] = first_terminal
                ids = list(decoded["ids"])
                logps = list(decoded["logps"])
                prompt = prompts[mode]
                record = {
                    "id": task.reaction_id,
                    "kind": "rl",
                    "input_ids": prompt + ids,
                    "loss_mask": [0] * len(prompt) + [1] * len(ids),
                    "old_logps": [0.0] * len(prompt) + logps,
                    "advantage": 0.0,
                    "reward": float(score["reward"]),
                    "candidate_index": candidate_index,
                    "terminated": bool(decoded["terminated"]),
                    "prediction": str(decoded["text"]),
                    "score": score,
                    "prompt_mode": mode,
                    "action_fingerprint": fingerprint,
                    "anchor": _task_record(task, state_only=getattr(args, "state_only_observation", False)),
                }
                if not ids:
                    record["loss_mask"] = [0] * len(record["loss_mask"])
                if not (
                    len(record["input_ids"])
                    == len(record["loss_mask"])
                    == len(record["old_logps"])
                ):
                    raise ValueError("rollout token/mask/logprob misalignment")
                records.append(record)
            summary = (
                assign_sibling_advantages(
                    records,
                    method=args.vnext_credit,
                    dynamic_all_negative=True,
                    allow_private_reference=args.vnext_private_reference_credit,
                )
                if args.vnext_credit
                else assign_local_advantages(
                    records, success_gated=args.success_gated_advantages
                )
            )
            if args.vnext_credit:
                teacher = search_teacher_distribution(records)
                for record in records:
                    record["search_teacher"] = teacher
            if no_correction_frontier and not args.evaluation:
                for record in records:
                    record["advantage"] = 0.0
                    record["update_eligible"] = False
                summary["effective"] = False
                summary["eligible_records"] = 0
            needs_replay = (
                not args.evaluation
                and not no_correction_frontier
                and (
                    not args.protocol_v2
                    or not (summary["successor_success"] or summary["endpoint_success"])
                )
            )
            if needs_replay:
                records.append(
                    _verified_replay_record(
                        tokenizer, task, row, episode, args.max_context, args
                    )
                )
            log_fields = {
                "id": task.reaction_id,
                "horizon": task.horizon,
                "full_episode": task.is_full_episode,
                "first_divergence_index": (
                    first_divergence_index if args.protocol_v2 else None
                ),
                "no_correction_frontier": no_correction_frontier,
                **summary,
            }
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                raise
            records = _collector_error_records(row, args, total, exc)
            detail = {
                "id": str(row.get("id") or row.get("source_id") or "UNKNOWN"),
                "rank": args.rank,
                "error": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(),
            }
            log_fields = {
                "id": detail["id"],
                "horizon": -1,
                "full_episode": bool(args.evaluation or args.full_only),
                "unique_actions": 1,
                "effective": False,
                "endpoint_success": False,
                "candidate_success_rate": 0.0,
                "collector_error": detail["error"],
            }
        log_fields.update(
            elapsed_wall_s=round(time.monotonic() - group_started, 3),
            collector_thread_cpu_s=round(time.thread_time() - cpu_started, 3),
            probe_wall_s=round(probe_wall, 3),
            first_generate_wall_s=round(first_generate_wall, 3),
            continuation_wall_s=round(continuation_wall, 3),
        )
        if hasattr(llm, "end_reaction"):
            log_fields.update(llm.end_reaction())
        return records, log_fields, detail

    def persist(handle, error_handle, results):
        for number, (records, log_fields, detail) in enumerate(results, 1):
            if detail is not None:
                error_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
                error_handle.flush()
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            log(
                stage="nl-anchor-group",
                rank=args.rank,
                done=number,
                total=len(rows),
                **log_fields,
            )

    error_path = output.with_suffix(output.suffix + ".errors.jsonl")
    with output.open("x", encoding="utf-8") as handle, error_path.open(
        "x", encoding="utf-8"
    ) as error_handle:
        if getattr(args, "async_reactions", 1) > 1:
            with ThreadPoolExecutor(max_workers=args.async_reactions) as pool:
                persist(
                    handle, error_handle,
                    _bounded_results(pool, process_row, enumerate(rows), args.async_reactions),
                )
        else:
            persist(handle, error_handle, map(process_row, enumerate(rows)))


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
    parser.add_argument("--invalid-penalty", type=float, default=0.1)
    parser.add_argument("--wrong-terminal-penalty", type=float, default=0.5)
    parser.add_argument("--endpoint-similarity-weight", type=float, default=0.45)
    parser.add_argument("--first-successor-progress-weight", type=float, default=0.25)
    parser.add_argument("--nonexact-reward-ceiling", type=float, default=0.01)
    parser.add_argument("--target-retained-penalty", type=float, default=0.5)
    parser.add_argument("--reference-first-successor-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-metric", choices=["full", "structural"], default="full")
    parser.add_argument("--value-adapter")
    parser.add_argument(
        "--value-kind", choices=["state_abc", "successor_pn"], default="state_abc"
    )
    parser.add_argument("--continuation-candidates-per-mode", type=int, default=1)
    parser.add_argument("--continuation-temperature", type=float, default=0.7)
    parser.add_argument("--value-score-weight", type=float, default=1.0)
    parser.add_argument("--policy-score-weight", type=float, default=0.1)
    parser.add_argument("--continuation-beam-width", type=int, default=1)
    parser.add_argument("--success-gated-advantages", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--max-imports", type=int, default=32)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--reject-target-retained-finish", action="store_true")
    parser.add_argument(
        "--legacy-dual-prompt",
        action="store_true",
        help="reproduce the historical gold-action-conditioned prompt split",
    )
    parser.add_argument("--protocol-v2", action="store_true")
    parser.add_argument("--state-only-observation", action="store_true")
    parser.add_argument("--vnext-credit", choices=["grpo", "gspo", "tree"])
    parser.add_argument("--vnext-private-reference-credit", action="store_true")
    parser.add_argument("--engine-mode", choices=["eager", "cuda_graph"], default="eager")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--async-reactions", type=int, default=1)
    args = parser.parse_args()
    if args.async_reactions < 1 or args.async_reactions > 32:
        raise ValueError("async reactions must be between 1 and 32 per GPU")
    if args.protocol_v2 and (args.legacy_dual_prompt or not args.success_gated_advantages):
        raise ValueError("EARHO v2 requires unified history prompts and success-gated advantages")
    if args.endpoint_metric == "structural" and not args.protocol_v2:
        raise ValueError("structural EARHO reward requires protocol-v2 mapped tasks")
    if args.vnext_credit and not args.protocol_v2:
        raise ValueError("vNext sibling credit requires protocol-v2 unified prompts")
    if args.vnext_credit == "gspo" and args.mode == "train":
        args.ratio_mode = "sequence"
    args.memory_efficient_logps = True
    (collect if args.mode == "collect" else train)(args)


if __name__ == "__main__":
    main()
