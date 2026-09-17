#!/usr/bin/env python3
"""Collect natural-language anchor branches and run the shared PPO learner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(REPO / "scripts")]

from anchor_branch_stage import train
from python_continual_stage import log, read_rows
from mechet.assistant_masking import encode_assistant_only_conversation, render_chat
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
from scripts.run_natural_language_value_search import Action, Node, execute, visible


PROMPT_MODES = ("action", "event")


def _messages(task, mode: str) -> list[dict[str, Any]]:
    if mode not in PROMPT_MODES:
        raise ValueError(f"unsupported prompt mode: {mode}")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": _prompt(
                task.target,
                task.anchor_state,
                include_inventory=mode == "event",
            ),
        },
    ]


def _render_prompt(tokenizer, task, state: str, mode: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": _prompt(task.target, state, include_inventory=mode == "event"),
        },
    ]
    rendered = render_chat(tokenizer, messages, tools=TOOLS, add_generation_prompt=True)
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
        visited={visible(state)},
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
    if (
        name == "finish_trace"
        and reject_target_retained_finish
        and contains_unchanged_target(node.state, node.target)
    ):
        return None, "TARGET_RETAINED_NO_TRANSFORM"
    action = Action(
        name=name,
        arguments=arguments,
        raw=str(decoded["text"]),
        logprob=sum(float(value) for value in decoded["logps"]),
        tokens=max(len(decoded["ids"]), 1),
    )
    return execute(node, action, max_imports=max_imports)


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
    for mode in PROMPT_MODES:
        prompt = _render_prompt(tokenizer, task, node.state, mode)
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
    """Replan after every executed event while retaining fallback branches.

    Unlike the historical continuation loop, this keeps several chemically
    distinct executor states alive.  A locally preferred branch can fail at a
    later step without destroying the alternatives.  Prompts are regenerated
    from each executor-owned successor, so this is receding-horizon search, not
    one-shot program sampling.
    """

    if start.terminal or remaining_decisions <= 0:
        return start, "" if start.terminal else "DECISION_BUDGET"
    width = max(int(args.continuation_beam_width), 1)
    frontier = [start]
    terminals = []
    last_errors: list[str] = []
    for _ in range(int(remaining_decisions)):
        jobs = []
        prompts = []
        for parent_index, node in enumerate(frontier):
            for mode in PROMPT_MODES:
                prompt = _render_prompt(tokenizer, task, node.state, mode)
                if len(prompt) + args.max_new_tokens > args.max_context:
                    last_errors.append("CONTEXT_BUDGET")
                    continue
                jobs.append((parent_index, node, mode))
                prompts.append({"prompt_token_ids": prompt})
        if not jobs:
            break
        generated = llm.generate(
            prompts, parameters, lora_request=lora, use_tqdm=False
        )
        candidates = []
        for (_, node, mode), output in zip(jobs, generated, strict=True):
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
                        last_errors.append(error)
                    continue
                candidates.append(child)
        if not candidates:
            break

        # Pool surface forms and convergent paths before critic evaluation.
        unique = {}
        for child in candidates:
            key = (visible(child.state), bool(child.terminal))
            incumbent = unique.get(key)
            if incumbent is None or child.policy_score > incumbent.policy_score:
                unique[key] = child
        candidates = list(unique.values())
        critic_scores = _critic_scores(
            llm,
            tokenizer,
            value_lora,
            value_parameters,
            task,
            candidates,
            value_kind=args.value_kind,
        )
        ranked = []
        for child, critic_score in zip(candidates, critic_scores, strict=True):
            child.value = float(critic_score)
            score = (
                float(args.value_score_weight) * float(critic_score)
                + float(args.policy_score_weight) * float(child.policy_score)
            )
            ranked.append((score, child))
        ranked.sort(key=lambda item: item[0], reverse=True)
        terminals.extend(item for item in ranked if item[1].terminal)
        terminals = sorted(terminals, key=lambda item: item[0], reverse=True)[:width]
        frontier = [
            child for _, child in ranked if not child.terminal
        ][:width]
        if not frontier:
            break

    if terminals:
        return terminals[0][1], ""
    if frontier:
        best = max(
            frontier,
            key=lambda child: (
                float(args.value_score_weight) * float(child.value)
                + float(args.policy_score_weight) * float(child.policy_score)
            ),
        )
        return best, "DECISION_BUDGET"
    return None, last_errors[-1] if last_errors else "NO_EXECUTABLE_CONTINUATION"


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
):
    terminal = bool(node is not None and node.terminal)
    precursor = visible(node.state) if node is not None else ""
    correct = bool(terminal and precursor == task.expected_precursor)
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
        anchor_state=task.anchor_state,
        first_successor_state=first_successor_state,
        final_state=node.state if node is not None else "",
        expected_precursor=task.expected_precursor,
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
        "precursor_smiles": precursor,
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


def _verified_replay_record(tokenizer, task, row, episode, max_context: int):
    mode, name, arguments, _ = _reference_first_decision(row, episode)
    messages = _messages(task, mode) + [
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
        "anchor": task_record(task),
    }


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
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm

    if vllm.__version__ != "0.8.5":
        raise ValueError(f"expected vLLM 0.8.5, got {vllm.__version__}")
    if args.k < 2 or args.k % 2:
        raise ValueError("k must be an even integer >= 2")
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"refusing overwrite: {output}")
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
        max_loras=2 if args.value_adapter else 1,
        max_cpu_loras=2 if args.value_adapter else 1,
        enforce_eager=True,
        seed=(args.seed + args.rank) % (2**32),
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
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
        n=args.k // 2,
        temperature=0.0 if args.evaluation else args.temperature,
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

    error_path = output.with_suffix(output.suffix + ".errors.jsonl")
    with output.open("x", encoding="utf-8") as handle, error_path.open(
        "x", encoding="utf-8"
    ) as error_handle:
        for number, row in enumerate(rows, 1):
            total = len(((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
            try:
                if total < 1:
                    raise ValueError(f"{row.get('id')}: empty trace")
                rng = stable_rng(args.seed, args.round_index, str(row["id"]))
                horizon = (
                    total
                    if args.evaluation or args.full_only
                    else choose_horizon(
                        total,
                        args.frontier,
                        rng,
                        full_episode_fraction=args.full_episode_fraction,
                    )
                )
                episode = reference_episode(row, horizon)
                task = task_from_episode(episode)
                _, _, _, reference_first_successor = _reference_first_decision(
                    row, episode
                )
                prompts = {
                    mode: _render_prompt(tokenizer, task, task.anchor_state, mode)
                    for mode in PROMPT_MODES
                }
                if any(
                    len(prompt) + args.max_new_tokens > args.max_context
                    for prompt in prompts.values()
                ):
                    raise ValueError(
                        f"{task.reaction_id}: first-action prompt exceeds context"
                    )
                generations = llm.generate(
                    [{"prompt_token_ids": prompts[mode]} for mode in PROMPT_MODES],
                    first_parameters,
                    lora_request=lora,
                    use_tqdm=False,
                )
                records = []
                candidate_index = 0
                for mode, generated in zip(PROMPT_MODES, generations, strict=True):
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
                        fingerprint = successor_fingerprint(
                            prompt_mode=mode,
                            action_name=str(decoded["name"]),
                            successor_state=first_state,
                            terminal=first_terminal,
                            invalid_text=str(decoded["text"]),
                        )
                        limit = min(args.max_decisions, 2 * task.horizon + 2)
                        if (
                            node is not None
                            and not node.terminal
                            and decisions < limit
                        ):
                            before = len(node.actions)
                            node, continuation_error = _beam_continue(
                                llm,
                                tokenizer,
                                lora,
                                continuation_parameters,
                                value_lora,
                                value_parameters,
                                eos_ids,
                                task,
                                node,
                                args,
                                remaining_decisions=limit - decisions,
                            )
                            decisions += (
                                max(len(node.actions) - before, 0)
                                if node is not None
                                else 0
                            )
                            if continuation_error:
                                error = continuation_error
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
                            "anchor": task_record(task),
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
                        candidate_index += 1
                summary = assign_local_advantages(
                    records, success_gated=args.success_gated_advantages
                )
                if not args.evaluation:
                    records.append(
                        _verified_replay_record(
                            tokenizer, task, row, episode, args.max_context
                        )
                    )
                log_fields = {
                    "id": task.reaction_id,
                    "horizon": task.horizon,
                    "full_episode": task.is_full_episode,
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
                error_handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
                error_handle.flush()
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
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--max-imports", type=int, default=8)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--reject-target-retained-finish", action="store_true")
    args = parser.parse_args()
    args.memory_efficient_logps = True
    (collect if args.mode == "collect" else train)(args)


if __name__ == "__main__":
    main()
