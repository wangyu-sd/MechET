#!/usr/bin/env python3
"""Short closed-loop suffix diagnostic for natural-language electron events.

For every selected reaction, the environment replays a trusted reference
prefix, supplies reference-scheduled exogenous fragments, and then lets the
model predict the remaining 1, 2, 3, or all electron events without any
reference state feedback.  The event horizon and imports are controlled so
this measures electron-event error accumulation rather than stopping or
fragment-retrieval errors.  It is a validation diagnostic, not product-only
endpoint accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from mechet.a7_rescue import canonical_event, stable_sample_key
from mechet.assistant_masking import render_chat
from mechet.forward_expert import verify_electron_step
from mechet.in_place_grounded_flow import (
    append_mapped_fragments_verbatim,
    deterministic_unmapped_state,
    extract_import_fragments,
    mapped_atom_numbers,
    mapped_state_signature,
    retain_mapped_components,
    schedule_imports,
)
from mechet.natural_language_electron_flow import compile_event_arguments
from scripts.build_natural_language_event_sft import SYSTEM, TOOLS, _prompt, convert_row
from scripts.eval_natural_language_event_local import (
    MODEL_REVISION,
    _trim_completion,
    distributed_coordinates,
    prediction_call,
    read_jsonl,
    sha256,
)


def select_reactions(
    rows: Iterable[Mapping[str, Any]], *, size: int, seed: int, minimum_events: int
) -> list[dict[str, Any]]:
    """Return one deterministic cohort shared by every requested horizon."""

    eligible = [
        dict(row)
        for row in rows
        if len(((row.get("metadata") or {}).get("trace_plan") or {}).get("steps") or [])
        >= minimum_events
    ]
    eligible.sort(key=lambda row: stable_sample_key(str(row["id"]), seed))
    if len(eligible) < size:
        raise ValueError(f"requested {size} rows from only {len(eligible)} eligible rows")
    return sorted(eligible[:size], key=lambda row: str(row["id"]))


def reference_episode(row: Mapping[str, Any], horizon: int | None) -> dict[str, Any]:
    """Build the trusted prefix and reference suffix without model-visible maps."""

    plan = dict((row.get("metadata") or {}).get("trace_plan") or {})
    steps = [dict(value) for value in plan.get("steps") or []]
    fragments = extract_import_fragments(row)
    scheduled = schedule_imports(fragments, steps)
    public = convert_row(row)
    event_public_indices = [
        index
        for index, value in enumerate(public)
        if value["metadata"]["decision_type"] == "event"
    ]
    if len(event_public_indices) != len(steps):
        raise ValueError(f"{row['id']}: public/reference event mismatch")
    start_event = 0 if horizon is None else len(steps) - horizon
    if start_event < 0:
        raise ValueError(f"{row['id']}: horizon exceeds reference event count")

    present_maps = set(mapped_atom_numbers(str(row["target_smiles"])))
    current = retain_mapped_components(str(steps[0]["state_before"]), present_maps)
    events: list[dict[str, Any]] = []
    start_state = current
    for event_index, (step, imports) in enumerate(zip(steps, scheduled, strict=True)):
        if event_index == start_event:
            start_state = current
        event_state = append_mapped_fragments_verbatim(current, imports)
        for fragment in imports:
            present_maps.update(mapped_atom_numbers(fragment))
        successor = retain_mapped_components(str(step["state_after"]), present_maps)
        if event_index >= start_event:
            events.append(
                {
                    "event_index": event_index,
                    "imports": list(imports),
                    "event_state": event_state,
                    "reference_successor": successor,
                    "gold_arguments": public[event_public_indices[event_index]]["messages"][2]
                    ["tool_calls"][0]["function"]["arguments"],
                }
            )
        current = successor

    first_public = event_public_indices[start_event]
    if first_public and public[first_public - 1]["metadata"]["decision_type"] == "import":
        first_public -= 1
    prefix_groups = [
        list(value["messages"][1:4]) for value in public[:first_public]
    ]
    expected = deterministic_unmapped_state(
        str(row.get("full_precursor_state") or row["expected_precursor"])
    ).text
    return {
        "reaction_id": str(row["source_id"]),
        "total_events": len(steps),
        "horizon": len(events),
        "start_state": start_state,
        "target": deterministic_unmapped_state(str(row["target_smiles"])).text,
        "expected_precursor": expected,
        "prefix_groups": prefix_groups,
        "events": events,
    }


def _visible_fragments(fragments: Sequence[str]) -> list[str]:
    return [deterministic_unmapped_state(value).text for value in fragments]


def _tool_call(name: str, arguments: Mapping[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": dict(arguments)},
    }


def _render_fitting_prompt(
    tokenizer: Any,
    history_groups: Sequence[Sequence[Mapping[str, Any]]],
    user: Mapping[str, Any],
    *,
    history_window: int,
    max_prompt_tokens: int,
) -> tuple[str, int, int]:
    kept = [list(group) for group in history_groups[-history_window:]]
    while True:
        messages = [{"role": "system", "content": SYSTEM}]
        for group in kept:
            messages.extend(group)
        messages.append(dict(user))
        prompt = render_chat(tokenizer, messages, tools=TOOLS, add_generation_prompt=True)
        tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if tokens <= max_prompt_tokens or not kept:
            return prompt, len(kept), tokens
        kept.pop(0)


def _episode_key(reaction_id: str, horizon_name: str) -> str:
    return f"{reaction_id}::suffix::{horizon_name}"


def run(args: argparse.Namespace) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rank, world, local_rank = distributed_coordinates()
    torch.cuda.set_device(local_rank)
    rows = read_jsonl(args.data)
    numeric_horizons = [int(value) for value in args.horizons if value != "full"]
    minimum = max(numeric_horizons or [1])
    selected_rows = select_reactions(
        rows, size=args.sample_reactions, seed=args.seed, minimum_events=minimum
    )
    episodes: list[tuple[str, dict[str, Any]]] = []
    for row in selected_rows:
        for value in args.horizons:
            horizon = None if value == "full" else int(value)
            episode = reference_episode(row, horizon)
            episodes.append((value, episode))
    episodes = [value for index, value in enumerate(episodes) if index % world == rank]

    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"suffix.shard-{rank:02d}-of-{world:02d}.jsonl"
    completed = {str(row["key"]) for row in read_jsonl(shard)} if shard.exists() else set()
    episodes = [
        value
        for value in episodes
        if _episode_key(value[1]["reaction_id"], value[0]) not in completed
    ]
    if rank == 0:
        reaction_ids = [str(row["source_id"]) for row in selected_rows]
        (args.output / "suffix_selection.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "sample_reactions": args.sample_reactions,
                    "minimum_events": minimum,
                    "horizons": args.horizons,
                    "reaction_ids": reaction_ids,
                    "reaction_ids_sha256": hashlib.sha256(
                        "\n".join(reaction_ids).encode()
                    ).hexdigest(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=MODEL_REVISION, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.float16
    try:
        bnb_version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("bitsandbytes is required") from exc
    print(
        f"[meteor-suffix-eval] rank={rank}/{world} gpu={torch.cuda.get_device_name(local_rank)} "
        f"episodes={len(episodes)} bnb={bnb_version}",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": local_rank},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    device = next(model.parameters()).device

    with shard.open("a", encoding="utf-8") as sink:
        for number, (horizon_name, episode) in enumerate(episodes, 1):
            started = time.time()
            current = str(episode["start_state"])
            history_groups = list(episode["prefix_groups"])
            event_records: list[dict[str, Any]] = []
            failure = ""
            for local_step, reference in enumerate(episode["events"]):
                imports = list(reference["imports"])
                try:
                    current = append_mapped_fragments_verbatim(current, imports)
                    imported = _visible_fragments(imports)
                    content = _prompt(
                        str(episode["target"]), current, include_inventory=True
                    )
                    content += (
                        "\n\nDIAGNOSTIC CONTRACT: predict apply_electron_flow only; "
                        "the environment controls exogenous fragments and the remaining horizon."
                    )
                    if imported:
                        content += "\nEXOGENOUS FRAGMENTS SUPPLIED NOW: " + json.dumps(
                            imported, ensure_ascii=False
                        )
                    user = {"role": "user", "content": content}
                    prompt, kept_history, input_tokens = _render_fitting_prompt(
                        tokenizer,
                        history_groups,
                        user,
                        history_window=args.history_window,
                        max_prompt_tokens=args.max_context - args.max_new_tokens,
                    )
                    encoded = tokenizer(
                        prompt, return_tensors="pt", add_special_tokens=False
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    with torch.inference_mode():
                        output = model.generate(
                            **encoded,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    completion_ids = _trim_completion(
                        [
                            int(value)
                            for value in output[0, encoded["input_ids"].shape[1] :].tolist()
                        ],
                        tokenizer,
                    )
                    generated = tokenizer.decode(completion_ids, skip_special_tokens=False)
                    name, arguments, parse_error = prediction_call(generated, tokenizer)
                    if parse_error:
                        raise ValueError(parse_error)
                    if name != "apply_electron_flow":
                        raise ValueError(f"WRONG_TOOL:{name or 'NONE'}")
                    predicted_moves = compile_event_arguments(current, arguments)
                    execution = verify_electron_step(current, predicted_moves)
                    if not execution.get("ok"):
                        raise ValueError(
                            "EXECUTION_FAILED:"
                            + str(execution.get("code") or execution.get("message") or "UNKNOWN")
                        )
                    predicted_successor = str(execution["state_smiles"])
                    reference_successor = str(reference["reference_successor"])
                    gold_moves = compile_event_arguments(
                        str(reference["event_state"]), reference["gold_arguments"]
                    )
                    event_exact = canonical_event(predicted_moves) == canonical_event(gold_moves)
                    map_exact = mapped_state_signature(predicted_successor) == mapped_state_signature(
                        reference_successor
                    )
                    chemical_exact = deterministic_unmapped_state(
                        predicted_successor
                    ).text == deterministic_unmapped_state(reference_successor).text
                    call_id = f"rollout_{local_step:03d}"
                    tool_result = {
                        "ok": True,
                        "code": "PASS",
                        "current_state": deterministic_unmapped_state(predicted_successor).text,
                    }
                    history_groups.append(
                        [
                            user,
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [_tool_call(name, arguments, call_id)],
                            },
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": name,
                                "content": json.dumps(tool_result, separators=(",", ":")),
                            },
                        ]
                    )
                    event_records.append(
                        {
                            "local_step": local_step + 1,
                            "reference_event_index": int(reference["event_index"]),
                            "oracle_imports": imported,
                            "predicted_arguments": arguments,
                            "generated_text": generated,
                            "generated_tokens": len(completion_ids),
                            "input_tokens": input_tokens,
                            "history_actions_kept": kept_history,
                            "formal_execute": True,
                            "event_exact": event_exact,
                            "successor_map_exact": map_exact,
                            "successor_chemical_exact": chemical_exact,
                        }
                    )
                    current = predicted_successor
                except Exception as exc:
                    failure = f"step={local_step + 1}:{type(exc).__name__}:{exc}"
                    event_records.append(
                        {
                            "local_step": local_step + 1,
                            "reference_event_index": int(reference["event_index"]),
                            "oracle_imports": _visible_fragments(imports),
                            "formal_execute": False,
                            "event_exact": False,
                            "successor_map_exact": False,
                            "successor_chemical_exact": False,
                            "error": failure,
                        }
                    )
                    break
            completed_events = sum(bool(value["formal_execute"]) for value in event_records)
            endpoint = deterministic_unmapped_state(current).text
            endpoint_exact = (
                completed_events == int(episode["horizon"])
                and endpoint == str(episode["expected_precursor"])
            )
            record = {
                "key": _episode_key(str(episode["reaction_id"]), horizon_name),
                "reaction_id": episode["reaction_id"],
                "horizon_name": horizon_name,
                "horizon_events": episode["horizon"],
                "total_reference_events": episode["total_events"],
                "completed_events": completed_events,
                "all_events_formally_executable": completed_events == int(episode["horizon"]),
                "all_events_exact": len(event_records) == int(episode["horizon"])
                and all(bool(value["event_exact"]) for value in event_records),
                "all_successors_chemical_exact": len(event_records) == int(episode["horizon"])
                and all(bool(value["successor_chemical_exact"]) for value in event_records),
                "endpoint_exact": endpoint_exact,
                "failure": failure,
                "elapsed_seconds": time.time() - started,
                "events": event_records,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[meteor-suffix-eval] rank={rank} done={number}/{len(episodes)} "
                f"h={horizon_name} reaction={episode['reaction_id']} "
                f"executed={completed_events}/{episode['horizon']} endpoint={endpoint_exact} "
                f"seconds={record['elapsed_seconds']:.1f}",
                flush=True,
            )
    return 0


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    count = sum(bool(row.get(field)) for row in rows)
    return {"count": count, "denominator": len(rows), "rate": count / max(len(rows), 1)}


def aggregate(args: argparse.Namespace) -> int:
    source = read_jsonl(args.data)
    numeric_horizons = [int(value) for value in args.horizons if value != "full"]
    selected = select_reactions(
        source,
        size=args.sample_reactions,
        seed=args.seed,
        minimum_events=max(numeric_horizons or [1]),
    )
    expected = {
        _episode_key(str(row["source_id"]), horizon)
        for row in selected
        for horizon in args.horizons
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(args.output.glob("suffix.shard-*-of-*.jsonl")):
        rows.extend(read_jsonl(path))
    observed = [str(row["key"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate suffix episode keys")
    missing = sorted(expected - set(observed))
    report = {
        "artifact_type": "natural_language_event_oracle_prefix_suffix_k1_v1",
        "claim_boundary": (
            "Fixed validation diagnostic with trusted reference prefix, oracle exogenous "
            "fragment schedule, fixed event horizon, and no reference state feedback after "
            "the branch point. Not product-only closed-loop endpoint accuracy."
        ),
        "seed": args.seed,
        "sample_reactions": args.sample_reactions,
        "horizons": args.horizons,
        "planned_episodes": len(expected),
        "completed_episodes": len(rows),
        "missing_episodes": len(missing),
        "complete": not missing and len(rows) == len(expected),
        "data": str(args.data),
        "data_sha256": sha256(args.data),
        "adapter": str(args.adapter),
        "adapter_model_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "model": args.model,
        "model_revision": MODEL_REVISION,
        "by_horizon": {},
        "failure_codes": dict(Counter(row["failure"] for row in rows if row["failure"])),
    }
    for horizon in args.horizons:
        subset = [row for row in rows if row["horizon_name"] == horizon]
        report["by_horizon"][horizon] = {
            "episodes": len(subset),
            "mean_horizon_events": sum(int(row["horizon_events"]) for row in subset)
            / max(len(subset), 1),
            "mean_completed_events": sum(int(row["completed_events"]) for row in subset)
            / max(len(subset), 1),
            "all_events_formally_executable": _rate(subset, "all_events_formally_executable"),
            "all_events_exact": _rate(subset, "all_events_exact"),
            "all_successors_chemical_exact": _rate(subset, "all_successors_chemical_exact"),
            "endpoint_exact": _rate(subset, "endpoint_exact"),
        }
    (args.output / "suffix_evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--sample-reactions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizons", nargs="+", default=["1", "2", "3", "full"])
    parser.add_argument("--history-window", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=4096)
    args = parser.parse_args()
    if any(value != "full" and int(value) <= 0 for value in args.horizons):
        parser.error("horizons must be positive integers or full")
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
