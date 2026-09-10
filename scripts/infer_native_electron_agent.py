#!/usr/bin/env python3
"""Run original Qwen3 in a gold-free, multi-turn electron-tool loop."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from mechet.native_electron_agent import (
    NONTHINKING_PROMPT,
    SYSTEM_PROMPT,
    TOOLS,
    NativeElectronAgent,
)
from mechet.reaction_mapping import fragment_multiset_unmapped


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_tool_call(text: str) -> dict:
    visible = text.rsplit("</think>", 1)[-1]
    calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", visible, re.S)
    if len(calls) != 1:
        raise ValueError("Return exactly one complete native tool-call JSON object")
    value = json.loads(calls[0])
    if set(value) != {"name", "arguments"} or not isinstance(
        value["arguments"], dict
    ):
        raise ValueError("Tool call requires name and object-valued arguments")
    return value


def load_cases(path: Path, limit: int) -> tuple[list[dict], dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = rows[:limit]
    if not rows or any("id" not in row or "target_smiles" not in row for row in rows):
        raise ValueError("Cases require id and target_smiles")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate case IDs")
    # References are deliberately held outside actor_cases and enter only in scoring.
    references = {
        row["id"]: str(row.get("expected_precursor") or "") for row in rows
    }
    actor_cases = [
        {"id": row["id"], "target_smiles": row["target_smiles"]} for row in rows
    ]
    return actor_cases, references


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-episode-tokens", type=int)
    parser.add_argument("--max-context", type=int, default=40960)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.limit <= 0 or args.max_turns <= 0:
        raise ValueError("limit and max-turns must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    actor_cases, references = load_cases(args.cases, args.limit)
    per_turn = args.max_new_tokens or (8192 if args.thinking else 2048)
    per_episode = args.max_episode_tokens or (32768 if args.thinking else 16384)
    plan = {
        "condition": "original_Qwen_native_electron_tools_"
        + ("thinking" if args.thinking else "nonthinking"),
        "model": args.model,
        "model_adapter": None,
        "case_sha256": sha256(args.cases),
        "ids": [row["id"] for row in actor_cases],
        "input_contract": "unmapped_product_only",
        "gold_feedback": False,
        "max_turns": args.max_turns,
        "max_new_tokens_per_turn": per_turn,
        "max_generated_tokens_per_episode": per_episode,
        "max_context": args.max_context,
        "seed": args.seed,
        "thinking": args.thinking,
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Existing output has a different frozen plan")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from transformers.generation.streamers import BaseStreamer

    class ProgressStreamer(BaseStreamer):
        def __init__(self):
            self.prompt = True
            self.count = 0

        def put(self, value):
            if self.prompt:
                self.prompt = False
                return
            self.count += value.numel()
            if self.count % 128 == 0:
                print(f"[meteor-native-tokens] generated={self.count}", flush=True)

        def end(self):
            print(f"[meteor-native-tokens] completed={self.count}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    ).eval()
    print("[meteor-native] original model loaded; adapter=None", flush=True)

    prediction_path = args.output / "predictions.jsonl"
    done = (
        [json.loads(line) for line in prediction_path.read_text().splitlines()]
        if prediction_path.exists()
        else []
    )
    if len({row["id"] for row in done}) != len(done):
        raise ValueError("Duplicate prediction IDs")
    planned_ids = {row["id"] for row in actor_cases}
    if {row["id"] for row in done} - planned_ids:
        raise ValueError("Existing predictions contain IDs outside the frozen plan")

    for case_index, row in enumerate(actor_cases):
        if row["id"] in {item["id"] for item in done}:
            continue
        set_seed(args.seed + case_index)
        environment = NativeElectronAgent(row["target_smiles"])
        inspected = args.thinking
        generated_tokens = 0
        events = []
        started = time.time()
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT if args.thinking else NONTHINKING_PROMPT,
            },
            {
                "role": "user",
                "content": "Infer a precursor pathway from this product only. "
                "Initial structure:\n" + json.dumps(environment.inspect()),
            },
        ]
        stop = "turn_budget"
        for turn in range(args.max_turns):
            available_tools = TOOLS if inspected else TOOLS[:1]
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=available_tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.thinking,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            limit = min(
                per_turn,
                per_episode - generated_tokens,
                args.max_context - inputs.input_ids.shape[1],
            )
            if limit <= 0:
                stop = "token_or_context_budget"
                break
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=limit,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        streamer=ProgressStreamer(),
                        pad_token_id=tokenizer.pad_token_id
                        or tokenizer.eos_token_id,
                    )
                token_ids = generated[0, inputs.input_ids.shape[1] :]
                n_tokens = len(token_ids)
                text = tokenizer.decode(token_ids, skip_special_tokens=False)
                text = text.replace("<|im_end|>", "").replace(
                    "<|endoftext|>", ""
                )
                generated_tokens += n_tokens
                del generated, token_ids, inputs
                assistant = (
                    text
                    if not args.thinking or text.lstrip().startswith("<think>")
                    else "<think>\n" + text
                )
                try:
                    tool_call = parse_tool_call(text)
                    if not inspected and tool_call["name"] != "inspect":
                        raise ValueError("The first operation must be inspect")
                    feedback = environment.call(
                        tool_call["name"], tool_call["arguments"]
                    )
                    if tool_call["name"] == "inspect" and feedback["ok"]:
                        inspected = True
                    messages.append({"role": "assistant", "content": assistant})
                    messages.append(
                        {
                            "role": "tool",
                            "name": tool_call["name"],
                            "content": json.dumps(feedback),
                        }
                    )
                except (ValueError, KeyError, json.JSONDecodeError) as exc:
                    tool_call = None
                    feedback = {
                        "ok": False,
                        "error": str(exc),
                        "state_id": environment.state_id,
                    }
                    if n_tokens == limit:
                        stop = "generation_length_cap"
                        feedback["error"] = (
                            "Generation ended before a complete tool call; "
                            "this is not a chemical execution error"
                        )
                    else:
                        messages.append(
                            {"role": "assistant", "content": assistant}
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": "Protocol feedback: "
                                + json.dumps(feedback),
                            }
                        )
                event = {
                    "turn": turn,
                    "generated_text": text,
                    "tool_call": tool_call,
                    "feedback": feedback,
                    "tokens": n_tokens,
                }
                events.append(event)
                with (args.output / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"id": row["id"], **event}) + "\n")
                print(
                    f"[meteor-native] id={row['id']} turn={turn + 1} "
                    f"tokens={n_tokens} tool="
                    f"{tool_call['name'] if tool_call else 'parse_error'} "
                    f"ok={feedback['ok']} steps={len(environment.trace.transitions)}",
                    flush=True,
                )
                if environment.finished:
                    stop = "finish"
                    break
                if stop == "generation_length_cap":
                    break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                stop = "CUDA_OOM"
                break
        record = {
            **row,
            "submitted": environment.finished,
            "precursor_smiles": environment.precursor_smiles,
            "accepted_steps": len(environment.trace.transitions),
            "stop": stop,
            "generated_tokens": generated_tokens,
            "turns": len(events),
            "tool_errors": sum(not event["feedback"]["ok"] for event in events),
            "seconds": time.time() - started,
            "messages": messages,
        }
        with prediction_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        done.append(record)

    scored = []
    for record in done:
        row = {key: value for key, value in record.items() if key != "messages"}
        expected = references[record["id"]]
        for name, neutralize in (("full_exact", False), ("neutralized_exact", True)):
            row[name] = bool(
                expected
                and record["submitted"]
                and fragment_multiset_unmapped(
                    record["precursor_smiles"], neutralize=neutralize
                )
                == fragment_multiset_unmapped(expected, neutralize=neutralize)
            )
        scored.append(row)
    report = {
        "n": len(actor_cases),
        "completed": len(scored),
        "submitted": sum(row["submitted"] for row in scored),
        "full_exact": sum(row["full_exact"] for row in scored),
        "neutralized_exact": sum(row["neutralized_exact"] for row in scored),
        "stop_reasons": dict(Counter(row["stop"] for row in scored)),
        "protocol": "development pilot; complete-mixture endpoint metrics",
        "rows": scored,
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
