#!/usr/bin/env python3
"""Score grounded electron-flow event options with an unadapted Qwen checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.grounded_event_smoke import render_grounded_event_prompt


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def coordinates() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world, local_rank


def render_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def score_labels(model, tokenizer, prompt: str, labels: list[str], device) -> dict[str, float]:
    """Return conditional log probability for each randomized option label."""
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ][0]
    label_ids = {
        label: tokenizer(label, add_special_tokens=False)["input_ids"] for label in labels
    }
    scores: dict[str, float] = {}

    if all(len(ids) == 1 for ids in label_ids.values()):
        ids = prompt_ids.unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(input_ids=ids).logits[0, -1]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        for label, ids_for_label in label_ids.items():
            scores[label] = float(log_probs[int(ids_for_label[0])].item())
        return scores

    for label, suffix in label_ids.items():
        full = torch.tensor(
            list(prompt_ids.tolist()) + list(suffix),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        with torch.inference_mode():
            logits = model(input_ids=full).logits[0]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
        start = len(prompt_ids) - 1
        total = 0.0
        for offset, token_id in enumerate(suffix):
            total += float(log_probs[start + offset, int(token_id)].item())
        scores[label] = total
    return scores


def run(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank, world, local_rank = coordinates()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    tasks = read_jsonl(args.data)
    selected = [task for index, task in enumerate(tasks) if index % world == rank]
    if args.limit_events:
        selected = selected[: args.limit_events]
    args.output.mkdir(parents=True, exist_ok=True)
    shard = args.output / f"predictions.shard-{rank:02d}-of-{world:02d}.jsonl"
    completed = {row["key"] for row in read_jsonl(shard)} if shard.exists() else set()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        and torch.cuda.get_device_capability(local_rank)[0] >= 8
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    ).eval()
    device = next(model.parameters()).device

    with shard.open("a", encoding="utf-8") as sink:
        for ordinal, task in enumerate(selected, 1):
            if task["key"] in completed:
                continue
            started = time.time()
            system, user = render_grounded_event_prompt(task)
            prompt = render_chat(tokenizer, system, user)
            labels = [str(item["label"]) for item in task["options"]]
            scores = score_labels(model, tokenizer, prompt, labels, device)
            ranked = sorted(labels, key=lambda label: scores[label], reverse=True)
            gold = str(task["correct_label"])
            rank_index = ranked.index(gold) + 1
            record = {
                "key": task["key"],
                "id": task["id"],
                "event_index": task["event_index"],
                "event_depth": task["event_depth"],
                "trajectory_events": task["trajectory_events"],
                "candidate_count": task["candidate_count"],
                "correct_label": gold,
                "ranked_labels": ranked,
                "gold_rank": rank_index,
                "scores": scores,
                "top1": rank_index == 1,
                "recall_at_2": rank_index <= min(2, len(ranked)),
                "recall_at_4": rank_index <= min(4, len(ranked)),
                "recall_at_8": rank_index <= min(8, len(ranked)),
                "random_top1": 1.0 / len(ranked),
                "seconds": time.time() - started,
                "claim_boundary": task.get("claim_boundary"),
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()
            print(
                f"[grounded-event] rank={rank} {ordinal}/{len(selected)} "
                f"{task['key']} gold_rank={rank_index}/{len(ranked)}",
                flush=True,
            )
    return 0


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "top1": sum(bool(row["top1"]) for row in rows) / len(rows),
        "recall_at_2": sum(bool(row["recall_at_2"]) for row in rows) / len(rows),
        "recall_at_4": sum(bool(row["recall_at_4"]) for row in rows) / len(rows),
        "recall_at_8": sum(bool(row["recall_at_8"]) for row in rows) / len(rows),
        "mean_gold_rank": sum(float(row["gold_rank"]) for row in rows) / len(rows),
        "mean_random_top1": sum(float(row["random_top1"]) for row in rows) / len(rows),
    }


def aggregate(args: argparse.Namespace) -> int:
    tasks = read_jsonl(args.data)
    expected = {task["key"] for task in tasks}
    rows: list[dict] = []
    for path in sorted(args.output.glob("predictions.shard-*-of-*.jsonl")):
        rows.extend(read_jsonl(path))
    observed = [row["key"] for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate grounded-event predictions")
    missing = sorted(expected - set(observed))

    by_depth = {
        str(depth): _summary([row for row in rows if row["event_depth"] == depth])
        for depth in sorted({int(row["event_depth"]) for row in rows})
    }
    by_candidates = {
        str(count): _summary([row for row in rows if row["candidate_count"] == count])
        for count in sorted({int(row["candidate_count"]) for row in rows})
    }
    at_least_four = [row for row in rows if int(row["candidate_count"]) >= 4]
    report = {
        "artifact_type": "pure_qwen_grounded_event_ranking_v1",
        "model": str(args.model),
        "adapter": None,
        "condition": "Qwen3-8B_zero_shot_F-oracle_grounded_event_ranking",
        "planned_events": len(expected),
        "completed_events": len(rows),
        "missing_events": len(missing),
        "complete": not missing,
        "overall": _summary(rows),
        "main_at_least_4_candidates": _summary(at_least_four),
        "by_event_depth": by_depth,
        "by_candidate_count": by_candidates,
        "claim_boundary": (
            "Validation-only pure-Qwen capability diagnostic on gold executor states. "
            "No adapter is loaded; candidates include a gold-derived reference event "
            "and formally executable hard negatives. This is not endpoint accuracy."
        ),
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)
    return 0 if report["complete"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit-events", type=int, default=0)
    args = parser.parse_args()
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
