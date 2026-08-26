#!/usr/bin/env python3
"""Teacher-force sampled agent trajectories and rank them without gold labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from mechet.endpoints import structural_exact
from mechet.knowledge_ablation import read_jsonl


RANKER_NAME = "formal_trace_gate__assistant_mean_nll_v1"
DIRECT_RANKER_NAME = "assistant_mean_nll_v1"


def _assistant_mask(tokenizer: Any, input_ids: list[int], n_expected: int) -> list[bool]:
    """Mask Qwen assistant bodies, including serialized tool calls."""

    start = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if not start or not end:
        raise ValueError("chat boundary tokens could not be encoded")
    mask = [False] * len(input_ids)
    cursor = 0
    found = 0
    while cursor <= len(input_ids) - len(start):
        if input_ids[cursor : cursor + len(start)] != start:
            cursor += 1
            continue
        body_start = cursor + len(start)
        body_end = body_start
        while body_end <= len(input_ids) - len(end):
            if input_ids[body_end : body_end + len(end)] == end:
                break
            body_end += 1
        if body_end > len(input_ids) - len(end):
            raise ValueError("assistant message has no closing chat token")
        # Include the closing token: the policy also decided to terminate this turn.
        for index in range(body_start, body_end + len(end)):
            mask[index] = True
        found += 1
        cursor = body_end + len(end)
    if found != n_expected:
        raise ValueError(f"expected {n_expected} assistant spans, found {found}")
    return mask


def _rank_key(
    score: dict[str, Any], candidate: dict[str, Any], *, direct: bool = False
) -> tuple[Any, ...]:
    nll = score.get("assistant_mean_nll")
    if direct:
        return (
            int(nll is not None and score.get("score_status") == "ok"),
            -float(nll) if nll is not None else float("-inf"),
            -int(candidate.get("sample_index") or 0),
        )
    state = dict(candidate.get("rollout_state") or {})
    final = dict(state.get("final_result") or {})
    return (
        int(bool(final.get("formal_execute") or final.get("ok"))),
        int(bool(final.get("trace_bound"))),
        -float(nll) if nll is not None else float("-inf"),
        -int(state.get("failed_steps") or 0),
        -int(state.get("tool_calls") or 0),
        -int(candidate.get("sample_index") or 0),
    )


def score(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    if not torch.cuda.is_available():
        raise RuntimeError("candidate NLL scoring requires a CUDA device")
    compute_major, _ = torch.cuda.get_device_capability()
    model_dtype = torch.bfloat16 if compute_major >= 8 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        device_map="auto",
        # The Taiji H20 image does not ship the optional flash_attn package.
        # PyTorch SDPA selects its fused CUDA backend without that dependency.
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    device = next(model.parameters()).device

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open() as source, args.output.open("w", encoding="utf-8") as sink:
        for row_index, line in enumerate(source):
            if not line.strip() or row_index % args.shard_count != args.shard_index:
                continue
            row = json.loads(line)
            scores: list[dict[str, Any]] = []
            for candidate in row.get("candidates") or []:
                messages = candidate.get("messages") or []
                n_assistant = sum(message.get("role") == "assistant" for message in messages)
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tools=row.get("tools") or None,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                ids = encoded["input_ids"][0].tolist()
                item: dict[str, Any] = {
                    "sample_index": int(candidate.get("sample_index") or 0),
                    "sequence_tokens": len(ids),
                    "assistant_turns": n_assistant,
                    "score_status": "ok",
                }
                if len(ids) > args.max_length:
                    item.update(
                        {
                            "score_status": "sequence_too_long",
                            "assistant_tokens": 0,
                            "assistant_logprob_sum": None,
                            "assistant_mean_logprob": None,
                            "assistant_mean_nll": None,
                        }
                    )
                    scores.append(item)
                    continue
                mask = _assistant_mask(tokenizer, ids, n_assistant)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                with torch.inference_mode():
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                    targets = input_ids[:, 1:]
                    losses = F.cross_entropy(
                        logits[:, :-1, :].transpose(1, 2),
                        targets,
                        reduction="none",
                    )[0]
                shifted_mask = torch.tensor(mask[1:], dtype=torch.bool, device=device)
                selected = losses[shifted_mask]
                n_tokens = int(selected.numel())
                nll_sum = float(selected.sum().cpu()) if n_tokens else None
                mean_nll = nll_sum / n_tokens if n_tokens and nll_sum is not None else None
                item.update(
                    {
                        "assistant_tokens": n_tokens,
                        "assistant_logprob_sum": -nll_sum if nll_sum is not None else None,
                        "assistant_mean_logprob": -mean_nll if mean_nll is not None else None,
                        "assistant_mean_nll": mean_nll,
                    }
                )
                scores.append(item)
                del logits, losses, input_ids, attention_mask
            order = sorted(
                range(len(scores)),
                key=lambda index: _rank_key(
                    scores[index],
                    row["candidates"][index],
                    direct=str(row.get("prediction_mode") or "") == "direct",
                ),
                reverse=True,
            )
            ranker = (
                DIRECT_RANKER_NAME
                if str(row.get("prediction_mode") or "") == "direct"
                else RANKER_NAME
            )
            sink.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "artifact_type": "candidate_nll_ranking",
                        "ranker": ranker,
                        "ranked_candidate_indices": order,
                        "selected_candidate_index": order[0] if order else None,
                        "candidate_scores": scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
    return 0


def _neutral_key(smiles: str) -> tuple[str, ...] | None:
    uncharger = rdMolStandardize.Uncharger()
    parts: list[str] = []
    for value in str(smiles or "").split("."):
        if not value:
            continue
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol = uncharger.uncharge(mol)
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return tuple(sorted(parts))


def aggregate(args: argparse.Namespace) -> int:
    references = {row["id"]: row for row in read_jsonl(args.reference)}
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    rankings: dict[str, dict[str, Any]] = {}
    for path in sorted(args.ranking_dir.glob("nll_scores.shard-*.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in rankings:
                raise ValueError(f"duplicate ranking row: {row['id']}")
            rankings[row["id"]] = row
    if set(rankings) != set(references) or set(predictions) != set(references):
        raise ValueError(
            f"ID mismatch: references={len(references)} predictions={len(predictions)} rankings={len(rankings)}"
        )

    counts = Counter()
    status = Counter()
    selected_rows = []
    for identifier, reference in references.items():
        prediction = predictions[identifier]
        ranking = rankings[identifier]
        index = int(ranking["selected_candidate_index"])
        candidate = prediction["candidates"][index]
        final = (candidate.get("rollout_state") or {}).get("final_result") or {}
        predicted = str(final.get("structural_precursor") or "")
        expected = str(reference.get("structural_precursor") or "")
        strict = structural_exact(predicted, expected)
        neutral = bool(predicted and _neutral_key(predicted) == _neutral_key(expected))
        counts["strict_selected"] += strict
        counts["neutralized_selected"] += neutral
        for item in ranking["candidate_scores"]:
            status[item["score_status"]] += 1
        selected_rows.append(
            {
                "id": identifier,
                "selected_candidate_index": index,
                "strict_exact": strict,
                "neutralized_exact": neutral,
                "ranking": ranking,
            }
        )
    report = {
        "artifact_type": "candidate_nll_ranking_evaluation",
        "ranker": RANKER_NAME,
        "n_rows": len(references),
        "n_candidates": sum(status.values()),
        "score_status": dict(status),
        "strict_selected_hits": counts["strict_selected"],
        "strict_selected_accuracy": counts["strict_selected"] / len(references),
        "neutralized_selected_hits": counts["neutralized_selected"],
        "neutralized_selected_accuracy": counts["neutralized_selected"] / len(references),
        "selection_uses_ground_truth": False,
        "evaluation_uses_ground_truth": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    (args.output.parent / "nll_selected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected_rows),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    scorer = sub.add_parser("score")
    scorer.add_argument("--predictions", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.add_argument("--model", required=True)
    scorer.add_argument("--revision", required=True)
    scorer.add_argument("--adapter", type=Path, required=True)
    scorer.add_argument("--shard-count", type=int, required=True)
    scorer.add_argument("--shard-index", type=int, required=True)
    scorer.add_argument("--max-length", type=int, default=24576)
    scorer.set_defaults(func=score)
    aggregator = sub.add_parser("aggregate")
    aggregator.add_argument("--reference", type=Path, required=True)
    aggregator.add_argument("--predictions", type=Path, required=True)
    aggregator.add_argument("--ranking-dir", type=Path, required=True)
    aggregator.add_argument("--output", type=Path, required=True)
    aggregator.set_defaults(func=aggregate)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
