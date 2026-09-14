#!/usr/bin/env python3
"""Evaluate ranked Retro-MTGR candidates on the complete raw test split.

Every raw test case remains in the denominator. Reactions rejected by native
preprocessing, and supported reactions whose labels are absent from the
train-only vocabulary, receive explicit failed candidate slots.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
from rdkit import RDLogger
from torch import nn

from mechet_retro_mtgr import (
    apply_retro_edit,
    graph_tensors,
    read_jsonl,
    stable_id,
    structural_key,
    write_jsonl,
)
from retro_mtgr_adapter_model import RetroMTGRAdapter


KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--raw-test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument(
        "--mechet-evaluator",
        type=Path,
        default=Path("/home/estar/pxy/mechet/MechET/scripts/evaluate_endpoint_candidates.py"),
    )
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--center-budget", type=int, default=10)
    parser.add_argument("--label-budget", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def failed_candidates(count: int, reason: str) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "prediction": "",
            "precursors": "",
            "score": None,
            "generation_error": reason,
        }
        for rank in range(1, count + 1)
    ]


def ranked_candidates(
    model: RetroMTGRAdapter,
    graph: dict[str, Any],
    product: str,
    labels: list[dict[str, Any]],
    candidate_count: int,
    center_budget: int,
    label_budget: int,
) -> list[dict[str, Any]]:
    hidden, pooled = model.encode(graph)
    center_logits = model.score_bonds(graph, hidden, pooled)
    if center_logits.numel() == 0:
        return []
    center_scores = torch.log_softmax(center_logits, dim=0)
    center_order = torch.topk(
        center_scores, min(center_budget, center_scores.numel())
    ).indices
    score_blocks: list[torch.Tensor] = []
    combinations: list[tuple[int, int, int]] = []
    for center_tensor in center_order:
        center_index = int(center_tensor)
        left_logits, right_logits = model.score_leaving_groups(
            graph, hidden, pooled, center_index
        )
        left_scores = torch.log_softmax(left_logits, dim=0)
        right_scores = torch.log_softmax(right_logits, dim=0)
        left_order = torch.topk(
            left_scores, min(label_budget, left_scores.numel())
        ).indices
        right_order = torch.topk(
            right_scores, min(label_budget, right_scores.numel())
        ).indices
        block = (
            center_scores[center_index]
            + left_scores[left_order][:, None]
            + right_scores[right_order][None, :]
        ).reshape(-1)
        score_blocks.append(block)
        combinations.extend(
            (center_index, int(left_index), int(right_index))
            for left_index in left_order
            for right_index in right_order
        )
    scores = torch.cat(score_blocks)
    order = torch.argsort(scores, descending=True).cpu().tolist()
    score_values = scores.detach().cpu().tolist()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for combination_index in order:
        center_index, left_index, right_index = combinations[combination_index]
        bond = graph["bonds"][center_index]
        prediction = apply_retro_edit(
            product, bond, labels[left_index], labels[right_index]
        )
        key = structural_key(prediction)
        if key is None or key in seen:
            continue
        seen.add(key)
        candidates.append({
            "rank": len(candidates) + 1,
            "prediction": prediction,
            "precursors": prediction,
            "score": score_values[combination_index],
            "center_atom_indices": list(bond),
            "left_label_id": left_index,
            "right_label_id": right_index,
            "generation_error": "",
        })
        if len(candidates) == candidate_count:
            break
    return candidates


def main() -> int:
    args = parse_args()
    if min(args.candidate_count, args.center_budget, args.label_budget, args.threads) < 1:
        raise ValueError("candidate, center, label, and thread counts must be positive")
    if args.candidate_count < max(KS):
        raise ValueError(f"--candidate-count must be at least {max(KS)}")
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format") != "mechet-retro-mtgr-adapter-v1":
        raise ValueError("unsupported checkpoint format")
    model = RetroMTGRAdapter(**checkpoint["model_configuration"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    labels = checkpoint["labels"]
    label_to_id = {item["key"]: int(item["id"]) for item in labels}

    targets = read_jsonl(args.data_dir / "test_targets.jsonl")
    source_rows = read_jsonl(args.raw_test)
    if len(targets) != len(source_rows):
        raise ValueError(f"target/source length mismatch: {len(targets)} != {len(source_rows)}")

    predictions: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter(total=len(source_rows))
    component_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    started = time.monotonic()

    with torch.no_grad():
        for source_index, (source_row, target) in enumerate(zip(source_rows, targets)):
            identifier = stable_id(source_row)
            if identifier != target.get("stable_id"):
                raise ValueError(f"stable ID mismatch at source index {source_index}")
            reason = ""
            generated: list[dict[str, Any]] = []
            component: dict[str, Any] = {}
            if target.get("status") != "supported":
                reason = f"preprocessing:{target.get('reason', 'unsupported')}"
                counts["preprocessing_unsupported"] += 1
            elif (
                target["left_label_key"] not in label_to_id
                or target["right_label_key"] not in label_to_id
            ):
                reason = "label_not_in_train_vocabulary"
                counts["unseen_label"] += 1
            else:
                product = target["product_mapped"]
                graph = graph_tensors(product, device)
                gold_center = tuple(int(value) for value in target["center_atom_indices"])
                try:
                    gold_center_index = graph["bonds"].index(gold_center)
                except ValueError:
                    reason = "center_not_found_in_product_graph"
                    counts["missing_center"] += 1
                else:
                    generated = ranked_candidates(
                        model,
                        graph,
                        product,
                        labels,
                        args.candidate_count,
                        args.center_budget,
                        args.label_budget,
                    )
                    hidden, pooled = model.encode(graph)
                    center_logits = model.score_bonds(graph, hidden, pooled)
                    left_logits, right_logits = model.score_leaving_groups(
                        graph, hidden, pooled, gold_center_index
                    )
                    gold_left = label_to_id[target["left_label_key"]]
                    gold_right = label_to_id[target["right_label_key"]]
                    component_loss += float(
                        loss_fn(
                            center_logits.unsqueeze(0),
                            torch.tensor([gold_center_index], device=device),
                        )
                        + loss_fn(
                            left_logits.unsqueeze(0),
                            torch.tensor([gold_left], device=device),
                        )
                        + loss_fn(
                            right_logits.unsqueeze(0),
                            torch.tensor([gold_right], device=device),
                        )
                    )
                    center_correct = int(center_logits.argmax().item() == gold_center_index)
                    left_correct = int(left_logits.argmax().item() == gold_left)
                    right_correct = int(right_logits.argmax().item() == gold_right)
                    counts["center_top1_correct"] += center_correct
                    counts["left_label_top1_correct"] += left_correct
                    counts["right_label_top1_correct"] += right_correct
                    counts["joint_target_top1_correct"] += (
                        center_correct and left_correct and right_correct
                    )
                    counts["evaluable"] += 1
                    component = {
                        "gold_center_atom_indices": list(gold_center),
                        "center_top1_correct": bool(center_correct),
                        "left_label_top1_correct": bool(left_correct),
                        "right_label_top1_correct": bool(right_correct),
                    }
            if reason:
                generated = failed_candidates(args.candidate_count, reason)
            elif len(generated) < args.candidate_count:
                generated.extend(failed_candidates(
                    args.candidate_count - len(generated),
                    "candidate_space_exhausted",
                ))
                for rank, candidate in enumerate(generated, start=1):
                    candidate["rank"] = rank

            expected_key = structural_key(source_row.get("precursor_unmapped", ""))
            hits = [
                structural_key(candidate.get("prediction", "")) == expected_key
                and expected_key is not None
                for candidate in generated
            ]
            for k in KS:
                counts[f"success_at_{k}"] += int(any(hits[:k]))
            counts["valid_candidates"] += sum(
                structural_key(candidate.get("prediction", "")) is not None
                for candidate in generated
            )
            predictions.append({
                "id": identifier,
                "stable_id": identifier,
                "source_index": source_index,
                "product": source_row.get("product_unmapped", ""),
                "reference_precursors": source_row.get("precursor_unmapped", ""),
                "candidates": generated,
                "evaluation_status": "unsupported" if reason else "ok",
                "evaluation_failure_reason": reason,
                "source_method": "Retro-MTGR",
                "checkpoint": str(args.checkpoint.resolve()),
                "candidate_semantics": "joint_ranked_center_and_endpoint_leaving_group",
                "candidate_budget": args.candidate_count,
                "gold_used_for_candidate_ranking": False,
            })
            evaluation_rows.append({
                "id": identifier,
                "stable_id": identifier,
                "source_index": source_index,
                "preprocessing_status": target.get("status"),
                "failure_reason": reason,
                **component,
                **{f"success_at_{k}": bool(any(hits[:k])) for k in KS},
            })
            if args.progress_every and (source_index + 1) % args.progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"[test] {source_index + 1}/{len(source_rows)} cases, "
                    f"{(source_index + 1) / elapsed:.1f} cases/s",
                    flush=True,
                )

    total = max(counts["total"], 1)
    evaluable = max(counts["evaluable"], 1)
    total_candidates = total * args.candidate_count
    report = {
        "artifact_type": "retro_mtgr_full_raw_test_endpoint_evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_metrics": checkpoint.get("metrics", {}),
        "raw_test": str(args.raw_test.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "device": str(device),
        "candidate_count": args.candidate_count,
        "candidate_semantics": "joint ranked center and endpoint leaving-group combinations",
        "unsupported_rows_counted_as_incorrect": True,
        "unseen_train_vocabulary_labels_counted_as_incorrect": True,
        "gold_used_for_candidate_ranking": False,
        "counts": dict(counts),
        "overall_metrics": {
            **{f"structural_success_at_{k}": counts[f"success_at_{k}"] / total for k in KS},
            "candidate_validity_rate": counts["valid_candidates"] / total_candidates,
            "center_top1_accuracy": counts["center_top1_correct"] / total,
            "left_label_top1_accuracy": counts["left_label_top1_correct"] / total,
            "right_label_top1_accuracy": counts["right_label_top1_correct"] / total,
            "joint_target_top1_accuracy": counts["joint_target_top1_correct"] / total,
        },
        "evaluable_component_metrics": {
            "center_top1_accuracy": counts["center_top1_correct"] / evaluable,
            "left_label_top1_accuracy": counts["left_label_top1_correct"] / evaluable,
            "right_label_top1_accuracy": counts["right_label_top1_correct"] / evaluable,
            "joint_target_top1_accuracy": counts["joint_target_top1_correct"] / evaluable,
            "loss": component_loss / evaluable,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output = args.predictions_output or args.output.with_suffix(".predictions.jsonl")
    write_jsonl(predictions_output, predictions)
    write_jsonl(args.output.with_suffix(".rows.jsonl"), evaluation_rows)
    report["predictions"] = str(predictions_output.resolve())
    report["evaluation_rows"] = str(args.output.with_suffix(".rows.jsonl").resolve())
    mechet_output = args.output.with_name(args.output.stem + ".mechet.json")
    subprocess.run([
        sys.executable,
        "-u",
        str(args.mechet_evaluator),
        "--reference",
        str(args.raw_test),
        "--predictions",
        str(predictions_output),
        "--output",
        str(mechet_output),
        "--expected-rows",
        str(len(source_rows)),
        "--expected-candidates",
        str(args.candidate_count),
    ], check=True)
    report["mechet_endpoint_evaluation"] = str(mechet_output.resolve())
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
