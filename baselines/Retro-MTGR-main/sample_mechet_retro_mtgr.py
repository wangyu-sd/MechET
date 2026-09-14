#!/usr/bin/env python3
"""Product-only ranked inference for the MechET Retro-MTGR adapter."""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import time
from typing import Any

import torch
from rdkit import RDLogger

from mechet_retro_mtgr import apply_retro_edit, graph_tensors, read_jsonl, stable_id, structural_key, write_jsonl
from retro_mtgr_adapter_model import RetroMTGRAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--center-budget", type=int, default=10)
    parser.add_argument("--label-budget", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> int:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("format") != "mechet-retro-mtgr-adapter-v1":
        raise ValueError("unsupported checkpoint format")
    model = RetroMTGRAdapter(**checkpoint["model_configuration"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    labels = checkpoint["labels"]
    rows_out: list[dict[str, Any]] = []

    with torch.no_grad():
        for source_index, row in enumerate(read_jsonl(args.input)):
            identifier = stable_id(row)
            product = row.get("product_mapped") or row.get("product_unmapped") or ""
            started = time.perf_counter()
            graph = graph_tensors(product, device)
            hidden, pooled = model.encode(graph)
            center_scores = torch.log_softmax(model.score_bonds(graph, hidden, pooled), dim=0)
            center_order = torch.argsort(center_scores, descending=True)[:args.center_budget]
            combinations: list[tuple[float, int, int, int]] = []
            for center_tensor in center_order:
                center_index = int(center_tensor)
                left_logits, right_logits = model.score_leaving_groups(
                    graph, hidden, pooled, center_index
                )
                left_scores = torch.log_softmax(left_logits, dim=0)
                right_scores = torch.log_softmax(right_logits, dim=0)
                left_order = torch.argsort(left_scores, descending=True)[:args.label_budget]
                right_order = torch.argsort(right_scores, descending=True)[:args.label_budget]
                for left_tensor in left_order:
                    for right_tensor in right_order:
                        left_index = int(left_tensor)
                        right_index = int(right_tensor)
                        score = float(
                            center_scores[center_index]
                            + left_scores[left_index]
                            + right_scores[right_index]
                        )
                        combinations.append((score, center_index, left_index, right_index))
            candidates: list[dict[str, Any]] = []
            seen: set[tuple[str, ...]] = set()
            for score, center_index, left_index, right_index in heapq.nlargest(
                len(combinations), combinations
            ):
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
                    "score": score,
                    "center_atom_indices": list(bond),
                    "left_label_id": left_index,
                    "right_label_id": right_index,
                })
                if len(candidates) == args.candidate_count:
                    break
            fallback = row.get("product_unmapped") or product
            while len(candidates) < args.candidate_count:
                candidates.append({
                    "rank": len(candidates) + 1,
                    "prediction": fallback,
                    "precursors": fallback,
                    "score": None,
                    "generation_error": "candidate_space_exhausted_fallback_product",
                })
            rows_out.append({
                "id": identifier,
                "stable_id": identifier,
                "source_index": source_index,
                "product": row.get("product_unmapped") or product,
                "reference_precursors": row.get("precursor_unmapped", ""),
                "candidates": candidates,
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
                "source_method": "Retro-MTGR",
                "checkpoint": str(args.checkpoint.resolve()),
                "candidate_semantics": "joint_ranked_center_and_endpoint_leaving_group",
                "candidate_budget": args.candidate_count,
                "gold_used_for_decoding": False,
            })
    write_jsonl(args.output, rows_out)
    print(json.dumps({
        "rows": len(rows_out),
        "candidates_per_row": args.candidate_count,
        "output": str(args.output.resolve()),
        "gold_used_for_decoding": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
