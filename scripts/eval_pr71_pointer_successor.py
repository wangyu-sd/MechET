#!/usr/bin/env python3
"""Executor-grounded local successor comparator for the frozen PR71 8B head.

This loads the existing Stage-II adapter and conditional pointer head without
training. It uses the same current-state executor replay and count-selection
rules as PR81; neither model receives reference move counts at inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.electron_pointer import candidate_keys, locate_marker_tokens
from mechet.electron_pointer_model import CoupledPointerHead, pairs_for_observation
from mechet.natural_language_electron_flow import execute_event_arguments
from mechet.structural_overlap import canonical_unmapped_smiles
from mechet.system_one_replay import execute_pair_indices, reconstruct_mapped_state
from scripts.eval_system_one_successor import select_flow_counts
from scripts.train_system_one_electron_flow import file_sha256, load, recall_at_k, verify_source


MODES = ("fixed1", "fixed2", "validity_backoff_2_to_1", "oracle_count")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("valid", "test", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def summarize(bucket: Counter) -> dict:
    n = bucket["n"]
    return {
        "n": n,
        "execute_ok": bucket["execute_ok"],
        "execute_rate": bucket["execute_ok"] / n,
        "successor_exact": bucket["successor_exact"],
        "successor_exact_rate": bucket["successor_exact"] / n,
        "codes": {key[5:]: count for key, count in bucket.items() if key.startswith("code_")},
    }


def coupled_target_indices(example) -> list[int]:
    """Preserve the original source-to-sink pairing in multi-flow events."""
    n = len(example.atom_names)
    source = candidate_keys(n, example.bonds, source=True)
    sink = candidate_keys(n, example.bonds, source=False)
    source_index = {key: index for index, key in enumerate(source)}
    sink_index = {key: index for index, key in enumerate(sink)}
    return sorted({
        source_index[src] * len(sink) + sink_index[dst]
        for src, dst in zip(example.source_targets, example.sink_targets, strict=True)
    })


def evaluate_split(split, source, examples, policy, head, tokenizer, device,
                   checkpoint_info, output, log_every, published_valid):
    import torch

    metrics = {mode: Counter() for mode in MODES}
    strata = {mode: {} for mode in MODES}
    paired = Counter()
    rows = []
    token_count = 0
    forward_elapsed = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, example in enumerate(examples, 1):
            mapped = reconstruct_mapped_state(example)
            gold_args = example.assistant_message["tool_calls"][0]["function"]["arguments"]
            gold = execute_event_arguments(mapped, gold_args)
            if not gold.get("ok"):
                raise ValueError(f"{example.row_id}: reference event failed executor replay")
            gold_successor = canonical_unmapped_smiles(gold["state_smiles"])

            prefix = render_qwen_sft_tool_prefix(tokenizer, example.messages, tools=example.tools)
            encoded = tokenizer(prefix, add_special_tokens=False)
            marker_positions = locate_marker_tokens(tokenizer, prefix, example)
            if len(encoded["input_ids"]) > 4096:
                raise ValueError(f"{example.row_id}: PR71 input exceeds its 4096-token cap")
            token_count += len(encoded["input_ids"])
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            forward_start = time.perf_counter()
            states = policy(input_ids=input_ids, output_hidden_states=True,
                            use_cache=False).hidden_states[-1][0]
            source_pairs, sink_pairs = pairs_for_observation(example, device)
            pair_logits = head(states[-1], states[marker_positions],
                               source_pairs, sink_pairs)[2]
            ranked = pair_logits.flatten().topk(min(8, pair_logits.numel())).indices.tolist()
            forward_elapsed += time.perf_counter() - forward_start

            targets = coupled_target_indices(example)
            paired.update(recall_at_k(ranked, targets))
            gold_pairs = execute_pair_indices(mapped, example, targets)
            if gold_pairs != gold:
                raise ValueError(f"{example.row_id}: pointer target pair replay differs from reference")
            flow_count = len(targets)
            executed = {
                count: execute_pair_indices(mapped, example, ranked[:count])
                for count in {1, 2, flow_count}
            }
            selected_counts = select_flow_counts(flow_count, executed)
            row = {
                "id": example.row_id,
                "gold_flow_count": flow_count,
                "pair_targets": targets,
                "ranked_top8": ranked,
                "gold_successor": gold_successor,
                "policies": {},
            }
            for mode in MODES:
                chosen = selected_counts[mode]
                result = executed[chosen]
                executable = bool(result.get("ok"))
                successor = (canonical_unmapped_smiles(result["state_smiles"])
                             if executable else None)
                exact = executable and successor == gold_successor
                for bucket in (metrics[mode], strata[mode].setdefault(str(flow_count), Counter())):
                    bucket["n"] += 1
                    bucket["execute_ok"] += int(executable)
                    bucket["successor_exact"] += int(exact)
                    bucket[f"code_{result.get('code', 'UNKNOWN')}"] += 1
                row["policies"][mode] = {
                    "selected_flow_count": chosen,
                    "pair_indices": ranked[:chosen],
                    "execute_ok": executable,
                    "code": result.get("code"),
                    "successor": successor,
                    "successor_exact": bool(exact),
                }
            rows.append(row)
            if index % log_every == 0 or index == len(examples):
                print(json.dumps({"phase": "pr71_successor_eval", "split": split,
                                  "events": index, "total": len(examples),
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)

    report = {
        "artifact_type": "pr71_conditional_pointer_local_successor_evaluation",
        "scope": "reference_current_state_not_product_start",
        "split": split,
        "source": source,
        "checkpoint": checkpoint_info,
        "evaluated_events": len(examples),
        "gold_replay_ok": len(examples),
        "pair_recall": {key: value / len(examples) for key, value in paired.items()},
        "policies": {
            mode: {
                "overall": summarize(metrics[mode]),
                "by_gold_flow_count": {key: summarize(value) for key, value in strata[mode].items()},
                "gold_independent": mode != "oracle_count",
                "validation_post_hoc": mode == "validity_backoff_2_to_1",
            }
            for mode in MODES
        },
        "model_forward_elapsed_s": forward_elapsed,
        "input_tokens": token_count,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_s": time.perf_counter() - started,
        "promotion_note": "PR71 comparator only; local reference-state successor, not product-start endpoint.",
    }
    if split == "valid" and len(examples) == source["event_decisions"]:
        deltas = {
            key: report["pair_recall"][key] - published_valid[key]
            for key in ("pair_r1", "pair_r4", "pair_r8",
                        "pair_all_r1", "pair_all_r4", "pair_all_r8")
        }
        if any(abs(value) > 2 / len(examples) for value in deltas.values()):
            raise ValueError(f"PR71 pair metrics differ materially from the frozen validation: {deltas}")
        report["published_valid_pair_metric_deltas"] = deltas
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "cases.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"phase": "pr71_successor_complete", "split": split,
                      "events": len(examples), "report": str(output / "report.json")}), flush=True)


def main() -> int:
    args = arguments()
    if args.limit < 0 or args.log_every < 1 or args.output.exists():
        raise ValueError("invalid limit/log interval or existing output directory")
    manifest_path = args.checkpoint / "pointer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("artifact_type") != "pr71_p0_stage2_frozen_policy_source_sink_pointer":
        raise ValueError("not the frozen PR71 conditional pointer baseline")
    if manifest.get("conditional_pair") is not True or manifest.get("lm_loss_weight") != 0.0:
        raise ValueError("not the source-conditioned paired PR71 head")
    adapter_path = args.adapter / "adapter_model.safetensors"
    head_path = args.checkpoint / "pointer_head_epoch1.pt"
    adapter_sha = file_sha256(adapter_path)
    head_sha = file_sha256(head_path)
    if adapter_sha != manifest["adapter_sha256"] or args.adapter.resolve() != Path(manifest["adapter"]).resolve():
        raise ValueError("PR71 adapter differs from the pointer checkpoint manifest")
    if args.data_dir.resolve() != Path(manifest["data_dir"]).resolve():
        raise ValueError("PR71 data directory differs from pointer checkpoint lineage")
    train_source = verify_source(args.data_dir / "train.jsonl")
    valid_source = verify_source(args.data_dir / "valid.jsonl")
    if train_source["sha256"] != manifest["train_sha256"] or valid_source["sha256"] != manifest["valid_sha256"]:
        raise ValueError("PR71 checkpoint source hashes do not match frozen inputs")
    splits = ("valid", "test") if args.split == "all" else (args.split,)
    selected = {}
    for split in splits:
        path = args.data_dir / f"{split}.jsonl"
        source = verify_source(path)
        if source["declared_sha256"] != source["sha256"] or source["artifact_id"] != valid_source["artifact_id"]:
            raise ValueError(f"{split}: data does not match the frozen pointer lineage")
        examples, counts = load(path)
        if counts["input_rows"] != source["decision_rows"] or counts["flow_events"] != source["event_decisions"]:
            raise ValueError(f"{split}: decision or event denominator mismatch")
        selected[split] = (source, examples[:args.limit] if args.limit else examples)
    checkpoint_info = {
        "pointer_manifest": str(manifest_path.resolve()),
        "adapter_sha256": adapter_sha,
        "pointer_head_sha256": head_sha,
        "base_model": manifest["base_model"],
        "base_revision": manifest["base_revision"],
    }
    published_valid = json.loads((args.checkpoint / "validation_epoch1.json").read_text())
    if published_valid.get("n") != valid_source["event_decisions"]:
        raise ValueError("published PR71 validation denominator mismatch")
    print(json.dumps({"phase": "pr71_audit", "checkpoint": checkpoint_info,
                      "splits": {key: len(value[1]) for key, value in selected.items()}}), flush=True)
    if args.audit_only:
        for split, (_, examples) in selected.items():
            for example in examples:
                mapped = reconstruct_mapped_state(example)
                gold_args = example.assistant_message["tool_calls"][0]["function"]["arguments"]
                gold = execute_event_arguments(mapped, gold_args)
                paired = execute_pair_indices(mapped, example, coupled_target_indices(example))
                if not gold.get("ok") or paired != gold:
                    raise ValueError(f"{example.row_id}: paired target replay differs from reference")
            print(json.dumps({"phase": "pr71_gold_replay_audit", "split": split,
                              "events": len(examples)}), flush=True)
        return 0

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("PR71 comparator requires a BF16 CUDA GPU")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(manifest["base_model"],
                                               revision=manifest["base_revision"],
                                               trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("PR71 pointer requires a fast tokenizer")
    base = AutoModelForCausalLM.from_pretrained(
        manifest["base_model"], revision=manifest["base_revision"],
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    policy = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    head = CoupledPointerHead(int(policy.config.hidden_size)).to(device)
    head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True), strict=True)
    head.eval()
    torch.cuda.reset_peak_memory_stats(device)
    for split in splits:
        source, examples = selected[split]
        evaluate_split(split, source, examples, policy, head, tokenizer, device,
                       checkpoint_info, args.output / split, args.log_every, published_valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
