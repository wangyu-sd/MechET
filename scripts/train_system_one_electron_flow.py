#!/usr/bin/env python3
"""Train a prefill-only source/sink decision policy on Stage-II observations.

The assistant answer and reference successor are never model inputs.  The
causal backbone is used as an encoder, without an LM projection or generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.electron_pointer import (
    PointerExample,
    UnsupportedPointerEvent,
    candidate_keys,
    parse_pointer_example,
)
from mechet.electron_pointer_model import pairs_for_observation
from mechet.system_one_decision import (
    ElectronFlowDecisionHead,
    append_option_anchors,
    locate_option_anchor_tokens,
    multi_target_nll,
)


@dataclass(frozen=True)
class PreparedDecision:
    example: PointerExample
    input_ids: list[int]
    marker_indices: list[int]
    pair_targets: list[int]
    pair_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    status_path = path.parent / "ARTIFACT_STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if status and not status.get("training_allowed", False):
        raise ValueError(f"source artifact forbids training: {status_path}")
    actual = file_sha256(path)
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    split = path.stem
    declared = str((manifest.get("splits") or {}).get(split, {}).get("output_sha256") or "")
    if declared and declared != actual:
        raise ValueError(f"{split} file differs from its frozen manifest: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "declared_sha256": declared or None,
        "artifact_id": status.get("artifact_id") or manifest.get("artifact_type"),
        "reaction_denominator": (manifest.get("reaction_denominator") or {}).get(split),
        "decision_rows": (manifest.get("decision_rows") or {}).get(split),
        "event_decisions": (manifest.get("splits") or {}).get(split, {}).get("event_decisions"),
    }


def load(path: Path) -> tuple[list[PointerExample], dict[str, int]]:
    rows: list[PointerExample] = []
    counts = {"input_rows": 0, "flow_events": 0, "nonflow": 0, "unsupported": 0}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError(f"blank JSONL row in {path}")
            counts["input_rows"] += 1
            row = json.loads(line)
            try:
                example = parse_pointer_example(row)
            except UnsupportedPointerEvent:
                counts["unsupported"] += 1
                continue
            if example is None:
                counts["nonflow"] += 1
                continue
            rows.append(example)
            counts["flow_events"] += 1
    return rows, counts


def prepare(example: PointerExample, tokenizer: Any) -> PreparedDecision:
    messages = append_option_anchors(example.messages, example.atom_names)
    prefix = render_qwen_sft_tool_prefix(tokenizer, messages, tools=example.tools)
    encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
    markers = locate_option_anchor_tokens(
        tokenizer, prefix, example.atom_names, offsets=encoded["offset_mapping"]
    )
    source = candidate_keys(len(example.atom_names), example.bonds, source=True)
    sink = candidate_keys(len(example.atom_names), example.bonds, source=False)
    source_index = {key: index for index, key in enumerate(source)}
    sink_index = {key: index for index, key in enumerate(sink)}
    targets = sorted({
        source_index[src] * len(sink) + sink_index[dst]
        for src, dst in zip(example.source_targets, example.sink_targets, strict=True)
    })
    if not targets:
        raise ValueError(f"{example.row_id}: empty paired electron-flow target")
    return PreparedDecision(
        example=example,
        input_ids=list(encoded["input_ids"]),
        marker_indices=markers,
        pair_targets=targets,
        pair_count=len(source) * len(sink),
    )


def recall_at_k(ranked: list[int], targets: list[int]) -> dict[str, int]:
    gold = set(targets)
    return {
        f"pair_r{k}": int(bool(gold & set(ranked[:k])))
        for k in (1, 4, 8)
    } | {
        f"pair_all_r{k}": int(gold <= set(ranked[:k]))
        for k in (1, 4, 8)
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--pointer-dim", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.epochs < 1 or args.max_length < 1 or args.max_train_events < 0 or args.valid_limit < 0:
        raise ValueError("epochs/max-length must be positive; event limits cannot be negative")
    if any(args.output.glob("decision_head_epoch*.pt")):
        raise FileExistsError(f"existing System-One training output: {args.output}")
    from transformers import AutoTokenizer

    train_source = verify_source(args.train)
    valid_source = verify_source(args.valid)
    if args.train.resolve() == args.valid.resolve():
        raise ValueError("train and validation must be different source files")
    train, train_counts = load(args.train)
    valid, valid_counts = load(args.valid)
    if train_source["decision_rows"] is not None and train_counts["input_rows"] != train_source["decision_rows"]:
        raise ValueError("train decision-row denominator mismatch")
    if valid_source["decision_rows"] is not None and valid_counts["input_rows"] != valid_source["decision_rows"]:
        raise ValueError("valid decision-row denominator mismatch")
    if train_source["event_decisions"] is not None and train_counts["flow_events"] + train_counts["unsupported"] != train_source["event_decisions"]:
        raise ValueError("train event-decision denominator mismatch")
    if valid_source["event_decisions"] is not None and valid_counts["flow_events"] + valid_counts["unsupported"] != valid_source["event_decisions"]:
        raise ValueError("valid event-decision denominator mismatch")

    random.Random(args.seed).shuffle(train)
    if args.max_train_events:
        train = train[: args.max_train_events]
    if args.valid_limit:
        valid = valid[: args.valid_limit]
    if not train or not valid:
        raise ValueError("both splits need at least one supported electron event")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for option alignment")
    prepared_train = [prepare(example, tokenizer) for example in train]
    prepared_valid = [prepare(example, tokenizer) for example in valid]
    lengths = [len(item.input_ids) for item in (*prepared_train, *prepared_valid)]
    overlength = [
        item.example.row_id
        for item in (*prepared_train, *prepared_valid)
        if len(item.input_ids) > args.max_length
    ]
    audit = {
        "artifact_type": "system_one_phase0_preflight",
        "train_source": train_source,
        "valid_source": valid_source,
        "train_counts": train_counts,
        "valid_counts": valid_counts,
        "selected_train_events": len(prepared_train),
        "selected_valid_events": len(prepared_valid),
        "model": args.model,
        "model_revision": args.revision,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_observed_length": max(lengths),
        "mean_observed_length": sum(lengths) / len(lengths),
        "overlength_count": len(overlength),
        "overlength_example_ids": overlength[:10],
        "max_pair_options": max(item.pair_count for item in (*prepared_train, *prepared_valid)),
        "input_contract": "full_executor_state_plus_gold_independent_post_state_atom_anchors_v1",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"phase": "preflight", **audit}), flush=True)
    if overlength:
        raise ValueError(f"{len(overlength)} decision prefixes exceed --max-length")
    if args.audit_only:
        return 0

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    base = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    policy = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.FEATURE_EXTRACTION,
        ),
    )
    head = ElectronFlowDecisionHead(int(policy.config.hidden_size), args.pointer_dim).to(device)
    params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    params += list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def score(item: PreparedDecision, train_mode: bool) -> tuple[float, dict[str, int]]:
        input_ids = torch.tensor([item.input_ids], dtype=torch.long, device=device)
        output = policy(input_ids=input_ids, use_cache=False)
        states = output.last_hidden_state[0]
        source_pairs, sink_pairs = pairs_for_observation(item.example, device)
        decision = head(
            states[-1], states[item.marker_indices], source_pairs, sink_pairs
        )
        loss = multi_target_nll(decision.pair_logits, item.pair_targets)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        ranked = decision.pair_logits.detach().flatten().topk(
            min(8, decision.pair_logits.numel())
        ).indices.tolist()
        return float(loss.detach()), recall_at_k(ranked, item.pair_targets)

    for epoch in range(args.epochs):
        policy.train()
        head.train()
        running = 0.0
        started = time.perf_counter()
        for index, item in enumerate(prepared_train, 1):
            loss, _ = score(item, True)
            running += loss
            if index % 100 == 0 or index == len(prepared_train):
                denominator = 100 if index % 100 == 0 else index % 100
                print(json.dumps({
                    "phase": "train", "epoch": epoch + 1, "step": index,
                    "total": len(prepared_train), "loss": running / denominator,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)
                running = 0.0

        policy.eval()
        head.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = {f"pair_{kind}_r{k}": 0 for kind in ("any", "all") for k in (1, 4, 8)}
        by_flows: dict[str, dict[str, int]] = {}
        valid_tokens = 0
        loss_sum = 0.0
        valid_started = time.perf_counter()
        with torch.no_grad():
            for item in prepared_valid:
                loss, recalls = score(item, False)
                loss_sum += loss
                valid_tokens += len(item.input_ids)
                flow_count = str(len(item.example.source_targets))
                group = by_flows.setdefault(flow_count, {"n": 0, "pair_r1": 0,
                    "pair_r4": 0, "pair_r8": 0, "pair_all_r8": 0})
                group["n"] += 1
                for k in (1, 4, 8):
                    metrics[f"pair_any_r{k}"] += recalls[f"pair_r{k}"]
                    metrics[f"pair_all_r{k}"] += recalls[f"pair_all_r{k}"]
                    group[f"pair_r{k}"] += recalls[f"pair_r{k}"]
                group["pair_all_r8"] += recalls["pair_all_r8"]
        elapsed = time.perf_counter() - valid_started
        denominator = len(prepared_valid)
        report = {
            "epoch": epoch + 1,
            "valid_events": denominator,
            "train_events": len(prepared_train),
            "loss": loss_sum / denominator,
            **{name: count / denominator for name, count in metrics.items()},
            "by_coupled_flows": by_flows,
            "validation_elapsed_s": elapsed,
            "latency_s_per_decision": elapsed / denominator,
            "validation_input_tokens": valid_tokens,
            "input_tokens_per_s": valid_tokens / elapsed if elapsed else None,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "autoregressive_generation": False,
            "backbone": args.model,
            "base_revision": args.revision,
            "model_dtype": str(model_dtype),
            "train_sha256": train_source["sha256"],
            "valid_sha256": valid_source["sha256"],
            "executed_successor_agreement": None,
            "promotion_ready": False,
            "promotion_note": "Pair selection alone is not a complete executor action; successor agreement is not yet measured.",
        }
        (args.output / f"validation_epoch{epoch + 1}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        adapter_dir = args.output / f"adapter_epoch{epoch + 1}"
        policy.save_pretrained(adapter_dir)
        head_path = args.output / f"decision_head_epoch{epoch + 1}.pt"
        torch.save(head.state_dict(), head_path)
        (args.output / f"run_manifest_epoch{epoch + 1}.json").write_text(
            json.dumps({
                "artifact_type": "system_one_phase0_decision_policy",
                "train_source": train_source,
                "valid_source": valid_source,
                "selected_train_events": len(prepared_train),
                "selected_valid_events": len(prepared_valid),
                "model": args.model,
                "model_revision": args.revision,
                "input_contract": audit["input_contract"],
                "adapter_model_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
                "decision_head_sha256": file_sha256(head_path),
                "validation_report": f"validation_epoch{epoch + 1}.json",
            }, indent=2) + "\n"
        )
        print(json.dumps({"phase": "validation", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
