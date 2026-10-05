#!/usr/bin/env python3
"""Train a typed Jev-style electron-flow decision policy.

One shared chemical state is packed with two independent typed questions:
SOURCE and SINK. Explicit option spans are isolated by a block-causal mask and
read out at <decide> tokens. No assistant answer is generated.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mechet.electron_pointer import PointerExample, candidate_keys
from mechet.jev_style_decision import (
    CONTROL_TOKENS,
    JevEncoding,
    TypedElectronFlowHead,
    TypedQuestion,
    block_causal_option_mask,
    encode_typed_record,
    required_target_nll,
)
from scripts.train_system_one_electron_flow import (
    file_sha256,
    load,
    recall_at_k,
    verify_source,
)


def option_label(key: tuple[str, int, int]) -> str:
    kind, left, right = key
    if kind == "atom":
        return f"ATOM A{left + 1:02d}"
    return f"BOND A{left + 1:02d} A{right + 1:02d}"


def state_text(example: PointerExample) -> str:
    chunks = []
    for message in example.messages:
        if message.get("role") in {"system", "user"} and isinstance(message.get("content"), str):
            chunks.append(f"{message['role'].upper()}:\n{message['content']}")
    if not chunks:
        raise ValueError(f"{example.row_id}: no model-visible state text")
    return "\n\n".join(chunks)


@dataclass(frozen=True)
class PreparedTypedDecision:
    example: PointerExample
    encoding: JevEncoding
    pair_targets: tuple[int, ...]
    source_targets: tuple[int, ...]
    sink_targets: tuple[int, ...]
    source_count: int
    sink_count: int

    @property
    def pair_count(self) -> int:
        return self.source_count * self.sink_count


def prepare_typed(example: PointerExample, tokenizer) -> PreparedTypedDecision:
    source = candidate_keys(len(example.atom_names), example.bonds, source=True)
    sink = candidate_keys(len(example.atom_names), example.bonds, source=False)
    source_index = {key: index for index, key in enumerate(source)}
    sink_index = {key: index for index, key in enumerate(sink)}
    src_targets = tuple(sorted({source_index[key] for key in example.source_targets}))
    dst_targets = tuple(sorted({sink_index[key] for key in example.sink_targets}))
    pair_targets = tuple(sorted({
        source_index[src] * len(sink) + sink_index[dst]
        for src, dst in zip(example.source_targets, example.sink_targets, strict=True)
    }))
    encoding = encode_typed_record(
        tokenizer,
        state=state_text(example),
        questions=(
            TypedQuestion("SELECT ELECTRON SOURCE", tuple(option_label(key) for key in source)),
            TypedQuestion("SELECT ELECTRON SINK", tuple(option_label(key) for key in sink)),
        ),
        option_isolation=True,
    )
    return PreparedTypedDecision(
        example=example,
        encoding=encoding,
        pair_targets=pair_targets,
        source_targets=src_targets,
        sink_targets=dst_targets,
        source_count=len(source),
        sink_count=len(sink),
    )


def arguments():
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
    parser.add_argument("--marginal-weight", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--valid-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.epochs < 1 or args.max_length < 1 or args.marginal_weight < 0:
        raise ValueError("invalid training configuration")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"existing typed-v2 output: {args.output}")

    from transformers import AutoTokenizer

    train_source = verify_source(args.train)
    valid_source = verify_source(args.valid)
    train, train_counts = load(args.train)
    valid, valid_counts = load(args.valid)
    random.Random(args.seed).shuffle(train)
    if args.max_train_events:
        train = train[: args.max_train_events]
    if args.valid_limit:
        valid = valid[: args.valid_limit]
    if not train or not valid:
        raise ValueError("empty typed-v2 train/valid selection")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    )
    prepared_train = [prepare_typed(example, tokenizer) for example in train]
    prepared_valid = [prepare_typed(example, tokenizer) for example in valid]
    all_rows = prepared_train + prepared_valid
    lengths = [len(item.encoding.input_ids) for item in all_rows]
    overlength = [
        item.example.row_id for item in all_rows
        if len(item.encoding.input_ids) > args.max_length
    ]
    preflight = {
        "artifact_type": "system_one_jev_typed_v2_preflight",
        "train_source": train_source,
        "valid_source": valid_source,
        "train_counts": train_counts,
        "valid_counts": valid_counts,
        "selected_train_events": len(prepared_train),
        "selected_valid_events": len(prepared_valid),
        "model": args.model,
        "model_revision": args.revision,
        "seed": args.seed,
        "input_contract": "shared_state_two_typed_questions_option_isolated_block_causal_v2",
        "control_tokens": list(CONTROL_TOKENS),
        "max_length": args.max_length,
        "max_observed_length": max(lengths),
        "mean_observed_length": sum(lengths) / len(lengths),
        "overlength_count": len(overlength),
        "overlength_example_ids": overlength[:20],
        "max_source_options": max(item.source_count for item in all_rows),
        "max_sink_options": max(item.sink_count for item in all_rows),
        "max_pair_options": max(item.pair_count for item in all_rows),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps({"phase": "preflight", **preflight}), flush=True)
    if overlength:
        raise ValueError(f"{len(overlength)} typed records exceed --max-length")
    if args.audit_only:
        return 0

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModel

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    base = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
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
    head = TypedElectronFlowHead(int(policy.config.hidden_size), args.pointer_dim).to(device)
    params = [p for p in policy.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    def score(item: PreparedTypedDecision, train_mode: bool):
        enc = item.encoding
        ids = torch.tensor([enc.input_ids], dtype=torch.long, device=device)
        positions = torch.tensor([enc.position_ids], dtype=torch.long, device=device)
        mask = block_causal_option_mask(enc, device=device, dtype=dtype)
        states = policy(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
        ).last_hidden_state[0]
        output = head(
            states[enc.decide_indices[0]],
            states[list(enc.option_indices[0])],
            states[enc.decide_indices[1]],
            states[list(enc.option_indices[1])],
        )
        pair_loss = required_target_nll(output.pair_logits, item.pair_targets)
        source_loss = required_target_nll(output.source_logits, item.source_targets)
        sink_loss = required_target_nll(output.sink_logits, item.sink_targets)
        loss = pair_loss + args.marginal_weight * 0.5 * (source_loss + sink_loss)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        ranked = output.pair_logits.detach().flatten().topk(
            min(8, output.pair_logits.numel())
        ).indices.tolist()
        return float(loss.detach()), recall_at_k(ranked, list(item.pair_targets))

    for epoch in range(args.epochs):
        policy.train()
        head.train()
        started = time.perf_counter()
        running = 0.0
        for index, item in enumerate(prepared_train, 1):
            loss, _ = score(item, True)
            running += loss
            if index % 100 == 0 or index == len(prepared_train):
                denom = 100 if index % 100 == 0 else index % 100
                print(json.dumps({
                    "phase": "train", "architecture": "jev_typed_v2",
                    "epoch": epoch + 1, "step": index, "total": len(prepared_train),
                    "loss": running / max(denom, 1),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)
                running = 0.0

        policy.eval()
        head.eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = {f"pair_{kind}_r{k}": 0 for kind in ("any", "all") for k in (1, 4, 8)}
        loss_sum = 0.0
        token_count = 0
        valid_started = time.perf_counter()
        with torch.no_grad():
            for item in prepared_valid:
                loss, recalls = score(item, False)
                loss_sum += loss
                token_count += len(item.encoding.input_ids)
                for k in (1, 4, 8):
                    totals[f"pair_any_r{k}"] += recalls[f"pair_r{k}"]
                    totals[f"pair_all_r{k}"] += recalls[f"pair_all_r{k}"]
        elapsed = time.perf_counter() - valid_started
        n = len(prepared_valid)
        report = {
            "artifact_type": "system_one_jev_typed_v2_validation",
            "architecture": "shared_state_typed_questions_option_isolated_block_causal",
            "epoch": epoch + 1,
            "train_events": len(prepared_train),
            "valid_events": n,
            "loss": loss_sum / n,
            **{name: value / n for name, value in totals.items()},
            "validation_elapsed_s": elapsed,
            "latency_s_per_decision": elapsed / n,
            "validation_input_tokens": token_count,
            "input_tokens_per_s": token_count / elapsed if elapsed else None,
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "autoregressive_generation": False,
            "model": args.model,
            "model_revision": args.revision,
        }
        (args.output / f"validation_epoch{epoch + 1}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        adapter_dir = args.output / f"adapter_epoch{epoch + 1}"
        policy.save_pretrained(adapter_dir)
        head_path = args.output / f"typed_head_epoch{epoch + 1}.pt"
        torch.save(head.state_dict(), head_path)
        manifest = {
            "artifact_type": "system_one_jev_typed_v2_decision_policy",
            "architecture": report["architecture"],
            "input_contract": preflight["input_contract"],
            "train_source": train_source,
            "valid_source": valid_source,
            "selected_train_events": len(prepared_train),
            "selected_valid_events": len(prepared_valid),
            "model": args.model,
            "model_revision": args.revision,
            "pointer_dim": args.pointer_dim,
            "marginal_weight": args.marginal_weight,
            "adapter_model_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
            "decision_head_sha256": file_sha256(head_path),
            "validation_report": f"validation_epoch{epoch + 1}.json",
        }
        (args.output / f"run_manifest_epoch{epoch + 1}.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps({"phase": "validation", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
