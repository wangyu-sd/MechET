#!/usr/bin/env python3
"""PR71 source/sink pointer: frozen pilot, evaluation, or opt-in joint LoRA.

The joint mode adds assistant-only LM loss to source/sink auxiliary losses;
neither mode changes executor semantics or claims autonomous endpoint gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechet.assistant_masking import (
    encode_assistant_only_conversation, render_qwen_sft_tool_prefix,
)
from mechet.electron_pointer import (
    PointerExample,
    UnsupportedPointerEvent,
    candidate_keys,
    locate_marker_tokens,
    parse_pointer_example,
    target_indices,
)
from mechet.electron_pointer_model import (
    CoupledPointerHead, PointerHead, coupled_pair_recall_metrics,
    pair_recall_metrics, pairs_for_observation,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_split(directory: Path, split: str) -> list[PointerExample]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "validated_trace_view" or manifest.get("training_allowed") is not True:
        raise ValueError("dataset is not a validated trainable trace view")
    expected = {"train": 10152, "valid": 1319, "test": 1253}
    decisions = {"train": 32401, "valid": 4288, "test": 4006}
    if manifest.get("reaction_denominator") != expected or manifest.get("decision_rows") != decisions:
        raise ValueError("31k Stage-II denominator mismatch")
    if manifest.get("full_reaction_denominator") != {"train": 24959, "valid": 3120, "test": 3120}:
        raise ValueError("full endpoint denominator mismatch")
    if manifest.get("decision_contract") != "unified_inventory_compressed_history_tool_decision_v2":
        raise ValueError("not Stage-II v2 decision contract")
    path = directory / f"{split}.jsonl"
    if sha256(path) != manifest["splits"][split]["output_sha256"]:
        raise ValueError(f"{split} data hash mismatch")
    examples = []
    row_count = 0
    unsupported_events = 0
    with path.open() as stream:
        for line in stream:
            row_count += 1
            try:
                example = parse_pointer_example(json.loads(line))
            except UnsupportedPointerEvent:
                unsupported_events += 1
                continue
            if example is not None:
                target_indices(example, source=True)
                target_indices(example, source=False)
                examples.append(example)
    expected_events = int(manifest["splits"][split]["event_decisions"])
    if row_count != decisions[split] or len(examples) + unsupported_events != expected_events:
        raise ValueError(f"{split} event/decision count mismatch")
    print(json.dumps({
        "pointer_split": split,
        "decision_rows": row_count,
        "supported_pointer_events": len(examples),
        "unsupported_pointer_events": unsupported_events,
        "source_event_decisions": expected_events,
    }), file=sys.stderr, flush=True)
    return examples


def configure_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--valid-limit", type=int, default=0, help="0 means full validation")
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--eval-checkpoint", type=Path,
                   help="Evaluate an existing pointer head without optimizer updates")
    p.add_argument("--joint-policy", action="store_true",
                   help="train the Stage-II LoRA jointly with LM and pointer losses")
    p.add_argument("--lambda-lm", type=float, default=1.0)
    p.add_argument("--conditional-pair", action="store_true",
                   help="train a source-conditioned sink head and paired-move objective")
    return p.parse_args()


def main() -> int:
    args = configure_args()
    if args.epochs < 1 or args.max_length < 512:
        raise ValueError("invalid epochs or sequence budget")
    if args.lambda_lm <= 0 or (args.eval_checkpoint is not None and args.joint_policy):
        raise ValueError("joint LM weight must be positive; evaluation cannot update policy")
    train = load_split(args.data_dir, "train")
    valid = load_split(args.data_dir, "valid")
    train_ids = {e.row_id.split("::", 1)[0] for e in train}
    valid_ids = {e.row_id.split("::", 1)[0] for e in valid}
    if train_ids & valid_ids:
        raise ValueError("reaction leakage between train and validation")
    if args.valid_limit:
        valid = valid[: args.valid_limit]
    if args.audit_only:
        print(json.dumps({"gate": "passed", "train_events": len(train), "valid_events": len(valid),
                          "train_reactions": len(train_ids), "valid_reactions": len(valid_ids)}), flush=True)
        return 0

    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModelForCausalLM, AutoTokenizer

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if not (args.adapter / "adapter_config.json").is_file() or not (args.adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("completed Stage-II PEFT adapter is required")
    adapter_hash = sha256(args.adapter / "adapter_model.safetensors")
    adapter_manifest_path = args.adapter / "adapter_manifest.json"
    adapter_manifest = json.loads(adapter_manifest_path.read_text())
    if adapter_manifest.get("adapter_sha256") != adapter_hash:
        raise ValueError("adapter manifest/weight hash mismatch")
    if adapter_manifest.get("base_model_revision") != args.revision:
        raise ValueError("adapter/base revision mismatch")
    if args.eval_checkpoint is not None:
        pointer_manifest = json.loads((args.eval_checkpoint.parent / "pointer_manifest.json").read_text())
        if pointer_manifest.get("adapter_sha256") != adapter_hash:
            raise ValueError("evaluation pointer head and frozen adapter mismatch")
        if pointer_manifest.get("valid_sha256") != sha256(args.data_dir / "valid.jsonl"):
            raise ValueError("evaluation pointer head and valid data contract mismatch")
        if bool(pointer_manifest.get("conditional_pair", False)) != args.conditional_pair:
            raise ValueError("evaluation pointer architecture differs from checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    if not tokenizer.is_fast:
        raise ValueError("fast Qwen tokenizer is required for marker-offset audit")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    policy = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=args.joint_policy)
    if args.joint_policy:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.enable_input_require_grads()
    else:
        policy.eval()
        for param in policy.parameters():
            param.requires_grad_(False)
    hidden = int(policy.config.hidden_size)
    if world > 1 and args.joint_policy:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False)

    head = (CoupledPointerHead(hidden) if args.conditional_pair else PointerHead(hidden)).to(device)
    if args.eval_checkpoint is not None:
        head.load_state_dict(torch.load(args.eval_checkpoint, map_location="cpu", weights_only=True), strict=True)
    if world > 1:
        head = DDP(head, device_ids=[local_rank])
    parameters = list(head.parameters())
    if args.joint_policy:
        parameters += [param for param in policy.parameters() if param.requires_grad]
    optimizer = None if args.eval_checkpoint is not None else torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=0.01
    )

    def score(example: PointerExample, *, train_mode: bool):
        prefix = render_qwen_sft_tool_prefix(tokenizer, example.messages, tools=example.tools)
        encoded = tokenizer(prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded["input_ids"]
        if len(ids) > args.max_length:
            raise ValueError(f"{example.row_id}: prompt {len(ids)} tokens exceeds max_length={args.max_length}")
        marker_positions = locate_marker_tokens(tokenizer, prefix, example)
        lm_loss = None
        if args.joint_policy:
            full_row = {"messages": example.messages + [example.assistant_message], "tools": example.tools}
            completed, _ = encode_assistant_only_conversation(
                tokenizer, full_row, max_length=args.max_length,
            )
            if completed["input_ids"][:len(ids)] != ids:
                raise ValueError(f"{example.row_id}: SFT prefix/full-row token mismatch")
            if len(completed["input_ids"]) > args.max_length:
                raise ValueError(f"{example.row_id}: completed row exceeds max_length")
            inputs = torch.tensor([completed["input_ids"]], device=device, dtype=torch.long)
            labels = torch.tensor([completed["labels"]], device=device, dtype=torch.long)
            output = policy(input_ids=inputs, labels=labels, output_hidden_states=True, use_cache=False)
            states = output.hidden_states[-1][0]
            query, atoms = states[len(ids) - 1], states[marker_positions]
            lm_loss = output.loss
        else:
            inputs = torch.tensor([ids], device=device, dtype=torch.long)
            with torch.inference_mode():
                output = policy(input_ids=inputs, output_hidden_states=True, use_cache=False)
                states = output.hidden_states[-1][0]
                frozen_query = states[-1]
                frozen_atoms = states[marker_positions]
            # Inference-mode tensors cannot be saved by trainable Linear layers.
            query, atoms = frozen_query.clone(), frozen_atoms.clone()
        n = len(example.atom_names)
        source_pairs, sink_pairs = pairs_for_observation(example, device)
        head_output = head(query, atoms, source_pairs, sink_pairs)
        logits_source, logits_sink = head_output[:2]
        pair_logits = head_output[2] if args.conditional_pair else None
        targets_source = torch.tensor(target_indices(example, source=True), device=device)
        targets_sink = torch.tensor(target_indices(example, source=False), device=device)
        pointer_loss = -0.5 * (torch.log_softmax(logits_source, 0)[targets_source].mean()
                               + torch.log_softmax(logits_sink, 0)[targets_sink].mean())
        source_lookup = {key: i for i, key in enumerate(candidate_keys(n, example.bonds, source=True))}
        sink_lookup = {key: i for i, key in enumerate(candidate_keys(n, example.bonds, source=False))}
        paired_sources = [source_lookup[key] for key in example.source_targets]
        paired_sinks = [sink_lookup[key] for key in example.sink_targets]
        if pair_logits is not None:
            coupled_targets = torch.tensor(
                [source * pair_logits.shape[1] + sink
                 for source, sink in zip(paired_sources, paired_sinks, strict=True)],
                device=device,
            )
            coupled_loss = -torch.log_softmax(pair_logits.flatten(), 0)[coupled_targets].mean()
            pointer_loss = 0.5 * (pointer_loss + coupled_loss)
        loss = pointer_loss + args.lambda_lm * lm_loss if lm_loss is not None else pointer_loss
        with torch.no_grad():
            recalls = []
            for logits, targets in ((logits_source, targets_source), (logits_sink, targets_sink)):
                for k in (1, 4, 8):
                    selected = logits.topk(min(k, logits.numel())).indices
                    recalls.append(bool(torch.isin(selected, targets).any()))
                recalls.append(bool(torch.isin(targets, logits.topk(min(8, logits.numel())).indices).all()))
            paired = (
                coupled_pair_recall_metrics(pair_logits, paired_sources, paired_sinks)
                if pair_logits is not None else
                pair_recall_metrics(logits_source, logits_sink, paired_sources, paired_sinks)
            )
            recalls.extend(paired[key] for key in (
                "pair_r1", "pair_r4", "pair_r8", "pair_all_r1", "pair_all_r4", "pair_all_r8"
            ))
        if train_mode:
            if optimizer is None:
                raise ValueError("evaluation checkpoint cannot update the optimizer")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        return float(loss.detach()), [int(x) for x in recalls], len(ids), len(example.source_targets)

    start = time.monotonic()
    if rank == 0:
        print(json.dumps({"phase": "optimizer_start", "world": world, "train_events": len(train),
                          "valid_events": len(valid), "adapter_sha256": adapter_hash,
                          "train_file_sha256": sha256(args.data_dir / "train.jsonl")}), flush=True)
    for epoch in range(args.epochs):
        # Equal-length ranks keep DDP collectives aligned without dropping rows.
        permutation = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(permutation)
        padding = (-len(permutation)) % world
        permutation += permutation[:padding]
        assigned = permutation[rank::world]
        if args.eval_checkpoint is None:
            if args.joint_policy:
                policy.train()
            head.train()
            for step, index in enumerate(assigned, 1):
                loss, recalls, tokens, _ = score(train[index], train_mode=True)
                if rank == 0 and (step % args.log_every == 0 or step == len(assigned)):
                    print(json.dumps({"phase": "train", "epoch": epoch + 1, "rank_step": step,
                                      "rank_total": len(assigned), "loss": round(loss, 5),
                                      "source_r1": recalls[0], "sink_r1": recalls[4],
                                      "tokens": tokens, "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)
        if world > 1:
            dist.barrier()
        policy.eval()
        head.eval()
        totals = torch.zeros(16 + 4 * 5, dtype=torch.float64, device=device)
        with torch.no_grad():
            for index in range(rank, len(valid), world):
                loss, recalls, _, flow_count = score(valid[index], train_mode=False)
                totals[0] += loss
                totals[1:15] += torch.tensor(recalls, device=device)
                totals[15] += 1
                bucket = min(flow_count, 4) - 1
                offset = 16 + bucket * 5
                totals[offset:offset + 5] += torch.tensor(
                    [1, recalls[0], recalls[4], recalls[10], recalls[13]], device=device
                )
        if world > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            count = int(totals[15].item())
            metric = {"phase": "validation", "epoch": epoch + 1, "n": count,
                      "loss": float(totals[0] / count), "elapsed_s": round(time.monotonic() - start, 1)}
            for name, value in zip(("source_r1", "source_r4", "source_r8", "source_all_r8",
                                    "sink_r1", "sink_r4", "sink_r8", "sink_all_r8",
                                    "pair_r1", "pair_r4", "pair_r8", "pair_all_r1",
                                    "pair_all_r4", "pair_all_r8"), totals[1:15]):
                metric[name] = float(value / count)
            metric["by_coupled_flows"] = {}
            for bucket, label in enumerate(("1", "2", "3", "4+")):
                values = totals[16 + bucket * 5:21 + bucket * 5]
                denominator = max(int(values[0].item()), 1)
                metric["by_coupled_flows"][label] = {
                    "n": int(values[0].item()),
                    "source_r1": float(values[1] / denominator),
                    "sink_r1": float(values[2] / denominator),
                    "pair_r8": float(values[3] / denominator),
                    "pair_all_r8": float(values[4] / denominator),
                }
            args.output.mkdir(parents=True, exist_ok=True)
            if args.eval_checkpoint is None:
                state = head.module.state_dict() if world > 1 else head.state_dict()
                torch.save(state, args.output / f"pointer_head_epoch{epoch + 1}.pt")
                if args.joint_policy:
                    actor = policy.module if world > 1 else policy
                    actor.save_pretrained(args.output / f"joint_adapter_epoch{epoch + 1}")
            (args.output / f"validation_epoch{epoch + 1}.json").write_text(json.dumps(metric, indent=2))
            print(json.dumps(metric), flush=True)
    if rank == 0 and args.eval_checkpoint is None:
        trained_adapter = args.output / f"joint_adapter_epoch{args.epochs}" if args.joint_policy else args.adapter
        (args.output / "pointer_manifest.json").write_text(json.dumps({
            "artifact_type": (
                "pr71_p1_stage2_joint_lm_source_sink_pointer" if args.joint_policy
                else "pr71_p0_stage2_frozen_policy_source_sink_pointer"
            ),
            "scope": "joint_lm_pointer" if args.joint_policy else "localization_pilot_not_joint_policy_or_tree_rl",
            "lm_loss_weight": args.lambda_lm if args.joint_policy else 0.0,
            "conditional_pair": args.conditional_pair,
            "adapter": str(trained_adapter),
            "adapter_sha256": sha256(trained_adapter / "adapter_model.safetensors"),
            "initial_adapter": str(args.adapter), "initial_adapter_sha256": adapter_hash,
            "base_model": args.model, "base_revision": args.revision,
            "data_dir": str(args.data_dir),
            "train_sha256": sha256(args.data_dir / "train.jsonl"),
            "valid_sha256": sha256(args.data_dir / "valid.jsonl"),
            "train_event_count": len(train), "valid_event_count": len(valid),
            "train_source_event_decisions": int(json.loads((args.data_dir / "manifest.json").read_text())["splits"]["train"]["event_decisions"]),
            "valid_source_event_decisions": int(json.loads((args.data_dir / "manifest.json").read_text())["splits"]["valid"]["event_decisions"]),
            "train_unsupported_pointer_events": int(json.loads((args.data_dir / "manifest.json").read_text())["splits"]["train"]["event_decisions"]) - len(train),
            "valid_unsupported_pointer_events": int(json.loads((args.data_dir / "manifest.json").read_text())["splits"]["valid"]["event_decisions"]) - len(valid),
            "model_visible_gold": False, "epochs": args.epochs,
        }, indent=2))
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
