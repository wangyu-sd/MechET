#!/usr/bin/env python3
"""Train a small prefill-only System-One electron-flow policy.

The backbone performs one forward pass over the current MechET state.  A
hierarchical pointer head selects source/sink options; no assistant text is
generated and the gold action is never part of the model input.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mechet.assistant_masking import render_qwen_sft_tool_prefix
from mechet.electron_pointer import (
    UnsupportedPointerEvent,
    candidate_keys,
    locate_marker_tokens,
    parse_pointer_example,
)
from mechet.electron_pointer_model import pairs_for_observation
from mechet.system_one_decision import ElectronFlowDecisionHead, multi_target_nll


def load(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            try:
                ex = parse_pointer_example(row)
            except UnsupportedPointerEvent:
                continue
            if ex is not None:
                rows.append(ex)
    return rows


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--pointer-dim", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-train-events", type=int, default=0)
    ap.add_argument("--valid-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def main() -> int:
    a = args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = load(a.train)
    valid = load(a.valid)
    random.shuffle(train)
    if a.max_train_events:
        train = train[:a.max_train_events]
    if a.valid_limit:
        valid = valid[:a.valid_limit]

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, trust_remote_code=True)
    if not tok.is_fast:
        raise ValueError("a fast tokenizer is required for atom-marker alignment")
    base = AutoModelForCausalLM.from_pretrained(
        a.model,
        revision=a.revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device)
    peft = get_peft_model(
        base,
        LoraConfig(
            r=a.lora_r,
            lora_alpha=2 * a.lora_r,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    head = ElectronFlowDecisionHead(int(peft.config.hidden_size), a.pointer_dim).to(device)
    params = [p for p in peft.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)

    def score(ex, train_mode: bool):
        prefix = render_qwen_sft_tool_prefix(tok, ex.messages, tools=ex.tools)
        enc = tok(prefix, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        if len(ids) > a.max_length:
            raise ValueError(f"{ex.row_id}: {len(ids)} tokens exceed max-length")
        marker = locate_marker_tokens(tok, prefix, ex)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        out = peft(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        states = out.hidden_states[-1][0]
        src_pairs, sink_pairs = pairs_for_observation(ex, device)
        decision = head(states[-1], states[marker], src_pairs, sink_pairs)

        src_keys = candidate_keys(len(ex.atom_names), ex.bonds, source=True)
        sink_keys = candidate_keys(len(ex.atom_names), ex.bonds, source=False)
        src_index = {k: i for i, k in enumerate(src_keys)}
        sink_index = {k: i for i, k in enumerate(sink_keys)}
        pair_targets = [
            src_index[s] * len(sink_keys) + sink_index[t]
            for s, t in zip(ex.source_targets, ex.sink_targets, strict=True)
        ]
        loss = multi_target_nll(decision.pair_logits, pair_targets)
        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        ranked = decision.pair_logits.detach().flatten().topk(min(8, decision.pair_logits.numel())).indices.tolist()
        gold = set(pair_targets)
        return float(loss.detach()), [int(bool(gold & set(ranked[:k]))) for k in (1, 4, 8)]

    for epoch in range(a.epochs):
        peft.train(); head.train()
        running = 0.0
        for i, ex in enumerate(train, 1):
            loss, _ = score(ex, True)
            running += loss
            if i % 100 == 0:
                print(json.dumps({"phase": "train", "epoch": epoch + 1, "step": i, "loss": running / 100}))
                running = 0.0

        peft.eval(); head.eval()
        totals = [0, 0, 0]
        loss_sum = 0.0
        with torch.no_grad():
            for ex in valid:
                loss, rec = score(ex, False)
                loss_sum += loss
                totals = [x + y for x, y in zip(totals, rec)]
        n = max(len(valid), 1)
        report = {
            "epoch": epoch + 1,
            "valid_events": len(valid),
            "loss": loss_sum / n,
            "pair_recall_at_1": totals[0] / n,
            "pair_recall_at_4": totals[1] / n,
            "pair_recall_at_8": totals[2] / n,
            "autoregressive_generation": False,
            "backbone": a.model,
        }
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / f"validation_epoch{epoch + 1}.json").write_text(json.dumps(report, indent=2))
        peft.save_pretrained(a.output / f"adapter_epoch{epoch + 1}")
        torch.save(head.state_dict(), a.output / f"decision_head_epoch{epoch + 1}.pt")
        print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
