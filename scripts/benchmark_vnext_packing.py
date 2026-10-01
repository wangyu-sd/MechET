#!/usr/bin/env python3
"""Matched Qwen Stage-II assistant-only packing training microbenchmark.

Uses the same rows and trainable adapter in two modes, with no optimizer
updates. It checks first that a block-diagonal packed forward matches the
independent-row loss; only then does it time forward+backward passes. This
does not switch the frozen paper trainer to packing automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from mechet.assistant_masking import encode_assistant_only_conversation
from mechet.segment_packing import block_causal_inputs, pack_encoded_rows


def sample_rows(path: Path, limit: int) -> list[dict]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == limit:
                    break
    if len(rows) != limit:
        raise ValueError("source has fewer rows than requested")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--packs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for packing training benchmark")
    manifest = json.loads((args.adapter / "adapter_manifest.json").read_text())
    if manifest.get("base_model_revision") != args.revision:
        raise ValueError("Stage-II adapter and pinned base revision mismatch")
    data_manifest = json.loads((args.data.parent / "manifest.json").read_text())
    if data_manifest.get("decision_contract") != "unified_inventory_compressed_history_tool_decision_v2":
        raise ValueError("benchmark requires Stage-II v2 assistant-only decisions")
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if digest != data_manifest["splits"][args.data.stem]["output_sha256"]:
        raise ValueError("Stage-II source hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    source = sample_rows(args.data, args.rows)
    encoded = [encode_assistant_only_conversation(tokenizer, row, max_length=args.max_length)[0]
               for row in source]
    packs = pack_encoded_rows(encoded, max_length=args.max_length)[:args.packs]
    if len(packs) < args.packs:
        raise ValueError("too few packs for requested benchmark")
    device = torch.device("cuda:0")
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()

    def independent(pack, *, backward: bool):
        offset = 0
        total_loss = 0.0
        supervised_total = sum(label != -100 for label in pack["labels"])
        for length in pack["segments"]:
            ids = pack["input_ids"][offset:offset + length]
            labels = pack["labels"][offset:offset + length]
            count = sum(label != -100 for label in labels)
            output = model(
                input_ids=torch.tensor([ids], device=device),
                labels=torch.tensor([labels], device=device), use_cache=False,
            )
            contribution = output.loss * (count / supervised_total)
            if backward:
                contribution.backward()
            total_loss += float(contribution.detach())
            offset += length
        return total_loss

    def packed(pack, *, backward: bool):
        inputs = block_causal_inputs(pack, device=device, dtype=torch.bfloat16)
        output = model(**inputs)
        if backward:
            output.loss.backward()
        return float(output.loss.detach())

    # Exact same samples; the assistant-only target mask and reset positions
    # should make the independent and packed losses match before timing.
    with torch.no_grad():
        independent_loss = independent(packs[0], backward=False)
        packed_loss = packed(packs[0], backward=False)
    difference = abs(independent_loss - packed_loss)
    if difference > 0.03:
        raise ValueError(f"packed/independent assistant loss mismatch: {difference:.6f}")
    report = {"scope": "matched_gpu_training_microbenchmark_not_full_epoch_speedup",
              "source_sha256": digest, "adapter_sha256": hashlib.sha256(
                  (args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
              "rows": args.rows, "packs": args.packs, "max_length": args.max_length,
              "independent_loss": independent_loss, "packed_loss": packed_loss,
              "loss_absolute_difference": difference, "modes": {}}
    for mode, run in (("independent", independent), ("block_packed", packed)):
        for _ in range(args.warmup):
            model.zero_grad(set_to_none=True)
            run(packs[0], backward=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for pack in packs:
            model.zero_grad(set_to_none=True)
            run(pack, backward=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        tokens = sum(len(pack["input_ids"]) for pack in packs)
        supervised = sum(sum(label != -100 for label in pack["labels"]) for pack in packs)
        report["modes"][mode] = {
            "wall_seconds": elapsed, "input_tokens_per_second": tokens / elapsed,
            "supervised_tokens_per_second": supervised / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        print(json.dumps({"mode": mode, **report["modes"][mode]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
