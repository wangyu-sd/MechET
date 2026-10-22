#!/usr/bin/env python3
"""Eight-device compatibility audit; scratch updates are discarded, never SFT."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from mechet.agent_inference import parse_tool_calls
    from mechet.collator import AssistantOnlyCollator
    from mechet.pr96_artifacts import validate_preflight
    from mechet.pr96_eval_runtime import HFBackend, sha
    from mechet.pr96_trainer import validate_deployed_hardware
    from mechet.qwen35_protocol import (OFFICIAL_REVISION, LORA_TARGETS, encode_tool_decision,
                                        load_language_model, supervised_loss)
    cfg = yaml.safe_load(args.config.read_text())
    names = validate_deployed_hardware(cfg["pr96"]["track"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("gloo")
    args.output.mkdir(parents=True, exist_ok=True)
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "transformers", "peft", "accelerate", "datasets", "tokenizers", "fla-core", "triton", "rdkit")}
    print(json.dumps({"stage": "pr96-deployed-preflight-start", "rank": rank, "gpu_names": names,
                      "versions": versions, "cuda": torch.version.cuda}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], revision=OFFICIAL_REVISION,
                                              local_files_only=True)
    samples = {}
    for line in Path(cfg["train_file"]).open():
        row = json.loads(line)
        name = row["messages"][2]["tool_calls"][0]["function"]["name"]
        samples.setdefault(name, row)
        if len(samples) == 3:
            break
    if set(samples) != {"import_fragments", "apply_electron_flow", "finish_trace"}:
        raise ValueError("PR96_PREFLIGHT_MISSING_TOOL_CLASS")
    encoded = []
    for sample in samples.values():
        value, audit = encode_tool_decision(tokenizer, sample, max_length=cfg["training"]["max_length"])
        if audit["exceeds_max_length"]:
            raise ValueError("PR96_PREFLIGHT_WOULD_TRUNCATE")
        text = tokenizer.decode([v for v,l in zip(value["input_ids"], value["labels"]) if l != -100])
        expected = sample["messages"][2]["tool_calls"][0]["function"]
        calls = parse_tool_calls(text)
        if len(calls) != 1 or calls[0].name != expected["name"] or calls[0].arguments != expected["arguments"]:
            raise ValueError("PR96_PREFLIGHT_XML_PARSE_ROUNDTRIP_FAILED")
        encoded.append(value)
    torch.manual_seed(17)
    model, info = load_language_model(cfg["model_name_or_path"], dtype=torch.bfloat16)
    targets = {n.rsplit(".",1)[-1] for n,m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    if not set(LORA_TARGETS) <= targets:
        raise ValueError("PR96_PREFLIGHT_LANGUAGE_TARGETS_MISSING")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    model.to(int(os.environ["LOCAL_RANK"]))
    if cfg["training"].get("gradient_checkpointing"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    if any("visual" in n or (p.requires_grad and "lora_" not in n) for n,p in model.named_parameters()):
        raise ValueError("PR96_PREFLIGHT_NONLANGUAGE_TRAINABLE")
    before = {n:p.detach().clone() for n,p in model.named_parameters() if "lora_B" in n}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, fused=True)
    batch = AssistantOnlyCollator(tokenizer)(encoded)
    batch = {k:v.to(next(model.parameters()).device) for k,v in batch.items()}
    begin = time.monotonic()
    loss, _ = supervised_loss(model, batch)
    if not torch.isfinite(loss):
        raise ValueError("PR96_PREFLIGHT_NONFINITE_LOSS")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    finite = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
    nonzero = any(torch.count_nonzero(g).item() for g in gradients)
    optimizer.step()
    changed = any(not torch.equal(p.detach(), before[n]) for n,p in model.named_parameters() if n in before)
    model.eval()
    backend = HFBackend(model, tokenizer, max_new_tokens=512, batch_size=1,
                        raw_path=args.output / f"generation.rank{rank:02d}.jsonl")
    generated = backend.generate([next(iter(samples.values()))["messages"][:2]])[0]
    record = {"rank": rank, "gpu_names": names, "versions": versions, "loss": float(loss.detach()),
              "finite_nonzero_gradients": bool(finite and nonzero), "parameters_updated": changed,
              "assistant_mask_verified": True, "tool_prefix_parity": True,
              "all_language_targets_present": True, "language_only": True,
              "generation_completed": bool(generated["generated_token_ids"]),
              "base_generation_parse_success": len(parse_tool_calls(generated["raw"])) == 1,
              "base_generation_truncated": generated["truncated"],
              "elapsed_seconds": time.monotonic() - begin,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
              "scratch_optimizer_updates": 1, "scratch_weights_saved": False}
    (args.output / f"rank{rank:02d}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"stage": "pr96-deployed-preflight-rank-complete", **record}), flush=True)
    dist.barrier()
    if rank == 0:
        result = {"track": cfg["pr96"]["track"], "base_revision": OFFICIAL_REVISION,
                  "base_manifest_sha256": sha(cfg["pr96"]["base_manifest"]),
                  "ranks": [json.loads((args.output / f"rank{i:02d}.json").read_text()) for i in range(8)],
                  "purpose": "compatibility_only_discarded_weights_not_formal_training"}
        validate_preflight(result, cfg["pr96"]["track"])
        (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
