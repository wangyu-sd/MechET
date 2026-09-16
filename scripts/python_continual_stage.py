#!/usr/bin/env python3
"""Isolated vLLM collection and clipped-PPO + supervised replay stages.

Each collection round uses one frozen actor. Stored behavior log probabilities
are retained for the single optimization epoch, then discarded for fresh data.
Generation uses temperature=1, top_p=1, top_k=-1 so vLLM's raw log probabilities
are the sampling probabilities. Tool observations never contribute to policy loss.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]

from mechet.assistant_masking import render_chat
from mechet.python_continual import (
    curriculum_example, detailed_feedback, evaluate, feedback_message, supervised_record,
)
from mechet.python_repair_rollout import continuation_tokens


def read_rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def log(**record):
    print(json.dumps(record, ensure_ascii=False), flush=True)


def gold_examples(tokenizer, row, index):
    prompt = tokenizer.encode(render_chat(tokenizer, row["messages"][:-1], add_generation_prompt=True), add_special_tokens=False)
    yield supervised_record(tokenizer, prompt, row["messages"][-1]["content"], row["id"], "gold_replay")
    state, answer = curriculum_example(row, index)
    messages = [row["messages"][0], {"role": "user", "content":
        "Auxiliary continuation exercise: TARGET=" + state +
        "\nComplete the remaining inverse electron steps from this supplied intermediate state. Output only STEPS."}]
    ids = tokenizer.encode(render_chat(tokenizer, messages, add_generation_prompt=True), add_special_tokens=False)
    yield supervised_record(tokenizer, ids, answer, row["id"], "local_curriculum")


def collect(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import vllm
    if vllm.__version__ != "0.8.5":
        raise ValueError(f"Expected tested vLLM 0.8.5, got {vllm.__version__}")
    rows = read_rows(args.data)[args.rank::8]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(f"Refusing to overwrite rollout: {output}")
    llm = LLM(model=args.model, tokenizer=args.model, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=.85, max_model_len=12288, max_num_seqs=32,
              enable_prefix_caching=True, enable_lora=True, max_lora_rank=16,
              enforce_eager=True, seed=(args.seed + args.rank) % (2**32), trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    lora = LoRARequest("continual_actor", 1, str(Path(args.adapter).resolve()))
    eos = [151643, 151645]

    def generate(prompts, n, greedy=False):
        params = SamplingParams(n=n, temperature=0. if greedy else 1., top_p=1., top_k=-1,
            repetition_penalty=1., max_tokens=2048, stop_token_ids=eos, logprobs=0)
        return llm.generate([{"prompt_token_ids": ids} for ids in prompts], params,
                            lora_request=lora, use_tqdm=False)

    def completion(value):
        ids = list(value.token_ids)
        if value.logprobs is None or len(value.logprobs) != len(ids):
            raise ValueError("Missing behavior token likelihoods")
        lp = [float(item[token].logprob) for token, item in zip(ids, value.logprobs)]
        if not all(math.isfinite(x) for x in lp):
            raise ValueError("Nonfinite rollout log probabilities")
        terminated = bool(ids and ids[-1] in eos and value.finish_reason == "stop")
        return ids, lp, terminated, tokenizer.decode(ids, skip_special_tokens=True)

    with output.open("x") as handle:
        for offset in range(0, len(rows), 4):
            batch = rows[offset:offset + 4]
            prompts = [tokenizer.encode(render_chat(tokenizer, r["messages"][:-1], add_generation_prompt=True),
                                       add_special_tokens=False) for r in batch]
            drafts = generate(prompts, args.k, args.evaluation)
            records, repair_jobs, repair_prompts = [], [], []
            for row, prompt, generated in zip(batch, prompts, drafts, strict=True):
                for c in generated.outputs:
                    ids, lp, done, text = completion(c)
                    scored = evaluate(row["target_smiles"], row["expected_precursor"], text, done)
                    r = {"id": row["id"], "kind": "rl", "input_ids": prompt + ids,
                         "loss_mask": [0] * len(prompt) + [int(done)] * len(ids),
                         "old_logps": [0.] * len(prompt) + lp, "advantage": 0.,
                         "turns": [{"text": text, "terminated": done, "score": scored}], "reward": scored["reward"]}
                    records.append(r)
                    if not scored["feedback"]["ok"]:
                        env = continuation_tokens(tokenizer, feedback_message(scored["feedback"]), ids)
                        context = r["input_ids"] + env
                        if len(context) + 2048 <= 12288:
                            repair_jobs.append((r, row, len(env)))
                            repair_prompts.append(context)
                        else:
                            r["repair_skipped"] = "context_budget"
            if repair_prompts:
                repairs = generate(repair_prompts, 1, args.evaluation)
                for (r, row, nenv), context, generated in zip(repair_jobs, repair_prompts, repairs, strict=True):
                    ids, lp, done, text = completion(generated.outputs[0])
                    scored = evaluate(row["target_smiles"], row["expected_precursor"], text, done,
                                      previous=r["turns"][0]["text"])
                    r["repair_prompt_ids"] = context
                    r["input_ids"] = context + ids
                    r["loss_mask"].extend([0] * nenv + [int(done)] * len(ids))
                    r["old_logps"].extend([0.] * nenv + lp)
                    r["turns"].append({"text": text, "terminated": done, "score": scored})
                    # Correct immediately must never pay less than deliberately
                    # failing first and repairing. Charge a small extra-turn cost.
                    r["reward"] = scored["reward"] - .02
                    if not done:
                        r["loss_mask"] = [0] * len(r["input_ids"])
            for row in batch:
                group = [r for r in records if r["id"] == row["id"]]
                if len(group) != args.k:
                    raise ValueError("Incomplete candidate group")
                rewards = [r["reward"] for r in group]
                mean = sum(rewards) / len(rewards)
                std = (sum((r - mean) ** 2 for r in rewards) / max(len(rewards) - 1, 1)) ** .5
                has_correct = any(r["turns"][-1]["score"]["correct"] for r in group)
                for r in group:
                    # Without a correct candidate, do not turn tiny execution/
                    # turn-cost differences into unit-scale policy advantages.
                    r["advantage"] = (r["reward"] - mean) / (std + 1e-4) if has_correct else 0.
                    handle.write(json.dumps(r) + "\n")
                if not args.evaluation:
                    for item in gold_examples(tokenizer, row, args.seed + offset):
                        handle.write(json.dumps(item) + "\n")
                    # At most one verified repair demonstration per product. Gold is
                    # an output label, NEVER appended to the model-visible error.
                    failed = next((r for r in group if "repair_prompt_ids" in r), None)
                    if failed:
                        correct = [r for r in group if r["turns"][-1]["score"]["correct"]]
                        answer = correct[0]["turns"][-1]["score"]["effective_program"] if correct else row["messages"][-1]["content"]
                        if not evaluate(row["target_smiles"], row["expected_precursor"], answer, True)["correct"]:
                            raise ValueError("Repair supervision failed verification")
                        item = supervised_record(tokenizer, failed["repair_prompt_ids"], answer, row["id"], "verified_repair")
                        handle.write(json.dumps(item) + "\n")
            handle.flush()
            log(stage="vllm-continual", rank=args.rank, products=min(offset + 4, len(rows)), total=len(rows),
                draft_correct=sum(r["turns"][0]["score"]["correct"] for r in records),
                final_correct=sum(r["turns"][-1]["score"]["correct"] for r in records),
                candidates=len(records), automatic_import_restorations=sum(len(r["turns"][-1]["score"]["tool_repairs"]) for r in records))


def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import PeftModel
    from python_repair_grpo import frozen_sft_adapter
    from trl.trainer.utils import selective_log_softmax
    rank = int(os.environ.get("LOCAL_RANK", 0))
    cpu_smoke = bool(getattr(args, "cpu_smoke", False))
    if not cpu_smoke:
        torch.cuda.set_device(rank)
    rows = read_rows(args.data)
    for r in rows:
        if not (len(r["input_ids"]) == len(r["loss_mask"]) == len(r["old_logps"])):
            raise ValueError("Trajectory mask/logprob misalignment")
        if len(r["input_ids"]) > 12288:
            raise ValueError("Training trajectory exceeds context (no silent truncation)")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32 if cpu_smoke else torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True)
    memory_efficient = bool(getattr(args, "memory_efficient_logps", False))
    if memory_efficient:
        from mechet.selected_policy_logps import install_selected_logps_forward
        install_selected_logps_forward(base)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.load_adapter(args.reference, adapter_name="sft_reference", is_trainable=False)
    model.set_adapter("default")
    for name, p in model.named_parameters():
        if ".sft_reference." in name:
            p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.use_cache = False

    class ContinualTrainer(Trainer):
        def policy_logps(self, model, ids, inputs):
            if memory_efficient:
                return model(input_ids=ids, attention_mask=inputs["attention_mask"],
                    loss_token_mask=inputs["loss_mask"], use_cache=False)["selected_logps"]
            logits = model(input_ids=ids, attention_mask=inputs["attention_mask"], use_cache=False).logits[:, :-1].float()
            return selective_log_softmax(logits, ids[:, 1:])

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            ids = inputs["input_ids"]
            mask = inputs["loss_mask"][:, 1:]
            logps = self.policy_logps(model, ids, inputs)
            if bool(inputs["supervised"].item()):
                loss = -(logps * mask).sum() / mask.sum().clamp(min=1)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite supervised replay loss")
                return .5 * loss
            actor = self.accelerator.unwrap_model(model)
            with torch.no_grad(), frozen_sft_adapter(actor):
                ref = self.policy_logps(actor, ids, inputs)
            old = inputs["old_logps"][:, 1:]
            ratio = torch.exp(logps - old)
            advantage = inputs["advantage"][:, None]
            objective = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
            delta = ref - logps
            # Squared log-ratio is a bounded-cost reference regularizer. Importance
            # ratios use recorded vLLM behavior probabilities, not recomputed new ones.
            per_token = -objective + .01 * delta.square()
            loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite continual PPO loss")
            return loss

    def collate(batch):
        if len(batch) != 1:
            raise ValueError("Current memory-safe PPO microbatch is one sequence")
        r = batch[0]
        return {"input_ids": torch.tensor([r["input_ids"]]),
                "attention_mask": torch.ones((1, len(r["input_ids"])), dtype=torch.long),
                "loss_mask": torch.tensor([r["loss_mask"]], dtype=torch.float32),
                "old_logps": torch.tensor([r["old_logps"]], dtype=torch.float32),
                "advantage": torch.tensor([r["advantage"]], dtype=torch.float32),
                "supervised": torch.tensor([r["kind"] != "rl"])}

    # All ranks must follow the same branch per microstep for DDP collective
    # ordering. RL and supervised phases run consecutively with the same optimizer
    # semantics per phase, never mixed across ranks.
    output = Path(args.output)
    if (output / "stage_done.json").exists():
        raise ValueError("Stage already completed; driver should resume at next stage")
    updates = 0
    for phase, phase_rows in [("ppo", [r for r in rows if r["kind"] == "rl"]),
                              ("replay", [r for r in rows if r["kind"] != "rl"])]:
        if not phase_rows:
            continue
        if phase == "ppo" and not any(r["advantage"] for r in phase_rows):
            log(stage="zero-task-reward-variance", action="supervised-recovery", rows=len(phase_rows))
            continue
        training_args = TrainingArguments(output_dir=str(output / phase), num_train_epochs=1,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=1e-6 if phase == "ppo" else 3e-6, lr_scheduler_type="constant",
            logging_steps=1, save_strategy="no", bf16=not cpu_smoke, tf32=not cpu_smoke, use_cpu=cpu_smoke,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False, report_to=[], remove_unused_columns=False,
            seed=args.seed, data_seed=args.seed, disable_tqdm=True)
        trainer = ContinualTrainer(model=model, args=training_args, data_collator=collate,
                                  train_dataset=Dataset.from_list(phase_rows), processing_class=tok)
        trainer.model_accepts_loss_kwargs = False
        log(stage="continual-train", phase=phase, rows=len(phase_rows), rank=rank)
        result = trainer.train()
        updates += trainer.state.global_step
        if trainer.is_world_process_zero():
            log(stage="phase-done", phase=phase, metrics=result.metrics)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        model.save_pretrained(output / "adapter", selected_adapters=["default"])
        tok.save_pretrained(output / "adapter")
        (output / "stage_done.json").write_text(json.dumps({"adapter": str(output / "adapter"),
            "initial_adapter": args.adapter, "reference": args.reference, "updates": updates,
            "rows": len(rows), "algorithm": "group_relative_clipped_policy_update_then_verified_supervised_replay",
            "memory_efficient_logps": memory_efficient}, indent=2))
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["collect", "train"])
    for key in ("data", "output", "model", "adapter"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--reference")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--evaluation", action="store_true")
    args = p.parse_args()
    (collect if args.mode == "collect" else train)(args)


if __name__ == "__main__":
    main()
