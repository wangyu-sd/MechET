"""Two-turn on-policy GRPO: draft -> verifier error -> full program repair.

TRL 0.21 supplies distributed sampling/optimization/checkpointing. We separate
attention from the policy-action mask: error messages remain visible but never
receive a policy gradient. The frozen reference is the chosen SFT LoRA, not the
unadapted Qwen base. This implementation deliberately uses HF generation with
the same temperature distribution as the token log probabilities.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
from trl import GRPOTrainer
from accelerate.utils import gather_object

from mechet.python_repair_rollout import (
    continuation_tokens, execution_feedback, repair_message, sampled_span, stop_token_ids,
)
from mechet.python_template_rlvr import score_template_rlvr_candidate


@contextmanager
def frozen_sft_adapter(model):
    """PEFT set_adapter changes requires_grad; restore it exactly on exit."""
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    active = model.active_adapter
    try:
        model.set_adapter("sft_reference")
        for p in model.parameters():
            p.requires_grad_(False)
        yield model
    finally:
        model.set_adapter(active)
        for n, p in model.named_parameters():
            p.requires_grad_(flags[n])


def group_advantages(rewards, ids, group_size):
    if len(ids) % group_size:
        raise ValueError("Incomplete distributed rollout group")
    for i in range(0, len(ids), group_size):
        if len(set(ids[i:i + group_size])) != 1:
            raise ValueError("Distributed GRPO group mixes different products")
    grouped = rewards.view(-1, group_size)
    std = grouped.std(dim=1)
    advantage = (grouped - grouped.mean(dim=1, keepdim=True)) / (std[:, None] + 1e-4)
    return advantage.reshape(-1), std


def masked_grpo_loss(logps, old_logps, ref_logps, mask, advantages, beta, epsilon):
    ratio = (logps - old_logps).exp()
    objective = torch.minimum(ratio * advantages[:, None],
                              ratio.clamp(1 - epsilon, 1 + epsilon) * advantages[:, None])
    delta = ref_logps - logps
    kl = delta.exp() - delta - 1
    per_token = -objective + beta * kl
    loss = ((per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
    return loss, (kl * mask).sum() / mask.sum().clamp(min=1)


class PythonRepairGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, repair_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.repair_config = dict(repair_config or {})
        if self.use_vllm or self.num_iterations != 1:
            raise ValueError("Repair v1 requires HF rollout and one on-policy update")
        if self.args.steps_per_generation > self.args.gradient_accumulation_steps:
            raise ValueError("Rollouts cannot span optimizer updates")
        if self.top_p != 1.0 or self.top_k not in (0, None) or self.repetition_penalty != 1.0:
            raise ValueError("Sampling/log-probability consistency requires unfiltered temperature sampling")
        self.zero_groups_in_a_row = 0

    @torch.no_grad()
    def _generate_and_score_completions(self, inputs):
        started = time.monotonic()
        tokenizer = self.processing_class
        model = self.accelerator.unwrap_model(self.model_wrapped)
        was_training = model.training
        model.eval()
        device = self.accelerator.device
        stops = stop_token_ids(tokenizer, model.generation_config)
        budget = int(self.repair_config.get("tokens_per_turn", 2048))
        context_limit = int(self.repair_config.get("max_context_tokens", 8192))
        records = []
        try:
            for row in inputs:
                prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
                if len(prompt) > self.max_prompt_length:
                    raise ValueError(f"Prompt exceeds untruncated budget: {row['id']} {len(prompt)}")
                suffix, action_mask, turns = [], [], []
                for turn in range(2):
                    context = prompt + suffix
                    if len(context) + budget > context_limit:
                        raise ValueError(f"Repair context exceeds budget: {row['id']}")
                    ids = torch.tensor([context], device=device)
                    generated = model.generate(
                        input_ids=ids, attention_mask=torch.ones_like(ids),
                        do_sample=True, temperature=self.temperature, top_p=1.0, top_k=0,
                        repetition_penalty=1.0, max_new_tokens=budget,
                        eos_token_id=stops, pad_token_id=tokenizer.pad_token_id,
                        use_cache=True, synced_gpus=False,
                    )[0, len(context):].tolist()
                    generated, terminated = sampled_span(generated, stops)
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    suffix.extend(generated)
                    # Do not positively/negatively train on a censored response.
                    action_mask.extend([int(terminated)] * len(generated))
                    feedback = execution_feedback(row["target_smiles"], text, terminated=terminated)
                    scored = score_template_rlvr_candidate(row, text) if terminated else {
                        "reward": -1.0, "formal_execute": False, "structural_precursor_exact": False}
                    turns.append({"text": text, "tokens": len(generated), "terminated": terminated,
                                  "feedback": feedback, "score": scored})
                    print(json.dumps({"stage": "repair-rollout", "rank": self.accelerator.process_index,
                                      "step": self.state.global_step, "id": row["id"], "turn": turn,
                                      "tokens": len(generated), "terminated": terminated,
                                      "execute": feedback["ok"], "reward": scored["reward"]}), flush=True)
                    if feedback["ok"] or turn == 1:
                        break
                    env_ids = continuation_tokens(tokenizer, repair_message(feedback), generated)
                    suffix.extend(env_ids)
                    action_mask.extend([0] * len(env_ids))
                final = turns[-1]
                # Terminal reward is attached to the complete sampled interaction;
                # a small repair cost discourages intentional first-turn failures.
                reward = float(final["score"]["reward"]) - 0.05 * (len(turns) - 1)
                # A final truncated rollout has no complete outcome for credit assignment.
                if not final["terminated"]:
                    action_mask = [0] * len(action_mask)
                records.append({"id": row["id"], "prompt": prompt, "suffix": suffix,
                                "mask": action_mask, "reward": reward, "turns": turns})

            def pad(values, value, left=False):
                width = max(map(len, values))
                return torch.tensor([([value] * (width - len(v)) + v if left else
                                      v + [value] * (width - len(v))) for v in values], device=device)

            prompt_ids = pad([r["prompt"] for r in records], tokenizer.pad_token_id, True)
            prompt_mask = pad([[1] * len(r["prompt"]) for r in records], 0, True)
            completion_ids = pad([r["suffix"] for r in records], tokenizer.pad_token_id)
            action_mask = pad([r["mask"] for r in records], 0)
            attention = pad([[1] * len(r["suffix"]) for r in records], 0)
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            full_attention = torch.cat([prompt_mask, attention], dim=1)
            # Store behavior logps, including the preceding generated/environment context.
            old, _ = self._get_per_token_logps_and_entropies(
                model, full_ids, full_attention, completion_ids.shape[1], batch_size=1)
            with frozen_sft_adapter(model):
                ref, _ = self._get_per_token_logps_and_entropies(
                    model, full_ids, full_attention, completion_ids.shape[1], batch_size=1)
        finally:
            model.train(was_training)

        reward = torch.tensor([r["reward"] for r in records], device=device)
        all_rewards = self.accelerator.gather(reward)
        all_ids = gather_object([r["id"] for r in records])
        advantages, std = group_advantages(all_rewards, all_ids, self.num_generations)
        start = self.accelerator.process_index * len(records)
        local_stats = torch.tensor([[len(r["turns"]) == 2,
                    r["turns"][0]["score"]["formal_execute"],
                    r["turns"][-1]["score"]["formal_execute"],
                    r["turns"][-1]["score"]["structural_precursor_exact"],
                    not r["turns"][-1]["terminated"],
                    sum(t["tokens"] for t in r["turns"])] for r in records], device=device).float()
        stats = self.accelerator.gather(local_stats).mean(0).tolist()
        names = ["repair_fraction", "draft_execution", "final_execution", "structural_exact",
                 "final_truncated", "generated_tokens"]
        summary = dict(zip(names, stats))
        summary.update(reward=float(all_rewards.mean()), reward_std=float(std.mean()),
                       zero_std_groups=float((std == 0).float().mean()), seconds=time.monotonic() - started)
        for key, value in summary.items():
            self._metrics["train"]["repair/" + key].append(value)
        out = Path(self.args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"rollouts_rank{self.accelerator.process_index}.jsonl").open("a") as handle:
            for r in records:
                handle.write(json.dumps({"step": self.state.global_step, "id": r["id"],
                                         "reward": r["reward"], "turns": r["turns"]}) + "\n")
        if self.accelerator.is_main_process:
            print(json.dumps({"stage": "repair-group", "step": self.state.global_step, **summary}), flush=True)
        self.zero_groups_in_a_row = self.zero_groups_in_a_row + 1 if bool((std == 0).all()) else 0
        if stats[4] == 1.0:
            raise RuntimeError("All final responses truncated: refusing another zero-signal RLVR run")
        if self.zero_groups_in_a_row >= int(self.repair_config.get("max_zero_signal_batches", 8)):
            raise RuntimeError("Repeated zero-variance rewards: inspect rollout errors before continuing")
        return {"prompt_ids": prompt_ids, "prompt_mask": prompt_mask,
                "completion_ids": completion_ids, "completion_mask": action_mask,
                "completion_attention_mask": attention, "advantages": advantages[start:start + len(records)],
                "old_per_token_logps": old, "ref_per_token_logps": ref}

    def _compute_loss(self, model, inputs):
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention = torch.cat([inputs["prompt_mask"], inputs["completion_attention_mask"]], dim=1)
        logps, _ = self._get_per_token_logps_and_entropies(
            model, ids, attention, inputs["completion_ids"].shape[1], compute_entropy=False)
        loss, kl = masked_grpo_loss(logps, inputs["old_per_token_logps"], inputs["ref_per_token_logps"],
                                   inputs["completion_mask"], inputs["advantages"], self.beta, self.args.epsilon)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite repair-GRPO loss")
        self._metrics["train"]["kl_to_sft"].append(self.accelerator.gather(kl.detach()).mean().item())
        return loss
