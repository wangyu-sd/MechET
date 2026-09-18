#!/usr/bin/env python3
"""Deterministic one-step optimizer smoke for endpoint-process RLVR."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mechet.endpoint_process_rl_env import EndpointProcessRLEnv
from mechet.endpoint_process_rollout import EpisodeRollout, SampledEventSpan
from mechet.grounded_event_search import GroundedProposal
from mechet.rlvr import (
    EventPolicySample,
    event_level_policy_loss,
    event_rloo_advantages,
    event_token_log_probs_batch,
)


class TinyCausalLM(torch.nn.Module):
    """Small differentiable LM used only to verify objective plumbing."""

    def __init__(self, vocab_size: int = 128, width: int = 24) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, width)
        self.output = torch.nn.Linear(width, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        del attention_mask
        return SimpleNamespace(logits=self.output(self.embedding(input_ids)))


def read_rows(path: Path, count: int = 8) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"optimizer smoke requires {count} pilot rows")
    return rows


def proposal(action: dict, raw: str) -> GroundedProposal:
    return GroundedProposal(
        name=str(action["name"]),
        arguments=dict(action.get("arguments") or {}),
        raw_response=raw,
        logprob_sum=0.0,
        token_count=2,
        seed=17,
    )


def build_group(row: dict, group_index: int) -> list[EpisodeRollout]:
    private = row["private_reward"]
    prefix = len(private["gold_actions"]) - 1
    success = EndpointProcessRLEnv(
        row, prefix_events=prefix, start_horizon="near_end"
    )
    success.assert_no_reward_leakage()
    spans_ok: list[SampledEventSpan] = []
    action = private["gold_actions"][prefix]
    success.step(proposal(action, "gold-last-event"))
    spans_ok.append(SampledEventSpan((1, 2, 3), (10 + group_index, 20)))
    success.step(proposal({"name": "finish_trace", "arguments": {}}, "finish"))
    spans_ok.append(SampledEventSpan((1, 2, 3, 4), (30 + group_index, 21)))

    rejected = EndpointProcessRLEnv(
        row, prefix_events=prefix, start_horizon="near_end"
    )
    rejected.assert_no_reward_leakage()
    spans_bad: list[SampledEventSpan] = []
    for attempt in range(2):
        rejected.reject_unparsed(
            raw_response="not-a-tool-call",
            token_count=2,
            message="deterministic optimizer-smoke rejection",
        )
        spans_bad.append(
            SampledEventSpan((1, 2, 3), (50 + group_index, 40 + attempt))
        )
    output = [
        EpisodeRollout(success, spans_ok),
        EpisodeRollout(rejected, spans_bad),
    ]
    for rollout in output:
        rollout.validate()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(17)
    rows = read_rows(args.pilot, count=8)
    all_samples: list[EventPolicySample] = []
    endpoint_rewards: list[float] = []
    rejected_rewards: list[float] = []
    nonzero_advantages = 0
    for group_index, row in enumerate(rows[:4]):
        group = build_group(row, group_index)
        rewards = [[credit.total_reward for credit in item.env.credits] for item in group]
        _, advantages = event_rloo_advantages(rewards, gamma=0.95)
        endpoint_rewards.append(group[0].env.credits[-1].terminal_reward)
        rejected_rewards.extend(
            credit.total_reward for credit in group[1].env.credits
        )
        for rollout, values in zip(group, advantages, strict=True):
            for span, credit, advantage in zip(
                rollout.spans, rollout.env.credits, values, strict=True
            ):
                nonzero_advantages += int(abs(advantage) > 1e-12)
                all_samples.append(
                    EventPolicySample(
                        span.prompt_token_ids,
                        span.completion_token_ids,
                        advantage,
                        rejected=not credit.accepted,
                    )
                )

    model = TinyCausalLM()
    reference = copy.deepcopy(model).eval()
    with torch.no_grad():
        ref_logps, _ = event_token_log_probs_batch(
            reference,
            [item.prompt_token_ids for item in all_samples],
            [item.completion_token_ids for item in all_samples],
            max_length=64,
        )
    samples = [
        EventPolicySample(
            item.prompt_token_ids,
            item.completion_token_ids,
            item.advantage,
            reference_logprob=float(ref),
            rejected=item.rejected,
        )
        for item, ref in zip(all_samples, ref_logps.tolist(), strict=True)
    ]
    before = [value.detach().clone() for value in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss, stats = event_level_policy_loss(
        model,
        samples,
        beta_kl=0.01,
        max_length=64,
        microbatch_size=8,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    parameters_changed = any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )
    result = {
        "artifact_type": "endpoint_process_optimizer_smoke_v1",
        "fixture_rows": len(rows),
        "active_groups": 4,
        "episodes": 8,
        "optimizer_steps": 1,
        "loss": float(loss.detach()),
        "loss_finite": math.isfinite(float(loss.detach())),
        "mean_kl": float(stats["mean_kl"]),
        "kl_finite": math.isfinite(float(stats["mean_kl"])),
        "nonzero_advantage_terms": nonzero_advantages,
        "event_terms": int(stats["event_terms"]),
        "rejected_event_terms": int(stats["rejected_event_terms"]),
        "rejected_credit_negative": all(value < 0 for value in rejected_rewards),
        "endpoint_terminal_dominant": min(endpoint_rewards) > 1.0,
        "gradient_norm": gradient_norm,
        "parameters_changed": parameters_changed,
        "gold_prompt_leakage": False,
    }
    required = (
        result["loss_finite"]
        and result["kl_finite"]
        and result["nonzero_advantage_terms"] > 0
        and result["rejected_event_terms"] > 0
        and result["rejected_credit_negative"]
        and result["endpoint_terminal_dominant"]
        and result["gradient_norm"] > 0
        and result["parameters_changed"]
    )
    result["passed"] = bool(required)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
