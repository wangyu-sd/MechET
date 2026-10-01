"""Matched token-ratio GRPO and length-normalized sequence-ratio GSPO losses."""
from __future__ import annotations

import torch


def clipped_group_policy_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    mask: torch.Tensor,
    advantage: torch.Tensor,
    *,
    mode: str,
    clip_epsilon: float = 0.2,
    reference_logps: torch.Tensor | None = None,
    reference_weight: float = 0.01,
) -> torch.Tensor:
    if mode not in {"token", "sequence"}:
        raise ValueError(f"unknown importance-ratio mode: {mode}")
    if not (new_logps.shape == old_logps.shape == mask.shape):
        raise ValueError("new/old log probabilities and action masks must align")
    if not 0 <= clip_epsilon < 1 or reference_weight < 0:
        raise ValueError("invalid clip or reference coefficient")
    active = mask.float()
    count = active.sum(dim=-1, keepdim=True).clamp(min=1.0)
    if mode == "sequence":
        # GSPO applies one importance ratio per entire generated action, with
        # length normalization so long tool calls do not dominate by exponent.
        log_ratio = ((new_logps - old_logps) * active).sum(dim=-1, keepdim=True) / count
        ratio = log_ratio.clamp(min=-20.0, max=20.0).exp()
    else:
        # Preserve the historical token-GRPO objective exactly on the default
        # path. vNext must remain opt-in; only sequence-ratio GSPO uses the
        # additional numerical clamp above.
        ratio = torch.exp(new_logps - old_logps)
    adv = advantage.reshape(-1, 1).float()
    surrogate = torch.minimum(
        ratio * adv,
        ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv,
    )
    loss = -(surrogate * active).sum() / active.sum().clamp(min=1.0)
    if reference_logps is not None:
        if reference_logps.shape != new_logps.shape:
            raise ValueError("reference log probabilities do not align")
        # Preserve the existing squared-log-ratio stabilizer for matched runs.
        delta = reference_logps - new_logps
        loss = loss + reference_weight * (delta.square() * active).sum() / active.sum().clamp(min=1.0)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite group policy objective")
    return loss
