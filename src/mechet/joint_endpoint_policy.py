"""Token-bound delegated Qwen/graph actions for conditional endpoint RL.

Scores are SUMMED log probabilities. Invalid/truncated proposal token paths
remain in support and receive their real failure return; no gold insertion.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class JointProposal:
    name: str
    arguments: dict[str, Any]
    token_ids: tuple[int, ...]
    raw: str
    moves: tuple[dict[str, Any], ...] = ()
    error: str = ''


def freeze_joint_support(proposals):
    paths = {}
    for proposal in proposals:
        key = (proposal.name, proposal.token_ids,
               json.dumps(proposal.moves, sort_keys=True))
        paths.setdefault(key, proposal)
    return tuple(paths.values())


def import_generation_budget(prompt_tokens, prefix_tokens, *, max_context, max_tokens):
    if prompt_tokens+max_tokens > max_context:
        raise ValueError('CONTEXT_BUDGET_EXCEEDED')
    if prefix_tokens >= max_tokens:
        raise ValueError('import prefix consumes entire token budget')
    return max_tokens-prefix_tokens


def qwen_continuation_log_probability(model, input_ids, continuation_ids):
    """Teacher-force original suffix IDs, including its EOS if generated.

Request only suffix logits from Qwen. No re-tokenization, length mean,
generation autograd or stale KV cache. Tiny/general test models may return
full logits; both paths have exactly the same causal indices.
"""
    import torch
    import torch.nn.functional as F
    if not input_ids or not continuation_ids:
        raise ValueError('empty prompt or continuation')
    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids) + list(continuation_ids)],
                       dtype=torch.long, device=device)
    count = len(continuation_ids)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                   logits_to_keep=count + 1, use_cache=False).logits
    if logits.shape[1] == ids.shape[1]:
        positions = logits[:, len(input_ids)-1:len(input_ids)+count-1]
    elif logits.shape[1] == count + 1:
        positions = logits[:, :-1]
    else:
        raise ValueError('unsupported causal suffix logits shape')
    target = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    score = -F.cross_entropy(positions[0].float(), target, reduction='sum')
    if not torch.isfinite(score):
        raise FloatingPointError('nonfinite Qwen suffix probability')
    return score


def joint_action_log_probability(proposal, qwen_logp, graph_logp=None):
    if proposal.name == 'apply_electron_flow':
        if graph_logp is None:
            raise ValueError('FLOW requires graph probability')
        return qwen_logp + graph_logp
    if graph_logp is not None:
        raise ValueError('non-FLOW action cannot own graph probability')
    return qwen_logp


def conditional_score_gradients(scores, draw_indices, returns, *, temperature,
                                actor_enabled, kl_weight, original_scores=None):
    """Exact score-space chain rule permits one LM activation graph at a time."""
    import torch
    from mechet.graph_endpoint_rl import (
        conditional_endpoint_policy_loss, truncated_slate_reference_kl,
    )
    values = scores.detach().clone().requires_grad_(True)
    loss = conditional_endpoint_policy_loss(list(values), draw_indices, returns,
        temperature=temperature, actor_enabled=actor_enabled)
    if kl_weight:
        if original_scores is None:
            raise ValueError('original complete support missing for KL')
        loss = loss + kl_weight * truncated_slate_reference_kl(
            list(values / temperature), list(original_scores.detach() / temperature))
    coefficients, = torch.autograd.grad(loss, values)
    return coefficients.detach(), float(loss.detach())
