"""Self-MechVR: teacher-free RLVR with local MECH_ET verifier rewards."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from mechet.chat_template import apply_mechet_chat_template
from mechet.metrics import extract_gold_answer, extract_product_from_user
from mechet.sft import MECH_ET_SYSTEM_PROMPT, parse_mech_cot_output
from mechet.verifier import compute_mech_et_reward


def extract_prompt_messages(row: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Return (product_smiles, system+user messages only)."""
    messages = row.get("messages") or []
    product = ""
    user_content = ""
    for msg in messages:
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("TARGET:"):
            user_content = str(msg["content"])
            product = extract_product_from_user(user_content)
            break
    if not user_content:
        raise ValueError(f"row {row.get('id')} missing TARGET user message")
    prompt_messages = [
        {"role": "system", "content": MECH_ET_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return product, prompt_messages


def mechvr_gate(verified: dict[str, Any]) -> bool:
    """Hard process gate for a self-consistent, locally executable rollout."""
    return bool(
        verified.get("format_ok")
        and verified.get("target_state_matches_product")
        and verified.get("reachability_ok")
        and verified.get("state_maps_consistent")
        and verified.get("local_transition_exact")
        and verified.get("electron_conserved")
        and verified.get("answer_state_agree")
    )


def compute_mechvr_reward(
    row: dict[str, Any],
    prediction: str,
    *,
    reward_config: dict[str, Any] | None = None,
    gate_penalty: float = -1.0,
    use_gold_answer: bool = True,
) -> dict[str, Any]:
    """Score one rollout; outcome supervision uses dataset gold, not an LLM teacher.

    Failed rollouts retain stage-specific verifier rewards. ``gate_penalty`` is
    only a fallback for malformed payloads, avoiding the all-equal reward groups
    caused by assigning one constant penalty to every type of process failure.
    """
    product, _ = extract_prompt_messages(row)
    gold = extract_gold_answer(row) if use_gold_answer else None
    parsed = parse_mech_cot_output(prediction)
    mechanism = str(parsed.get("mechanism") or "")
    answer = str(parsed.get("answer") or "").strip()

    payload = compute_mech_et_reward(
        prediction,
        product,
        expected_precursor=gold,
        config=reward_config,
    )
    verified = payload.get("verified") or {}
    gate_ok = mechvr_gate(verified)
    if not gate_ok:
        failure_reward = payload.get("rlvr_failure_reward")
        if failure_reward is None:
            failure_reward = payload.get("total", gate_penalty)
        return {
            **payload,
            "gate_ok": False,
            "rlvr_total": float(failure_reward),
            "product": product,
            "gold_answer": gold,
            "answer": answer,
            "mechanism_len": len(mechanism),
        }
    return {
        **payload,
        "gate_ok": True,
        "rlvr_total": float(payload.get("total") or 0.0),
        "product": product,
        "gold_answer": gold,
        "answer": answer,
        "mechanism_len": len(mechanism),
    }


def grpo_advantages(rewards: Sequence[float], *, eps: float = 1e-6) -> list[float]:
    """Group-relative advantages (GRPO-style normalization)."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < eps:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + eps) for r in rewards]


def rloo_advantages(rewards: Sequence[float]) -> list[float]:
    """Leave-one-out baseline (RLOO)."""
    n = len(rewards)
    if n <= 1:
        return [0.0 for _ in rewards]
    total = sum(rewards)
    out: list[float] = []
    for r in rewards:
        baseline = (total - r) / (n - 1)
        out.append(r - baseline)
    return out


def compute_advantages(rewards: Sequence[float], *, method: str = "grpo") -> list[float]:
    if method == "rloo":
        return rloo_advantages(rewards)
    return grpo_advantages(rewards)


def build_assistant_completion(row: dict[str, Any], prediction: str) -> str:
    """Normalize model output to assistant message body for logprob training."""
    parsed = parse_mech_cot_output(prediction)
    mechanism = str(parsed.get("mechanism") or "").strip()
    answer = str(parsed.get("answer") or "").strip()
    if parsed.get("format_ok"):
        return (
            f"<mechanism>\n{mechanism}\n</mechanism>\n"
            f"<answer>\n{answer}\n</answer>"
        )
    return prediction.strip()


def completion_log_probs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completion_text: str,
    *,
    max_length: int = 8192,
) -> tuple[torch.Tensor, int]:
    """Sum log-prob of completion tokens (assistant span only)."""
    assistant_messages = prompt_messages + [{"role": "assistant", "content": completion_text}]
    prompt_text = apply_mechet_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = apply_mechet_chat_template(tokenizer, assistant_messages, add_generation_prompt=False)

    prompt_enc = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    prompt_len = prompt_enc["input_ids"].shape[1]
    input_ids = full_enc["input_ids"]
    if input_ids.shape[1] <= prompt_len:
        return torch.tensor(0.0, device=_model_device(model)), 0

    device = _model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = full_enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    start = max(prompt_len - 1, 0)
    completion_logp = token_log_probs[0, start:]
    n_tokens = int(completion_logp.numel())
    if n_tokens == 0:
        return torch.tensor(0.0, device=device), 0
    return completion_logp.sum(), n_tokens


def normalize_sequence_log_prob(
    log_prob_sum: torch.Tensor,
    n_tokens: int,
    *,
    length_normalize: bool = True,
) -> torch.Tensor:
    """Remove the otherwise dominant preference for shorter/longer mechanisms."""
    if not length_normalize:
        return log_prob_sum
    return log_prob_sum / max(int(n_tokens), 1)


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def policy_loss_from_advantages(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completions: Sequence[str],
    advantages: Sequence[float],
    *,
    max_length: int = 8192,
    length_normalize: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Group-relative REINFORCE loss over assistant completion tokens."""
    if not completions:
        z = torch.tensor(0.0, device=_model_device(model))
        return z, {"n_completions": 0, "loss_terms": 0}

    total_loss = torch.tensor(0.0, device=_model_device(model))
    n_terms = 0
    logprob_sum = 0.0
    token_count = 0
    for completion_raw, adv in zip(completions, advantages):
        if abs(float(adv)) < 1e-8:
            continue
        completion = build_assistant_completion({}, completion_raw)
        logp, n_tok = completion_log_probs(
            model,
            tokenizer,
            prompt_messages,
            completion,
            max_length=max_length,
        )
        if n_tok == 0:
            continue
        sequence_logp = normalize_sequence_log_prob(
            logp,
            n_tok,
            length_normalize=length_normalize,
        )
        total_loss = total_loss - float(adv) * sequence_logp
        n_terms += 1
        logprob_sum += float(sequence_logp.detach().cpu())
        token_count += n_tok

    if n_terms == 0:
        total_loss = torch.tensor(0.0, device=_model_device(model), requires_grad=True)
    stats = {
        "n_completions": len(completions),
        "loss_terms": n_terms,
        "avg_logprob": logprob_sum / max(n_terms, 1),
        "avg_completion_tokens": token_count / max(n_terms, 1),
        "length_normalized": bool(length_normalize),
    }
    return total_loss / max(n_terms, 1), stats
