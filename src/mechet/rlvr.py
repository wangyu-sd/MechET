"""Teacher-free group-relative RLVR for state-annotated and proof-only MechET."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from mechet.chat_template import apply_mechet_chat_template
from mechet.metrics import extract_gold_answer, extract_product_from_user
from mechet.proof_program import extract_proof_body, verify_proof
from mechet.sft import MECH_ET_SYSTEM_PROMPT, parse_mech_cot_output
from mechet.verifier import compute_mech_et_reward


def extract_prompt_messages(
    row: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Return ``(product_smiles, system+user messages only)``.

    Proof rows retain their proof-specific system prompt; legacy rows fall back
    to the MECH_ET v3 prompt for backward compatibility.
    """
    messages = row.get("messages") or []
    product = ""
    user_content = ""
    system_content = ""
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "")
        if role == "system" and not system_content:
            system_content = content
        if role == "user" and content.startswith("TARGET:"):
            user_content = content
            product = extract_product_from_user(content)
            break
    if not user_content:
        raise ValueError(f"row {row.get('id')} missing TARGET user message")
    return product, [
        {
            "role": "system",
            "content": system_content or MECH_ET_SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]


def mechvr_gate(verified: dict[str, Any]) -> bool:
    """Hard process gate for a self-consistent MECH_ET v3 rollout."""
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
    """Score one rollout, dispatching proof rows to executor rewards."""
    task_type = str(
        row.get("task_type")
        or (row.get("metadata") or {}).get("task_type")
        or ""
    )
    if task_type == "mech_proof_retro":
        return compute_proofvr_reward(
            row,
            prediction,
            gate_penalty=gate_penalty,
        )

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
            "reward_mode": "mech_et_v3",
        }
    return {
        **payload,
        "gate_ok": True,
        "rlvr_total": float(payload.get("total") or 0.0),
        "product": product,
        "gold_answer": gold,
        "answer": answer,
        "mechanism_len": len(mechanism),
        "reward_mode": "mech_et_v3",
    }


def _proof_gold(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return str(
        metadata.get("derived_precursor")
        or metadata.get("initial_reactants")
        or ""
    )


def compute_proofvr_reward(
    row: dict[str, Any],
    prediction: str,
    *,
    gate_penalty: float = -1.0,
) -> dict[str, Any]:
    """Reward an action-only proof without teacher-trace matching.

    The process gate depends only on deterministic execution. The endpoint
    reward compares the executor-derived precursor with the dataset endpoint.
    """
    product, _ = extract_prompt_messages(row)
    gold = _proof_gold(row)
    proof_body = extract_proof_body(prediction)
    if not proof_body:
        return {
            "gate_ok": False,
            "rlvr_total": -4.0,
            "format_ok": False,
            "execute_ok": False,
            "endpoint_exact": False,
            "product": product,
            "gold_answer": gold,
            "reward_mode": "mech_proof_v1",
            "diagnostics": [
                {
                    "code": "MISSING_PROOF",
                    "message": "missing <proof> block",
                }
            ],
        }
    verified = verify_proof(
        prediction,
        expected_precursor=gold or None,
    )
    if not verified.get("execute_ok"):
        return {
            **verified,
            "gate_ok": False,
            "rlvr_total": (
                -2.0
                if verified.get("format_ok")
                else float(gate_penalty)
            ),
            "product": product,
            "gold_answer": gold,
            "reward_mode": "mech_proof_v1",
        }
    endpoint_exact = bool(verified.get("endpoint_exact"))
    # Execution is the proof-level objective; endpoint correctness is an
    # additional outcome channel. No intermediate teacher trace is required.
    reward = 3.0 + (4.0 if endpoint_exact else 0.0)
    return {
        **verified,
        "gate_ok": True,
        "rlvr_total": reward,
        "product": product,
        "gold_answer": gold,
        "reward_mode": "mech_proof_v1",
    }


def compute_rollout_reward(
    row: dict[str, Any],
    prediction: str,
    *,
    gate_penalty: float = -1.0,
    reward_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch RLVR rewards by task type."""
    return compute_mechvr_reward(
        row,
        prediction,
        gate_penalty=gate_penalty,
        reward_config=reward_config,
    )


def grpo_advantages(
    rewards: Sequence[float],
    *,
    eps: float = 1e-6,
) -> list[float]:
    """Group-relative advantages (GRPO-style normalization)."""
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(
        rewards
    )
    std = math.sqrt(variance)
    if std < eps:
        return [0.0 for _ in rewards]
    return [(reward - mean) / (std + eps) for reward in rewards]


def rloo_advantages(rewards: Sequence[float]) -> list[float]:
    """Leave-one-out baseline (RLOO)."""
    n = len(rewards)
    if n <= 1:
        return [0.0 for _ in rewards]
    total = sum(rewards)
    return [
        reward - (total - reward) / (n - 1)
        for reward in rewards
    ]


def compute_advantages(
    rewards: Sequence[float],
    *,
    method: str = "grpo",
) -> list[float]:
    return (
        rloo_advantages(rewards)
        if method == "rloo"
        else grpo_advantages(rewards)
    )


def build_assistant_completion(
    row: dict[str, Any],
    prediction: str,
) -> str:
    """Normalize proof or v3 output for assistant-span log-prob training."""
    proof = extract_proof_body(prediction)
    if proof:
        return f"<proof>\n{proof}\n</proof>"
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
    assistant_messages = prompt_messages + [
        {"role": "assistant", "content": completion_text}
    ]
    prompt_text = apply_mechet_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_text = apply_mechet_chat_template(
        tokenizer,
        assistant_messages,
        add_generation_prompt=False,
    )
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
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
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    completion_logp = token_log_probs[0, max(prompt_len - 1, 0) :]
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
        zero = torch.tensor(0.0, device=_model_device(model))
        return zero, {"n_completions": 0, "loss_terms": 0}
    total_loss = torch.tensor(0.0, device=_model_device(model))
    n_terms = 0
    logprob_sum = 0.0
    token_count = 0
    for completion_raw, advantage in zip(completions, advantages):
        if abs(float(advantage)) < 1e-8:
            continue
        completion = build_assistant_completion({}, completion_raw)
        logp, n_tokens = completion_log_probs(
            model,
            tokenizer,
            prompt_messages,
            completion,
            max_length=max_length,
        )
        if n_tokens == 0:
            continue
        sequence_logp = normalize_sequence_log_prob(
            logp,
            n_tokens,
            length_normalize=length_normalize,
        )
        total_loss = total_loss - float(advantage) * sequence_logp
        n_terms += 1
        logprob_sum += float(sequence_logp.detach().cpu())
        token_count += n_tokens
    if n_terms == 0:
        total_loss = torch.tensor(
            0.0,
            device=_model_device(model),
            requires_grad=True,
        )
    stats = {
        "n_completions": len(completions),
        "loss_terms": n_terms,
        "avg_logprob": logprob_sum / max(n_terms, 1),
        "avg_completion_tokens": token_count / max(n_terms, 1),
        "length_normalized": bool(length_normalize),
    }
    return total_loss / max(n_terms, 1), stats
