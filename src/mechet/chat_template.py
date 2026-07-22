"""Unified Qwen3 chat template helpers for MechET train/inference."""

from __future__ import annotations

from typing import Any


def apply_mechet_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render chat text with Qwen3 thinking disabled for consistent train/infer."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def find_assistant_start(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    """Return token index where assistant JSON target begins (after thinking block)."""
    prefix = apply_mechet_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full = apply_mechet_chat_template(tokenizer, messages, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    start = len(prefix_ids)
    if start > len(full_ids):
        start = len(full_ids)
    decoded = tokenizer.decode(full_ids[start:], skip_special_tokens=False)
    if decoded.lstrip().startswith("{"):
        return start
    # Fallback: skip empty thinking block tokens if prefix alignment drifted.
    assistant_text = messages[-1]["content"]
    for probe in range(start, min(start + 32, len(full_ids))):
        if tokenizer.decode(full_ids[probe:], skip_special_tokens=False).lstrip().startswith("{"):
            return probe
    needle = assistant_text[: min(24, len(assistant_text))]
    if needle:
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        idx = full_text.find(needle)
        if idx >= 0:
            return len(tokenizer(full_text[:idx], add_special_tokens=False)["input_ids"])
    return start


def build_generation_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return apply_mechet_chat_template(tokenizer, messages, add_generation_prompt=True)
