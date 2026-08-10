"""Assistant-only supervision masks that do not depend on template generation blocks."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def flatten_token_ids(value: Any, *, field: str = "input_ids") -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of token ids")
    while len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if any(isinstance(item, list) for item in value):
        raise ValueError(f"{field} contains nested token ids")
    return [int(item) for item in value]


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a Qwen-style chat once, tolerating minor Transformers API drift."""

    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools:
        kwargs["tools"] = tools
    attempts: list[dict[str, Any]] = [
        {"enable_thinking": False, **kwargs},
        dict(kwargs),
    ]
    if tools:
        no_tools = dict(kwargs)
        no_tools.pop("tools", None)
        attempts.extend(
            [
                {"enable_thinking": False, **no_tools},
                no_tools,
            ]
        )
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return str(tokenizer.apply_chat_template(messages, **attempt))
        except TypeError as exc:
            last_error = exc
        try:
            return str(
                tokenizer.apply_chat_template(conversation=messages, **attempt)
            )
        except TypeError as exc:
            last_error = exc
    raise last_error or TypeError("chat template rendering failed")


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer did not return input_ids")
    return flatten_token_ids(encoded["input_ids"])


def _find_subsequence(
    values: list[int], needle: list[int], start: int = 0
) -> int:
    if not needle:
        return -1
    stop = len(values) - len(needle) + 1
    for index in range(max(start, 0), max(stop, 0)):
        if values[index : index + len(needle)] == needle:
            return index
    return -1


def qwen_chatml_assistant_mask(
    tokenizer: Any,
    input_ids: list[int],
    *,
    expected_assistant_turns: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Find assistant spans directly in the final ChatML token sequence.

    Qwen3's shipped chat template does not provide a ``{% generation %}`` block,
    so Transformers cannot produce an assistant token mask. Instead of rendering
    prefixes repeatedly, this function scans the already-tokenized final sequence
    for the ChatML assistant/end markers. This avoids prefix/full-template drift
    around tool calls and thinking-mode branches.
    """

    if expected_assistant_turns <= 0:
        raise ValueError("conversation has no assistant messages")
    start_marker = tokenize_text(tokenizer, "<|im_start|>assistant")
    end_marker = tokenize_text(tokenizer, "<|im_end|>")
    if not start_marker or not end_marker:
        raise ValueError("QWEN_CHATML_MARKERS_UNAVAILABLE")

    mask = [0] * len(input_ids)
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        marker_start = _find_subsequence(input_ids, start_marker, cursor)
        if marker_start < 0:
            break
        content_start = marker_start + len(start_marker)
        marker_end = _find_subsequence(input_ids, end_marker, content_start)
        if marker_end < 0:
            raise ValueError(
                f"QWEN_ASSISTANT_END_MARKER_MISSING:{marker_start}"
            )
        content_end = marker_end + len(end_marker)
        for index in range(content_start, content_end):
            mask[index] = 1
        spans.append((content_start, content_end))
        cursor = content_end

    if len(spans) != expected_assistant_turns:
        raise ValueError(
            "QWEN_ASSISTANT_SPAN_COUNT_MISMATCH: "
            f"expected={expected_assistant_turns} observed={len(spans)}"
        )
    if not any(mask):
        raise ValueError("QWEN_ASSISTANT_MASK_EMPTY")
    return mask, spans


def encode_assistant_only_conversation(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_length: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    messages = [dict(item) for item in row.get("messages") or []]
    if not messages:
        raise ValueError("row has no messages")
    tools = [dict(item) for item in row.get("tools") or []]
    rendered = render_chat(
        tokenizer,
        messages,
        tools=tools or None,
        add_generation_prompt=False,
    )
    input_ids = tokenize_text(tokenizer, rendered)
    assistant_turns = sum(
        str(message.get("role") or "") == "assistant" for message in messages
    )
    mask, spans = qwen_chatml_assistant_mask(
        tokenizer,
        input_ids,
        expected_assistant_turns=assistant_turns,
    )
    labels = [
        token_id if supervised else -100
        for token_id, supervised in zip(input_ids, mask)
    ]
    supervised_tokens = sum(value != -100 for value in labels)
    if supervised_tokens <= 0:
        raise ValueError("row has zero supervised assistant tokens")
    return (
        {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        },
        {
            "raw_length": len(input_ids),
            "exceeds_max_length": len(input_ids) > int(max_length),
            "assistant_turns": assistant_turns,
            "assistant_spans": [list(item) for item in spans],
            "supervised_tokens": supervised_tokens,
            "mask_method": "final_chatml_token_scan_v1",
        },
    )


def percentile_nearest_rank(values: Iterable[int], percentile: float) -> int:
    ordered = sorted(int(item) for item in values)
    if not ordered:
        return 0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be in [0, 1]")
    rank = max(1, int((percentile * len(ordered)) + 0.999999999))
    return ordered[min(rank - 1, len(ordered) - 1)]
