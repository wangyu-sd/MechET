"""Assistant-only loss encoding and collator utilities for ORBIT-Qwen SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mechet.chat_template import find_assistant_start as _find_assistant_start


def _flatten_token_sequence(value: Any, *, field: str) -> list[int]:
    """Return a one-dimensional integer token sequence or raise a clear error."""
    if value is None:
        raise ValueError(f"{field} is missing")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list of token ids")
    unwraps = 0
    while len(value) == 1 and isinstance(value[0], list):
        unwraps += 1
        if unwraps > 1:
            raise ValueError(f"{field} excessive nesting")
        value = value[0]
    if any(isinstance(item, list) for item in value):
        raise ValueError(f"{field} excessive nesting")
    try:
        return [int(item) for item in value]
    except Exception as exc:
        raise ValueError(f"{field} contains non-integer token ids") from exc


def encode_assistant_only(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_length: int = 2048,
) -> dict[str, list[int]]:
    from mechet.chat_template import apply_mechet_chat_template

    text = apply_mechet_chat_template(tokenizer, messages, add_generation_prompt=False)
    assistant_start = _find_assistant_start(tokenizer, messages)
    encoded = tokenizer(text, truncation=False, add_special_tokens=False)
    full_ids = _flatten_token_sequence(encoded["input_ids"], field="input_ids")
    assistant_start = min(max(assistant_start, 0), len(full_ids))
    prompt_ids = full_ids[:assistant_start]
    assistant_ids = full_ids[assistant_start:]
    if len(assistant_ids) >= max_length:
        input_ids = assistant_ids[:max_length]
        assistant_start = 0
    else:
        prompt_budget = max_length - len(assistant_ids)
        prompt_tail = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_tail + assistant_ids
        assistant_start = len(prompt_tail)
    labels = [-100] * len(input_ids)
    for index in range(assistant_start, len(input_ids)):
        labels[index] = input_ids[index]
    if len(input_ids) != len(labels):
        raise ValueError("input_ids and labels length mismatch")
    if not any(value != -100 for value in labels):
        raise ValueError("truncation removed all assistant labels")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class AssistantOnlyCollator:
    tokenizer: Any
    max_length: int = 2048

    def build_labels(self, input_ids: list[int], assistant_start: int) -> list[int]:
        labels = [-100] * len(input_ids)
        for index in range(assistant_start, len(input_ids)):
            labels[index] = input_ids[index]
        return labels

    def _pad_id(self) -> int:
        pad = getattr(self.tokenizer, "pad_token_id", None)
        if pad is None:
            pad = getattr(self.tokenizer, "eos_token_id", None)
        return int(pad if pad is not None else 0)

    def _from_text_features(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row_index, item in enumerate(features):
            assistant_start = int(item.get("assistant_start", 0))
            for col_index in range(labels.shape[1]):
                if col_index < assistant_start:
                    labels[row_index, col_index] = -100
        batch["labels"] = labels
        return batch

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("empty feature batch")
        if "text" in features[0]:
            return self._from_text_features(features)

        import torch

        max_len = max(len(_flatten_token_sequence(item["input_ids"], field="input_ids")) for item in features)
        pad_id = self._pad_id()
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            input_ids = _flatten_token_sequence(item["input_ids"], field="input_ids")
            attention_mask = _flatten_token_sequence(
                item.get("attention_mask", [1] * len(input_ids)),
                field="attention_mask",
            )
            labels = _flatten_token_sequence(item["labels"], field="labels")
            if len(input_ids) != len(labels):
                raise ValueError("input_ids and labels length mismatch")
            if len(attention_mask) != len(input_ids):
                raise ValueError("input_ids and attention_mask length mismatch")
            if not any(value != -100 for value in labels):
                raise ValueError("batch item has zero supervised label tokens")
            pad = max_len - len(input_ids)
            batch["input_ids"].append(input_ids + [pad_id] * pad)
            batch["attention_mask"].append(attention_mask + [0] * pad)
            batch["labels"].append(labels + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def find_assistant_start(tokenizer, messages: list[dict[str, str]]) -> int:
    return _find_assistant_start(tokenizer, messages)
