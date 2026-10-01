"""Exact assistant-only segment packing for a gated training microbenchmark.

Each packed sample has a block-diagonal causal mask and reset positions, so
no reaction can attend to another reaction. This does not modify the frozen
paper trainer until Qwen logits and throughput parity pass on GPU.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def pack_encoded_rows(rows: Iterable[Mapping[str, Any]], *, max_length: int) -> list[dict[str, Any]]:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    packed: list[dict[str, Any]] = []
    ids: list[int] = []
    labels: list[int] = []
    segments: list[int] = []
    for row in rows:
        new_ids = [int(value) for value in row["input_ids"]]
        new_labels = [int(value) for value in row["labels"]]
        if len(new_ids) != len(new_labels) or not new_ids or len(new_ids) > max_length:
            raise ValueError("invalid encoded row length or pack budget")
        if not any(value != -100 for value in new_labels):
            raise ValueError("assistant-only row has no supervised tokens")
        if ids and len(ids) + len(new_ids) > max_length:
            packed.append({"input_ids": ids, "labels": labels, "segments": segments})
            ids, labels, segments = [], [], []
        ids.extend(new_ids)
        labels.extend(new_labels)
        segments.append(len(new_ids))
    if ids:
        packed.append({"input_ids": ids, "labels": labels, "segments": segments})
    return packed


def block_causal_inputs(row: Mapping[str, Any], *, device: Any, dtype: Any):
    """Return input/label IDs, reset positions, and a 4D additive SDPA mask."""
    import torch

    ids = list(row["input_ids"])
    labels = list(row["labels"])
    segments = [int(length) for length in row["segments"]]
    if sum(segments) != len(ids) or len(ids) != len(labels) or any(length <= 0 for length in segments):
        raise ValueError("invalid packed segment boundaries")
    n = len(ids)
    mask = torch.full((1, 1, n, n), torch.finfo(dtype).min, dtype=dtype, device=device)
    positions = []
    offset = 0
    for length in segments:
        triangular = torch.tril(torch.ones((length, length), dtype=torch.bool, device=device))
        block = mask[0, 0, offset:offset + length, offset:offset + length]
        block.masked_fill_(triangular, 0)
        positions.extend(range(length))
        offset += length
    return {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
        "position_ids": torch.tensor([positions], dtype=torch.long, device=device),
        "attention_mask": mask,
        "use_cache": False,
    }
