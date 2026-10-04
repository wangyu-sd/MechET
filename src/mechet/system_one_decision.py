"""Prefill-only decision heads for System-One retrosynthesis.

This module is intentionally small and chemistry-agnostic at the readout layer.
It reuses a causal LM only as a state encoder: no autoregressive tokens are
sampled.  The executor remains authoritative for chemistry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class DecisionOutput:
    source_logits: torch.Tensor
    sink_logits: torch.Tensor
    pair_logits: torch.Tensor


OPTION_ANCHOR = "SYSTEM_ONE_ATOM_OPTIONS: "


def append_option_anchors(
    messages: Sequence[dict[str, Any]], atom_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Place gold-independent atom handles after the complete current state.

    A causal LM's hidden state at an in-SMILES marker cannot see the atom text
    to its right.  Repeating only the handles at the end of the current user
    observation lets each option state attend to the full chemical state.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("the decision prefix must end with a user observation")
    if not atom_names:
        raise ValueError("at least one atom option is required")
    result = [dict(message) for message in messages]
    content = result[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("the current user observation must be text")
    if OPTION_ANCHOR in content:
        raise ValueError("option anchors are already present")
    result[-1]["content"] = (
        content + "\n" + OPTION_ANCHOR
        + " ".join(f"<{name}>" for name in atom_names)
    )
    return result


def locate_option_anchor_tokens(
    tokenizer: Any,
    prefix: str,
    atom_names: Sequence[str],
    *,
    offsets: Sequence[tuple[int, int]] | None = None,
) -> list[int]:
    """Find each post-state handle's final token using tokenizer offsets."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("option alignment requires a fast tokenizer")
    anchor_start = prefix.rfind(OPTION_ANCHOR)
    if anchor_start < 0:
        raise ValueError("rendered prefix lost option anchors")
    start = anchor_start + len(OPTION_ANCHOR)
    rendered = " ".join(f"<{name}>" for name in atom_names)
    if prefix[start : start + len(rendered)] != rendered:
        raise ValueError("rendered atom options do not match the inventory")
    if offsets is None:
        offsets = tokenizer(
            prefix, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
    output: list[int] = []
    cursor = start
    for name in atom_names:
        end = cursor + len(name) + 2
        indices = [i for i, (a, b) in enumerate(offsets) if a < end <= b]
        if len(indices) != 1:
            raise ValueError(f"cannot locate option handle <{name}>")
        output.append(indices[0])
        cursor = end + 1
    if len(set(output)) != len(output) or output != sorted(output):
        raise ValueError("option handles do not map to unique ordered tokens")
    return output


class OptionPointerHead(nn.Module):
    """Jev/Kev-style query-to-option pointer readout."""

    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.key = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.scale = 1.0 / math.sqrt(pointer_dim)
        self.temperature = 1.0

    def forward(self, query_state: torch.Tensor, option_states: torch.Tensor) -> torch.Tensor:
        logits = self.key(option_states.float()) @ self.query(query_state.float())
        logits = logits * self.scale
        if not self.training and self.temperature != 1.0:
            logits = logits / self.temperature
        return logits


class ElectronFlowDecisionHead(nn.Module):
    """Hierarchical source/sink decision head over current-state atom/bond options.

    Atom options use the marker-token hidden state.  Bond options use the mean
    of their endpoint states plus a learned type embedding.  The coupled pair
    score is the primary move-selection objective.
    """

    def __init__(self, hidden_size: int, pointer_dim: int = 256):
        super().__init__()
        self.atom = nn.Linear(hidden_size, hidden_size, bias=False)
        self.type_embed = nn.Embedding(2, hidden_size)
        self.source = OptionPointerHead(hidden_size, pointer_dim)
        self.sink = OptionPointerHead(hidden_size, pointer_dim)
        self.pair_source = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.pair_sink = nn.Linear(hidden_size, pointer_dim, bias=False)
        self.pair_scale = 1.0 / math.sqrt(pointer_dim)

    def _options(self, atom_states: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        atom = self.atom(atom_states.float())
        atom_options = atom + self.type_embed.weight[0]
        if pairs.numel() == 0:
            return atom_options
        bonds = 0.5 * (atom[pairs[:, 0]] + atom[pairs[:, 1]]) + self.type_embed.weight[1]
        return torch.cat((atom_options, bonds), dim=0)

    def forward(
        self,
        query_state: torch.Tensor,
        atom_states: torch.Tensor,
        source_pairs: torch.Tensor,
        sink_pairs: torch.Tensor,
    ) -> DecisionOutput:
        source_options = self._options(atom_states, source_pairs)
        sink_options = self._options(atom_states, sink_pairs)
        source_logits = self.source(query_state, source_options)
        sink_logits = self.sink(query_state, sink_options)
        pair_logits = (
            self.pair_source(source_options)
            @ self.pair_sink(sink_options).T
        ) * self.pair_scale
        pair_logits = pair_logits + source_logits[:, None] + sink_logits[None, :]
        return DecisionOutput(source_logits, sink_logits, pair_logits)


def multi_target_nll(logits: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Mean NLL of every required flow, not merely any one flow.

    A multi-flow event is one decision whose entire unordered pair set must be
    recovered. Summed target probability would reward only its easiest flow.
    """
    if not indices:
        raise ValueError("at least one target index is required")
    logp = torch.log_softmax(logits.reshape(-1).float(), dim=0)
    target = torch.tensor(sorted(set(indices)), dtype=torch.long, device=logits.device)
    return -logp[target].mean()
