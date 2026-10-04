"""Prefill-only decision heads for System-One retrosynthesis.

This module is intentionally small and chemistry-agnostic at the readout layer.
It reuses a causal LM only as a state encoder: no autoregressive tokens are
sampled.  The executor remains authoritative for chemistry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DecisionOutput:
    source_logits: torch.Tensor
    sink_logits: torch.Tensor
    pair_logits: torch.Tensor


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
    """Negative log probability mass assigned to one or more valid targets."""
    if not indices:
        raise ValueError("at least one target index is required")
    logp = torch.log_softmax(logits.reshape(-1).float(), dim=0)
    target = torch.tensor(sorted(set(indices)), dtype=torch.long, device=logits.device)
    return -torch.logsumexp(logp[target], dim=0)
