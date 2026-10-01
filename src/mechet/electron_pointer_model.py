"""Trainable source/sink site scorer; checkpoint-compatible with PR71 pilot."""
from __future__ import annotations

import math
import torch
from torch import nn


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, width: int = 192):
        super().__init__()
        self.atom = nn.Linear(hidden_size, width)
        self.query = nn.Linear(hidden_size, width)
        self.type_embed = nn.Embedding(2, width)
        self.source = nn.Sequential(nn.LayerNorm(width), nn.Tanh(), nn.Linear(width, 1))
        self.sink = nn.Sequential(nn.LayerNorm(width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, query, atom_states, src_pairs, sink_pairs):
        atom = self.atom(atom_states.float())
        query = self.query(query.float())

        def scores(pairs, scorer):
            atom_candidate = atom + self.type_embed.weight[0]
            if pairs.numel():
                bond_candidate = (atom[pairs[:, 0]] + atom[pairs[:, 1]]) * 0.5 + self.type_embed.weight[1]
                choices = torch.cat((atom_candidate, bond_candidate), 0)
            else:
                choices = atom_candidate
            return scorer(choices + query).squeeze(-1)

        return scores(src_pairs, self.source), scores(sink_pairs, self.sink)


def pairs_for_observation(observation, device):
    source = torch.tensor(observation.bonds, dtype=torch.long, device=device).reshape(-1, 2)
    sink = torch.triu_indices(len(observation.atom_names), len(observation.atom_names), offset=1, device=device).T.contiguous()
    return source, sink


def pair_recall_metrics(source_logits, sink_logits, source_indices, sink_indices):
    """Recall of *coupled* source/sink moves, not two independent marginals.

    The highest K Cartesian pairs can only use a top-K source and a top-K
    sink, so this avoids constructing a potentially large full pair matrix.
    """
    if len(source_indices) != len(sink_indices) or not source_indices:
        raise ValueError("each electron move needs one source and one sink")
    kmax = min(8, source_logits.numel() * sink_logits.numel())
    src = source_logits.detach().float().topk(min(8, source_logits.numel())).indices.tolist()
    sink = sink_logits.detach().float().topk(min(8, sink_logits.numel())).indices.tolist()
    ranked = sorted(
        ((i, j) for i in src for j in sink),
        key=lambda pair: (-float(source_logits[pair[0]] + sink_logits[pair[1]]), pair),
    )[:kmax]
    gold = set(zip(source_indices, sink_indices, strict=True))
    result = {}
    for k in (1, 4, 8):
        selected = set(ranked[:k])
        result[f"pair_r{k}"] = int(bool(gold & selected))
        result[f"pair_all_r{k}"] = int(gold <= selected)
    return result


class CoupledPointerHead(PointerHead):
    """Adds a source-conditioned sink term to the independent site heads."""

    def __init__(self, hidden_size: int, width: int = 192):
        super().__init__(hidden_size, width=width)
        self.pair_source = nn.Linear(width, width, bias=False)
        self.pair_sink = nn.Linear(width, width, bias=False)

    def forward(self, query, atom_states, src_pairs, sink_pairs):
        source_logits, sink_logits = super().forward(query, atom_states, src_pairs, sink_pairs)
        atom = self.atom(atom_states.float())
        query_vector = self.query(query.float())
        atom_candidate = atom + self.type_embed.weight[0]

        def candidates(pairs):
            if not pairs.numel():
                return atom_candidate + query_vector
            bonds = (atom[pairs[:, 0]] + atom[pairs[:, 1]]) * 0.5 + self.type_embed.weight[1]
            return torch.cat((atom_candidate, bonds), 0) + query_vector

        source = self.pair_source(candidates(src_pairs))
        sink = self.pair_sink(candidates(sink_pairs))
        pair_logits = source @ sink.T / math.sqrt(source.shape[-1])
        pair_logits = pair_logits + source_logits[:, None] + sink_logits[None, :]
        return source_logits, sink_logits, pair_logits


def coupled_pair_recall_metrics(pair_logits, source_indices, sink_indices):
    if len(source_indices) != len(sink_indices) or not source_indices:
        raise ValueError("each electron move needs one source and one sink")
    flat = pair_logits.detach().float().reshape(-1)
    ranked = flat.topk(min(8, flat.numel())).indices.tolist()
    sink_count = pair_logits.shape[1]
    gold = {int(i) * sink_count + int(j) for i, j in zip(source_indices, sink_indices, strict=True)}
    output = {}
    for k in (1, 4, 8):
        chosen = set(ranked[:k])
        output[f"pair_r{k}"] = int(bool(gold & chosen))
        output[f"pair_all_r{k}"] = int(gold <= chosen)
    return output
