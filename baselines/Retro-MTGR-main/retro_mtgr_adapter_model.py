"""Trainable Retro-MTGR adapter model used by the MechET diagnostics."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class GraphLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(adjacency @ self.linear(features)))


class RetroMTGRAdapter(nn.Module):
    """Shared graph encoder with center, endpoint-LG, and alignment objectives."""

    def __init__(self, atom_feature_dim: int, hidden_dim: int, label_count: int) -> None:
        super().__init__()
        self.atom_feature_dim = atom_feature_dim
        self.hidden_dim = hidden_dim
        self.label_count = label_count
        self.gcn1 = GraphLayer(atom_feature_dim, hidden_dim)
        self.gcn2 = GraphLayer(hidden_dim, hidden_dim)
        pair_dim = hidden_dim * 5 + 6
        endpoint_dim = hidden_dim * 3 + 6
        self.center_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.leaving_group_head = nn.Sequential(
            nn.Linear(endpoint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, label_count),
        )

    def encode(self, graph: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gcn1(graph["x"], graph["adj"])
        hidden = self.gcn2(hidden, graph["adj"])
        return hidden, hidden.mean(dim=0)

    def encode_batch(self, graph: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.gcn1(graph["x"], graph["adj"])
        hidden = self.gcn2(hidden, graph["adj"])
        mask = graph["node_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden, pooled

    def score_bonds(
        self,
        graph: dict[str, Any],
        hidden: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        if not graph["bonds"]:
            return torch.empty(0, device=hidden.device)
        rows = []
        for bond_index, (left, right) in enumerate(graph["bonds"]):
            left_hidden = hidden[left]
            right_hidden = hidden[right]
            rows.append(torch.cat((
                left_hidden,
                right_hidden,
                torch.abs(left_hidden - right_hidden),
                left_hidden * right_hidden,
                pooled,
                graph["bond_features"][bond_index],
            )))
        return self.center_head(torch.stack(rows)).squeeze(-1)

    def score_leaving_groups(
        self,
        graph: dict[str, Any],
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        bond_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = graph["bonds"][bond_index]
        bond_features = graph["bond_features"][bond_index]
        left_input = torch.cat((hidden[left], hidden[right], pooled, bond_features))
        right_input = torch.cat((hidden[right], hidden[left], pooled, bond_features))
        return (
            self.leaving_group_head(left_input),
            self.leaving_group_head(right_input),
        )

    def score_bonds_batch(
        self,
        graph: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        left = graph["bonds"][..., 0]
        right = graph["bonds"][..., 1]
        left_hidden = hidden[batch_indices, left]
        right_hidden = hidden[batch_indices, right]
        pooled_expanded = pooled[:, None, :].expand(-1, left.shape[1], -1)
        inputs = torch.cat((
            left_hidden,
            right_hidden,
            torch.abs(left_hidden - right_hidden),
            left_hidden * right_hidden,
            pooled_expanded,
            graph["bond_features"],
        ), dim=-1)
        logits = self.center_head(inputs).squeeze(-1)
        return logits.masked_fill(~graph["bond_mask"], -1e9)

    def score_leaving_groups_batch(
        self,
        graph: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        pooled: torch.Tensor,
        center_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        selected_bonds = graph["bonds"][batch_indices, center_indices]
        left = selected_bonds[:, 0]
        right = selected_bonds[:, 1]
        left_hidden = hidden[batch_indices, left]
        right_hidden = hidden[batch_indices, right]
        bond_features = graph["bond_features"][batch_indices, center_indices]
        left_input = torch.cat((left_hidden, right_hidden, pooled, bond_features), dim=-1)
        right_input = torch.cat((right_hidden, left_hidden, pooled, bond_features), dim=-1)
        return self.leaving_group_head(left_input), self.leaving_group_head(right_input)

    def configuration(self) -> dict[str, int]:
        return {
            "atom_feature_dim": self.atom_feature_dim,
            "hidden_dim": self.hidden_dim,
            "label_count": self.label_count,
        }
