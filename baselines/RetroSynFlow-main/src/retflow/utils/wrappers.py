from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class GraphDimensions:
    node_dim: int
    edge_dim: int
    y_dim: int


@dataclass
class GraphModelLayerInfo:
    dim_X: int
    dim_E: int
    dim_y: int


class GraphWrapper:
    def __init__(self, X, E, y=None):
        self.X = X
        self.E = E
        self.y = (
            y
            if y is not None
            else torch.zeros(
                size=(self.X.shape[0], 0), dtype=torch.float, device=X.device
            )
        )

    def type_as(self, x: torch.Tensor):
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

        return self

    def clone(self):
        if self.y is not None:
            return GraphWrapper(self.X.clone(), self.E.clone(), self.y.clone())
        else:
            return GraphWrapper(self.X.clone(), self.E.clone())

    def detach(self):
        if self.y is not None:
            return GraphWrapper(
                self.X.clone().detach(),
                self.E.clone().detach(),
                self.y.clone().detach(),
            )
        else:
            return GraphWrapper(self.X.clone().detach(), self.E.clone().detach())

class GraphModelWrapper:
    def __init__(
        self,
        torch_model: nn.Module,
        extra_graph_features: "ExtraFeatures",
        extra_mol_features: "ExtraMolecularFeatures" = None
    ) -> None:
        self.torch_model = torch_model
        self.extra_graph_features = extra_graph_features
        self.extra_mol_features = extra_mol_features

    def __call__(self, noisy_graph: GraphWrapper, node_mask, context, t) -> Any:
        extra_data = self.compute_extra_data(noisy_graph, node_mask, context)
        extra_data.y = torch.cat((extra_data.y, t), dim=-1)

        if self.extra_mol_features is not None:
            extra_mol_data = self.extra_mol_features(noisy_graph.X, noisy_graph.E, node_mask)

            X = torch.cat((noisy_graph.X, extra_data.X, extra_mol_data.X), dim=-1).float()
            E = torch.cat((noisy_graph.E, extra_data.E), dim=-1).float()
            y = torch.hstack((noisy_graph.y, extra_data.y, extra_mol_data.y)).float()
        else:
            X = torch.cat((noisy_graph.X, extra_data.X), dim=-1).float()
            E = torch.cat((noisy_graph.E, extra_data.E), dim=-1).float()
            y = torch.hstack((noisy_graph.y, extra_data.y)).float()
        return self.torch_model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_graph: GraphWrapper, node_mask, context=None):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_graph_features(
            noisy_graph.X, noisy_graph.E, node_mask
        )

        extra_X = extra_features.X
        extra_E = extra_features.E
        extra_y = extra_features.y

        if context is not None:
            if isinstance(context, list):
                for c in context:
                    extra_X = torch.cat((extra_X, c.X), dim=-1)
                    extra_E = torch.cat((extra_E, c.E), dim=-1)
            else:
                extra_X = torch.cat((extra_X, context.X), dim=-1)
                extra_E = torch.cat((extra_E, context.E), dim=-1)

        return GraphWrapper(X=extra_X, E=extra_E, y=extra_y)




