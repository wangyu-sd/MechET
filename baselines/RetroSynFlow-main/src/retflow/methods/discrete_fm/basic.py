from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from retflow.datasets.retro import RetrosynthesisInfo
from retflow.methods.discrete_fm.scheduler import (LinearTimeScheduler,
                                                   TimeScheduler)
from retflow.methods.loss_functions import TrainLossDiscrete
from retflow.methods.method import Method
from retflow.methods.method_utils import pad_t_like_x, sample_discrete_features
from retflow.methods.time_sampler import TimeSampler, UniformTimeSampler
from retflow.models.model import Model
from retflow.utils.wrappers import GraphWrapper


@dataclass
class GraphDiscreteFM(Method):
    steps: int = 50
    edge_time_sched: TimeScheduler = LinearTimeScheduler()
    node_time_sched: TimeScheduler = LinearTimeScheduler()
    time_sampler: TimeSampler = UniformTimeSampler()
    edge_weight_loss: float = 1.0

    def setup(self, dataset_info: RetrosynthesisInfo, model: Model, device: str):
        self.dataset_info = dataset_info
        self.input_dim = dataset_info.input_dim
        self.output_dim = dataset_info.output_dim

        torch_model = model.load_model(dataset_info.input_dim, dataset_info.output_dim)

        self.loss_function = TrainLossDiscrete(edge_weight=self.edge_weight_loss)

        return torch_model

    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        t = self.time_sampler.sample(
            num_samples=X.size(0), dtype=torch.float32, device=X.device
        )
        edge_kappa = self.edge_time_sched.kappa(t)
        node_kappa = self.node_time_sched.kappa(t)

        padded_edge_kappa = pad_t_like_x(edge_kappa, E)
        prob_E = torch.clamp(
            (1 - padded_edge_kappa) * E + padded_edge_kappa * E_T, min=0.0, max=1.0
        )

        padded_node_kappa = pad_t_like_x(node_kappa, X)
        prob_X = torch.clamp(
            (1 - padded_node_kappa) * X + padded_node_kappa * X_T, min=0.0, max=1.0
        )

        sampled_t = sample_discrete_features(
            probX=prob_X, probE=prob_E, node_mask=node_mask
        )

        E_t = F.one_hot(sampled_t.E, num_classes=self.output_dim.edge_dim)
        X_t = F.one_hot(sampled_t.X, num_classes=self.output_dim.node_dim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        noisy_graph = GraphWrapper(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return noisy_graph, node_mask, t.unsqueeze(-1)

    def compute_loss(self, reactants, pred, node_mask, noisy_graph, t_float):
        loss, node_loss, edge_loss = self.loss_function(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=reactants.X,
            true_E=reactants.E,
        )
        return loss, node_loss, edge_loss

    def sample(self, initial_graph, node_mask, context, predictor: Callable):
        # Masks for fixed and modifiable nodes
        fixed_nodes = (initial_graph.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (initial_graph.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X, E, y = (
            initial_graph.X,
            initial_graph.E,
            torch.empty((node_mask.shape[0], 0), device=initial_graph.X.device),
        )

        assert (E == torch.transpose(E, 1, 2)).all()

        t = 0.0
        h = 1.0 / self.steps
        pbar = tqdm(total=1 - h)
        while t < 1 - h:
            tb = t * torch.ones((X.size(0),)).to(X.device)
            hb = h * torch.ones((X.size(0),)).to(X.device)

            hb_node = torch.minimum(
                hb,
                (1 - self.node_time_sched.kappa(tb))
                / self.node_time_sched.kappa_prime(tb),
            )

            hb_edge = torch.minimum(
                hb,
                (1 - self.edge_time_sched.kappa(tb))
                / self.edge_time_sched.kappa_prime(tb),
            )

            noisy_graph = GraphWrapper(X, E, y)
            X_p, E_p = predictor(noisy_graph, node_mask, context, tb.unsqueeze(-1))
            pred = GraphWrapper(X_p, E_p, y=initial_graph.y).mask(node_mask)
            X_ut = self._compute_node_vector_field(X, pred.X, tb)
            E_ut = self._compute_edge_vector_field(E, pred.E, tb)

            prob_X = X + pad_t_like_x(hb_node, X) * X_ut
            prob_E = E + pad_t_like_x(hb_edge, E) * E_ut

            prob_X[torch.sum(prob_X, dim=-1) == 0] = 1e-5
            prob_X = prob_X / torch.sum(prob_X, dim=-1, keepdim=True)

            prob_E[torch.sum(prob_E, dim=-1) == 0] = 1e-5
            prob_E = prob_E / torch.sum(prob_E, dim=-1, keepdim=True)

            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
            assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

            sampled_t = sample_discrete_features(
                probX=prob_X, probE=prob_E, node_mask=node_mask
            )
            # Categorical outputs tokens so we must convert to one hot
            E_raw = F.one_hot(sampled_t.E, num_classes=self.output_dim.edge_dim)
            X_raw = F.one_hot(sampled_t.X, num_classes=self.output_dim.node_dim)

            assert (E_raw == torch.transpose(E_raw, 1, 2)).all()
            assert (X_raw.shape == X.shape) and (E_raw.shape == E.shape)

            graph = (
                GraphWrapper(X_raw, E_raw, y).mask(node_mask).type_as(initial_graph.X)
            )

            graph.X = graph.X * modifiable_nodes + initial_graph.X * fixed_nodes
            graph = graph.mask(node_mask)
            X, E, y = graph.X, graph.E, graph.y

            t += hb_edge[0].item()
            pbar.update(hb_edge[0].item())
        pbar.update(hb_node[0].item())
        pbar.close()

        final_graph = GraphWrapper(X, E, y)
        final_graph = final_graph.mask(node_mask, collapse=True)
        X, E, y = final_graph.X, final_graph.E, final_graph.y

        return X, E

    def _compute_node_vector_field(self, X_t, X_1_logits, t):
        X_1_probs = F.softmax(X_1_logits, dim=-1)
        mult_factor = self.node_time_sched.kappa_prime(t) / (
            1.0 - self.node_time_sched.kappa(t)
        )

        return pad_t_like_x(mult_factor, X_t) * (X_1_probs - X_t)

    def _compute_edge_vector_field(self, E_t, E_1_logits, t):
        E_1_probs = F.softmax(E_1_logits, dim=-1)
        mult_factor = self.edge_time_sched.kappa_prime(t) / (
            1.0 - self.edge_time_sched.kappa(t)
        )
        return pad_t_like_x(mult_factor, E_t) * (E_1_probs - E_t)

