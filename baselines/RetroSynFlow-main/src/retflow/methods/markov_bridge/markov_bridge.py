from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from retflow.datasets.retro import RetrosynthesisInfo
from retflow.methods.loss_functions import TrainLossDiscrete, TrainLossVLB
from retflow.methods.markov_bridge.noise_schedule import (
    InterpolationTransition, PredefinedNoiseScheduleDiscrete)
from retflow.methods.method import Method
from retflow.methods.method_utils import sample_discrete_features
from retflow.models.model import Model
from retflow.utils.wrappers import GraphWrapper


@dataclass
class GraphMarkovBridge(Method):
    steps: int = 50
    noise_schedule: str = "cosine"
    lambda_train: float = 5.0
    vlb_loss: bool = True

    def setup(self, dataset_info: RetrosynthesisInfo, model: Model, device: str):
        self.dataset_info = dataset_info
        self.input_dim = dataset_info.input_dim
        self.output_dim = dataset_info.output_dim

        torch_model = model.load_model(dataset_info.input_dim, dataset_info.output_dim)

        self.loss_function = (
            TrainLossVLB(lambda_train=self.lambda_train)
            if self.vlb_loss
            else TrainLossDiscrete(edge_weight=self.lambda_train)
        )

        self.noise_sched = PredefinedNoiseScheduleDiscrete(
            noise_schedule=self.noise_schedule, timesteps=self.steps, device=device
        )

        self.transition_model = InterpolationTransition(
            x_classes=dataset_info.output_dim.node_dim,
            e_classes=dataset_info.output_dim.edge_dim,
            y_classes=dataset_info.output_dim.y_dim,
        )

        return torch_model

    def compute_loss(
        self, reactants, pred, node_mask, noisy_graph, t
    ) -> Tuple[torch.Tensor]:
        if self.vlb_loss:
            true_pX, true_pE = self.compute_q_zs_given_q_zt(
                noisy_graph, reactants, node_mask, t=t
            )
            pred_pX, pred_pE = self.compute_p_zs_given_p_zt(
                noisy_graph, pred, node_mask, t=t
            )

            loss, node_loss, edge_loss = self.loss_function(
                masked_pred_X=pred_pX,
                masked_pred_E=pred_pE,
                true_X=true_pX,
                true_E=true_pE,
            )
        else:
            loss, node_loss, edge_loss = self.loss_function(
                masked_pred_X=pred.X,
                masked_pred_E=pred.E,
                true_X=reactants.X,
                true_E=reactants.E,
            )
        return loss, node_loss, edge_loss

    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        # Sample a timestep t.
        lowest_t = 0
        t_int = torch.randint(
            lowest_t, self.steps + 1, size=(X.size(0), 1), device=X.device
        ).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.steps

        alpha_t_bar = self.noise_sched.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(
            alpha_bar_t=alpha_t_bar,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T,
            node_mask=node_mask,
            device=X.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        assert len(Qtb.X.shape) == 4 and len(Qtb.E.shape) == 5
        assert (abs(Qtb.X.sum(dim=3) - 1.0) < 1e-4).all(), Qtb.X.sum(dim=3) - 1
        assert (abs(Qtb.E.sum(dim=4) - 1.0) < 1e-4).all()

        probX = (X.unsqueeze(-2) @ Qtb.X).squeeze(-2)  # (bs, n, dx_out)
        probE = (E.unsqueeze(-2) @ Qtb.E).squeeze(-2)  # (bs, n, n, de_out)

        sampled_t = sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.output_dim.node_dim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.output_dim.edge_dim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        noisy_graph = GraphWrapper(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        return noisy_graph, node_mask, t_float

    @torch.no_grad()
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

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.steps)), total=self.steps):
            s_array = s_int * torch.ones((len(X), 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.steps
            t_norm = t_array / self.steps

            # Sample z_s
            t_norm = 1 - t_norm
            beta_t = self.noise_sched(t_normalized=t_norm)
            noisy_graph = GraphWrapper(X, E, y)
            X_p, E_p = predictor(noisy_graph, node_mask, context, t_norm)
            pred = GraphWrapper(X_p, E_p).mask(node_mask)

            sampled_s, _, _, _ = self.sample_p_zs_given_zt(
                pred=pred,
                beta_t=beta_t,
                X_t=X,
                E_t=E,
                y_t=y,
                X_T=initial_graph.X,
                E_T=initial_graph.E,
                y_T=initial_graph.y,
                node_mask=node_mask,
            )

            sampled_s.X = sampled_s.X * modifiable_nodes + initial_graph.X * fixed_nodes
            sampled_s = sampled_s.mask(node_mask)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        return X, E

    def sample_p_zs_given_zt(
        self,
        pred,
        beta_t,
        X_t,
        E_t,
        y_t,
        X_T,
        E_T,
        y_T,
        node_mask,
    ):
        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=pred_X,
            E_T=pred_E,
            y_T=y_T,
            node_mask=node_mask,
            device=pred_X.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        # Edge transition probabilities
        E_T_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t-1, d_t)
        unnormalized_prob_E = E_T_flat.unsqueeze(-2) @ Qt_E_flat  # bs, N, 1, d_t
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)  # bs, N, d_t
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.output_dim.node_dim).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.output_dim.edge_dim).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = GraphWrapper(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = GraphWrapper(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        # Likelihood
        node_log_likelihood = torch.log(prob_X) + torch.log(pred_X)
        node_log_likelihood = (node_log_likelihood * X_s).sum(-1).sum(-1)

        edge_log_likelihood = torch.log(prob_E) + torch.log(pred_E)
        edge_log_likelihood = (edge_log_likelihood * E_s).sum(-1).sum(-1).sum(-1)

        return (
            out_one_hot.mask(node_mask).type_as(y_t),
            out_discrete.mask(node_mask, collapse=True).type_as(y_t),
            node_log_likelihood,
            edge_log_likelihood,
        )

    def compute_q_zs_given_q_zt(self, z_t, z_T, node_mask, t):
        X_t = z_t.X.to(torch.float32)
        E_t = z_t.E.to(torch.float32)

        # Hack: in direct MB we consider flipped time flow
        bs, n = X_t.shape[:2]
        beta_t = self.noise_sched(t_normalized=t)  # (bs, 1)

        # Normalize predictions
        X_T = z_T.X.to(torch.float32)  # bs, n, d0
        E_T = z_T.E.to(torch.float32)  # bs, n, n, d0
        y_T = z_T.y

        # Compute transition matrices given prediction
        Qt = self.transition_model.get_Qt(
            beta_t=beta_t,
            X_T=X_T,
            E_T=E_T,
            y_T=y_T,
            node_mask=node_mask,
            device=X_T.device,
        )  # (bs, n, dx_in, dx_out), (bs, n, n, de_in, de_out)

        # Node transition probabilities
        unnormalized_prob_X = X_t.unsqueeze(-2) @ Qt.X  # bs, n, 1, d_t
        unnormalized_prob_X = unnormalized_prob_X.squeeze(-2)  # bs, n, d_t
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )  # bs, n, d_t-1

        # Edge transition probabilities
        E_T_flat = E_t.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t)
        Qt_E_flat = Qt.E.flatten(start_dim=1, end_dim=2)  # (bs, N, d_t-1, d_t)
        unnormalized_prob_E = E_T_flat.unsqueeze(-2) @ Qt_E_flat  # bs, N, 1, d_t
        unnormalized_prob_E = unnormalized_prob_E.squeeze(-2)  # bs, N, d_t
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, E_T.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E

    def compute_p_zs_given_p_zt(self, z_t, pred, node_mask, t):
        p_X_T = F.softmax(pred.X, dim=-1)  # bs, n, d
        p_E_T = F.softmax(pred.E, dim=-1)  # bs, n, n, d

        prob_X = torch.zeros_like(p_X_T)  # bs, n, d
        prob_E = torch.zeros_like(p_E_T)  # bs, n, n, d

        for i in range(self.output_dim.node_dim):
            X_T_i = F.one_hot(
                torch.ones_like(p_X_T[..., 0]).long() * i,
                num_classes=self.output_dim.node_dim,
            ).float()
            E_T_i = F.one_hot(
                torch.zeros_like(p_E_T[..., 0]).long(),
                num_classes=self.output_dim.edge_dim,
            ).float()
            z_T = GraphWrapper(X_T_i, E_T_i)
            prob_X_i, _ = self.compute_q_zs_given_q_zt(
                z_t, z_T, node_mask, t
            )  # bs, n, d
            prob_X += prob_X_i * p_X_T[..., i].unsqueeze(-1)  # bs, n, d

        for i in range(self.output_dim.edge_dim):
            X_T_i = F.one_hot(
                torch.zeros_like(p_X_T[..., 0]).long(),
                num_classes=self.output_dim.node_dim,
            ).float()
            E_T_i = F.one_hot(
                torch.ones_like(p_E_T[..., 0]).long() * i,
                num_classes=self.output_dim.edge_dim,
            ).float()
            z_T = GraphWrapper(X_T_i, E_T_i)
            _, prob_E_i = self.compute_q_zs_given_q_zt(
                z_t, z_T, node_mask, t
            )  # bs, n, n, d
            prob_E += prob_E_i * p_E_T[..., i].unsqueeze(-1)  # bs, n, n, d

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        return prob_X, prob_E
