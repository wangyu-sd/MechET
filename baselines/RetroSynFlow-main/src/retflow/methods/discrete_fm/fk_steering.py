from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rdkit import Chem
from tqdm import tqdm

from retflow.datasets.retro import RetrosynthesisInfo
from retflow.methods.discrete_fm.basic import GraphDiscreteFM
from retflow.methods.discrete_fm.rewards import QEDReward, Reward
from retflow.methods.method_utils import pad_t_like_x, sample_discrete_features
from retflow.utils.data import build_molecule
from retflow.utils.wrappers import GraphWrapper


@dataclass
class FKSteeringDiscreteFM(GraphDiscreteFM):
    num_particles: int = 4
    resample_freq: int = 10
    lmbda: float = 2.0
    reward_min_value: float = 0.1
    potential_type: str = "add"
    initial_temperature: float = 1.1
    reward_fn: Reward = QEDReward()

    def setup(self, dataset_info, model, device):
        self.reward_fn.initialize_reward()
        return super().setup(dataset_info, model, device)

    def sample(self, initial_graph, node_mask, context, predictor):
        bs = len(initial_graph.X)
        self.population_rs = (
            torch.ones((bs, self.num_particles), device=initial_graph.X.device)
            * self.reward_min_value
        )
        self.product_of_potentials = torch.ones((bs, self.num_particles)).to(
            initial_graph.X.device
        )

        # Masks for fixed and modifiable nodes
        fixed_nodes = (initial_graph.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (initial_graph.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # z_T – starting state (product)
        X_start, E_start, y = (
            initial_graph.X,
            initial_graph.E,
            torch.empty((node_mask.shape[0], 0), device=initial_graph.X.device),
        )

        fixed_nodes_repeated = torch.repeat_interleave(
            fixed_nodes, self.num_particles, dim=0
        )
        modifiable_nodes_repeated = torch.repeat_interleave(
            modifiable_nodes, self.num_particles, dim=0
        )
        product_X_repeated = torch.repeat_interleave(
            initial_graph.X, self.num_particles, dim=0
        )

        X = torch.repeat_interleave(X_start, self.num_particles, dim=0)
        E = torch.repeat_interleave(E_start, self.num_particles, dim=0)
        y = torch.repeat_interleave(y, self.num_particles, dim=0)
        repeated_node_mask = torch.repeat_interleave(
            node_mask, self.num_particles, dim=0
        )

        product_mols = self.convert_graphs_to_mols(
            torch.argmax(X, dim=-1), torch.argmax(E, dim=-1), repeated_node_mask
        )
        product_smiles = [Chem.MolToSmiles(mol) for mol in product_mols]

        context.X = torch.repeat_interleave(context.X, self.num_particles, dim=0)
        context.E = torch.repeat_interleave(context.E, self.num_particles, dim=0)
        context.y = torch.repeat_interleave(context.y, self.num_particles, dim=0)

        assert (E == torch.transpose(E, 1, 2)).all()

        t = 0.0
        step = 0
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
            X_p, E_p = predictor(
                noisy_graph, repeated_node_mask, context, tb.unsqueeze(-1)
            )

            if step == 0:
                X_p = X_p / self.initial_temperature
                E_p = E_p / self.initial_temperature

            pred = GraphWrapper(X_p, E_p, y=initial_graph.y).mask(repeated_node_mask)
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
                probX=prob_X, probE=prob_E, node_mask=repeated_node_mask
            )
            if step % self.resample_freq == 0:
                E1_pred = torch.argmax(E_p, dim=-1)

                X1_pred = F.one_hot(
                    torch.argmax(X_p, dim=-1), num_classes=self.output_dim.node_dim
                )
                X1_pred = (
                    X1_pred * modifiable_nodes_repeated
                    + product_X_repeated * fixed_nodes_repeated
                )
                X1_pred = torch.argmax(X1_pred, dim=-1)

                pred_mols = self.convert_graphs_to_mols(
                    X1_pred, E1_pred, repeated_node_mask
                )
                pred_smiles = [Chem.MolToSmiles(mol) for mol in pred_mols]

                rewards = self.reward_fn.compute_reward(
                    pred_smiles, product_smiles, X.device
                )
                Xt, Et = self.resample(sampled_t.X, sampled_t.E, rewards, bs, False)
            else:
                Xt, Et = sampled_t.X, sampled_t.E
            # Categorical outputs tokens so we must convert to one hot
            E_raw = F.one_hot(Et, num_classes=self.output_dim.edge_dim)
            X_raw = F.one_hot(Xt, num_classes=self.output_dim.node_dim)

            assert (E_raw == torch.transpose(E_raw, 1, 2)).all()
            assert (X_raw.shape == X.shape) and (E_raw.shape == E.shape)

            graph = (
                GraphWrapper(X_raw, E_raw, y).mask(repeated_node_mask).type_as(initial_graph.X)
            )

            graph.X = (
                graph.X * modifiable_nodes_repeated
                + product_X_repeated * fixed_nodes_repeated
            )
            graph = graph.mask(repeated_node_mask)
            X, E, y = graph.X, graph.E, graph.y

            t += hb_edge[0].item()
            step += 1
            pbar.update(hb_edge[0].item())
        pbar.update(hb_node[0].item())
        pbar.close()

        X, E = self.final_step(X, E, bs)
        final_graph = GraphWrapper(X, E, y)
        final_graph = final_graph.mask(node_mask, collapse=True)
        X, E, y = final_graph.X, final_graph.E, final_graph.y

        return X, E
            

    def final_step(self, Xt, Et, original_bs):
        max_indices = torch.argmax(self.population_rs, dim=-1)
        Xt = Xt.reshape((original_bs, self.num_particles, *Xt.shape[1:]))
        Et = Et.reshape((original_bs, self.num_particles, *Et.shape[1:]))
        Xt = Xt[torch.arange(Xt.size(0)), max_indices]
        Et = Et[torch.arange(Et.size(0)), max_indices]

        return Xt, Et

    def resample(self, Xt, Et, rewards, original_bs, end_step):
        rs_candidates = rewards.reshape((original_bs, self.num_particles))
        Xt = Xt.reshape((original_bs, self.num_particles, *Xt.shape[1:]))
        Et = Et.reshape((original_bs, self.num_particles, *Et.shape[1:]))

        # Compute importance weights
        if self.potential_type == "max":
            w = torch.exp(self.lmbda * torch.max(rs_candidates, self.population_rs))
        elif self.potential_type == "add":
            rs_candidates = rs_candidates + self.population_rs
            w = torch.exp(self.lmbda * rs_candidates)
        elif self.potential_type == "diff":
            diffs = rs_candidates - self.population_rs
            w = torch.exp(self.lmbda * diffs)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if end_step:
            if self.potential_type == "max" or self.potential_type == "add":
                w = torch.exp(self.lmbda * rs_candidates) / self.product_of_potentials

        w = torch.clamp(w, 0, 1e10)
        w[torch.isnan(w)] = 0.0

        if end_step:
            # compute effective sample size
            normalized_w = w / w.sum(dim=-1, keepdim=True)
            ess = 1.0 / (normalized_w.pow(2).sum())

            if ess < 0.5 * self.num_particles:
                # Resample indices based on weights
                indices = torch.multinomial(
                    w, num_samples=self.num_particles, replacement=True
                )
                resampled_Xt = Xt[indices]
                resampled_Et = Et[indices]
                self.population_rs = rs_candidates[indices]

                # Update product of potentials; used for max and add potentials
                self.product_of_potentials = (
                    self.product_of_potentials[indices] * w[indices]
                )
            else:
                # No resampling
                resampled_Xt = Xt
                resampled_Et = Et
                self.population_rs = rs_candidates

        else:
            # Resample indices based on weights
            indices = torch.multinomial(
                w, num_samples=self.num_particles, replacement=True
            )
            self.population_rs = torch.gather(rs_candidates, -1, indices)

            batch_indices = (
                torch.arange(original_bs).unsqueeze(-1).repeat(1, self.num_particles)
            )

            resampled_Xt = Xt[batch_indices, indices]
            resampled_Et = Et[batch_indices, indices]

            # Update product of potentials; used for max and add potentials
            self.product_of_potentials = torch.gather(
                self.product_of_potentials, -1, indices
            ) * torch.gather(w, -1, indices)

        return (
            resampled_Xt.reshape(original_bs * self.num_particles, *Xt.shape[2:]),
            resampled_Et.reshape(original_bs * self.num_particles, *Et.shape[2:]),
        )

    def convert_graphs_to_mols(self, X, E, node_mask):
        mol_list = []
        n_nodes = node_mask.sum(dim=-1)

        for i in range(len(X)):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            mol = build_molecule(
                atom_types, edge_types, RetrosynthesisInfo.atom_decoder
            )
            mol_list.append(mol)
        return mol_list
