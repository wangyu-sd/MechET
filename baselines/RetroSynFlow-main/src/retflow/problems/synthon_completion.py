from dataclasses import dataclass

import torch
from tqdm import tqdm

from retflow import config
from retflow.problems.retrosynthesis import Retrosynthesis
from retflow.runner import DistributedHelper
from retflow.utils import (GraphWrapper, get_molecule_list,
                           get_molecule_smi_list, to_dense, top_k_accuracy)


@dataclass
class SynthonCompletion(Retrosynthesis):
    use_product_context: bool = False
    gradient_accumulation_steps: int = 1

    def one_epoch(
        self,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler._LRScheduler,
        dist_helper: DistributedHelper | None,
    ):
        self.torch_model.train()
        num_points = 0
        total_loss = 0.0
        edge_loss = 0.0
        node_loss = 0.0

        enable = dist_helper is None or (
            dist_helper is not None and dist_helper.get_rank() == 0
        )
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        optim.zero_grad()
        total_batches = len(self.train_loader)
        for batch_index, data in enumerate(
            tqdm(self.train_loader, leave=False, disable=not enable)
        ):
            data = (
                data.to(dist_helper.device, non_blocking=True)
                if dist_helper
                else data.to(config.get_device())
            )
            reactants, r_node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            reactants = reactants.mask(
                r_node_mask
            )  # mask out fake nodes and makes sure adj matrix is upper triangular

            synthons, s_node_mask = to_dense(
                data.s_x, data.s_edge_index, data.s_edge_attr, data.batch
            )
            synthons = synthons.mask(s_node_mask)

            if self.use_product_context:
                product, p_node_mask = to_dense(
                    data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
                )
                product = product.mask(p_node_mask)

            assert torch.allclose(r_node_mask, s_node_mask)
            if self.use_product_context:
                assert torch.allclose(r_node_mask, p_node_mask)
            node_mask = r_node_mask

            noisy_graph, node_mask, t_float = self.method.apply_noise(
                X=synthons.X,
                E=synthons.E,
                y=synthons.y,
                X_T=reactants.X,
                E_T=reactants.E,
                y_T=reactants.y,
                node_mask=node_mask,
            )
            if self.use_product_context:
                context = [synthons.clone(), product.clone()]
            else:
                context = synthons.clone()

            X, E = self.model_wrapper(noisy_graph, node_mask, context, t_float)
            prediction = GraphWrapper(X, E, synthons.y).mask(node_mask)

            fixed_nodes = (synthons.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (synthons.X[..., -1] == 1).unsqueeze(-1)  # dummy nodes
            assert torch.all(fixed_nodes | modifiable_nodes)
            prediction.X = prediction.X * modifiable_nodes + synthons.X * fixed_nodes
            prediction.X = prediction.X * node_mask.unsqueeze(-1)

            batch_loss, batch_node_loss, batch_edge_loss = self.method.compute_loss(
                reactants,
                prediction,
                node_mask,
                noisy_graph,
                t_float,
            )

            group_start = (
                batch_index // self.gradient_accumulation_steps
            ) * self.gradient_accumulation_steps
            group_size = min(
                self.gradient_accumulation_steps,
                total_batches - group_start,
            )
            (batch_loss / group_size).backward()

            if (
                (batch_index + 1) % self.gradient_accumulation_steps == 0
                or batch_index + 1 == total_batches
            ):
                optim.step()
                optim.zero_grad()

            total_loss += batch_loss.detach() * len(data)
            node_loss += batch_node_loss.detach() * len(data)
            edge_loss += batch_edge_loss.detach() * len(data)

            num_points += len(data)

        sched.step()

        metrics = {}

        metrics["train_loss"] = total_loss
        metrics["train_node_loss"] = node_loss
        metrics["train_edge_loss"] = edge_loss

        if dist_helper:
            metric_dict = {
                "metrics": metrics,
                "num_points": torch.tensor(num_points, device=dist_helper.device),
            }
        else:
            for metric in metrics.keys():
                metrics[metric] = metrics[metric].item() / num_points
            metric_dict = {"metrics": metrics, "num_points": num_points}
        return metric_dict

    @torch.no_grad()
    def validation(self, dist_helper: DistributedHelper | None):
        self.torch_model.eval()

        total_loss = 0.0
        edge_loss = 0.0
        node_loss = 0.0
        num_points = 0

        for _, data in enumerate(tqdm(self.val_loader, leave=False)):
            if dist_helper:
                data = data.to(dist_helper.device, non_blocking=True)
            else:
                data = data.to(config.get_device())

            reactants, r_node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            reactants = reactants.mask(r_node_mask)

            synthons, s_node_mask = to_dense(
                data.s_x, data.s_edge_index, data.s_edge_attr, data.batch
            )
            synthons = synthons.mask(s_node_mask)

            if self.use_product_context:
                product, p_node_mask = to_dense(
                    data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
                )
                product = product.mask(p_node_mask)

            assert torch.allclose(r_node_mask, s_node_mask)
            if self.use_product_context:
                assert torch.allclose(r_node_mask, p_node_mask)
            node_mask = r_node_mask

            noisy_graph, node_mask, t_float = self.method.apply_noise(
                X=synthons.X,
                E=synthons.E,
                y=synthons.y,
                X_T=reactants.X,
                E_T=reactants.E,
                y_T=reactants.y,
                node_mask=node_mask,
            )

            if self.use_product_context:
                context = [synthons.clone(), product.clone()]
            else:
                context = synthons.clone()

            X, E = self.model_wrapper(noisy_graph, node_mask, context, t_float)
            prediction = GraphWrapper(X, E, synthons.y).mask(node_mask)

            fixed_nodes = (synthons.X[..., -1] == 0).unsqueeze(-1)
            modifiable_nodes = (synthons.X[..., -1] == 1).unsqueeze(-1)  # dummy nodes
            assert torch.all(fixed_nodes | modifiable_nodes)
            prediction.X = prediction.X * modifiable_nodes + synthons.X * fixed_nodes
            prediction.X = prediction.X * node_mask.unsqueeze(-1)

            batch_loss, batch_node_loss, batch_edge_loss = self.method.compute_loss(
                reactants, prediction, node_mask, noisy_graph, t_float
            )

            total_loss += batch_loss.detach() * len(data)
            node_loss += batch_node_loss.detach() * len(data)
            edge_loss += batch_edge_loss.detach() * len(data)
            num_points += len(data)

        metrics = {}
        metrics["val_loss"] = total_loss
        metrics["val_node_loss"] = node_loss
        metrics["val_edge_loss"] = edge_loss

        if dist_helper:
            return {
                "metrics": metrics,
                "num_points": torch.tensor(num_points, device=config.get_device()),
            }

        for metric in metrics.keys():
            metrics[metric] = metrics[metric].item() / num_points
        return {"metrics": metrics, "num_points": num_points}

    @torch.no_grad()
    def sample_generation(
        self,
        num_samples: int,
        examples_per_sample: int,
        dist_helper: DistributedHelper | None = None,
    ):
        self.torch_model.eval()
        num_samples_generated = 0

        batch_id = 0

        grouped_samples = []
        grouped_scores = []
        ground_truth = []

        for _, data in enumerate(tqdm(self.val_loader, leave=False)):
            if num_samples_generated >= num_samples:
                break

            if dist_helper:
                data = data.to(dist_helper.device, non_blocking=True)
            else:
                data = data.to(config.get_device())

            to_generate = len(data)
            batch_groups = []
            batch_scores = []

            synthons, node_mask = to_dense(
                data.s_x, data.s_edge_index, data.s_edge_attr, data.batch
            )
            synthons = synthons.mask(node_mask)
            reactants, node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch)
            reactants = reactants.mask(node_mask)

            if self.use_product_context:
                product, p_node_mask = to_dense(
                    data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
                )
                product = product.mask(p_node_mask)

            ground_truth.extend(get_molecule_list(reactants.X, reactants.E, node_mask, self.info.atom_decoder))

            if self.use_product_context:
                context = [synthons.clone(), product.clone()]
            else:
                context = synthons.clone()

            for _ in range(examples_per_sample):
                X, E = self.method.sample(
                    initial_graph=synthons,
                    node_mask=node_mask,
                    context=context,
                    predictor=self.model_wrapper,
                )

                pred_molecule_list = get_molecule_list(X, E, node_mask, self.info.atom_decoder)

                scores = [0] * len(pred_molecule_list)

                batch_groups.append(pred_molecule_list)
                batch_scores.append(scores)

            num_samples_generated += to_generate
            batch_id += 1

            for mol_idx_in_batch in range(to_generate):  # batch size
                mol_samples_group = []
                mol_scores_group = []
                for batch_group, scores_group in zip(
                    batch_groups, batch_scores
                ):  # K times
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                    mol_scores_group.append(scores_group[mol_idx_in_batch])

                assert len(mol_samples_group) == examples_per_sample
                grouped_samples.append(mol_samples_group)
                grouped_scores.append(mol_scores_group)

        metrics = top_k_accuracy(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.info.atom_decoder,
            grouped_scores=grouped_scores,
        )

        return metrics

    @torch.no_grad()
    def sample_generation_eval(
        self,
        examples_per_sample: int,
    ):
        self.torch_model.eval()

        num_samples = 0
        products = [] # these are intermediate synthons
        true_reactants = []
        predicted_reactants = []
        pred_reactant_scores = []
        stable_ids = []

        for i, data in enumerate(self.test_loader):
            config.get_logger().info(
                f"Generated reactants for {num_samples} product molecules so far."
            )
            config.get_logger().info(f"Generating samples for batch {i} of data.")

            data = data.to(config.get_device())
            batch_stable_ids = data.stable_id
            if isinstance(batch_stable_ids, (list, tuple)):
                stable_ids.extend(str(value) for value in batch_stable_ids)
            else:
                stable_ids.append(str(batch_stable_ids))
            synthons, node_mask = to_dense(
                data.s_x, data.s_edge_index, data.s_edge_attr, data.batch
            )
            synthons = synthons.mask(node_mask)
            reactants, node_mask = to_dense(
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            reactants = reactants.mask(node_mask)

            true_reactants.extend(get_molecule_smi_list(reactants.X, reactants.E, node_mask, self.info.atom_decoder, onehot=True))
            products.extend(get_molecule_smi_list(synthons.X, synthons.E, node_mask, self.info.atom_decoder, onehot=True))

            synthons.X = synthons.X.repeat((examples_per_sample, 1, 1))
            synthons.E = synthons.E.repeat((examples_per_sample, 1, 1, 1))
            node_mask = node_mask.repeat((examples_per_sample, 1))

            if self.use_product_context:
                product, p_node_mask = to_dense(
                    data.p_x, data.p_edge_index, data.p_edge_attr, data.batch
                )
                product = product.mask(p_node_mask)

                product.X = product.X.repeat((examples_per_sample, 1, 1))
                product.E = product.E.repeat((examples_per_sample, 1, 1, 1))

            if self.use_product_context:
                context = [synthons.clone(), product.clone()]
            else:
                context = synthons.clone()

            X, E = self.method.sample(
                initial_graph=synthons,
                node_mask=node_mask,
                context=context,
                predictor=self.model_wrapper,
            )
            pred_reactants_smi_list = get_molecule_list(X, E, node_mask, self.info.atom_decoder)
            scores = [0] * len(pred_reactants_smi_list)

            predicted_reactants.append(pred_reactants_smi_list)
            pred_reactant_scores.append(scores)
            num_samples += 1

        sampled_data = {
            "reactants": true_reactants,
            "products": products,  # these are intermediate synthons, had to keep the name for consistency in the pandas dataframe
            "predicted_reactants": predicted_reactants,
            "scores": pred_reactant_scores,
            "stable_ids": stable_ids,
            "preprocessing_failures": getattr(
                self.dataset, "eval_preprocessing_failures", []
            ),
            "preprocessing_input_rows": getattr(
                self.dataset, "eval_preprocessing_report", {}
            ).get("input_rows", len(stable_ids)),
        }
        return sampled_data
