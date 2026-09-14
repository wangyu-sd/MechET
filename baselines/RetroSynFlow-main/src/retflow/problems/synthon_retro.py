import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data
from torchdrug import layers, models, tasks
from torchdrug.core import Registry as R

from retflow import config
from retflow.optimizers.optimizer import Optimizer
from retflow.problems.problem import Problem
from retflow.runner import DistributedHelper
from retflow.utils import (ExtraFeatures, GraphModelWrapper, build_molecule,
                           build_simple_molecule, to_dense)
from retflow.utils.data import (build_graph_from_mol,
                                build_graph_from_mol_with_mapping,
                                compute_joint_nodes_order_mapping,
                                get_molecule_smi_list)


@dataclass
class SynthonRetrosynthesis(Problem):
    synthon_topk: int = 1
    samples_per_synthon: List[int] = None
    product_context: bool = False
    center_checkpoint: str | None = None

    def setup_problem(self, dist_helper: DistributedHelper | None = None):
        self.train_loader, self.val_loader, self.info = self.dataset.load(
            dist_helper=dist_helper
        )
        if dist_helper:
            self.torch_model = self.method.setup(
                self.info, self.model, device=dist_helper.device
            )
            self.torch_model = dist_helper.dist_adapt_model(self.torch_model)
        else:
            self.torch_model = self.method.setup(
                self.info, self.model, device=config.get_device()
            ).to(config.get_device())

        self.sampling_metrics = self.dataset.get_metrics()

        self.model_wrapper = GraphModelWrapper(
            self.torch_model,
            ExtraFeatures(self.info.max_n_nodes),
        )

    def setup_problem_eval(self, model_checkpoint, on_valid=False):
        self.test_loader, self.info = self.dataset.load_eval(load_valid=on_valid)

        self.torch_model = self.method.setup(
            self.info, self.model, device=config.get_device()
        )

        self.torch_model.load_state_dict(torch.load(model_checkpoint))
        self.torch_model = self.torch_model.to(config.get_device())

        center_model = models.RGCN(
            input_dim=self.dataset.test_dataset.node_feature_dim,
            hidden_dims=[256, 256, 256, 256],
            num_relation=4,
            short_cut=True,
            concat_hidden=True,
        )

        synthon_pred_task = CenterIdentificationModified(
            center_model, feature=("graph", "atom", "bond")
        )
        synthon_pred_task.preprocess(None, None, self.dataset.test_dataset)

        center_checkpoint = (
            Path(self.center_checkpoint)
            if self.center_checkpoint is not None
            else config.get_models_directory() / "g2g_center.pth"
        )
        synthon_pred_task.load_state_dict(
            torch.load(center_checkpoint, weights_only=False)["model"]
        )
        self.synthon_pred_task = synthon_pred_task 

        self.model_wrapper = GraphModelWrapper(
            self.torch_model,
            ExtraFeatures(self.info.max_n_nodes),
        )

    def get_optimizer(self, optimizer: Optimizer) -> torch.optim.Optimizer:
        return optimizer.get_optim(self.torch_model)

    def one_epoch(
        self,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler._LRScheduler,
        log_every_n_batch: int,
        dist_helper: DistributedHelper | None,
    ):
        return {}

    @torch.no_grad()
    def validation(self, dist_helper: DistributedHelper | None):
        return {}

    def save_model(
        self, checkpoint_dir: Path, ddp: bool = False, epoch: int | None = None
    ):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if epoch:
            save_path = checkpoint_dir / f"model_epoch_{epoch}.pt"
        else:
            save_path = checkpoint_dir / "final_model.pt"
        if ddp:
            torch.save(self.torch_model.module.state_dict(), save_path)
        else:
            torch.save(self.torch_model.state_dict(), save_path)

    @torch.no_grad()
    def sample_generation(
        self,
        num_samples: int,
        examples_per_sample: int,
        dist_helper: DistributedHelper | None = None,
    ):
        return {}

    @torch.no_grad()
    def sample_generation_eval(
        self,
        examples_per_sample: int,
    ):
        self.torch_model.eval()

        num_samples = 0
        product_molecules_smiles = []
        synthon_molecule_smiles = []
        true_reactant_smiles = []
        pred_reactant_smiles = []
        computed_scores = []

        for i, data in enumerate(self.test_loader):
            config.get_logger().info(
                f"Generated reactants for {num_samples} product molecules so far."
            )
            config.get_logger().info(f"Generating samples for batch {i} of data.")

            assert data["graph"][0].batch_size == 1

            ground_truth = []
            input_products = []

            r_mols = data["graph"][0].to_molecule()
            p_mols = data["graph"][1].to_molecule()
            
            for r_mol, p_mol in zip(r_mols, p_mols):
                simple_rmol = build_simple_molecule(r_mol)
                simple_pmol = build_simple_molecule(p_mol)
                ground_truth.append(Chem.MolToSmiles(simple_rmol, canonical=True))
                input_products.append(Chem.MolToSmiles(simple_pmol, canonical=True))

            result = self.synthon_pred_task.predict_synthon(data, k=self.synthon_topk)
            grouped_synthon_smis = self.group_predicted_synthon_smiles(
                result["synthon"].to_smiles(), result["num_synthon"]
            )

            # for some edge cases, model the same reaction center twice (one unique) which requires manual duplication
            if len(grouped_synthon_smis) < 2:
                grouped_synthon_smis.append(grouped_synthon_smis[0])
            
            synthons_smi_list = []

            for n, synthon_smi in enumerate(grouped_synthon_smis):
                synthons_smi_list.extend([synthon_smi] * self.samples_per_synthon[n])
            
            if self.product_context:
                synthons_batch = self.create_synthon_graphs_with_products(
                    synthons_smi_list, data["graph"][1].to_molecule() * examples_per_sample
                )
            else:
                synthons_batch = self.create_synthon_graphs(synthons_smi_list)
            synthons_batch = synthons_batch.to(config.get_device())

            synthon, node_mask = to_dense(
                synthons_batch.x,
                synthons_batch.edge_index,
                synthons_batch.edge_attr,
                synthons_batch.batch,
            )
            synthon = synthon.mask(node_mask)
            
            predicted_synthon_smis = get_molecule_smi_list(
                synthon.X, 
                synthon.E, 
                node_mask, 
                self.info.atom_decoder, 
                onehot=True
            )
            
            if self.product_context:
                products, p_node_mask = to_dense(
                    synthons_batch.p_x,
                    synthons_batch.p_edge_index,
                    synthons_batch.p_edge_attr,
                    synthons_batch.batch,
                )
                products = products.mask(p_node_mask)

            assert len(synthon.X) == examples_per_sample
            context = [synthon.clone(), products.clone()] if self.product_context else synthon.clone()
            
            X, E = self.method.sample(
                initial_graph=synthon,
                node_mask=node_mask,
                context=context,
                predictor=self.model_wrapper,
            )
            
            pred_reactants_smis = get_molecule_smi_list(X, E, node_mask, self.info.atom_decoder)

            scores = [0] * len(pred_reactants_smis)
        
            true_reactant_smiles.append(ground_truth[0])
            product_molecules_smiles.append(input_products[0])

            synthon_molecule_smiles.append(predicted_synthon_smis)
            pred_reactant_smiles.append(pred_reactants_smis)
            computed_scores.append(scores)
            num_samples += 1

        sampled_data = {
            "reactants": true_reactant_smiles,
            "products": product_molecules_smiles,
            "synthons": synthon_molecule_smiles,
            "predicted_reactants": pred_reactant_smiles,
            "scores": computed_scores,
            "stable_ids": list(self.dataset.test_dataset.ids[:num_samples]),
            "preprocessing_failures": getattr(
                self.dataset, "eval_preprocessing_failures", []
            ),
            "preprocessing_input_rows": getattr(
                self.dataset, "eval_preprocessing_report", {}
            ).get("input_rows", num_samples),
        }
        return sampled_data

    def group_predicted_synthon_smiles(self, synthon_smi_list, num_synthon_list):
        curr_id = 0
        grouped_synthon_smi_list = []
        for i in range(len(num_synthon_list)):
            num_synthons = num_synthon_list[i].item()
            synthons_smi = synthon_smi_list[curr_id : curr_id + num_synthons]
            full_synthons_smi = ".".join([s for s in synthons_smi])
            grouped_synthon_smi_list.append(full_synthons_smi)
            curr_id += num_synthons
        return grouped_synthon_smi_list

    def create_synthon_graphs_with_products(self, synthon_smi_list, product_mol_list):
        data_list = []
        for i, (synthon_smi, product_mol) in enumerate(
            zip(synthon_smi_list, product_mol_list)
        ):
            synthons_mol = Chem.MolFromSmiles(synthon_smi, sanitize=False)
            mapping = compute_joint_nodes_order_mapping(synthons_mol, product_mol)
            num_nodes = (
                product_mol.GetNumAtoms()
                + self.info.max_n_product_dummy_nodes
            )
            if len(mapping) > num_nodes:
                raise ValueError(
                    "predicted synthon/product union exceeds configured node capacity"
                )

            s_x, s_edge_index, s_edge_attr = build_graph_from_mol_with_mapping(
                synthons_mol,
                mapping,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )
            p_x, p_edge_index, p_edge_attr = build_graph_from_mol_with_mapping(
                product_mol,
                mapping,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )
            y = torch.zeros(size=(1, 0), dtype=torch.float)
            synthon_mask = ~(s_x[:, -1].bool()).squeeze()
            new2old_idx = torch.randperm(num_nodes).long()
            old2new_idx = torch.empty_like(new2old_idx)
            old2new_idx[new2old_idx] = torch.arange(num_nodes)

            p_x = p_x[new2old_idx]
            p_edge_index = torch.stack(
                [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]],
                dim=0,
            )
            p_edge_index, p_edge_attr = self.sort_edges(
                p_edge_index, p_edge_attr, num_nodes
            )

            s_x = s_x[new2old_idx]
            s_edge_index = torch.stack(
                [old2new_idx[s_edge_index[0]], old2new_idx[s_edge_index[1]]],
                dim=0,
            )
            s_edge_index, s_edge_attr = self.sort_edges(
                s_edge_index, s_edge_attr, num_nodes
            )

            synthon_mask = ~(s_x[:, -1].bool()).squeeze()
            assert torch.allclose(p_x[synthon_mask], s_x[synthon_mask])

            data = Data(
                x=s_x,
                edge_index=s_edge_index,
                edge_attr=s_edge_attr,
                p_x=p_x,
                p_edge_index=p_edge_index,
                p_edge_attr=p_edge_attr,
                y=y,
                idx=i,
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def create_synthon_graphs(self, synthon_smi_list):
        data_list = []
        for i, synthon_smi in enumerate(synthon_smi_list):
            synthons_mol = Chem.MolFromSmiles(synthon_smi, sanitize=False)
            num_nodes = synthons_mol.GetNumAtoms() + self.info.max_n_dummy_nodes

            s_x, s_edge_index, s_edge_attr = build_graph_from_mol(
                synthons_mol,
                num_nodes,
                types=self.info.atom_encoder,
                bonds=self.info.bonds,
            )

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=s_x,
                edge_index=s_edge_index,
                edge_attr=s_edge_attr,
                y=y,
                idx=i,
            )
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def convert_reactant_graphs_to_smile(self, graph_list):
        smi_list = []
        for reactant_graph in graph_list:
            r_mol = build_molecule(
                reactant_graph[0], reactant_graph[1], self.info.atom_decoder
            )
            smi_list.append(Chem.MolToSmiles(r_mol))
        return smi_list

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


@R.register("tasks.CenterIdentificationModified")
class CenterIdentificationModified(tasks.CenterIdentification):
    def __init__(self, model, feature=("graph", "atom", "bond")):
        super().__init__(model, feature)
        self.name = "CenterIdentificationModified"

    def preprocess(self, train_set, valid_set, test_set):
        reaction_types = set()
        bond_types = set()
        for sample in test_set:
            reaction_types.add(sample["reaction"])
            for graph in sample["graph"]:
                bond_types.update(graph.edge_list[:, 2].tolist())
        self.num_reaction = len(reaction_types)
        self.num_relation = len(bond_types)
        node_feature_dim = test_set[0]["graph"][0].node_feature.shape[-1]
        edge_feature_dim = test_set[0]["graph"][0].edge_feature.shape[-1]

        node_dim = self.model.output_dim
        edge_dim = 0
        graph_dim = 0

        for _feature in sorted(self.feature):
            if _feature == "reaction":
                graph_dim += self.num_reaction
            elif _feature == "graph":
                graph_dim += self.model.output_dim
            elif _feature == "atom":
                node_dim += node_feature_dim
            elif _feature == "bond":
                edge_dim += edge_feature_dim
            else:
                raise ValueError("Unknown feature `%s`" % _feature)

        node_dim += graph_dim  # inherit graph features
        edge_dim += node_dim * 2  # inherit node features

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.edge_mlp = layers.MLP(edge_dim, hidden_dims + [1])
        self.node_mlp = layers.MLP(node_dim, hidden_dims + [1])
