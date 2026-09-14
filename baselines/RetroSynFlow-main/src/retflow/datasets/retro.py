import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch_geometric.loader import DataLoader

from retflow import config
from retflow.datasets.data.uspto import USPTO
from retflow.datasets.dataset import Dataset
from retflow.datasets.info import RetrosynthesisInfo
from retflow.runner import DistributedHelper
from retflow.utils import ExtraFeatures, GraphDimensions, to_dense


@dataclass
class RetroDataset(Dataset):
    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))
        val_dataset = USPTO(split="val", root=str(save_dir))

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)

        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = config.get_dataset_directory() / self.name
        for split in ["train", "val", "test"]:
            USPTO(split=split, root=str(save_dir), download_and_process=True)

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        save_dir = config.get_dataset_directory() / self.name

        eval_split = "val" if load_valid else "test"
        eval_dataset = USPTO(split=eval_split, root=str(save_dir))
        self.test_dataset = eval_dataset
        self._load_eval_preprocessing_metadata(save_dir, "direct", eval_split)

        self.info = self._get_info(
            USPTO(split="train", root=str(save_dir)),
            USPTO(split="val", root=str(save_dir)),
        )

        test_loader = DataLoader(eval_dataset, batch_size=self.batch_size)
        return test_loader, self.info

    def _load_eval_preprocessing_metadata(self, save_dir, builder, split):
        report_path = save_dir / "processed" / f"{builder}_{split}.preprocessing.json"
        if not report_path.is_file():
            raise FileNotFoundError(
                f"missing required preprocessing audit: {report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        failure_path = Path(report["failure_file"])
        if not failure_path.is_file():
            raise FileNotFoundError(
                f"missing required preprocessing failure ledger: {failure_path}"
            )
        failures = [
            json.loads(line)
            for line in failure_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if report["output_rows"] != len(self.test_dataset):
            raise RuntimeError("evaluation cache row count disagrees with audit")
        if report["failed_rows"] != len(failures):
            raise RuntimeError("evaluation failure ledger disagrees with audit")
        if report["input_rows"] != report["output_rows"] + report["failed_rows"]:
            raise RuntimeError("evaluation preprocessing audit has an invalid denominator")
        self.eval_preprocessing_report = report
        self.eval_preprocessing_failures = failures


    def _get_info(self, train_dataset, val_dataset):
        n_nodes_dist = self._node_counts(train_dataset, val_dataset)
        max_n_nodes = len(n_nodes_dist) - 1
        max_n_dummy_nodes = max(
            int(graph.p_x[:, -1].sum().item())
            for dataset in (train_dataset, val_dataset)
            for graph in dataset
        )

        input_dim, output_dim = self._compute_input_output_dims(
            max_n_nodes, train_dataset
        )

        return RetrosynthesisInfo(
            input_dim=input_dim,
            output_dim=output_dim,
            max_n_nodes=max_n_nodes,
            max_n_dummy_nodes=max_n_dummy_nodes,
            num_dummy_nodes=list(range(max_n_dummy_nodes + 1)),
        )

    @staticmethod
    def _compute_input_output_dims(max_n_nodes, train_dataset):
        extra_features = ExtraFeatures(max_n_nodes)

        ex = train_dataset[0]

        r_ex_dense, r_node_mask = to_dense(ex.x, ex.edge_index, ex.edge_attr, ex.batch)

        input_dim = GraphDimensions(
            node_dim=ex.x.shape[1], edge_dim=ex.edge_attr.shape[1], y_dim=ex.y.shape[1] + 1  # type: ignore
        )

        # doesn't matter if we use the product or reactant molecule to compute
        # the extra features to determine the dimension
        ex_extra_feat = extra_features(r_ex_dense.X, r_ex_dense.E, r_node_mask)

        input_dim.node_dim += ex_extra_feat.X.size(-1)
        input_dim.edge_dim += ex_extra_feat.E.size(-1)
        input_dim.y_dim += ex_extra_feat.y.size(-1)

        input_dim.node_dim += ex.x.shape[1]
        input_dim.edge_dim += ex.edge_attr.shape[1]

        output_dim = GraphDimensions(
            node_dim=ex.x.shape[1], edge_dim=ex.edge_attr.shape[1], y_dim=0  # type: ignore
        )

        return input_dim, output_dim

    @staticmethod
    def _node_counts(train_dataset, val_dataset, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)

        for dataset in [train_dataset, val_dataset]:
            num_nodes_list = [graph.num_nodes for graph in dataset]
            for num_node in num_nodes_list:
                all_counts[num_node] += 1

        max_index = max(all_counts.nonzero())  # type: ignore
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()

        return all_counts.to(config.get_device())


@dataclass
class SmallRetroDataset(RetroDataset):
    full_test: bool = False

    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))
        val_dataset = USPTO(split="val", root=str(save_dir))

        train_dataset = train_dataset[0 : int(0.2 * len(train_dataset))]
        val_dataset = val_dataset[0 : int(0.2 * len(val_dataset))]

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper=dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        save_dir = config.get_dataset_directory() / self.name

        eval_dataset = USPTO(split="val" if load_valid else "test", root=str(save_dir))
        if not self.full_test:
            eval_dataset = eval_dataset[0 : int(0.2 * len(eval_dataset))]

        self.info = self._get_info(
            USPTO(split="train", root=str(save_dir)),
            USPTO(split="val", root=str(save_dir)),
        )

        test_loader = DataLoader(eval_dataset, batch_size=self.batch_size)
        return test_loader, self.info


@dataclass
class ToyRetroDataset(RetroDataset):
    num_molecules: int = 2

    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))[0 : self.num_molecules]
        val_dataset = copy.deepcopy(train_dataset)

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info
