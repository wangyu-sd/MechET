import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torch_geometric.loader import DataLoader

from retflow import config
from retflow.datasets.data.uspto_synthon import SynthonUSPTO
from retflow.datasets.info import RetrosynthesisInfo
from retflow.datasets.retro import RetroDataset
from retflow.runner import DistributedHelper


@dataclass
class SynthonDataset(RetroDataset):
    dataset_root: str | None = None

    def _save_dir(self):
        if self.dataset_root is not None:
            return Path(self.dataset_root)
        return config.get_dataset_directory() / self.name

    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = self._save_dir()

        train_dataset = SynthonUSPTO(split="train", root=str(save_dir))
        val_dataset = SynthonUSPTO(split="val", root=str(save_dir))

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)

        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = self._save_dir()
        for split in ["train", "val", "test"]:
            SynthonUSPTO(
                split=split, root=str(save_dir), download_and_process=True
            )

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        save_dir = self._save_dir()

        eval_split = "val" if load_valid else "test"
        test_dataset = SynthonUSPTO(split=eval_split, root=str(save_dir))
        self.test_dataset = test_dataset
        self._load_eval_preprocessing_metadata(save_dir, "synthon", eval_split)
        self.info = self._get_info(
            SynthonUSPTO(split="train", root=str(save_dir)),
            SynthonUSPTO(split="val", root=str(save_dir)),
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader, self.info

    def _compute_input_output_dims(self, max_n_nodes, train_dataset):
        input_dim, output_dim = super()._compute_input_output_dims(
            max_n_nodes, train_dataset
        )

        ex = train_dataset[0]

        # for the extra context product molecule (it has the same dimension as the reactant molecule)
        input_dim.node_dim += ex.x.shape[1]
        input_dim.edge_dim += ex.edge_attr.shape[1]

        return input_dim, output_dim

    def _get_info(self, train_dataset, val_dataset):
        info = super()._get_info(train_dataset, val_dataset)
        max_synthon_dummy_nodes = max(
            int(graph.s_x[:, -1].sum().item())
            for dataset in (train_dataset, val_dataset)
            for graph in dataset
        )
        info.max_n_product_dummy_nodes = info.max_n_dummy_nodes
        info.max_n_dummy_nodes = max_synthon_dummy_nodes
        info.num_dummy_nodes = list(range(max_synthon_dummy_nodes + 1))
        return info


@dataclass
class ToySynthonDataset(RetroDataset):
    num_molecules: int = 2

    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = config.get_dataset_directory() / self.name

        if self.name == "SynthonUSPTO":
            train_dataset = SynthonUSPTO(split="train", root=str(save_dir))[0: self]
            val_dataset = copy.deepcopy(train_dataset)
        else:
            raise NotImplementedError

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper=dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info
