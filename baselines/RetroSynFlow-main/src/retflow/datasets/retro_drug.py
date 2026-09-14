import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from torchdrug.data import DataLoader

from retflow import config
from retflow.datasets.data.uspto_drug import _TorchDrugUSPTO
from retflow.datasets.dataset import Dataset
from retflow.datasets.info import RetrosynthesisInfo
from retflow.datasets.synthon import SynthonDataset
from retflow.runner import DistributedHelper


@dataclass
class TorchDrugRetroDataset(Dataset):
    product_context: bool = False
    synthon_dataset_name: str | None = None
    dataset_root: str | None = None

    def _save_dir(self) -> Path:
        if self.dataset_root is not None:
            return Path(self.dataset_root)
        return config.get_dataset_directory() / self.name

    def load(self, dist_helper: DistributedHelper | None = None):
        save_dir = self._save_dir()

        train_dataset = _TorchDrugUSPTO(
            data_split="train", path=str(save_dir / "raw"), atom_feature="center_identification"
        )
        val_dataset = _TorchDrugUSPTO(
            data_split="val", path=str(save_dir / "raw"), atom_feature="center_identification"
        )

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, DataLoader, dist_helper
        )
        dummy_dataset = SynthonDataset(
            name=self.synthon_dataset_name or self.name,
            batch_size=1,
            dataset_root=str(save_dir),
        )
        _, _, self.info = dummy_dataset.load()

        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = self._save_dir()
        for split in ["train", "val", "test"]:
            _TorchDrugUSPTO.download(split, str(save_dir / "raw"))

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        save_dir = self._save_dir()

        self.test_dataset = _TorchDrugUSPTO(
            data_split="val" if load_valid else "test",
            path=str(save_dir / "raw"),
            atom_feature="center_identification",
        )
        self.eval_preprocessing_report = self.test_dataset.preprocessing_report
        self.eval_preprocessing_failures = self.test_dataset.preprocessing_failures
        dummy_dataset = SynthonDataset(
            name=self.synthon_dataset_name or self.name,
            batch_size=1,
            dataset_root=str(save_dir),
        )
        _, _, self.info = dummy_dataset.load()

        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)
        return test_loader, self.info
