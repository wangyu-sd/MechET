import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader, DistributedSampler

from retflow.datasets.info import RetrosynthesisInfo
from retflow.utils.distributed_helper import DistributedHelper


@dataclass
class Dataset(ABC):
    name: str
    batch_size: int

    @abstractmethod
    def load(
        self, dist_helper: DistributedHelper | None = None
    ) -> Tuple[DataLoader, DataLoader, RetrosynthesisInfo]:
        pass

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def load_eval(self, load_valid=False) -> Tuple[DataLoader, RetrosynthesisInfo]:
        pass

    def _get_train_and_val_loaders(
        self, train_dataset, val_dataset, data_loader, dist_helper: DistributedHelper | None = None
    ):
        bs = self.batch_size if dist_helper is None else max(
            1, self.batch_size // dist_helper.get_ddp_status()[1]["WORLD_SIZE"]
        )

        train_loader = data_loader(
            dataset=train_dataset,
            sampler=DistributedSampler(train_dataset, shuffle=False) if dist_helper is not None else None,
            batch_size=bs,
            pin_memory=False if dist_helper is None else True,
            num_workers=min(6, os.cpu_count()),
        )

        val_loader = data_loader(
            dataset=val_dataset,
            sampler=DistributedSampler(val_dataset, shuffle=False) if dist_helper is not None else None,
            batch_size=bs,
            pin_memory=False if dist_helper is None else True,
            num_workers=min(6, os.cpu_count()),
        )
        
        return train_loader, val_loader
