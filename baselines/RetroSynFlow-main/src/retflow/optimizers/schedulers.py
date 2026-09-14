from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR,
                                      _LRScheduler)


@dataclass
class LearningRateSched(ABC):
    @abstractmethod
    def get_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        pass


@dataclass
class CosineAnnealing(LearningRateSched):
    t_max: int

    def get_scheduler(self, optimizer: Optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.t_max)

@dataclass
class ConsLR(LearningRateSched):
    factor: float = 1.0

    def get_scheduler(self, optimizer: Optimizer):
        return ConstantLR(optimizer, factor=self.factor)