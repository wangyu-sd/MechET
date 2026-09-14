from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim.optimizer import Optimizer as Optim

from retflow.optimizers.schedulers import ConsLR, LearningRateSched


@dataclass
class Optimizer(ABC):
    lr: float
    lr_sched: LearningRateSched

    @abstractmethod
    def get_optim(self, torch_model: nn.Module) -> Optim:
        pass


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999

    def get_optim(self, torch_model: nn.Module) -> Optim:
        return torch.optim.Adam(
            torch_model.parameters(), self.lr, betas=(self.beta1, self.beta2)
        )


@dataclass
class AdamW(Adam):
    weight_decay: float = 1e-12
    ams_grad: bool = True

    def get_optim(self, torch_model: nn.Module) -> Optim:
        return torch.optim.AdamW(
            torch_model.parameters(),
            self.lr,
            betas=(self.beta1, self.beta2),
            amsgrad=self.ams_grad,
            weight_decay=self.weight_decay,
        )

ADAMW = AdamW(lr=2e-4, lr_sched=ConsLR())