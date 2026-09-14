from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict

from torch import Tensor
from torch.nn import Module

from retflow.datasets.retro import RetrosynthesisInfo
from retflow.models.model import Model


@dataclass
class Method(ABC):
    @abstractmethod
    def setup(self, dataset_info: RetrosynthesisInfo, model: Model, device: str):
        pass

    @abstractmethod
    def apply_noise(self, X, E, y, X_T, E_T, y_T, node_mask):
        pass

    @abstractmethod
    def compute_loss(self, reactants, pred, node_mask, noisy_graph, t_float):
        pass

    @abstractmethod
    def sample(self, initial_graph, node_mask, context, predictor: Callable):
        pass

