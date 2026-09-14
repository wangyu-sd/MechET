from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TimeScheduler(ABC):

    @abstractmethod
    def kappa(self, t: Tensor) -> Tensor:
        pass

    @abstractmethod
    def kappa_prime(self, t: Tensor) -> Tensor:
        pass


@dataclass
class CubicTimeScheduler(TimeScheduler):
    a: float = 2.0
    b: float = 0.5

    def kappa(self, t: Tensor) -> Tensor:
        return (
            -2 * (t**3)
            + 3 * (t**2)
            + self.a * (t**3 - 2 * t**2 + t)
            + self.b * (t**3 - t**2)
        )

    def kappa_prime(self, t: Tensor) -> Tensor:
        return (
            -6 * (t**2)
            + 6 * t
            + self.a * (3 * t**2 - 4 * t + 1)
            + self.b * (3 * t**2 - 2 * t)
        )


@dataclass
class CosineSquareTimeScheduler(TimeScheduler):
    v: float = 1.0

    def kappa(self, t: Tensor) -> Tensor:
        return 1 - torch.cos((torch.pi / 2.0) * t**self.v) ** 2

    def kappa_prime(self, t: Tensor) -> Tensor:
        return (torch.pi / 2) * self.v * t**self.v * torch.sin(torch.pi * t**self.v)


@dataclass
class LogLinearTimeScheduler(TimeScheduler):
    def kappa(self, t: Tensor) -> Tensor:
        return -torch.log1p(-(1 - (1 / torch.e)) * t)

    def kappa_prime(self, t: Tensor) -> Tensor:
        return (1 - torch.e) / ((torch.e - 1) * t - torch.e)


@dataclass
class LinearTimeScheduler(TimeScheduler):
    def kappa(self, t: Tensor) -> Tensor:
        return t

    def kappa_prime(self, t: Tensor) -> Tensor:
        return torch.ones_like(t).to(t.device)
