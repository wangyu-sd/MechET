from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class TimeSampler(ABC):
    min_time: float = 0.0
    max_time: float | None = 1.0

    @abstractmethod
    def sample(self, num_samples, dtype, device) -> torch.Tensor:
        pass


@dataclass
class UniformTimeSampler(TimeSampler):
    def sample(self, num_samples, dtype, device) -> torch.Tensor:
        t = torch.rand(num_samples, device=device).type(dtype)
        return torch.clip(t, self.min_time, self.max_time)


@dataclass
class ExponentialTimeSampler(TimeSampler):
    rate: float = 1.0

    def sample(self, num_samples, dtype, device) -> torch.Tensor:
        t = (
            torch.distributions.Exponential(self.rate)
            .sample(torch.Size((num_samples,)))
            .to(dtype)
            .to(device)
        )
        return torch.clip(t, self.min_time, self.max_time)
