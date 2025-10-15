import torch
from torch import Tensor, nn


class PolarityInversion(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1).item() < self.p:
            return -data
        return data
