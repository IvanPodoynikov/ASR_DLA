import torch
from torch import Tensor, nn


class AddWhiteNoise(nn.Module):
    def __init__(self, amplitude: float = 0.01, p: float = 0.5):
        super().__init__()
        self.amplitude = amplitude
        self.p = p

    def __call__(self, data: Tensor):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(data) * self.amplitude
            data = data + noise
            data = torch.clamp(data, -1.0, 1.0)
        return data
