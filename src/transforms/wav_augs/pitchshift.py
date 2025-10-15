import torch_audiomentations
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(
        self, min_semitones: float = -4.0, max_semitones: float = 4.0, p: float = 0.5
    ):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(
            min_semitones=min_semitones, max_semitones=max_semitones, p=p
        )

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
