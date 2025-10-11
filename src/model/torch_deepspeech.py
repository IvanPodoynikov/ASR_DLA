import torch.nn as nn
import torchaudio


class DeepSpeech(torchaudio.models.DeepSpeech):
    def __init__(self, n_feats, n_tokens, n_hidden=512, **batch):
        super().__init__(n_feature=n_feats)
        self.n_class = n_tokens
        self.n_hidden = n_hidden

    def forward(self, spectrogram, spectrogram_length, **batch):
        output = super().forward(
            spectrogram.transpose(1, 2).unsqueeze(1)
        )  # (B, T, class)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": spectrogram_length}
