from torch import nn
from torch.nn import Sequential
from torchaudio.models import DeepSpeech


class DeepSpeechWrapper(nn.Module):
    """
    Wrapper of torchaudio's deepspeech
    """

    def __init__(self, n_feats, n_tokens, n_hidden=2048, dropout=0.0):
        """
        Args:
            n_feats (int): number of input features.
        """
        super().__init__()
        self.net = DeepSpeech(
            n_feature=n_feats, n_hidden=n_hidden, n_class=n_tokens, dropout=dropout
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.
        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        output = self.net(spectrogram.transpose(1, 2))
        print(f"Output: {output}")
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.
        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
