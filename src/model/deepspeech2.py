import math

import torch
import torch.nn as nn


def size_after_conv(dim, pad, k, s, d=1):
    return (dim + 2 * pad - d * (k - 1) - 1) // s + 1


class MaskConvs(nn.Module):
    # маскирует остатки после паддинга
    def __init__(self, seq_modules):
        super().__init__()
        self.seq_modules = seq_modules

    def forward(self, x, lengths):
        updated_lengths = lengths
        for module in self.seq_modules:
            x = module(x)
            if isinstance(module, nn.Conv2d):
                updated_lengths = size_after_conv(
                    dim=updated_lengths,
                    pad=module.padding[1],
                    k=module.kernel_size[1],
                    s=module.stride[1],
                    d=module.dilation[1],
                )
            mask = torch.zeros_like(x, dtype=torch.bool, device=x.device)
            for i, length in enumerate(updated_lengths):
                length = length.item()
                if (mask[i].size(-1) - length) > 0:
                    mask[i].narrow(-1, length, mask[i].size(-1) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, updated_lengths


class BatchRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type,
        bidirectional,
        batch_norm,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.cell = getattr(nn, rnn_type.upper())(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
            bias=True,
        )
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x, output_length):
        if self.batch_norm:
            x = (
                self.batch_norm(x.transpose(1, 2)).transpose(1, 2).contiguous()
            )  # (B, T, F)
        x = nn.utils.rnn.pack_padded_sequence(
            x, output_length, batch_first=True, enforce_sorted=False
        )
        x, _ = self.cell(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.cell.bidirectional:
            fwd, bwd = torch.chunk(x, 2, dim=-1)
            x = fwd + bwd
        return x


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        rnn_hidden=1024,
        rnn_layers=5,
        bidirectional=True,
        rnn_type="gru",
    ):
        super().__init__()
        self.conv = MaskConvs(
            nn.Sequential(
                nn.Conv2d(
                    1,
                    32,
                    kernel_size=(41, 11),
                    stride=(2, 2),
                    padding=(20, 5),
                    bias=False,
                ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(
                    32,
                    32,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                    bias=False,
                ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(
                    32,
                    96,
                    kernel_size=(21, 11),
                    stride=(2, 1),
                    padding=(10, 5),
                    bias=False,
                ),
                nn.BatchNorm2d(96),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.n_tokens = n_tokens
        assert rnn_type in [
            "gru",
            "lstm",
            "rnn",
        ], f"{rnn_type} not in ['gru', 'lstm', 'rnn]"
        self.rnn_type = rnn_type

        freq_dim = n_feats
        for pad, k, s in [(20, 41, 2), (10, 21, 2), (10, 21, 2)]:
            freq_dim = size_after_conv(freq_dim, pad, k, s)
        self._rnn_input_size = freq_dim * 96

        recurrent_blocks = []
        first_block = BatchRNN(
            input_size=self._rnn_input_size,
            hidden_size=rnn_hidden,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
            batch_norm=False,
        )
        recurrent_blocks.append(first_block)
        for _ in range(rnn_layers - 1):
            recurrent_blocks.append(
                BatchRNN(
                    input_size=rnn_hidden,
                    hidden_size=rnn_hidden,
                    rnn_type=rnn_type,
                    bidirectional=bidirectional,
                    batch_norm=True,
                )
            )
        self.reccurent_stack = nn.ModuleList(recurrent_blocks)
        self.bn_final = nn.BatchNorm1d(rnn_hidden)
        self.linear_final = nn.Linear(rnn_hidden, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x = spectrogram.unsqueeze(1)  # (B, F, T) -> (B, 1, F, T)
        conv_x, updated_lengths = self.conv(x, spectrogram_length)

        B, C, F, T = conv_x.shape
        rnn_input = conv_x.view(B, C * F, T).transpose(1, 2)  # (B, T, C*F)
        rnn_output = rnn_input
        for rnn_layer in self.reccurent_stack:
            rnn_output = rnn_layer(rnn_output, updated_lengths)

        normed = self.bn_final(rnn_output.transpose(1, 2)).transpose(1, 2).contiguous()
        output = self.linear_final(normed)
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
        output_length = input_lengths
        for m in self.conv.seq_modules:
            if isinstance(m, nn.Conv2d):
                output_length = size_after_conv(
                    output_length,
                    m.padding[1],
                    m.kernel_size[1],
                    m.stride[1],
                    m.dilation[1],
                )
        return output_length

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
