import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return F.silu(x)


class FeedForwardModule(nn.Module):
    # (B, T, D) -> (B, T, D)
    def __init__(self, dim, expansion_factor=4, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(normalized_shape=dim),
            nn.Linear(in_features=dim, out_features=dim * expansion_factor),
            Swish(),
            nn.Dropout(p=p),
            nn.Linear(in_features=dim * expansion_factor, out_features=dim),
            nn.Dropout(p=p),
        )
        self.scale = 0.5

    def forward(self, x):
        return x + self.scale * self.net(x)


class MultiHeadSelfAttentionModule(nn.Module):
    # (B, T, D) -> (B, T, D)
    def __init__(self, dim, num_heads, p=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x_ = x
        x = self.layer_norm(x)
        x, _ = self.mhsa(x, x, x)
        x = self.dropout(x)
        return x + x_


class ConvolutionModule(nn.Module):
    # (B, T, D) -> (B, T, D)
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)
        # Before should transpose to (B, D, T)
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=dim, out_channels=2 * dim, kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding="same",
            groups=dim,
        )
        self.batch_norm = nn.BatchNorm1d(num_features=dim)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x_nn = self.layer_norm(x)
        x_nn = x_nn.transpose(1, 2)  # (B, T, D) -> (B, D, T)
        x_nn = self.pointwise_conv1(x_nn)
        x_nn = self.glu(x_nn)
        x_nn = self.depthwise_conv(x_nn)
        x_nn = self.batch_norm(x_nn)
        x_nn = self.swish(x_nn)
        x_nn = self.pointwise_conv2(x_nn)
        x_nn = self.dropout(x_nn)
        x_nn = x_nn.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        return x + x_nn


class ConformerBlock(nn.Module):
    # (B, T, D) -> (B, T, D)
    def __init__(
        self, dim, expansion_factor=4, p=0.1, num_heads=4, kernel_size=31, **kwargs
    ):
        super().__init__()
        self.ffn1 = FeedForwardModule(dim=dim, expansion_factor=expansion_factor, p=p)
        self.mhsa = MultiHeadSelfAttentionModule(dim=dim, num_heads=num_heads, p=p)
        self.conv = ConvolutionModule(dim=dim, kernel_size=kernel_size)
        self.ffn2 = FeedForwardModule(dim=dim, expansion_factor=expansion_factor, p=p)
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.layer_norm(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_layers=16,
        encoder_dim=256,
        attention_heads=4,
        conv_kernel_size=31,
        expansion_factor=4,
        p=0.1,
        **kwargs,
    ):
        super().__init__()
        self.input_fc = nn.Linear(in_features=input_dim, out_features=encoder_dim)
        self.input_dropout = nn.Dropout(p=p)
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    dim=encoder_dim,
                    expansion_factor=expansion_factor,
                    p=p,
                    num_heads=attention_heads,
                    kernel_size=conv_kernel_size,
                    **kwargs,
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.input_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x


class ConformerModel(nn.Module):
    def __init__(
        self,
        n_feats,
        n_tokens,
        encoder_layers=16,
        encoder_dim=256,
        attention_heads=4,
        conv_kernel_size=31,
        expansion_factor=4,
        p=0.1,
    ):
        super().__init__()
        self.encoder = ConformerEncoder(
            input_dim=n_feats,
            encoder_layers=encoder_layers,
            encoder_dim=encoder_dim,
            attention_heads=attention_heads,
            conv_kernel_size=conv_kernel_size,
            expansion_factor=expansion_factor,
            p=p,
        )
        self.classifier = nn.Linear(in_features=encoder_dim, out_features=n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        output = self.classifier(
            self.encoder(spectrogram.transpose(1, 2))
        )  # (B, F, T) -> (B, T, F)
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
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
