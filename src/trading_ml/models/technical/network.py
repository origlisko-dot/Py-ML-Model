"""Cross-timeframe attention network for Model 1.

Two stages:

1. **Per-timeframe encoder** — a temporal CNN (TCN) or a Transformer encoder
   turns each timeframe's ``(window, n_features)`` sequence into one embedding.
2. **Cross-timeframe attention** — the per-timeframe embeddings are treated as a
   short token sequence (one token per timeframe) and passed through a
   multi-head self-attention block, so the model learns interactions between,
   e.g., the daily trend and the 5-minute microstructure.

A classification head then predicts the triple-barrier direction (down/flat/up).
"""

from __future__ import annotations

import torch
from torch import nn


class _TCNEncoder(nn.Module):
    """Small dilated temporal conv stack over one timeframe's sequence."""

    def __init__(self, n_features: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        layers: list[nn.Module] = []
        for i in range(n_layers):
            dilation = 2**i
            layers.append(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation, dilation=dilation)
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) -> (B, d_model)
        h = self.input_proj(x)  # (B, W, d)
        h = h.transpose(1, 2)  # (B, d, W)
        h = self.tcn(h)
        h = h.transpose(1, 2)  # (B, W, d)
        h = self.norm(h)
        return h[:, -1, :]  # last (most recent) step


class _TransformerEncoder(nn.Module):
    """Transformer encoder over one timeframe's sequence, mean-pooled."""

    def __init__(self, n_features: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, W, F) -> (B, d_model)
        h = self.input_proj(x)
        h = self.encoder(h)
        return h.mean(dim=1)


class CrossTimeframeNet(nn.Module):
    """Full Model 1 network."""

    def __init__(
        self,
        n_features: int,
        n_timeframes: int,
        n_classes: int = 3,
        encoder: str = "tcn",
        d_model: int = 64,
        n_heads: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_timeframes = n_timeframes
        if encoder == "transformer":
            self.encoders = nn.ModuleList(
                _TransformerEncoder(n_features, d_model, n_heads, encoder_layers, dropout)
                for _ in range(n_timeframes)
            )
        else:
            self.encoders = nn.ModuleList(
                _TCNEncoder(n_features, d_model, encoder_layers, dropout)
                for _ in range(n_timeframes)
            )

        # Learnable per-timeframe positional embedding (identifies which tf a token is).
        self.tf_embedding = nn.Parameter(torch.zeros(1, n_timeframes, d_model))
        nn.init.normal_(self.tf_embedding, std=0.02)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=1)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tf, W, F)
        token_list = [enc(x[:, i]) for i, enc in enumerate(self.encoders)]  # each (B, d_model)
        tokens = torch.stack(token_list, dim=1)  # (B, n_tf, d_model)
        tokens = tokens + self.tf_embedding
        fused = self.fusion(tokens)  # (B, n_tf, d_model)
        pooled = fused.mean(dim=1)  # (B, d_model)
        return self.head(pooled)  # (B, n_classes) logits
