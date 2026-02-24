from __future__ import annotations

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """Single-layer LSTM with optional initial hidden state.

    Input: (B, T, D)
    Output: (B, T, H), hidden
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, h0: tuple[torch.Tensor, torch.Tensor] | None = None):
        y, h = self.lstm(x, h0)
        return y, h

