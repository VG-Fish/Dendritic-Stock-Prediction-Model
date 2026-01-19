from typing import Self

import torch
import torch.nn as nn


class StockPredictionModel(nn.Module):
    def __init__(
        self: Self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        device,
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_dim: int = hidden_dim
        self.device: torch.device = device

        self.lstm: nn.LSTM = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2,
            device=device,
        )
        self.fc: nn.Linear = nn.Linear(
            hidden_dim,
            output_dim,
            device=device,
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        h0: torch.Tensor = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=self.device
        )
        c0: torch.Tensor = torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim, device=self.device
        )

        out, (_, _) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])

        return out
