from typing import Self

import torch
import torch.nn as nn

from parse_config import ModelConfig


class StockPredictionModel(nn.Module):
    def __init__(
        self: Self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        target_feature_idx: int,
        device: torch.device,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_dim: int = hidden_dim
        self.target_feature_idx: int = target_feature_idx
        self.device: torch.device = device
        # This variable is used outside of this script
        self.config: ModelConfig = model_config

        self.lstm: nn.LSTM = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            device=device,
        )

        # Apparently, batch norm and layer norm can help speed up training. They both adjust the mean and variance
        # of the data in different ways. Batch norm looks across an entire batch while adjusting features.
        # Layer norm looks across one day for each feature. Batch norm can handle a variety of different scales, and
        # scales everything to around the same scale level (I think).
        # This layer stabilizes/speeds up training by preventing vanishing/exploding gradients
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim, device=device)
        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(input_dim, device=device)

        # High dropout as financial data is noisy
        self.dropout: nn.Dropout = nn.Dropout(0.5).to(device)

        # Projection Head
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            # ReLU kills negative neurons, but having negative neurons could be valuable as they can be correlated to bearish trends
            nn.LeakyReLU().to(device),
            nn.Linear(hidden_dim // 2, output_dim, device=device),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        # Training data shape = (Batch Size, Sequence Length, Features)
        # PyTorch expects (Batch Size, Features, Sequence Length)
        x = x.permute(0, 2, 1)
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)

        # Take the last time step, which is the standard for many-to-one tasks, because the last step should
        # theoretically contain all of the information of the previous steps
        out = out[:, -1, :]
        out = self.layer_norm(out)

        out = self.dropout(out)

        # We're doing binary classification now
        out_logits: torch.Tensor = self.fc(out)

        return out_logits
