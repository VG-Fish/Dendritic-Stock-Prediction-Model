from typing import Self, Tuple

import torch
import torch.nn as nn


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
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.hidden_dim: int = hidden_dim
        self.target_feature_idx: int = target_feature_idx
        self.device: torch.device = device

        self.lstm: nn.LSTM = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            device=device,
        )
        # Doing bidirectional = True doubles the amount of hidden units
        # This layer stabilizes/speeds up training by preventing vanishing/exploding gradients
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(hidden_dim, device=device)
        self.dropout: nn.Dropout = nn.Dropout(0.2)

        # Projection Head
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            # ReLU kills negative neurons, but having negative neurons could be valuable as they can be correlated to bearish trends
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, output_dim, device=device),
        )

    def forward(
        self: Self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Doing instance normalization for each input tensor to force the model to generalize
        # This technique is called RevIN
        # x shape: [Batch, Sequence, Features]
        x_mean: torch.Tensor = x.mean(dim=1, keepdim=True)
        x_std: torch.Tensor = (
            x.std(dim=1, keepdim=True) + 1e-5
        )  # epsilon to avoid div/0

        x_norm: torch.Tensor = (x - x_mean) / x_std
        batch_size, sequence_length = x.shape[0], x.shape[1]
        x_norm_flat: torch.Tensor = x_norm.view(batch_size, sequence_length, -1)

        out, _ = self.lstm(x_norm_flat)

        # Take the last time step, which is the standard for many-to-one tasks, because the last step should
        # theoretically contain all of the information of the previous steps
        out = out[:, -1, :]
        out = self.layer_norm(out)

        # out = normalized Z-Score here
        out_z_score: torch.Tensor = self.fc(out)

        # Denormalize out and rescale to match the original magnitude
        target_mean: torch.Tensor = x_mean[:, :, self.target_feature_idx]
        target_std: torch.Tensor = x_std[:, :, self.target_feature_idx]

        # This turns it back into log returns
        out_denormalized: torch.Tensor = (out_z_score * target_std) + target_mean

        return out_denormalized, target_mean, target_std


class DirectionalMSELoss(nn.Module):
    def __init__(self, penalty_factor: float = 5.0, scale_factor: float = 100.0):
        super().__init__()
        self.mse: nn.MSELoss = nn.MSELoss()
        self.penalty_factor: float = penalty_factor
        self.scale_factor: float = scale_factor

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Scale up the inputs as log returns make the inputs very small
        pred_scaled: torch.Tensor = y_pred * self.scale_factor
        true_scaled: torch.Tensor = y_true * self.scale_factor

        mse_loss: torch.Tensor = self.mse(pred_scaled, true_scaled)

        # Penalty applies if signs are different
        direction_penalty: torch.Tensor = torch.mean(
            torch.relu(-pred_scaled * true_scaled)
        )

        return mse_loss + (self.penalty_factor * direction_penalty)
