import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # -> input: (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_length, input_size)
        """

        # out: (batch_size, seq_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        out, (h_n, c_n) = self.lstm(x)

        # We only get the output from the last timestep
        # out[:, -1, :] => (batch_size, hidden_size)
        last_hidden = out[:, -1, :]

        out = self.fc(last_hidden)

        return out
