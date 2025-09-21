import torch.nn as nn


class AttentionLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h, _ = self.attention(x, x, x)

        h, _ = self.lstm(h)

        batch_size, timestep, hidden_dim = h.shape

        h = h.reshape(-1, hidden_dim)

        y = self.fc(h)

        y = y.reshape(timestep, batch_size, -1)

        y = y[-1]

        return y
