import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):

        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h, _ = self.attention(x, x, x)

        batch_size, timestep, hidden_dim = h.shape

        h = h.reshape(-1, hidden_dim)

        h = self.fc1(h)

        y = self.fc2(h)

        y = y.reshape(timestep, batch_size, -1)

        y = y[-1]

        return y
