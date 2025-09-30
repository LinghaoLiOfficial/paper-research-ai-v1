import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):

        super().__init__()

        self.encoder_fc1 = nn.Linear(hidden_dim1, hidden_dim2)

        self.encoder_fc2 = nn.Linear(hidden_dim2, hidden_dim3)

        self.encoder_fc3 = nn.Linear(hidden_dim3, hidden_dim4)

        self.decoder_fc1 = nn.Linear(hidden_dim4, hidden_dim3)

        self.decoder_fc2 = nn.Linear(hidden_dim3, hidden_dim2)

        self.decoder_fc3 = nn.Linear(hidden_dim2, hidden_dim1)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = x.reshape(-1, x.shape[2])

        h = self.encoder_fc1(h)
        h = self.relu(h)

        h = self.encoder_fc2(h)
        h = self.relu(h)

        h = self.encoder_fc3(h)
        h = self.relu(h)

        h = self.decoder_fc1(h)
        h = self.relu(h)

        h = self.decoder_fc2(h)
        h = self.relu(h)

        h = self.decoder_fc3(h)
        h = self.sigmoid(h)

        y = h.reshape(-1, x.shape[1], x.shape[2])

        return y
