import torch
import torch.nn as nn


class RVAE(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):

        super().__init__()

        self.encoder_lstm = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)

        self.encoder_fc = nn.Linear(hidden_dim2, hidden_dim3)

        self.encoder_mu_fc = nn.Linear(hidden_dim3, hidden_dim4)

        self.encoder_log_var_fc = nn.Linear(hidden_dim3, hidden_dim4)

        self.decoder_fc1 = nn.Linear(hidden_dim4, hidden_dim3)

        self.decoder_fc2 = nn.Linear(hidden_dim3, hidden_dim2)

        self.decoder_fc3 = nn.Linear(hidden_dim2, hidden_dim1)

        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, log_var):

        std = torch.exp(log_var / 2)

        eps = torch.randn_like(std)

        return mu + eps * std

    def calculate_kl_divergence(self, mu, log_var):

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return kl_divergence

    def forward(self, x):

        h, _ = self.encoder_lstm(x)
        h = self.relu(h)

        batch_size, timestep, hidden_dim = h.shape

        h = h.reshape(-1, hidden_dim)

        h = self.encoder_fc(h)

        mu = self.encoder_mu_fc(h)

        log_var = self.encoder_log_var_fc(h)

        h = self.reparameterize(
            mu=mu,
            log_var=log_var
        )

        h = self.decoder_fc1(h)
        h = self.relu(h)

        h = self.decoder_fc2(h)
        h = self.relu(h)

        h = self.decoder_fc3(h)
        y = self.sigmoid(h)

        kl_divergence = self.calculate_kl_divergence(
            mu=mu,
            log_var=log_var
        )

        return y, kl_divergence
