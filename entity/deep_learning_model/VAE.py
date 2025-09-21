import torch.nn as nn
import torch


class VAE(nn.Module):

    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4):

        super().__init__()

        self.encoder_fc1 = nn.Linear(hidden_dim1, hidden_dim2)

        self.encoder_fc2 = nn.Linear(hidden_dim2, hidden_dim3)

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

        h = x.reshape(-1, x.shape[2])

        h = self.encoder_fc1(h)
        h = self.relu(h)

        h = self.encoder_fc2(h)
        h = self.relu(h)

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
        h = self.sigmoid(h)

        y = h.reshape(-1, x.shape[1], x.shape[2])

        kl_divergence = self.calculate_kl_divergence(
            mu=mu,
            log_var=log_var
        )

        return y, kl_divergence
