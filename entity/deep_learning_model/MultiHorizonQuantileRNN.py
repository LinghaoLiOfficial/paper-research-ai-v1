import torch.nn as nn
import torch


class MultiHorizonQuantileRNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, context_dim, quantiles):

        super().__init__()

        self.context_dim = context_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        self.global_decoder = GlobalDecoder(
            hidden_dim=hidden_dim,
            input_dim=input_dim,
            context_dim=context_dim
        )

        self.local_decoder = LocalDecoder(
            context_dim=hidden_dim,
            output_dim=quantiles
        )

        self.quantiles = quantiles

    def forward(self, x):

        old_x = x[:, : x.size(1) - self.context_dim, :]
        future_x = x[:, x.size(1) - self.context_dim: x.size(1), :]

        h, _ = self.encoder(old_x)

        h = h[-1]

        global_context = self.global_decoder(h, future_x[:, 0, :])

        local_context = self.local_decoder(global_context, future_x[:, 1:, :])

        # Predict all quantiles for each future horizon

        quantile_predictions = []
        for i in range(future_x.size(1)):

            quantiles = self.local_decoder(local_context, future_x[:, i, :])

            quantile_predictions.append(quantiles)

        y = torch.stack(quantile_predictions, dim=1)

        return y


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):

        _, (h, cell) = self.lstm(x)

        return h, cell


class GlobalDecoder(nn.Module):

    def __init__(self, hidden_dim, input_dim, context_dim):

        super(GlobalDecoder, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + input_dim * context_dim, context_dim)

        # self.fc2 = nn.Linear(context_dim)

        # self.fc3 = nn.Linear(, )

        self.relu = nn.ReLU()

    def forward(self, h, future_x):

        combined = torch.cat((h, future_x), dim=1)

        y = self.fc(combined)

        return y


class LocalDecoder(nn.Module):

    def __init__(self, context_dim, output_dim):

        super(LocalDecoder, self).__init__()

        self.fc = nn.Linear(context_dim + output_dim, output_dim)

    def forward(self, context, future_x):

        combined = torch.cat((context, future_x), dim=1)


        y = self.fc(combined)

        return y



