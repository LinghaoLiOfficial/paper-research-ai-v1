import torch.nn as nn
import torch
from torch_geometric.nn import GATConv


class AttentionSTAE(nn.Module):

    def __init__(self, turbine_num, embedding_dim, input_dim, temporal_encoder_hidden_dim, graph_encoder_hidden_dim1,
                 graph_encoder_hidden_dim2, timestep,
                 device, dropout):

        super().__init__()

        self.timestep = timestep

        self.temporal_encoder = TemporalEncoder(
            turbine_num=turbine_num,
            embedding_dim=embedding_dim,
            input_dim=input_dim,
            hidden_dim=temporal_encoder_hidden_dim,
            device=device,
            dropout=dropout
        )

        self.temporal_decoder = TemporalDecoder(
            input_dim=input_dim,
            hidden_dim=temporal_encoder_hidden_dim,
            dropout=dropout
        )

        self.graph_encoder = GraphEncoder(
            input_dim=temporal_encoder_hidden_dim,
            hidden_dim1=graph_encoder_hidden_dim1,
            hidden_dim2=graph_encoder_hidden_dim2,
            dropout=dropout
        )

        self.graph_decoder = GraphDecoder(
            input_dim=temporal_encoder_hidden_dim,
            hidden_dim1=graph_encoder_hidden_dim1,
            hidden_dim2=graph_encoder_hidden_dim2,
            turbine_num=turbine_num,
            dropout=dropout
        )

    def forward(self, x, distance_adj, time_context_adj):

        temporal_encoder_h_list = []
        for i, tensor in enumerate(x):
            temporal_encoder_h = self.temporal_encoder(tensor, i)

            temporal_encoder_h_list.append(temporal_encoder_h)

        total_temporal_encoder_h = torch.stack(temporal_encoder_h_list, 1)

        total_temporal_encoder_h = total_temporal_encoder_h.reshape(-1, total_temporal_encoder_h.shape[1],
                                                                    total_temporal_encoder_h.shape[3])

        spatial_encoder_h, distance_adj = self.graph_encoder(total_temporal_encoder_h, distance_adj)

        spatial_decoder_h = self.graph_decoder(spatial_encoder_h, distance_adj)

        temporal_decoder_h_list = [t.reshape(spatial_decoder_h.shape[0], spatial_decoder_h.shape[2]) for t in
                                   torch.split(spatial_decoder_h, 1, dim=1)]

        y = []
        for tensor in temporal_decoder_h_list:
            tensor = tensor.reshape(-1, self.timestep, tensor.shape[1])

            temporal_decoder_h = self.temporal_decoder(tensor)

            y.append(temporal_decoder_h)

        return y


class TemporalEncoder(nn.Module):

    def __init__(self, turbine_num, embedding_dim, input_dim, hidden_dim, device, dropout):
        super(TemporalEncoder, self).__init__()

        self.embedding = nn.Embedding(turbine_num, embedding_dim)

        self.lstm = nn.LSTM(input_dim + embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)

        self.device = device

    def forward(self, x, i):
        embedding = self.embedding(torch.tensor(i).to(self.device))

        embedding = embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], len(embedding))

        h = torch.cat((x, embedding), dim=2)

        y, _ = self.lstm(h)

        return y


class TemporalDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout):
        super(TemporalDecoder, self).__init__()

        self.lstm = nn.LSTM(hidden_dim, input_dim, num_layers=2, batch_first=True, dropout=dropout)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, _ = self.lstm(x)

        y = self.sigmoid(h)

        return y


class GraphEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout):
        super(GraphEncoder, self).__init__()

        self.gat1 = GATConv(input_dim, hidden_dim1, heads=2, dropout=dropout)

        self.gat2 = GATConv(input_dim, hidden_dim1, heads=1, dropout=dropout)

        self.fc = nn.Linear(hidden_dim1, hidden_dim2)

        self.relu = nn.ReLU()

    def forward(self, x, adj):
        adj = torch.stack([adj] * x.shape[0], 0)
        adj = adj.reshape(adj.shape[1], -1)

        x = x.reshape(-1, x.shape[2])

        h = self.gat1(x, adj)
        h = self.relu(h)

        h = self.gat2(h, adj)
        h = self.relu(h)

        h = self.fc(h)
        y = self.relu(h)

        return y, adj


class GraphDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim1, turbine_num, hidden_dim2, dropout):
        super(GraphDecoder, self).__init__()

        self.turbine_num = turbine_num

        self.fc = nn.Linear(hidden_dim2, hidden_dim1)

        self.gat1 = GATConv(hidden_dim1, hidden_dim1, heads=2, dropout=dropout)

        self.gat2 = GATConv(input_dim, input_dim, heads=1, dropout=dropout)

        self.relu = nn.ReLU()

    def forward(self, x, adj):
        h = self.fc(x)
        h = self.relu(h)

        h = self.gat1(h, adj)
        h = self.relu(h)

        h = self.gat2(h, adj)
        y = self.relu(h)

        y = y.reshape(-1, self.turbine_num, y.shape[1])

        return y
