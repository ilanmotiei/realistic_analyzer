import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, ReLU
from torch_geometric.utils import add_self_loops, degree
from torch import nn


class WeightedGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(WeightedGNNLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = Linear(in_channels, out_channels)
        self.relu = ReLU()

    def forward(self, x, edge_index, edge_weight):
        # Add self-loops to the adjacency matrix.
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))

        # Compute normalization.
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Scale the node features by the computed norm values.
        return norm.view(-1, 1) * self.lin(x_j)

    def update(self, aggr_out):
        # Apply a non-linearity to aggregated messages.
        return self.relu(aggr_out)


class WeightedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        # Input layer
        self.layers.append(WeightedGNNLayer(in_channels, hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(WeightedGNNLayer(hidden_channels, hidden_channels))
        # Output layer
        self.layers.append(WeightedGNNLayer(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_weight):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)

        return x


class DealPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden_channels, out_channels, address_embedding_size: int, num_layers=5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_channels + address_embedding_size, hidden_channels))
        for _ in range(num_layers - 3):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, deals_numerical_features: torch.Tensor, deals_address_embedding: torch.Tensor) -> torch.Tensor:
        deals_features: torch.Tensor = self.layers[0](deals_numerical_features)
        features: torch.Tensor = torch.cat([deals_features, deals_address_embedding], dim=1)
        for layer in self.layers[1:]:
            features: torch.Tensor = layer(features)

        return features
