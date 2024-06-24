from typing import List
import math

import numpy as np
import torch
import pandas as pd
from torch import optim, nn
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


from src.config import Config
from src.models.graph.models import WeightedGNN, DealPredictor


GRAPH_EMBEDDING_SIZE: int = 32
GNN_EPOCHS: int = 100
DEALS_EPOCHS: int = 100
DEALS_BATCH_SIZE: int = 64


deals_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_DEALS_LOCATION)
addresses_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_ADDRESSES_LOCATION)
edges_df: pd.DataFrame = pd.read_csv(Config.POST_PROCESSED_EDGES_LOCATION)

# -- addresses graph neural networks --
node_features = torch.tensor(addresses_df[['centroid_X_ITM', 'centroid_Y_ITM']].values, dtype=torch.float)
edge_index = torch.tensor(edges_df[['index_left', 'index_right']].values.T, dtype=torch.long)
edge_weight = torch.tensor(edges_df['distance'].values, dtype=torch.float)
# Create the graph
graph_data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
# -------------------------------------

deals_categorical_features: torch.Tensor = \
    torch.tensor(deals_df[[
        Config.DEAL_TYPE_FIELD_NAME, Config.SETTLEMENT_NUMERIC_FIELD_NAME,
        Config.DEAL_YEAR_FIELD_NAME, Config.ADDRESS_IDX_FIELD_NAME
    ]].values, dtype=torch.long)
# Normalize numerical features
scaler_encoder = StandardScaler()
deals_numerical_features_names: List[str] = [
    Config.DEAL_DATE_FIELD_NAME, Config.ROOMS_NUMBER_FIELD_NAME,
    Config.SIZE_FIELD_NAME, Config.BUILDING_BUILD_YEAR_FIELD_NAME, Config.BUILDING_NUM_FLOORS_FIELD_NAME,
    Config.STARTING_FLOOR_NUMBER_FIELD_NAME, Config.NUMBER_FLOORS_FIELD_NAME,
    Config.X_ITM_FIELD_NAME, Config.Y_ITM_FIELD_NAME
]
deals_numerical_features: np.ndarray = scaler_encoder.fit_transform(deals_df[deals_numerical_features_names])
deals_numerical_features: torch.Tensor = torch.tensor(deals_numerical_features, dtype=torch.float)
deals_target_data: torch.Tensor = torch.tensor(deals_df[Config.DEAL_SUM_FIELD_NAME].values, dtype=torch.float).view(-1, 1)

# -- start training --

dataset = TensorDataset(deals_numerical_features, deals_categorical_features, deals_target_data)
train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = WeightedGNN(in_channels=node_features.size(1), hidden_channels=32, out_channels=GRAPH_EMBEDDING_SIZE).to(device)
deal_predictor: DealPredictor = \
    DealPredictor(in_channels=deals_numerical_features.size(1), hidden_channels=32, out_channels=1, num_layers=2, address_embedding_size=GRAPH_EMBEDDING_SIZE)
criterion = nn.MSELoss()
optimizer = optim.Adam(list(gnn_model.parameters()) + list(deal_predictor.parameters()), lr=0.001)


def train_loop() -> float:
    for _ in tqdm(range(DEALS_EPOCHS), desc="Deals iteration"):
        loss_sum: float = 0
        num_batches: int = 0
        for batch_deals_numerical_features, batch_deals_categorical_features, batch_targets in tqdm(train_loader, total=math.ceil(len(train_dataset) / DEALS_BATCH_SIZE)):
            addresses_embeddings: torch.Tensor = \
                gnn_model(graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.edge_weight.to(device))
            optimizer.zero_grad()
            batch_deals_numerical_features, batch_deals_categorical_features, batch_targets = \
                batch_deals_numerical_features.to(device), batch_deals_categorical_features.to(device), batch_targets.to(device)
            addresses_indexes: torch.Tensor = batch_deals_categorical_features[:, -1]
            batch_addresses_embeddings: torch.Tensor = addresses_embeddings[addresses_indexes]
            outputs: torch.Tensor = deal_predictor(batch_deals_numerical_features, batch_addresses_embeddings)
            loss: torch.Tensor = criterion(outputs, batch_targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            num_batches += 1

        print(f"Training loss: {loss_sum / num_batches}")


for graph_epoch in range(GNN_EPOCHS):  # number of epochs
    epoch_loss: float = train_loop()
    print(f'Epoch {graph_epoch+1}, Loss: {epoch_loss}')


