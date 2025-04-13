import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# ----------------------------
# 1. Constants and Utility Functions
# ----------------------------
MIN_BET = 1.0
MAX_BET = 10.0  # adjust based on your data range
MAX_SEQ_LENGTH = 10
RATIOS = ["ratio1", "ratio2", "ratio3"]


def scale_features(features: pd.DataFrame) -> pd.DataFrame:
    return (features - MIN_BET) / (MAX_BET - 1)


def prepare_features(features: pd.DataFrame) -> np.array:
    features_scaled = scale_features(features)
    features_scaled = features_scaled.to_numpy()
    # Use the last MAX_SEQ_LENGTH timesteps
    features_scaled = features_scaled[-MAX_SEQ_LENGTH:, :]
    return features_scaled


def remove_consecutive_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df[RATIOS].shift()
    mask = ~((df[RATIOS] == shifted).all(axis=1))
    return df[mask]


def process_ratios(group_df: pd.DataFrame) -> pd.DataFrame:
    group_df[RATIOS] = group_df[RATIOS].astype(float)
    group_df = remove_consecutive_duplicates(group_df)
    return group_df


def prepare_data_for_model(grouped_data) -> Tuple[np.array, np.array]:
    x, y = [], []
    for _, group_df in grouped_data:
        group_df = process_ratios(group_df)
        if group_df.shape[0] < MAX_SEQ_LENGTH:
            continue
        features = group_df[RATIOS]
        # Extract target as the last row's outcome columns.
        # We assume exactly one of the outcomes is 1.
        target_vector = (
            group_df[["bet1_won", "bet2_won", "tie_won"]].values[-1].astype(float)
        )
        target = np.argmax(
            target_vector
        )  # Convert one-hot to a class index (0: bet1, 1: bet2, 2: tie)
        prepared_features = prepare_features(features)
        x.append(prepared_features)
        y.append(target)
    x = np.array(x)
    y = np.array(y)
    return x, y


# ----------------------------
# 2. Data Loading and Preparation
# ----------------------------
# Load your data; adjust the filename and delimiter as necessary.
df = data

# Ensure the odds columns are numeric
for col in RATIOS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert outcome columns to integers (0/1)
df["bet1_won"] = df["bet1_won"].astype(int)
df["tie_won"] = df["tie_won"].astype(int)
df["bet2_won"] = df["bet2_won"].astype(int)

# Group by a game identifier (here we use "unique_id")
grouped_data = df.groupby("unique_id")
x_data, y_data = prepare_data_for_model(grouped_data)

print("x_data shape:", x_data.shape)  # Expected shape: (num_samples, MAX_SEQ_LENGTH, 3)
print("y_data shape:", y_data.shape)  # Expected shape: (num_samples,)


# ----------------------------
# 3. PyTorch Dataset and DataLoader
# ----------------------------
class SoccerOddsTorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = SoccerOddsTorchDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# ----------------------------
# 4. Positional Encoding Module
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


# ----------------------------
# 5. Transformer Forecast Model
# ----------------------------
class TransformerForecastModel(nn.Module):
    def __init__(
        self,
        input_dim=3,
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        num_classes=3,
    ):
        super().__init__()
        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        # Use the last time step for classification
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        last_step = x[:, -1, :]  # Take last time step representation
        logits = self.fc(last_step)  # (batch, num_classes)
        return logits


# ----------------------------
# 6. Model Training Loop
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerForecastModel().to(device)
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Adjust as needed

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
