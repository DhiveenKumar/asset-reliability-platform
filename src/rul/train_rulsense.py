# =============================================================================
# train_rulsense.py — RULSense: Remaining Useful Life Regression
#
# Trains LSTM and GRU models to predict continuous RUL_hours from
# 60-hour sensor sequences.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.rul.rul_labels import derive_rul_labels
from src.rul.sequence_builder import build_sequences

logger = get_logger(__name__)


class SequenceDataset(Dataset):
    """
    PyTorch requires data wrapped in a Dataset class, which defines
    exactly two things: how many samples exist (__len__), and how
    to fetch one specific sample by index (__getitem__).

    This lets PyTorch's DataLoader automatically handle batching,
    shuffling, and iteration during training - we just need to
    tell it how to access one item.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """
    LSTM (Long Short-Term Memory) network for RUL regression.

    Plain terms: the LSTM reads the 60-hour sequence one timestep
    at a time, maintaining an internal "memory" that gets updated
    as it processes each hour. By the time it's read all 60 hours,
    that memory encodes a learned summary of the whole trajectory -
    we then pass that summary through a final linear layer to
    produce a single number: the predicted RUL.

    hidden_size: how much "memory capacity" the LSTM has - larger
    means it can capture more complex patterns, at the cost of
    more parameters to train.
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        # We only need the LAST timestep's output - it summarizes
        # the entire sequence after processing all 60 hours
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


class GRURegressor(nn.Module):
    """
    GRU (Gated Recurrent Unit) - a simpler, faster alternative to
    LSTM with fewer internal parameters (no separate "cell state",
    just a single hidden state). Often achieves similar accuracy
    to LSTM with less computation - useful to compare directly.
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, hidden = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


def prepare_rul_data(sequence_length: int = 60, stride: int = 6):
    """
    Loads validated data, derives RUL labels, builds sequences,
    scales features, and splits by asset (same leakage-prevention
    principle as AssetPulse).
    """
    logger.info("Preparing RUL training data...")

    sensor_df = pd.read_csv("data/validated/sensor_timeseries.csv")
    failure_events_df = pd.read_csv("data/raw/failure_events.csv")

    labeled_df = derive_rul_labels(sensor_df, failure_events_df)

    feature_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    # Scale features BEFORE sequencing - neural networks train far
    # more effectively when inputs are normalized to similar ranges,
    # since raw sensor units (RPM ~1800 vs vibration ~2.2) would
    # otherwise dominate the loss function purely due to scale,
    # not actual importance.
    scaler = StandardScaler()
    labeled_df[feature_columns] = scaler.fit_transform(labeled_df[feature_columns])

    X, y, asset_ids = build_sequences(
        labeled_df, feature_columns,
        sequence_length=sequence_length, stride=stride
    )

    # Scale the TARGET too - raw RUL values up to ~2573 hours make
    # for large, unstable gradients. Dividing by 1000 brings targets
    # into a much more neural-network-friendly range (roughly 0-2.5).
    # We'll need to multiply predictions back by 1000 at evaluation.
    y = y / 1000.0

    unique_assets = np.unique(asset_ids)
    np.random.seed(42)
    test_assets = np.random.choice(
        unique_assets, size=max(1, int(len(unique_assets) * 0.2)), replace=False
    )

    train_mask = ~np.isin(asset_ids, test_assets)
    test_mask = np.isin(asset_ids, test_assets)

    logger.info(f"Train sequences: {train_mask.sum():,} from "
                f"{len(unique_assets) - len(test_assets)} assets")
    logger.info(f"Test sequences:  {test_mask.sum():,} from "
                f"{len(test_assets)} assets")

    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask],
        len(feature_columns)
    )


def train_model(model, train_loader, epochs: int = 40, lr: float = 0.01):
    """
    Standard PyTorch training loop with learning rate scheduling.

    We normalize the TARGET (RUL_hours) too, not just features -
    predicting raw values in the 0-2573 range makes MSE loss huge
    and gradients unstable. Scaling y to a smaller range (done in
    prepare_rul_data now) combined with a higher initial learning
    rate lets the model converge meaningfully faster.

    ReduceLROnPlateau: automatically lowers the learning rate when
    the loss stops improving, letting us start aggressive (fast
    learning) and fine-tune more precisely as training progresses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{epochs} - MSE Loss: {avg_loss:.4f} "
                f"- LR: {current_lr:.5f}"
            )

    return model


def evaluate_rul_model(model, X_test, y_test, model_name: str) -> dict:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(X_tensor).numpy()

    # Un-scale both predictions and targets back to real hours
    predictions = predictions * 1000.0
    y_test = y_test * 1000.0

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1))) * 100

    logger.info(f"\\n--- {model_name} Results ---")
    logger.info(f"RMSE: {rmse:.2f} hours")
    logger.info(f"MAE:  {mae:.2f} hours")
    logger.info(f"MAPE: {mape:.2f}%")

    return {"model": model_name, "rmse": rmse, "mae": mae, "mape": mape}


def run_rulsense_training():
    logger.info("=" * 60)
    logger.info("RULSense - Remaining Useful Life Training")
    logger.info("=" * 60)

    X_train, y_train, X_test, y_test, n_features = prepare_rul_data()

    train_dataset = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    results = []

    logger.info("\\nTraining LSTM...")
    lstm_model = LSTMRegressor(n_features=n_features)
    lstm_model = train_model(lstm_model, train_loader)
    results.append(evaluate_rul_model(lstm_model, X_test, y_test, "LSTM"))

    logger.info("\\nTraining GRU...")
    gru_model = GRURegressor(n_features=n_features)
    gru_model = train_model(gru_model, train_loader)
    results.append(evaluate_rul_model(gru_model, X_test, y_test, "GRU"))

    results_df = pd.DataFrame(results)
    logger.info("\\n" + "=" * 60)
    logger.info("Model Comparison")
    logger.info("=" * 60)
    logger.info("\\n" + results_df.to_string(index=False))

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/rulsense_model_comparison.csv", index=False)

    return lstm_model, gru_model, results_df


if __name__ == "__main__":
    run_rulsense_training()
