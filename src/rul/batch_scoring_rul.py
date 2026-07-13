# =============================================================================
# batch_scoring_rul.py — RULSense Batch Scoring Pipeline
#
# Loads the best trained GRU model and scores all assets' latest
# 60-hour sequence to estimate Remaining Useful Life.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.rul.rul_labels import derive_rul_labels
from src.rul.train_rulsense import GRURegressor
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)


def load_rul_model(n_features: int, model_path: str = "mlflow/temp_models/GRU_state_dict.pt"):
    logger.info(f"Loading GRU model from {model_path}...")
    model = GRURegressor(n_features=n_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logger.info("Model loaded successfully")
    return model


def get_latest_sequences(features_path_sensor: str = "data/validated/sensor_timeseries.csv",
                          features_path_failures: str = "data/raw/failure_events.csv",
                          sequence_length: int = 60):
    """
    Builds ONE 60-hour sequence per asset - the most recent 60 hours
    of readings - representing 'what does each asset look like right
    now' for RUL estimation.
    """
    logger.info("Building latest sequences per asset...")

    sensor_df = pd.read_csv(features_path_sensor)
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])

    feature_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    scaler = StandardScaler()
    sensor_df[feature_columns] = scaler.fit_transform(sensor_df[feature_columns])

    sequences = []
    asset_ids = []

    for asset_id in sensor_df["asset_id"].unique():
        asset_df = sensor_df[sensor_df["asset_id"] == asset_id].sort_values("timestamp")
        if len(asset_df) < sequence_length:
            continue
        latest_window = asset_df[feature_columns].values[-sequence_length:]
        sequences.append(latest_window)
        asset_ids.append(asset_id)

    X = np.array(sequences, dtype=np.float32)
    logger.info(f"Built {len(X)} 'current state' sequences (one per asset)")

    return X, asset_ids


def score_rul(model, X: np.ndarray, asset_ids: list) -> pd.DataFrame:
    logger.info("Scoring RUL for all assets...")

    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        predictions = model(X_tensor).numpy()

    predictions_hours = predictions * 1000.0  # un-scale, matching training
    # Clip at 0 - negative RUL isn't operationally meaningful; it just
    # means the model's prediction error puts a near-failure asset
    # slightly past its estimated failure point. Clipping to 0 gives
    # a clean "failure imminent / already due" signal instead.
    predictions_hours = np.maximum(predictions_hours, 0)

    results = pd.DataFrame({
        "asset_id": asset_ids,
        "predicted_rul_hours": predictions_hours,
        "predicted_rul_days": predictions_hours / 24.0
    })
    results = results.sort_values("predicted_rul_hours")
    results["maintenance_urgency_rank"] = range(1, len(results) + 1)

    logger.info("Scoring complete")
    for _, row in results.iterrows():
        logger.info(
            f"  Rank {row['maintenance_urgency_rank']}: {row['asset_id']} - "
            f"RUL: {row['predicted_rul_hours']:.0f}h "
            f"(~{row['predicted_rul_days']:.1f} days)"
        )

    return results


def run_rul_batch_scoring():
    logger.info("=" * 60)
    logger.info("RULSense Batch Scoring Job")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    X, asset_ids = get_latest_sequences()
    model = load_rul_model(n_features=X.shape[2])
    results = score_rul(model, X, asset_ids)

    os.makedirs("data/processed", exist_ok=True)
    output_path = (
        f"data/processed/rul_batch_scoring_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info(f"RUL batch scoring complete")
    logger.info(f"   Assets scored: {len(results)}")
    logger.info(f"   Most urgent: {results.iloc[0]['asset_id']} "
                f"({results.iloc[0]['predicted_rul_hours']:.0f}h remaining)")
    logger.info(f"   Results saved to: {output_path}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_rul_batch_scoring()
