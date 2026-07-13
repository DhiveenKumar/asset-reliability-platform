# =============================================================================
# sequence_builder.py — Sliding Window Sequence Construction for RULSense
#
# LSTM/GRU models require input shaped as sequences: for each
# prediction, a window of the last N hours of sensor readings,
# not just a single instantaneous row.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_sequences(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = "RUL_hours",
    sequence_length: int = 60,
    stride: int = 1
):
    """
    Builds sliding-window sequences PER ASSET.

    For asset with readings [r1, r2, r3, ..., rN] and
    sequence_length=60:
        sequence 1 = [r1...r60]  -> predict RUL at r60
        sequence 2 = [r2...r61]  -> predict RUL at r61
        (stride=1 means we slide forward by 1 row each time)

    CRITICAL: sequences never cross asset boundaries - a window
    can only contain readings from ONE asset's continuous history.

    Returns:
        X: shape (n_sequences, sequence_length, n_features)
        y: shape (n_sequences,) - the RUL_hours target for the
           LAST timestep of each sequence
        asset_ids: which asset each sequence belongs to (for
                   splitting train/test by asset later)
    """
    logger.info(
        f"Building sequences: length={sequence_length}h, stride={stride}"
    )

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    X_list, y_list, asset_id_list = [], [], []

    for asset_id in df["asset_id"].unique():
        asset_df = df[df["asset_id"] == asset_id].reset_index(drop=True)
        n_rows = len(asset_df)

        if n_rows < sequence_length:
            continue

        feature_values = asset_df[feature_columns].values
        target_values = asset_df[target_column].values

        for start in range(0, n_rows - sequence_length + 1, stride):
            end = start + sequence_length
            X_list.append(feature_values[start:end])
            y_list.append(target_values[end - 1])
            asset_id_list.append(asset_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    asset_ids = np.array(asset_id_list)

    logger.info(
        f"Built {len(X):,} sequences of shape "
        f"({sequence_length}, {len(feature_columns)})"
    )

    return X, y, asset_ids


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    from src.rul.rul_labels import derive_rul_labels

    sensor_df = pd.read_csv("data/validated/sensor_timeseries.csv")
    failure_events_df = pd.read_csv("data/raw/failure_events.csv")

    labeled_df = derive_rul_labels(sensor_df, failure_events_df)

    feature_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    X, y, asset_ids = build_sequences(
        labeled_df,
        feature_columns,
        sequence_length=config["rulsense"]["sequence_length_hours"],
        stride=config["rulsense"]["stride"]
    )

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sample y values: {y[:10]}")
