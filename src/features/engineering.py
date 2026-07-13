# =============================================================================
# engineering.py — Feature Engineering Pipeline
#
# Two responsibilities:
# 1. Derive failure labels by looking backward from actual failure
#    events (the "real world" technique, not the operating_mode shortcut)
# 2. Engineer predictive features from raw sensor readings
#    (rolling stats, FFT, lag features)
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def derive_failure_labels(
    sensor_df: pd.DataFrame,
    failure_events_df: pd.DataFrame,
    horizons_days: list
) -> pd.DataFrame:
    """
    For every sensor reading, determines whether the asset failed
    within each specified horizon (7, 14, 30 days) AFTER that reading.

    This mirrors exactly how a real company builds training labels:
    look backward from actual historical failure records, not from
    any 'God mode' label. We deliberately do NOT use operating_mode
    here, even though we could — this keeps our methodology honest
    and realistic.
    """
    logger.info("Deriving failure labels from historical failure events...")

    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    failures = failure_events_df.copy()
    failures["failure_date"] = pd.to_datetime(failures["failure_date"])

    for horizon in horizons_days:
        col_name = f"failure_within_{horizon}d"
        df[col_name] = 0

    # Process asset by asset for efficiency and clarity
    for asset_id in df["asset_id"].unique():
        asset_mask = df["asset_id"] == asset_id
        asset_failures = failures[failures["asset_id"] == asset_id]["failure_date"].values

        if len(asset_failures) == 0:
            continue

        asset_timestamps = df.loc[asset_mask, "timestamp"].values

        for horizon in horizons_days:
            col_name = f"failure_within_{horizon}d"
            horizon_delta = np.timedelta64(horizon, 'D')

            # For each reading, check if ANY failure falls within
            # (reading_time, reading_time + horizon]
            labels = np.zeros(len(asset_timestamps), dtype=int)
            for failure_time in asset_failures:
                within_window = (
                    (asset_timestamps < failure_time) &
                    (asset_timestamps >= failure_time - horizon_delta)
                )
                labels = labels | within_window

            df.loc[asset_mask, col_name] = labels.astype(int)

    for horizon in horizons_days:
        col_name = f"failure_within_{horizon}d"
        positive_rate = df[col_name].mean() * 100
        logger.info(
            f"  {col_name}: {df[col_name].sum():,} positive labels "
            f"({positive_rate:.2f}% of all readings)"
        )

    return df


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    sensor_df = pd.read_csv("data/validated/sensor_timeseries.csv")
    failure_events_df = pd.read_csv("data/raw/failure_events.csv")

    labeled_df = derive_failure_labels(
        sensor_df,
        failure_events_df,
        config["assetpulse"]["prediction_horizons_days"]
    )

    print(labeled_df[[
        "asset_id", "timestamp", "operating_mode",
        "failure_within_7d", "failure_within_14d", "failure_within_30d"
    ]].tail(20))
