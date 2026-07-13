# =============================================================================
# engineering.py — Feature Engineering Pipeline
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def derive_failure_labels(sensor_df, failure_events_df, horizons_days):
    logger.info("Deriving failure labels from historical failure events...")

    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    failures = failure_events_df.copy()
    failures["failure_date"] = pd.to_datetime(failures["failure_date"])

    for horizon in horizons_days:
        col_name = f"failure_within_{horizon}d"
        df[col_name] = 0

    for asset_id in df["asset_id"].unique():
        asset_mask = df["asset_id"] == asset_id
        asset_failures = failures[failures["asset_id"] == asset_id]["failure_date"].values

        if len(asset_failures) == 0:
            continue

        asset_timestamps = df.loc[asset_mask, "timestamp"].values

        for horizon in horizons_days:
            col_name = f"failure_within_{horizon}d"
            horizon_delta = np.timedelta64(horizon, 'D')

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


def engineer_rolling_features(df, sensor_columns, windows=[6, 24, 72]):
    logger.info(f"Engineering rolling features for windows: {windows}")

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    for sensor in sensor_columns:
        for window in windows:
            mean_col = f"{sensor}_roll_mean_{window}h"
            std_col = f"{sensor}_roll_std_{window}h"

            df[mean_col] = (
                df.groupby("asset_id")[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            df[std_col] = (
                df.groupby("asset_id")[sensor]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )

    df = df.fillna(0)
    logger.info(f"Added {len(sensor_columns) * len(windows) * 2} rolling features")
    return df


def engineer_trend_features(df, sensor_columns, lag_hours=24):
    logger.info(f"Engineering trend features with {lag_hours}h lag...")

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    for sensor in sensor_columns:
        lag_col = f"{sensor}_lag_{lag_hours}h"
        trend_col = f"{sensor}_trend_{lag_hours}h"

        df[lag_col] = df.groupby("asset_id")[sensor].shift(lag_hours)
        df[trend_col] = df[sensor] - df[lag_col]

    df = df.fillna(0)
    logger.info(f"Added {len(sensor_columns) * 2} trend/lag features")
    return df


def engineer_vibration_fft_features(df, window=24):
    logger.info("Engineering vibration FFT feature...")

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    def rolling_fft_energy(series, window):
        result = np.zeros(len(series))
        values = series.values
        for i in range(len(values)):
            start = max(0, i - window + 1)
            segment = values[start:i + 1]
            if len(segment) < 4:
                result[i] = 0
                continue
            fft_vals = np.abs(np.fft.rfft(segment - np.mean(segment)))
            result[i] = np.sum(fft_vals ** 2)
        return result

    df["vibration_fft_energy"] = (
        df.groupby("asset_id")["vibration_mm"]
        .transform(lambda x: rolling_fft_energy(x, window))
    )

    logger.info("Added vibration_fft_energy feature")
    return df


def engineer_maintenance_features(df, work_orders_df):
    logger.info("Engineering maintenance interval features...")

    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    wo = work_orders_df.copy()
    wo["work_order_date"] = pd.to_datetime(wo["work_order_date"])

    days_since = np.zeros(len(df))

    for asset_id in df["asset_id"].unique():
        asset_mask = df["asset_id"] == asset_id
        asset_wo_dates = np.sort(
            wo[wo["asset_id"] == asset_id]["work_order_date"].values
        )
        asset_timestamps = df.loc[asset_mask, "timestamp"].values

        if len(asset_wo_dates) == 0:
            continue

        idx = np.searchsorted(asset_wo_dates, asset_timestamps) - 1
        idx = np.clip(idx, 0, len(asset_wo_dates) - 1)

        last_maintenance = asset_wo_dates[idx]
        deltas = (asset_timestamps - last_maintenance) / np.timedelta64(1, 'D')
        deltas = np.where(idx >= 0, deltas, 999)

        days_since[asset_mask] = deltas

    df["days_since_maintenance"] = days_since
    logger.info("Added days_since_maintenance feature")
    return df


def run_feature_pipeline():
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)

    sensor_df = pd.read_csv("data/validated/sensor_timeseries.csv")
    failure_events_df = pd.read_csv("data/raw/failure_events.csv")
    work_orders_df = pd.read_csv("data/raw/work_orders.csv")

    sensor_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    df = derive_failure_labels(
        sensor_df, failure_events_df,
        config["assetpulse"]["prediction_horizons_days"]
    )

    df = engineer_rolling_features(df, sensor_columns)
    df = engineer_trend_features(df, sensor_columns)
    df = engineer_vibration_fft_features(df)
    df = engineer_maintenance_features(df, work_orders_df)

    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/assetpulse_features.csv"
    df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info("Feature engineering complete")
    logger.info(f"   Total rows: {len(df):,}")
    logger.info(f"   Total columns: {len(df.columns)}")
    logger.info(f"   Saved to: {output_path}")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    run_feature_pipeline()
