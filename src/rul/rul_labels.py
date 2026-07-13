# =============================================================================
# rul_labels.py — Remaining Useful Life Label Derivation
#
# For every sensor reading, computes hours remaining until that
# asset's NEXT failure event. This is a continuous regression
# target, unlike AssetPulse's binary classification labels.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def derive_rul_labels(
    sensor_df: pd.DataFrame,
    failure_events_df: pd.DataFrame,
    max_rul_cap: int = 3000
) -> pd.DataFrame:
    """
    For each reading, finds the NEXT failure event (in time) for
    that asset, and computes RUL_hours = time until that failure.

    Readings that occur AFTER an asset's last known failure have
    no valid "next failure" to count down to - we drop these,
    since we cannot know their true remaining life from this data.

    max_rul_cap: readings very early in a long normal phase can
    have huge RUL values (e.g. 2000+ hours). We cap this because:
    1. Extremely large regression targets destabilize training
    2. Practically, "3000+ hours away" and "5000+ hours away"
       are operationally the same answer: "not a near-term concern"
    """
    logger.info("Deriving RUL labels...")

    df = sensor_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    failures = failure_events_df.copy()
    failures["failure_date"] = pd.to_datetime(failures["failure_date"])

    df["RUL_hours"] = np.nan

    for asset_id in df["asset_id"].unique():
        asset_mask = df["asset_id"] == asset_id
        asset_failure_times = np.sort(
            failures[failures["asset_id"] == asset_id]["failure_date"].values
        )

        if len(asset_failure_times) == 0:
            continue

        asset_timestamps = df.loc[asset_mask, "timestamp"].values

        # For each reading, find index of the NEXT failure after it
        next_failure_idx = np.searchsorted(
            asset_failure_times, asset_timestamps, side="right"
        )

        valid = next_failure_idx < len(asset_failure_times)

        rul = np.full(len(asset_timestamps), np.nan)
        next_failures = np.where(
            valid,
            asset_failure_times[np.clip(next_failure_idx, 0, len(asset_failure_times)-1)],
            np.datetime64('NaT')
        )

        rul_hours = (next_failures - asset_timestamps) / np.timedelta64(1, 'h')
        rul_hours = np.where(valid, rul_hours, np.nan)

        df.loc[asset_mask, "RUL_hours"] = rul_hours

    before_drop = len(df)
    df = df.dropna(subset=["RUL_hours"]).reset_index(drop=True)
    after_drop = len(df)

    logger.info(
        f"Dropped {before_drop - after_drop:,} rows with no future "
        f"failure to count down to ({after_drop:,} rows remain)"
    )

    n_capped = (df["RUL_hours"] > max_rul_cap).sum()
    df["RUL_hours"] = df["RUL_hours"].clip(upper=max_rul_cap)
    logger.info(f"Capped {n_capped:,} rows at max RUL of {max_rul_cap} hours")

    logger.info(
        f"RUL_hours stats: min={df['RUL_hours'].min():.1f}, "
        f"max={df['RUL_hours'].max():.1f}, "
        f"mean={df['RUL_hours'].mean():.1f}"
    )

    return df


if __name__ == "__main__":
    sensor_df = pd.read_csv("data/validated/sensor_timeseries.csv")
    failure_events_df = pd.read_csv("data/raw/failure_events.csv")

    labeled_df = derive_rul_labels(sensor_df, failure_events_df)

    print(labeled_df[["asset_id", "timestamp", "operating_mode", "RUL_hours"]].head(20))
