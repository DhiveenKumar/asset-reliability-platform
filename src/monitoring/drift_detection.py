# =============================================================================
# drift_detection.py — Data & Prediction Drift Monitoring for AssetPulse
#
# Compares current incoming sensor data against the original training
# data distribution, and monitors prediction patterns over time, to
# detect when the model may need retraining.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list,
    threshold: float = 0.05
) -> pd.DataFrame:
    """
    Detects data drift using the Kolmogorov-Smirnov (KS) test.

    Plain terms: for each sensor column, this checks whether the
    current data's distribution (shape, spread, typical values)
    still statistically resembles the distribution the model was
    ORIGINALLY trained on. If the current data looks meaningfully
    different (p-value below threshold), that sensor has "drifted"
    - meaning the model's learned patterns may no longer apply well.

    Why this matters: a model trained on data where healthy vibration
    averaged 2.2mm might start performing poorly if, say, a sensor
    recalibration shifts all future readings to average 3.0mm -
    even though nothing is actually mechanically wrong. Drift
    detection catches this kind of silent data shift.
    """
    logger.info("Computing data drift (KS test) across sensor columns...")

    results = []

    for col in columns:
        ref_values = reference_df[col].dropna()
        curr_values = current_df[col].dropna()

        ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)

        drifted = p_value < threshold

        results.append({
            "column": col,
            "ks_statistic": round(ks_statistic, 4),
            "p_value": round(p_value, 4),
            "drifted": drifted,
            "reference_mean": round(ref_values.mean(), 3),
            "current_mean": round(curr_values.mean(), 3)
        })

        status = "⚠️  DRIFT DETECTED" if drifted else "✅ stable"
        logger.info(
            f"  {col:<25} p={p_value:.4f}  {status}  "
            f"(ref_mean={ref_values.mean():.2f}, "
            f"curr_mean={curr_values.mean():.2f})"
        )

    return pd.DataFrame(results)


def compute_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    threshold: float = 0.05
) -> dict:
    """
    Detects prediction drift: is the MODEL'S OUTPUT distribution
    shifting over time, even if we don't have new ground truth
    labels yet to measure accuracy directly?

    Example: if a model used to predict an average 10% failure
    risk across assets, and now it's predicting an average 40%
    risk, that's worth investigating - either equipment genuinely
    got riskier, or something upstream changed unexpectedly.
    """
    logger.info("Computing prediction drift...")

    ks_statistic, p_value = stats.ks_2samp(
        reference_predictions, current_predictions
    )

    drifted = p_value < threshold

    result = {
        "ks_statistic": round(ks_statistic, 4),
        "p_value": round(p_value, 4),
        "drifted": drifted,
        "reference_mean_risk": round(reference_predictions.mean(), 4),
        "current_mean_risk": round(current_predictions.mean(), 4)
    }

    status = "⚠️  PREDICTION DRIFT DETECTED" if drifted else "✅ stable"
    logger.info(
        f"  {status}  "
        f"(ref_mean_risk={result['reference_mean_risk']:.1%}, "
        f"curr_mean_risk={result['current_mean_risk']:.1%})"
    )

    return result


def check_false_positive_rate(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    risk_threshold: float = 0.5
) -> dict:
    """
    Once actual outcomes are known (did the asset really fail or
    not, after enough time has passed), this checks how many
    'high risk' predictions turned out to be false alarms.

    This is a business-facing metric: too many false alarms erode
    trust in the system, even if the model's statistical metrics
    (F1, ROC-AUC) look fine in aggregate.
    """
    merged = predictions.merge(actuals, on="asset_id", how="inner")

    high_risk_predicted = merged["failure_risk_score"] >= risk_threshold
    actual_failure = merged["actual_failure"] == 1

    false_positives = (high_risk_predicted & ~actual_failure).sum()
    total_high_risk_predictions = high_risk_predicted.sum()

    false_positive_rate = (
        false_positives / total_high_risk_predictions
        if total_high_risk_predictions > 0 else 0
    )

    logger.info(
        f"False positive rate: {false_positive_rate:.1%} "
        f"({false_positives}/{total_high_risk_predictions} "
        f"high-risk predictions did not result in actual failure)"
    )

    return {
        "false_positives": int(false_positives),
        "total_high_risk_predictions": int(total_high_risk_predictions),
        "false_positive_rate": round(false_positive_rate, 4)
    }


def run_monitoring_pipeline(
    features_path: str = "data/processed/assetpulse_features.csv"
):
    """
    Simulates a monitoring cycle: splits the dataset temporally
    into an 'older reference period' and a 'more recent period',
    then checks whether sensor distributions and prediction
    patterns have drifted between them.
    """
    logger.info("=" * 60)
    logger.info("AssetPulse Monitoring & Drift Detection")
    logger.info("=" * 60)

    df = pd.read_csv(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    split_point = int(len(df) * 0.7)
    reference_df = df.iloc[:split_point]
    current_df = df.iloc[split_point:]

    logger.info(
        f"Reference period: {reference_df['timestamp'].min()} "
        f"to {reference_df['timestamp'].max()} ({len(reference_df):,} rows)"
    )
    logger.info(
        f"Current period:   {current_df['timestamp'].min()} "
        f"to {current_df['timestamp'].max()} ({len(current_df):,} rows)"
    )

    sensor_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    drift_report = compute_data_drift(reference_df, current_df, sensor_columns)

    n_drifted = drift_report["drifted"].sum()

    logger.info("=" * 60)
    logger.info(f"Monitoring complete: {n_drifted}/{len(sensor_columns)} "
                f"sensors show statistically significant drift")

    if n_drifted > 0:
        logger.warning(
            "⚠️  Recommendation: investigate drifted sensors and "
            "consider retraining if drift persists across multiple "
            "monitoring cycles"
        )
    else:
        logger.info("✅ No significant drift detected - model remains valid")

    logger.info("=" * 60)

    os.makedirs("data/processed", exist_ok=True)
    drift_report.to_csv("data/processed/drift_report.csv", index=False)

    return drift_report


if __name__ == "__main__":
    run_monitoring_pipeline()
