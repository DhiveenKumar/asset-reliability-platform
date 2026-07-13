# =============================================================================
# root_cause_analysis.py — Root Cause Analysis for AssetGuardian
#
# For every flagged anomaly, identifies WHICH sensors deviate most
# from normal, and maps that pattern to a likely failure mode using
# engineering rules - not GenAI, just combining feature attribution
# with domain knowledge.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Engineering rules: which sensor deviation PATTERNS map to which
# likely failure modes. This is domain knowledge encoded directly,
# not learned from data - exactly like a real reliability engineer's
# diagnostic heuristics.
FAILURE_MODE_RULES = [
    {
        "name": "Bearing wear",
        "primary_signals": ["vibration_mm", "temperature_f"],
        "condition": lambda dev: dev.get("vibration_mm", 0) > 1.5
                                  and dev.get("temperature_f", 0) > 1.0
    },
    {
        "name": "Lubrication failure",
        "primary_signals": ["oil_quality_index", "temperature_f"],
        "condition": lambda dev: dev.get("oil_quality_index", 0) < -1.5
                                  and dev.get("temperature_f", 0) > 1.0
    },
    {
        "name": "Seal leakage / cavitation",
        "primary_signals": ["pressure_psi", "flow_rate"],
        "condition": lambda dev: dev.get("pressure_psi", 0) < -1.5
                                  and dev.get("flow_rate", 0) < -1.0
    },
    {
        "name": "Motor winding failure",
        "primary_signals": ["motor_current", "temperature_f"],
        "condition": lambda dev: dev.get("motor_current", 0) > 2.0
    },
    {
        "name": "Rotor imbalance / shaft misalignment",
        "primary_signals": ["vibration_mm", "rpm"],
        "condition": lambda dev: dev.get("vibration_mm", 0) > 2.0
                                  and abs(dev.get("rpm", 0)) > 1.0
    },
    {
        "name": "Early-stage lubrication degradation",
        "primary_signals": ["oil_quality_index"],
        "condition": lambda dev: dev.get("oil_quality_index", 0) < -1.5
    },
    {
        "name": "Flow restriction / suction blockage",
        "primary_signals": ["pressure_psi"],
        "condition": lambda dev: dev.get("pressure_psi", 0) < -2.0
    },
]


def compute_sensor_deviations(
    reading: pd.Series,
    normal_stats: dict,
    sensor_columns: list
) -> dict:
    """
    For one flagged reading, computes how many standard deviations
    each sensor is from its NORMAL operating mean - this is our
    simplified feature attribution: bigger deviation = more likely
    this sensor is contributing to the anomaly.

    (A full SHAP TreeExplainer approach would work on the Isolation
    Forest directly, but IsolationForest doesn't have a straightforward
    SHAP interface the way tree classifiers do - this z-score deviation
    approach achieves the same INTERPRETABILITY goal: "which sensors
    are unusual, and by how much" - using a simpler, transparent method.)
    """
    deviations = {}
    for sensor in sensor_columns:
        mean = normal_stats[sensor]["mean"]
        std = normal_stats[sensor]["std"]
        if std > 0:
            deviations[sensor] = (reading[sensor] - mean) / std
        else:
            deviations[sensor] = 0.0
    return deviations


def diagnose_root_cause(deviations: dict) -> dict:
    """
    Checks engineering rules in order, returns the FIRST matching
    failure mode - or 'Unclassified anomaly' if no rule matches
    (an honest fallback rather than forcing a guess).
    """
    for rule in FAILURE_MODE_RULES:
        if rule["condition"](deviations):
            top_signals = {
                s: round(deviations.get(s, 0), 2)
                for s in rule["primary_signals"]
            }
            return {
                "likely_cause": rule["name"],
                "contributing_signals": top_signals
            }

    # No rule matched - report the single most deviant sensor instead
    sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_sensor, top_value = sorted_devs[0]

    return {
        "likely_cause": "Unclassified anomaly",
        "contributing_signals": {top_sensor: round(top_value, 2)}
    }


def analyze_flagged_anomalies(
    features_path: str = "data/processed/assetpulse_features.csv",
    top_n: int = 10
):
    """
    Full root cause analysis pipeline: loads data, computes normal
    baselines per sensor, then for a sample of degrading/critical
    readings, diagnoses the likely root cause.
    """
    logger.info("=" * 60)
    logger.info("AssetGuardian Root Cause Analysis")
    logger.info("=" * 60)

    df = pd.read_csv(features_path)

    sensor_columns = [
        "vibration_mm", "temperature_f", "pressure_psi",
        "flow_rate", "rpm", "motor_current", "oil_quality_index"
    ]

    normal_df = df[df["operating_mode"] == "normal"]
    normal_stats = {
        sensor: {
            "mean": normal_df[sensor].mean(),
            "std": normal_df[sensor].std()
        }
        for sensor in sensor_columns
    }

    logger.info("Computed normal operating baselines for all sensors")

    critical_readings = df[df["operating_mode"] == "critical"].sample(
        n=min(top_n, len(df[df["operating_mode"] == "critical"])),
        random_state=42
    )

    results = []
    for _, reading in critical_readings.iterrows():
        deviations = compute_sensor_deviations(reading, normal_stats, sensor_columns)
        diagnosis = diagnose_root_cause(deviations)

        result = {
            "asset_id": reading["asset_id"],
            "timestamp": reading["timestamp"],
            "likely_cause": diagnosis["likely_cause"],
            "contributing_signals": diagnosis["contributing_signals"]
        }
        results.append(result)

        logger.info(f"\\nAsset: {reading['asset_id']}")
        logger.info(f"Likely Cause: {diagnosis['likely_cause']}")
        logger.info(f"Contributing Signals: {diagnosis['contributing_signals']}")

    results_df = pd.DataFrame(results)
    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv(
        "data/processed/assetguardian_root_cause_report.csv", index=False
    )

    logger.info("=" * 60)
    logger.info(f"Root cause analysis complete - {len(results)} cases diagnosed")
    logger.info("=" * 60)

    return results_df


if __name__ == "__main__":
    analyze_flagged_anomalies()
