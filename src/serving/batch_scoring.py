# =============================================================================
# batch_scoring.py — AssetPulse Batch Scoring Pipeline
#
# Simulates an Azure ML Batch Endpoint scoring job: loads the
# registered production model from MLflow, scores all current
# asset readings, and outputs a maintenance priority report.
#
# In a real Azure deployment, this script's core logic is exactly
# what would run inside an Azure ML Batch Endpoint compute cluster,
# triggered on a schedule via Azure DevOps / Azure ML Pipelines.
# =============================================================================

import os
import sys
import mlflow
import pandas as pd
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
REGISTERED_MODEL_NAME = "AssetPulse-Production"


def load_production_model(stage: str = "Staging"):
    """
    Fetches the current registered model from MLflow Model Registry
    by name and stage — NOT by hardcoding a specific run ID or
    algorithm. This is exactly the abstraction the registry provides:
    the batch scoring job doesn't need to know whether the current
    production model is Random Forest, XGBoost, or something else
    entirely after a future retraining.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info(f"Loading model '{REGISTERED_MODEL_NAME}' (stage: {stage})...")

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)

    logger.info("Model loaded successfully")
    return model


def get_latest_readings_per_asset(features_path: str) -> pd.DataFrame:
    """
    In a real batch job, this would query the latest sensor readings
    for every asset from the data source (SCADA/PI Historian).

    For this portfolio project, we simulate that by taking the MOST
    RECENT reading per asset from our engineered features dataset -
    representing 'what does each asset look like right now, this
    scoring cycle.'
    """
    logger.info("Fetching latest readings per asset...")

    df = pd.read_csv(features_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    latest = (
        df.sort_values("timestamp")
        .groupby("asset_id")
        .tail(1)
        .reset_index(drop=True)
    )

    logger.info(f"Retrieved latest reading for {len(latest)} assets")
    return latest


def score_assets(model, latest_readings: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the model against all assets' latest readings in one batch,
    producing a failure risk score and priority ranking for each -
    exactly the output a maintenance planning dashboard would consume.
    """
    logger.info("Scoring all assets...")

    exclude_cols = [
        "asset_id", "timestamp", "operating_mode",
        "failure_within_7d", "failure_within_14d", "failure_within_30d"
    ]
    feature_cols = [c for c in latest_readings.columns if c not in exclude_cols]

    X = latest_readings[feature_cols]
    probabilities = model.predict_proba(X)[:, 1]

    results = latest_readings[["asset_id", "timestamp"]].copy()
    results["failure_risk_score"] = probabilities
    results["risk_category"] = pd.cut(
        results["failure_risk_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    results = results.sort_values("failure_risk_score", ascending=False)
    results["maintenance_priority_rank"] = range(1, len(results) + 1)

    logger.info("Scoring complete")
    for _, row in results.iterrows():
        logger.info(
            f"  Rank {row['maintenance_priority_rank']}: "
            f"{row['asset_id']} - Risk: {row['failure_risk_score']:.1%} "
            f"({row['risk_category']})"
        )

    return results


def run_batch_scoring_job():
    """
    Master function - simulates one complete Azure ML Batch Endpoint
    scoring run. In production this would be triggered on a schedule
    (e.g., nightly) via Azure ML Pipelines / Azure DevOps.
    """
    logger.info("=" * 60)
    logger.info("AssetPulse Batch Scoring Job")
    logger.info(f"Run timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    model = load_production_model()
    latest_readings = get_latest_readings_per_asset(
        "data/processed/assetpulse_features.csv"
    )
    results = score_assets(model, latest_readings)

    os.makedirs("data/processed", exist_ok=True)
    output_path = (
        f"data/processed/batch_scoring_results_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    results.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info(f"Batch scoring job complete")
    logger.info(f"   Assets scored: {len(results)}")
    logger.info(f"   High risk assets: {(results['risk_category']=='High').sum()}")
    logger.info(f"   Results saved to: {output_path}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_batch_scoring_job()
