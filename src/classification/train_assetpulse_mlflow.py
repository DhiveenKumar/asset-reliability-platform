# =============================================================================
# train_assetpulse_mlflow.py — AssetPulse Training with MLflow Tracking
#
# Wraps the AssetPulse training pipeline with MLflow experiment
# tracking and model registry, so every run is logged, comparable,
# and the best model can be formally promoted to production.
# =============================================================================

import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.classification.train_assetpulse import (
    load_features, prepare_train_test_split,
    train_random_forest, train_xgboost, train_lightgbm,
    evaluate_model
)

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT_NAME = "AssetPulse_Failure_Classification"


def setup_mlflow():
    """
    Configures MLflow to store experiment data locally in ./mlflow/mlruns.
    In a real Azure deployment, this URI would instead point to an
    Azure ML tracking server, but the code and logic stay identical —
    only the tracking backend changes.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")


def train_and_log_model(
    model_name: str,
    train_fn,
    log_fn,
    X_train, y_train, X_test, y_test,
    params: dict
):
    """
    Trains one model inside an MLflow "run" — a single tracked
    execution. Everything logged here (params, metrics, the model
    file itself) becomes permanently queryable in MLflow's UI/API.
    """
    with mlflow.start_run(run_name=model_name):
        logger.info(f"\\n{'='*60}")
        logger.info(f"MLflow Run: {model_name}")
        logger.info(f"{'='*60}")

        # Log hyperparameters BEFORE training
        mlflow.log_params(params)

        model = train_fn(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, model_name)

        # Log metrics — these become comparable across all runs
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])

        # Log the actual trained model file as an artifact
        log_fn(model, "model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Run ID: {run_id}")

        return model, metrics, run_id


def register_best_model(results: list, run_ids: dict):
    """
    Promotes the best-performing model (by F1 score) to the
    MLflow Model Registry under a formal name, with version
    tracking. This is what allows a deployment pipeline to
    always fetch 'the current production AssetPulse model'
    without hardcoding which algorithm or run that refers to.
    """
    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["f1"].idxmax()]
    best_model_name = best_row["model"]
    best_run_id = run_ids[best_model_name]

    logger.info(f"\\n{'='*60}")
    logger.info(f"Registering best model: {best_model_name} "
                f"(F1={best_row['f1']:.3f})")
    logger.info(f"{'='*60}")

    model_uri = f"runs:/{best_run_id}/model"
    registered_name = "AssetPulse-Production"

    result = mlflow.register_model(model_uri, registered_name)

    logger.info(f"Registered as: {registered_name}, "
                f"version {result.version}")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=registered_name,
        version=result.version,
        stage="Staging"
    )
    logger.info(f"Model version {result.version} moved to Staging")

    return registered_name, result.version


def run_assetpulse_mlflow_pipeline():
    setup_mlflow()

    logger.info("=" * 60)
    logger.info("AssetPulse Training Pipeline with MLflow Tracking")
    logger.info("=" * 60)

    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    results = []
    run_ids = {}

    rf_model, rf_metrics, rf_run_id = train_and_log_model(
        "Random Forest", train_random_forest, mlflow.sklearn.log_model,
        X_train, y_train, X_test, y_test,
        params={"n_estimators": 200, "max_depth": 12, "class_weight": "balanced"}
    )
    results.append(rf_metrics)
    run_ids["Random Forest"] = rf_run_id

    xgb_model, xgb_metrics, xgb_run_id = train_and_log_model(
        "XGBoost", train_xgboost, mlflow.xgboost.log_model,
        X_train, y_train, X_test, y_test,
        params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}
    )
    results.append(xgb_metrics)
    run_ids["XGBoost"] = xgb_run_id

    lgb_model, lgb_metrics, lgb_run_id = train_and_log_model(
        "LightGBM", train_lightgbm, mlflow.lightgbm.log_model,
        X_train, y_train, X_test, y_test,
        params={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                "class_weight": "balanced"}
    )
    results.append(lgb_metrics)
    run_ids["LightGBM"] = lgb_run_id

    registered_name, version = register_best_model(results, run_ids)

    logger.info("=" * 60)
    logger.info("MLflow pipeline complete")
    logger.info(f"All 3 runs logged under experiment: {EXPERIMENT_NAME}")
    logger.info(f"Best model registered as: {registered_name} v{version}")
    logger.info(f"View results: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    logger.info("=" * 60)

    return results, registered_name, version


if __name__ == "__main__":
    run_assetpulse_mlflow_pipeline()
