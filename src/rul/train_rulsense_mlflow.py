# =============================================================================
# train_rulsense_mlflow.py — RULSense Training with MLflow Tracking
# =============================================================================

import os
import sys
import mlflow
import mlflow.pytorch
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.rul.train_rulsense import (
    prepare_rul_data, train_model, evaluate_rul_model,
    LSTMRegressor, GRURegressor, SequenceDataset
)
from torch.utils.data import DataLoader

logger = get_logger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT_NAME = "RULSense_RUL_Regression"


def run_rulsense_mlflow_pipeline():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=" * 60)
    logger.info("RULSense Training Pipeline with MLflow Tracking")
    logger.info("=" * 60)

    X_train, y_train, X_test, y_test, n_features = prepare_rul_data()
    train_dataset = SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    results = []
    run_ids = {}

    for model_name, model_class in [("LSTM", LSTMRegressor), ("GRU", GRURegressor)]:
        with mlflow.start_run(run_name=model_name):
            logger.info(f"\\nTraining {model_name} (tracked)...")

            mlflow.log_params({
                "hidden_size": 64, "num_layers": 2,
                "epochs": 40, "lr": 0.01, "sequence_length": 60
            })

            model = model_class(n_features=n_features)
            model = train_model(model, train_loader, epochs=40, lr=0.01)
            metrics = evaluate_rul_model(model, X_test, y_test, model_name)

            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("mape", metrics["mape"])
            # Save model state_dict directly as an artifact rather than
            # using mlflow.pytorch.log_model's automatic export, which
            # has known issues tracing LSTM/GRU's dynamic batch dimension.
            # This is a common, valid alternative - MLflow still tracks
            # everything needed to reload the model (weights + architecture
            # metadata), just via a simpler manual artifact instead of
            # the auto-export mechanism.
            os.makedirs("mlflow/temp_models", exist_ok=True)
            model_path = f"mlflow/temp_models/{model_name}_state_dict.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("model_class", model_name)

            run_ids[model_name] = mlflow.active_run().info.run_id
            results.append(metrics)

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["rmse"].idxmin()]
    best_model_name = best_row["model"]
    best_run_id = run_ids[best_model_name]

    logger.info(f"\\nRegistering best model: {best_model_name} "
                f"(RMSE={best_row['rmse']:.2f})")

    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, "RULSense-Production")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="RULSense-Production", version=result.version, stage="Staging"
    )

    logger.info(f"Registered: RULSense-Production v{result.version} (Staging)")
    logger.info("=" * 60)

    return results_df, best_model_name, result.version


if __name__ == "__main__":
    run_rulsense_mlflow_pipeline()
