# =============================================================================
# train_assetguardian.py — AssetGuardian: Anomaly Detection
#
# Trains Isolation Forest (statistical baseline) to detect abnormal
# equipment behavior WITHOUT using failure labels - purely learning
# what "normal" operation looks like, then flagging deviations.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_anomaly_data(features_path: str = "data/processed/assetpulse_features.csv"):
    """
    Reuses AssetPulse's engineered features - rolling stats, trends,
    FFT energy - since these same features that predict failure risk
    also help distinguish normal from abnormal behavior patterns.

    CRITICAL: Isolation Forest trains ONLY on data we believe is
    "normal" (operating_mode == 'normal'), since the whole point is
    learning what healthy operation looks like. We then test against
    a MIX of normal + degrading + critical data to see if it
    correctly flags the abnormal periods.
    """
    logger.info("Preparing data for AssetGuardian anomaly detection...")

    df = pd.read_csv(features_path)

    exclude_cols = [
        "asset_id", "timestamp", "operating_mode",
        "failure_within_7d", "failure_within_14d", "failure_within_30d"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Split by asset first (same leakage prevention as before)
    assets = df["asset_id"].unique()
    np.random.seed(42)
    test_assets = np.random.choice(
        assets, size=max(1, int(len(assets) * 0.2)), replace=False
    )

    train_df = df[~df["asset_id"].isin(test_assets)]
    test_df = df[df["asset_id"].isin(test_assets)]

    # TRAIN only on normal data - this is what makes it "unsupervised"
    # in spirit: we never show the model examples of failure directly,
    # only what healthy operation looks like
    train_normal_df = train_df[train_df["operating_mode"] == "normal"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_normal_df[feature_cols])

    # TEST on a mix of all operating modes - this is where we check
    # if the model correctly flags degrading/critical periods
    X_test = scaler.transform(test_df[feature_cols])
    y_test_true_anomaly = (test_df["operating_mode"] != "normal").astype(int).values

    logger.info(f"Trained on {len(X_train):,} 'normal' readings only")
    logger.info(f"Testing on {len(X_test):,} readings "
                f"({y_test_true_anomaly.sum():,} true anomalies: "
                f"degrading/critical periods)")

    return X_train, X_test, y_test_true_anomaly, feature_cols, scaler


def train_isolation_forest(X_train: np.ndarray, contamination: float = 0.1):
    """
    Isolation Forest works by randomly partitioning data - anomalies
    are "easier to isolate" (require fewer random splits to separate
    from the rest) than normal points, since they sit further from
    the bulk of the data.

    contamination: our ESTIMATE of what fraction of data is anomalous.
    Since we train ONLY on data we labeled 'normal', this is really
    just a decision threshold - how conservative to be when flagging
    borderline points as anomalies vs normal.
    """
    logger.info("Training Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    logger.info("Isolation Forest trained")
    return model


def evaluate_anomaly_model(model, X_test, y_true, model_name: str) -> dict:
    """
    Isolation Forest predicts -1 for anomaly, 1 for normal - we
    convert to 1/0 to match our y_true convention (1 = anomaly).
    """
    raw_predictions = model.predict(X_test)
    y_pred = (raw_predictions == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logger.info(f"\\n--- {model_name} Results ---")
    logger.info(f"Precision: {precision:.3f}")
    logger.info(f"Recall:    {recall:.3f}")
    logger.info(f"F1 Score:  {f1:.3f}")
    logger.info(
        f"Flagged {y_pred.sum():,} anomalies out of {len(y_pred):,} readings "
        f"(true anomaly rate: {y_true.mean():.1%})"
    )

    return {
        "model": model_name, "precision": precision,
        "recall": recall, "f1": f1
    }


def run_assetguardian_training():
    logger.info("=" * 60)
    logger.info("AssetGuardian - Anomaly Detection Training")
    logger.info("=" * 60)

    X_train, X_test, y_test, feature_cols, scaler = prepare_anomaly_data()

    model = train_isolation_forest(X_train)
    metrics = evaluate_anomaly_model(model, X_test, y_test, "Isolation Forest")

    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        "data/processed/assetguardian_isoforest_results.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("AssetGuardian Isolation Forest training complete")
    logger.info("=" * 60)

    return model, metrics, scaler, feature_cols


if __name__ == "__main__":
    run_assetguardian_training()
