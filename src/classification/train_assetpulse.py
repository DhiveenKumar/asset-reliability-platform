# =============================================================================
# train_assetpulse.py — AssetPulse: Equipment Failure Classification
#
# Trains and compares 3 classifiers (Random Forest, XGBoost, LightGBM)
# to predict failure_within_7d, handling class imbalance properly.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(path="data/processed/assetpulse_features.csv") -> pd.DataFrame:
    logger.info(f"Loading engineered features from {path}...")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = "failure_within_7d",
    test_size: float = 0.2
):
    """
    Splits data into train/test sets.

    IMPORTANT: we split by ASSET, not by random row shuffling.
    If we randomly shuffled rows, readings from the same asset's
    same failure event could leak into both train and test sets,
    letting the model 'cheat' by seeing very similar data in both.
    Splitting by asset ensures test assets are genuinely unseen.
    """
    logger.info("Splitting data by asset (not random rows) to prevent leakage...")

    exclude_cols = [
        "asset_id", "timestamp", "operating_mode",
        "failure_within_7d", "failure_within_14d", "failure_within_30d"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    assets = df["asset_id"].unique()
    train_assets, test_assets = train_test_split(
        assets, test_size=test_size, random_state=42
    )

    train_df = df[df["asset_id"].isin(train_assets)]
    test_df = df[df["asset_id"].isin(test_assets)]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    logger.info(f"Train: {len(X_train):,} rows from {len(train_assets)} assets")
    logger.info(f"Test:  {len(X_test):,} rows from {len(test_assets)} assets")
    logger.info(f"Train positive rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Test positive rate:  {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test, feature_cols


def train_random_forest(X_train, y_train):
    """
    class_weight='balanced' automatically adjusts the loss function
    to penalize mistakes on the minority class (failures) more
    heavily, compensating for the 8.6% vs 91.4% imbalance.
    """
    logger.info("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """
    scale_pos_weight achieves the same class-imbalance correction
    as class_weight='balanced', but XGBoost's specific parameter
    for it — ratio of negative to positive examples.
    """
    logger.info("Training XGBoost...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    logger.info("Training LightGBM...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    logger.info(f"\\n--- {model_name} Results ---")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall:    {metrics['recall']:.3f}")
    logger.info(f"F1 Score:  {metrics['f1']:.3f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.3f}")

    return metrics


def run_assetpulse_training():
    logger.info("=" * 60)
    logger.info("AssetPulse - Failure Classification Training")
    logger.info("=" * 60)

    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    results = []
    models = {}

    rf_model = train_random_forest(X_train, y_train)
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))
    models["random_forest"] = rf_model

    xgb_model = train_xgboost(X_train, y_train)
    results.append(evaluate_model(xgb_model, X_test, y_test, "XGBoost"))
    models["xgboost"] = xgb_model

    lgb_model = train_lightgbm(X_train, y_train)
    results.append(evaluate_model(lgb_model, X_test, y_test, "LightGBM"))
    models["lightgbm"] = lgb_model

    results_df = pd.DataFrame(results)
    logger.info("\\n" + "=" * 60)
    logger.info("Model Comparison Summary")
    logger.info("=" * 60)
    logger.info("\\n" + results_df.to_string(index=False))

    best_model_name = results_df.loc[results_df["f1"].idxmax(), "model"]
    logger.info(f"\\nBest model by F1 score: {best_model_name}")

    os.makedirs("data/processed", exist_ok=True)
    results_df.to_csv("data/processed/assetpulse_model_comparison.csv", index=False)

    return models, results_df, X_test, y_test, feature_cols


if __name__ == "__main__":
    run_assetpulse_training()
