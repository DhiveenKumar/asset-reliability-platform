# =============================================================================
# shap_analysis.py — SHAP Explainability for AssetPulse
#
# Explains WHY the model made specific predictions, not just WHAT
# it predicted. Critical for maintenance engineers to trust alerts
# and for root-cause understanding.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.classification.train_assetpulse import (
    load_features, prepare_train_test_split,
    train_random_forest
)

logger = get_logger(__name__)


def compute_shap_values(model, X_test: pd.DataFrame, sample_size: int = 500):
    """
    Computes SHAP values for the trained model.

    SHAP works by asking: 'if I remove this feature, how much does
    the prediction change?' — repeated across every feature, in every
    possible combination/order, then averaged. This gives a fair,
    mathematically grounded attribution of how much each feature
    contributed to each individual prediction.

    We use TreeExplainer, which is specifically optimized for
    tree-based models like Random Forest/XGBoost/LightGBM — much
    faster than the generic SHAP explainer for these model types.

    sample_size limits computation to a subset of test rows, since
    SHAP computation is expensive for large datasets.
    """
    logger.info(f"Computing SHAP values on {sample_size} sample rows...")

    X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, Random Forest returns SHAP values
    # for both classes — we want the "positive" (failure) class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    logger.info("SHAP values computed successfully")
    return explainer, shap_values, X_sample


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Global feature importance: averaged across ALL predictions,
    which features matter most overall for this model?

    This answers: "In general, what does this model pay attention
    to?" — different from explaining one specific prediction.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    logger.info(f"\\nTop {top_n} most important features (global):")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']:<40} {row['mean_abs_shap']:.4f}")

    return importance_df


def explain_single_prediction(
    model,
    explainer,
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    row_index: int,
    top_n: int = 5
) -> dict:
    """
    Local explanation: for ONE specific prediction, which features
    pushed it toward 'failure' vs 'normal', and by how much?

    This is what a maintenance engineer would actually see for a
    specific alert: "Pump-04 flagged as high risk because of X, Y, Z"
    """
    row_shap = shap_values[row_index]
    row_features = X_sample.iloc[row_index]

    prediction_proba = model.predict_proba(
        X_sample.iloc[[row_index]]
    )[0, 1]

    contributions = pd.DataFrame({
        "feature": X_sample.columns,
        "value": row_features.values,
        "shap_contribution": row_shap
    }).sort_values("shap_contribution", key=abs, ascending=False).head(top_n)

    logger.info(f"\\nExplanation for row {row_index} "
                f"(predicted failure probability: {prediction_proba:.1%}):")
    for _, row in contributions.iterrows():
        direction = "increased" if row["shap_contribution"] > 0 else "decreased"
        logger.info(
            f"  {row['feature']:<35} = {row['value']:.2f} "
            f"({direction} risk by {abs(row['shap_contribution']):.4f})"
        )

    return {
        "predicted_probability": prediction_proba,
        "top_contributions": contributions.to_dict("records")
    }


def run_shap_analysis():
    logger.info("=" * 60)
    logger.info("SHAP Explainability Analysis")
    logger.info("=" * 60)

    df = load_features()
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)

    logger.info("Retraining Random Forest for SHAP analysis...")
    model = train_random_forest(X_train, y_train)

    explainer, shap_values, X_sample = compute_shap_values(model, X_test)

    importance_df = get_global_feature_importance(shap_values, feature_cols)

    # Find a genuinely high-risk prediction to explain as an example
    probas = model.predict_proba(X_sample)[:, 1]
    highest_risk_idx = int(np.argmax(probas))

    explanation = explain_single_prediction(
        model, explainer, shap_values, X_sample, highest_risk_idx
    )

    os.makedirs("data/processed", exist_ok=True)
    importance_df.to_csv(
        "data/processed/shap_global_importance.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("SHAP analysis complete")
    logger.info("=" * 60)

    return importance_df, explanation


if __name__ == "__main__":
    run_shap_analysis()
