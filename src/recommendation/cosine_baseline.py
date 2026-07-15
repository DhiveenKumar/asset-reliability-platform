# =============================================================================
# cosine_baseline.py — Cosine Similarity Recommendation Baseline
#
# Builds a failure-mode-to-action co-occurrence matrix and uses
# cosine similarity to recommend the top actions for any failure
# mode, including ranking by strength of association.
# =============================================================================

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a failure_mode x action matrix, where each cell counts
    how many times that action was performed for that failure mode
    across historical work orders.

    This is the "co-occurrence matrix" - the foundational data
    structure behind most classical recommendation systems, including
    collaborative filtering. Each row (failure mode) becomes a vector
    in "action space" - failure modes with similar repair patterns
    will have similar vectors.
    """
    logger.info("Building failure-mode x action co-occurrence matrix...")

    matrix = pd.crosstab(df["failure_mode"], df["action"])

    logger.info(f"Matrix shape: {matrix.shape[0]} failure modes x "
                f"{matrix.shape[1]} actions")

    return matrix


def recommend_actions_cosine(
    matrix: pd.DataFrame,
    failure_mode: str,
    top_k: int = 5
) -> list:
    """
    For a given failure mode, ranks ALL actions by how strongly
    associated they are - using the failure mode's row vector in
    the co-occurrence matrix directly (since we want actions FOR
    this failure mode, not similarity BETWEEN failure modes, we use
    the raw counts, normalized, rather than needing cosine similarity
    between rows here specifically).

    Cosine similarity's real value shows up when comparing NEW,
    unseen failure mode profiles against known ones - which is
    exactly what we'll demonstrate with cold-start handling next.
    """
    if failure_mode not in matrix.index:
        logger.warning(f"Failure mode '{failure_mode}' not in training data")
        return []

    action_scores = matrix.loc[failure_mode]
    action_scores = action_scores[action_scores > 0]
    top_actions = action_scores.sort_values(ascending=False).head(top_k)

    recommendations = [
        {"action": action, "association_strength": int(count)}
        for action, count in top_actions.items()
    ]

    return recommendations


def find_similar_failure_modes(
    matrix: pd.DataFrame,
    failure_mode: str,
    top_k: int = 3
) -> list:
    """
    THIS is where cosine similarity earns its place: given one
    failure mode's action pattern, find OTHER failure modes with
    similar repair signatures. Useful for cross-referencing -
    "failures that need similar interventions to this one."
    """
    if failure_mode not in matrix.index:
        return []

    similarity_matrix = cosine_similarity(matrix)
    sim_df = pd.DataFrame(
        similarity_matrix, index=matrix.index, columns=matrix.index
    )

    similar = sim_df[failure_mode].drop(failure_mode).sort_values(ascending=False).head(top_k)

    return [
        {"failure_mode": fm, "similarity": round(float(score), 3)}
        for fm, score in similar.items()
    ]


def run_cosine_baseline():
    logger.info("=" * 60)
    logger.info("Cosine Similarity Recommendation Baseline")
    logger.info("=" * 60)

    df = pd.read_csv("data/recommendation/work_order_actions.csv")
    matrix = build_cooccurrence_matrix(df)

    os.makedirs("data/recommendation", exist_ok=True)
    matrix.to_csv("data/recommendation/cooccurrence_matrix.csv")

    # Demonstrate on a few failure modes, including the one AssetGuardian
    # flagged for MOTO-003 earlier - genuine cross-project consistency
    test_modes = ["bearing_wear", "seal_leakage", "motor_winding_failure"]

    all_recommendations = []

    for mode in test_modes:
        logger.info(f"\\n--- Recommendations for: {mode} ---")
        recs = recommend_actions_cosine(matrix, mode, top_k=5)
        for r in recs:
            logger.info(f"  {r['action']} (strength: {r['association_strength']})")
            all_recommendations.append({"failure_mode": mode, **r})

        similar = find_similar_failure_modes(matrix, mode, top_k=2)
        logger.info(f"  Similar failure modes: {similar}")

    pd.DataFrame(all_recommendations).to_csv(
        "data/recommendation/cosine_recommendations.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("Cosine baseline complete")
    logger.info("=" * 60)

    return matrix


if __name__ == "__main__":
    run_cosine_baseline()
