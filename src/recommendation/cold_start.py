# =============================================================================
# cold_start.py — Cold-Start Fallback for Recommendation Engine
#
# Handles genuinely novel failure modes with zero historical data by
# falling back to content-based similarity using known failure mode
# CATEGORIES, rather than requiring direct co-occurrence history.
# =============================================================================

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
from src.recommendation.cosine_baseline import recommend_actions_cosine, build_cooccurrence_matrix

logger = get_logger(__name__)


# Content-based category mapping - encodes domain knowledge about
# WHICH KIND of failure mode this is, independent of any historical
# co-occurrence data. This is what a cold-start fallback relies on.
FAILURE_MODE_CATEGORIES = {
    "bearing_wear": "mechanical",
    "shaft_misalignment": "mechanical",
    "rotor_imbalance": "mechanical",
    "lubrication_failure": "mechanical",
    "seal_leakage": "hydraulic",
    "cavitation": "hydraulic",
    "impeller_erosion": "hydraulic",
    "motor_winding_failure": "electrical",
}


def get_recommendations_with_fallback(
    matrix: pd.DataFrame,
    failure_mode: str,
    top_k: int = 5
) -> dict:
    """
    Attempts direct historical lookup first. If the failure mode
    is genuinely unseen (cold-start), falls back to borrowing
    recommendations from the most similar KNOWN failure mode in
    the same category - explicitly flagging that this is a
    fallback, not a direct historical match, so a maintenance
    planner knows to treat it with appropriate caution.
    """
    if failure_mode in matrix.index:
        recs = recommend_actions_cosine(matrix, failure_mode, top_k)
        return {
            "failure_mode": failure_mode,
            "recommendations": recs,
            "method": "direct_historical_match",
            "confidence": "high"
        }

    logger.warning(f"'{failure_mode}' has no historical data - "
                    f"applying cold-start fallback")

    category = FAILURE_MODE_CATEGORIES.get(failure_mode)

    if category is None:
        logger.warning(f"Unknown category for '{failure_mode}' - "
                        f"cannot provide even a fallback recommendation")
        return {
            "failure_mode": failure_mode,
            "recommendations": [],
            "method": "no_data_available",
            "confidence": "none"
        }

    same_category_modes = [
        fm for fm, cat in FAILURE_MODE_CATEGORIES.items()
        if cat == category and fm in matrix.index
    ]

    if not same_category_modes:
        return {
            "failure_mode": failure_mode,
            "recommendations": [],
            "method": "no_similar_category_data",
            "confidence": "none"
        }

    borrowed_from = same_category_modes[0]
    recs = recommend_actions_cosine(matrix, borrowed_from, top_k)

    logger.info(f"Borrowed recommendations from '{borrowed_from}' "
                f"(same category: {category})")

    return {
        "failure_mode": failure_mode,
        "recommendations": recs,
        "method": f"category_fallback (borrowed from {borrowed_from})",
        "confidence": "low - verify with engineer before acting"
    }


def demonstrate_cold_start():
    logger.info("=" * 60)
    logger.info("Cold-Start Handling Demonstration")
    logger.info("=" * 60)

    df = pd.read_csv("data/recommendation/work_order_actions.csv")
    matrix = build_cooccurrence_matrix(df)

    # Test 1: A failure mode we HAVE seen (should be direct match)
    logger.info("\\n--- TEST 1: Known failure mode (bearing_wear) ---")
    result1 = get_recommendations_with_fallback(matrix, "bearing_wear")
    logger.info(f"Method: {result1['method']}, Confidence: {result1['confidence']}")

    # Test 2: A genuinely novel failure mode (simulating cold-start)
    logger.info("\\n--- TEST 2: Novel failure mode (coupling_failure) ---")
    FAILURE_MODE_CATEGORIES["coupling_failure"] = "mechanical"
    result2 = get_recommendations_with_fallback(matrix, "coupling_failure")
    logger.info(f"Method: {result2['method']}, Confidence: {result2['confidence']}")
    for r in result2["recommendations"]:
        logger.info(f"  {r['action']} (borrowed strength: {r['association_strength']})")

    import json
    with open("data/recommendation/cold_start_demo.json", "w") as f:
        json.dump([result1, result2], f, indent=2)

    logger.info("=" * 60)
    logger.info("Cold-start demonstration complete")
    logger.info("=" * 60)

    return result1, result2


if __name__ == "__main__":
    demonstrate_cold_start()
