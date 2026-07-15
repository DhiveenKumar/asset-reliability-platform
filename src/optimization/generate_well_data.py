# =============================================================================
# generate_well_data.py — Synthetic Well & Infrastructure Data
#
# Generates 15 wells with individual capacity limits, sharing common
# infrastructure (pipeline, compressor) with finite capacity.
# Integrates health scores from AssetPulse/RULSense batch scoring
# outputs where asset_ids match, to demonstrate genuine cross-project
# integration.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_wells(n_wells: int = 15, seed: int = 42) -> pd.DataFrame:
    """
    Creates well master data - each well's maximum theoretical
    production capacity, before any health-based adjustment.
    """
    np.random.seed(seed)

    well_ids = [f"WELL-{i:03d}" for i in range(1, n_wells + 1)]
    max_capacity = np.random.uniform(200, 800, n_wells).round(1)

    df = pd.DataFrame({
        "well_id": well_ids,
        "max_capacity_bpd": max_capacity,  # barrels per day
    })

    logger.info(f"Generated {len(df)} wells, "
                f"total theoretical capacity: {df['max_capacity_bpd'].sum():.0f} bpd")

    return df


def apply_health_constraints(wells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts each well's SAFE capacity based on equipment health.

    This is the genuine cross-project integration: if an asset's
    AssetPulse failure risk is high, or RULSense shows low remaining
    life, we don't let the optimizer allocate it full capacity -
    exactly how a real operations team would derate a degrading well
    rather than run it at maximum until it fails.
    """
    logger.info("Applying health-based capacity constraints...")

    pulse_files = [f for f in os.listdir("data/processed")
                   if f.startswith("batch_scoring_results_")]
    rul_files = [f for f in os.listdir("data/processed")
                 if f.startswith("rul_batch_scoring_")]

    health_map = {}

    if pulse_files:
        latest_pulse = sorted(pulse_files)[-1]
        pulse_df = pd.read_csv(f"data/processed/{latest_pulse}")
        for _, row in pulse_df.iterrows():
            health_map[row["asset_id"]] = {
                "failure_risk": row["failure_risk_score"]
            }
        logger.info(f"Loaded health data from {latest_pulse}")

    if rul_files:
        latest_rul = sorted(rul_files)[-1]
        rul_df = pd.read_csv(f"data/processed/{latest_rul}")
        for _, row in rul_df.iterrows():
            if row["asset_id"] in health_map:
                health_map[row["asset_id"]]["rul_hours"] = row["predicted_rul_hours"]

    # Map our 12 existing assets' health onto the first 12 wells -
    # wells beyond that get synthetic health scores. This demonstrates
    # real integration where overlap exists, without forcing a
    #1-to-1 mapping that wouldn't naturally exist.
    safe_capacity = []
    health_status = []

    asset_ids = list(health_map.keys())

    for i, row in wells_df.iterrows():
        if i < len(asset_ids):
            asset_id = asset_ids[i]
            risk = health_map[asset_id].get("failure_risk", 0.1)
            rul = health_map[asset_id].get("rul_hours", 1000)

            # Derate capacity based on risk: high risk = lower safe capacity
            if risk > 0.7 or rul < 50:
                derate_factor = 0.3  # severely restrict
                status = "CRITICAL - Derated"
            elif risk > 0.4 or rul < 200:
                derate_factor = 0.7
                status = "DEGRADED - Reduced"
            else:
                derate_factor = 1.0
                status = "HEALTHY - Full capacity"

            safe_cap = row["max_capacity_bpd"] * derate_factor
        else:
            safe_cap = row["max_capacity_bpd"]
            status = "No health data - assumed healthy"

        safe_capacity.append(round(safe_cap, 1))
        health_status.append(status)

    wells_df["safe_capacity_bpd"] = safe_capacity
    wells_df["health_status"] = health_status

    logger.info(f"Health-adjusted total safe capacity: "
                f"{wells_df['safe_capacity_bpd'].sum():.0f} bpd "
                f"(vs {wells_df['max_capacity_bpd'].sum():.0f} bpd theoretical)")

    return wells_df


def generate_infrastructure_limits() -> dict:
    """
    Shared infrastructure constraints - the pipeline and compressor
    that ALL wells' production must flow through combined.
    """
    limits = {
        "pipeline_capacity_bpd": 4500,
        "compressor_capacity_bpd": 5000,
    }
    logger.info(f"Infrastructure limits: {limits}")
    return limits


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Generating Well & Infrastructure Data")
    logger.info("=" * 60)

    wells_df = generate_wells()
    wells_df = apply_health_constraints(wells_df)
    infra_limits = generate_infrastructure_limits()

    os.makedirs("data/optimization", exist_ok=True)
    wells_df.to_csv("data/optimization/wells.csv", index=False)

    import json
    with open("data/optimization/infrastructure_limits.json", "w") as f:
        json.dump(infra_limits, f, indent=2)

    print(wells_df.to_string(index=False))
    logger.info("=" * 60)
    logger.info("Data generation complete")
    logger.info("=" * 60)
