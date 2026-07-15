# =============================================================================
# generate_maintenance_data.py — Maintenance Action Co-occurrence Data
#
# Generates realistic historical records of which maintenance actions
# (parts, inspections, tools) were performed together for each failure
# mode, reusing failure modes from the existing Asset Reliability
# Platform data.
# =============================================================================

import os
import sys
import random
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Real-world-inspired mapping: which actions are TYPICALLY needed
# for each failure mode, encoding genuine domain knowledge - this
# becomes the "ground truth" pattern our recommendation models
# should learn to reconstruct from historical co-occurrence alone
FAILURE_MODE_ACTIONS = {
    "bearing_wear": [
        "Bearing Replacement Kit", "Lubrication Service", "Alignment Check",
        "Vibration Analysis", "Oil Sample Test"
    ],
    "shaft_misalignment": [
        "Alignment Check", "Coupling Inspection", "Vibration Analysis",
        "Shaft Repair Kit"
    ],
    "rotor_imbalance": [
        "Rotor Balancing Kit", "Vibration Analysis", "Bearing Inspection"
    ],
    "lubrication_failure": [
        "Lubrication Service", "Oil Sample Test", "Filter Replacement",
        "Seal Inspection"
    ],
    "seal_leakage": [
        "Seal Replacement Kit", "Gasket Kit", "Pressure Test",
        "Housing Inspection"
    ],
    "cavitation": [
        "Impeller Inspection", "Suction Line Check", "Pressure Test",
        "Flow Rate Calibration"
    ],
    "impeller_erosion": [
        "Impeller Replacement Kit", "Flow Rate Calibration",
        "Housing Inspection"
    ],
    "motor_winding_failure": [
        "Motor Winding Repair Kit", "Insulation Test",
        "Motor Current Calibration", "Cooling System Check"
    ],
}


def generate_work_order_action_records(n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Simulates historical work order records - each row is one
    historical repair event, listing the failure mode and the
    SPECIFIC subset of actions actually performed (real historical
    data would have some variation, not every action every time).
    """
    random.seed(seed)
    np.random.seed(seed)

    records = []
    failure_modes = list(FAILURE_MODE_ACTIONS.keys())

    for i in range(n_records):
        failure_mode = random.choice(failure_modes)
        possible_actions = FAILURE_MODE_ACTIONS[failure_mode]

        # Not every historical repair used ALL possible actions -
        # simulate realistic partial overlap (2 to all actions)
        n_actions = random.randint(2, len(possible_actions))
        actions_taken = random.sample(possible_actions, n_actions)

        for action in actions_taken:
            records.append({
                "work_order_id": f"WO-{i+1:05d}",
                "failure_mode": failure_mode,
                "action": action
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} action records across "
                f"{n_records} historical work orders")
    logger.info(f"Unique failure modes: {df['failure_mode'].nunique()}")
    logger.info(f"Unique actions: {df['action'].nunique()}")

    return df


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Generating Maintenance Action Co-occurrence Data")
    logger.info("=" * 60)

    df = generate_work_order_action_records()

    os.makedirs("data/recommendation", exist_ok=True)
    df.to_csv("data/recommendation/work_order_actions.csv", index=False)

    print(df.head(20).to_string(index=False))
    logger.info("=" * 60)
    logger.info("Data generation complete")
    logger.info("=" * 60)
