# =============================================================================
# generate_dataset.py — Synthetic Industrial Asset Dataset Generator
# =============================================================================

import os
import sys
import yaml
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path="configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_asset_master(config: dict) -> pd.DataFrame:
    logger.info("Generating equipment master data...")

    n_assets = config["data"]["n_assets"]
    equipment_types = config["data"]["equipment_types"]
    manufacturers = ["Flowserve", "Sulzer", "Baker Hughes", "Siemens", "GE"]
    locations = [
        "Platform A - Deck 1", "Platform A - Deck 2",
        "Platform B - Deck 1", "Onshore Facility 1",
        "Onshore Facility 2"
    ]

    rows = []
    for i in range(1, n_assets + 1):
        eq_type = equipment_types[i % len(equipment_types)]
        prefix = eq_type[:4].upper()
        asset_id = f"{prefix}-{i:03d}"

        install_date = datetime(2019, 1, 1) + timedelta(
            days=random.randint(0, 1500)
        )

        rows.append({
            "asset_id": asset_id,
            "equipment_type": eq_type,
            "manufacturer": random.choice(manufacturers),
            "install_date": install_date.strftime("%Y-%m-%d"),
            "location": random.choice(locations),
            "rated_capacity": round(random.uniform(200, 800), 1)
        })

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} assets across "
                f"{df['equipment_type'].nunique()} equipment types")
    return df


def generate_sensor_phase(n_hours, sensor_config, phase, start_time, asset_id):
    timestamps = [start_time + timedelta(hours=h) for h in range(n_hours)]

    drift_map = {"normal": 0.0, "degrading": 1.0, "critical": 2.5}
    drift_strength = drift_map[phase]

    rows = {"asset_id": asset_id, "timestamp": timestamps, "operating_mode": phase}

    for sensor, params in sensor_config.items():
        mean = params["normal_mean"]
        std = params["normal_std"]

        progression = np.linspace(0, 1, n_hours)
        drift = drift_strength * progression * std * 2

        if sensor in ["pressure_psi", "rpm"]:
            values = mean - drift + np.random.normal(0, std, n_hours)
        elif sensor == "oil_quality_index":
            values = mean - drift + np.random.normal(0, std * 0.5, n_hours)
        else:
            values = mean + drift + np.random.normal(0, std, n_hours)

        # Clip to physically realistic bounds per sensor
        clip_bounds = {
            "vibration_mm": (0, 20),
            "temperature_f": (-20, 400),
            "pressure_psi": (0, 500),
            "flow_rate": (0, 2000),
            "rpm": (0, 5000),
            "motor_current": (0, 200),
            "oil_quality_index": (0, 100),
            "ambient_temp": (-40, 150),
            "ambient_humidity": (0, 100),
        }
        if sensor in clip_bounds:
            low, high = clip_bounds[sensor]
            values = np.clip(values, low, high)

        rows[sensor] = np.round(values, 2)

    return pd.DataFrame(rows)


def generate_asset_lifecycle(asset_id, config, start_time):
    data_cfg = config["data"]
    sensor_cfg = config["sensors"]

    n_cycles = random.randint(
        data_cfg["lifecycle_cycles_per_asset"]["min"],
        data_cfg["lifecycle_cycles_per_asset"]["max"]
    )

    all_phases = []
    failure_events = []
    current_time = start_time

    for cycle in range(n_cycles):
        normal_hours = random.randint(
            data_cfg["hours_per_normal_phase"]["min"],
            data_cfg["hours_per_normal_phase"]["max"]
        )
        degrading_hours = random.randint(
            data_cfg["hours_per_degrading_phase"]["min"],
            data_cfg["hours_per_degrading_phase"]["max"]
        )
        critical_hours = random.randint(
            data_cfg["hours_per_critical_phase"]["min"],
            data_cfg["hours_per_critical_phase"]["max"]
        )

        normal_df = generate_sensor_phase(
            normal_hours, sensor_cfg, "normal", current_time, asset_id
        )
        current_time += timedelta(hours=normal_hours)
        all_phases.append(normal_df)

        degrading_df = generate_sensor_phase(
            degrading_hours, sensor_cfg, "degrading", current_time, asset_id
        )
        current_time += timedelta(hours=degrading_hours)
        all_phases.append(degrading_df)

        critical_df = generate_sensor_phase(
            critical_hours, sensor_cfg, "critical", current_time, asset_id
        )
        current_time += timedelta(hours=critical_hours)
        all_phases.append(critical_df)

        failure_time = current_time
        failure_mode = random.choice(config["failure_modes"])
        failure_events.append({
            "asset_id": asset_id,
            "failure_date": failure_time.strftime("%Y-%m-%d %H:%M:%S"),
            "failure_mode": failure_mode,
            "downtime_hours": random.randint(4, 72),
            "maintenance_action_taken": f"Replaced/repaired due to {failure_mode}"
        })

        current_time += timedelta(hours=random.randint(4, 72))

    sensor_df = pd.concat(all_phases, ignore_index=True)
    return sensor_df, failure_events



def generate_work_orders(
    equipment_master: pd.DataFrame,
    failure_events: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Generates work_orders.csv — maintenance visit log.

    Includes:
    - Corrective work orders (tied to actual failures)
    - Preventive work orders (routine scheduled checks,
      not tied to any failure)
    """
    logger.info("Generating work order history...")

    rows = []
    wo_counter = 1

    # Corrective work orders — one per failure event
    for _, failure in failure_events.iterrows():
        rows.append({
            "work_order_id": f"WO-{wo_counter:05d}",
            "asset_id": failure["asset_id"],
            "work_order_date": failure["failure_date"],
            "work_order_type": "corrective",
            "description": f"Repair for {failure['failure_mode']}",
            "technician_notes": (
                f"Addressed {failure['failure_mode']}. "
                f"Downtime: {failure['downtime_hours']} hours."
            )
        })
        wo_counter += 1

    # Preventive work orders — random routine checks per asset
    for _, asset in equipment_master.iterrows():
        n_preventive = random.randint(4, 10)
        install = datetime.strptime(asset["install_date"], "%Y-%m-%d")

        for _ in range(n_preventive):
            visit_date = install + timedelta(
                days=random.randint(30, 900)
            )
            rows.append({
                "work_order_id": f"WO-{wo_counter:05d}",
                "asset_id": asset["asset_id"],
                "work_order_date": visit_date.strftime("%Y-%m-%d %H:%M:%S"),
                "work_order_type": "preventive",
                "description": "Routine scheduled inspection",
                "technician_notes": "No issues found. Lubrication checked."
            })
            wo_counter += 1

    df = pd.DataFrame(rows)
    df = df.sort_values("work_order_date").reset_index(drop=True)

    logger.info(f"Generated {len(df)} work orders "
                f"({len(failure_events)} corrective, "
                f"{len(df) - len(failure_events)} preventive)")
    return df


def run_full_generation():
    config = load_config()
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    logger.info("=" * 60)
    logger.info("Industrial Asset Reliability Platform - Data Generation")
    logger.info("=" * 60)

    equipment_master = generate_asset_master(config)

    all_sensor_data = []
    all_failure_events = []
    base_start = datetime(2024, 1, 1)

    for _, asset_row in equipment_master.iterrows():
        asset_id = asset_row["asset_id"]
        logger.info(f"Generating lifecycle data for {asset_id}...")

        sensor_df, failures = generate_asset_lifecycle(
            asset_id, config, base_start
        )
        all_sensor_data.append(sensor_df)
        all_failure_events.extend(failures)

        logger.info(
            f"  {asset_id}: {len(sensor_df)} hourly readings, "
            f"{len(failures)} failure events"
        )

    sensor_timeseries = pd.concat(all_sensor_data, ignore_index=True)
    failure_events = pd.DataFrame(all_failure_events)

    os.makedirs(config["data"]["raw_path"], exist_ok=True)

    equipment_master.to_csv(
        f"{config['data']['raw_path']}/equipment_master.csv", index=False
    )
    sensor_timeseries.to_csv(
        f"{config['data']['raw_path']}/sensor_timeseries.csv", index=False
    )
    failure_events.to_csv(
        f"{config['data']['raw_path']}/failure_events.csv", index=False
    )

    work_orders = generate_work_orders(equipment_master, failure_events, config)
    work_orders.to_csv(
        f"{config['data']['raw_path']}/work_orders.csv", index=False
    )

    logger.info("=" * 60)
    logger.info("Generation complete")
    logger.info(f"   Equipment master: {len(equipment_master)} assets")
    logger.info(f"   Sensor readings:  {len(sensor_timeseries):,} rows")
    logger.info(f"   Failure events:   {len(failure_events)} events")
    logger.info(f"   Saved to: {config['data']['raw_path']}/")
    logger.info("=" * 60)

    return equipment_master, sensor_timeseries, failure_events


if __name__ == "__main__":
    run_full_generation()
