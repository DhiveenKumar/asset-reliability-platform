# =============================================================================
# schema.py — Data Validation for Industrial Asset Reliability Platform
#
# Uses Pandera to validate sensor data before it enters the feature
# engineering pipeline. Catches physically impossible values, missing
# data, and timestamp inconsistencies before they corrupt model training.
# =============================================================================

import os
import sys
import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# SCHEMA DEFINITION
#
# Each Column defines: the data type, and Checks — rules the data
# must satisfy. If any row violates a check, Pandera flags it.
# =============================================================================

sensor_schema = DataFrameSchema({
    "asset_id": Column(str, nullable=False),

    "timestamp": Column(
        pa.dtypes.Timestamp,
        nullable=False
    ),

    "operating_mode": Column(
        str,
        Check.isin(["normal", "degrading", "critical"]),
        nullable=False
    ),

    "vibration_mm": Column(
        float,
        Check.in_range(0, 20),
        nullable=False
    ),

    "temperature_f": Column(
        float,
        Check.in_range(-20, 400),
        nullable=False
    ),

    "pressure_psi": Column(
        float,
        Check.in_range(0, 500),
        nullable=False
    ),

    "flow_rate": Column(
        float,
        Check.in_range(0, 2000),
        nullable=False
    ),

    "rpm": Column(
        float,
        Check.in_range(0, 5000),
        nullable=False
    ),

    "motor_current": Column(
        float,
        Check.in_range(0, 200),
        nullable=False
    ),

    "oil_quality_index": Column(
        float,
        Check.in_range(0, 100),
        nullable=False
    ),

    "ambient_temp": Column(
        float,
        Check.in_range(-40, 150),
        nullable=False
    ),

    "ambient_humidity": Column(
        float,
        Check.in_range(0, 100),
        nullable=False
    ),
})


def validate_sensor_data(df: pd.DataFrame) -> tuple:
    """
    Validates sensor time-series data against the schema.

    Returns (valid_df, validation_report) where validation_report
    contains any errors found. Rather than crashing on the first
    error, this collects ALL violations so we can see the full
    picture of data quality issues at once.
    """
    logger.info("Starting sensor data validation...")
    logger.info(f"Input rows: {len(df):,}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    errors = []

    # Run schema validation, collecting all failures (lazy=True)
    try:
        validated_df = sensor_schema.validate(df, lazy=True)
        logger.info("✅ All rows passed schema validation")
        return validated_df, {"status": "passed", "errors": []}

    except pa.errors.SchemaErrors as e:
        failure_cases = e.failure_cases
        logger.warning(
            f"⚠️  Validation found {len(failure_cases)} rule violations"
        )

        for _, row in failure_cases.iterrows():
            error_detail = {
                "column": row.get("column", "unknown"),
                "check": row.get("check", "unknown"),
                "failure_case": row.get("failure_case", "unknown"),
                "index": row.get("index", "unknown")
            }
            errors.append(error_detail)
            logger.warning(f"   Column '{error_detail['column']}': "
                          f"{error_detail['check']} failed "
                          f"(value: {error_detail['failure_case']})")

        # Additional custom checks beyond schema rules
        duplicate_check = check_duplicate_timestamps(df)
        if duplicate_check["count"] > 0:
            errors.append(duplicate_check)

        return df, {"status": "failed", "errors": errors}


def check_duplicate_timestamps(df: pd.DataFrame) -> dict:
    """
    Custom check: no asset should have two readings at the
    exact same timestamp — this would indicate a sensor
    data collection error.
    """
    duplicates = df.duplicated(subset=["asset_id", "timestamp"]).sum()

    if duplicates > 0:
        logger.warning(f"⚠️  Found {duplicates} duplicate timestamp entries")
    else:
        logger.info("✅ No duplicate timestamps found")

    return {
        "check": "duplicate_timestamps",
        "count": int(duplicates)
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    """
    Custom check: flags columns with significant missing data.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    problem_columns = missing_pct[missing_pct > 5].to_dict()

    if problem_columns:
        logger.warning(f"⚠️  Columns with >5% missing values: {problem_columns}")
    else:
        logger.info("✅ No columns with excessive missing values")

    return {
        "check": "missing_values",
        "problem_columns": problem_columns
    }


def run_validation_pipeline(input_path: str, output_path: str):
    """
    Master validation function — reads raw data, validates it,
    saves the validated version, and logs a summary report.
    """
    logger.info("=" * 60)
    logger.info("Data Validation Pipeline")
    logger.info("=" * 60)

    df = pd.read_csv(input_path)
    validated_df, report = validate_sensor_data(df)

    missing_check = check_missing_values(df)
    duplicate_check = check_duplicate_timestamps(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    validated_df.to_csv(output_path, index=False)

    logger.info("=" * 60)
    logger.info(f"Validation status: {report['status']}")
    logger.info(f"Total violations: {len(report['errors'])}")
    logger.info(f"Missing value issues: {len(missing_check['problem_columns'])}")
    logger.info(f"Duplicate timestamps: {duplicate_check['count']}")
    logger.info(f"Validated data saved to: {output_path}")
    logger.info("=" * 60)

    return validated_df, report


if __name__ == "__main__":
    run_validation_pipeline(
        input_path="data/raw/sensor_timeseries.csv",
        output_path="data/validated/sensor_timeseries.csv"
    )
