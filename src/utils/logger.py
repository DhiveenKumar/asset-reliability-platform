# =============================================================================
# logger.py — Centralized logging for the Asset Reliability Platform
#
# Every pipeline stage (data generation, validation, feature engineering,
# training, scoring) uses this instead of print() statements.
# =============================================================================

import logging
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("logs/pipeline.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
