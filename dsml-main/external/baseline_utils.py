#!/usr/bin/env python3
"""
Baseline State Estimation Utilities
Separate script to handle baseline SE creation with proper multiprocessing support
"""

import os
import sys
import multiprocessing
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.robusttest.core.grid_time_series import GridTimeSeries
from src.robusttest.core.SE.baseline_state_estimation import BaselineStateEstimation

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] - %(message)s'
)
logger = logging.getLogger("baseline_utils")


def create_baseline_se(grid_ts_path, baseline_save_path, n_jobs=18):
    """
    Create baseline state estimation from grid time series.

    Args:
        grid_ts_path: Path to grid time series file
        baseline_save_path: Path to save baseline SE
        n_jobs: Number of parallel jobs
    """
    multiprocessing.freeze_support()

    logger.info(f"Loading grid time series from: {grid_ts_path}")
    grid_ts = GridTimeSeries.load(str(grid_ts_path))

    logger.info("Creating baseline state estimation...")
    logger.info("This may take several minutes...")

    baseline_se = BaselineStateEstimation(grid_ts)
    baseline_se.run_parallel_state_estimation(n_jobs=n_jobs)

    # Save for future use
    baseline_se.save(str(baseline_save_path))
    logger.info(f"Baseline SE saved to: {baseline_save_path}")

    return True


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if len(sys.argv) != 4:
        print("Usage: python baseline_utils.py <grid_ts_path> <baseline_save_path> <n_jobs>")
        sys.exit(1)

    grid_ts_path = sys.argv[1]
    baseline_save_path = sys.argv[2]
    n_jobs = int(sys.argv[3])

    create_baseline_se(grid_ts_path, baseline_save_path, n_jobs)