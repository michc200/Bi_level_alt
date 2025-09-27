#!/usr/bin/env python3
import os
import warnings
import logging
import random
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from external.utils import (
    load_or_create_grid_ts,
    load_or_create_baseline_se,
    create_datasets_and_loaders
)
from external.train import train_se_methods
from external.evaluate import evaluate_model, calculate_evaluation_metrics

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s - %(levelname)s] - %(message)s'
)
logger = logging.getLogger("dsml_pipeline")

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device('cpu')

################################################################################
#### Configuration
################################################################################

# Grid Parameters
GRID_CODE = "1-LV-rural1--0-sw" # '1-MV-urban--0-sw' # 1-LV-rural1--0-sw
ERROR_TYPE = 'no_errors'
MEASUREMENT_RATE = 0.9
SEED = 15

# Model Parameters
MODEL_TYPE = 'gat_dsse'  # Options: 'gat_dsse', 'bi_level_gat_dsse'
EPOCHS = 20
BATCH_SIZE = 64

# Loss Configuration
LOSS_TYPE = 'wls_and_physical'  # Options: 'wls', 'physical', 'wls_and_physical', 'mse'
LOSS_KWARGS = {
    "lambda_physical" : 0,
    "lambda_wls" : 1,
    'lam_v': 1,
    'lam_p': 0.025,
    'lam_pf': 0.025,
    'lam_reg': 1,
}

# Directory Setup (relative to main.py location)
SCRIPT_DIR = Path(__file__).parent.resolve().parent.resolve()
BASE_DIR = SCRIPT_DIR.parent / "dsml-data"
GRID_TS_DIR = BASE_DIR / "grid-time-series"
BASELINE_SE_DIR = BASE_DIR / "baseline-state-estimation"
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = SCRIPT_DIR / "models"  # Save models inside dsml-main
PLOTS_DIR = SCRIPT_DIR.parent / "plots"

# Create directories
for directory in [GRID_TS_DIR, BASELINE_SE_DIR, DATASET_DIR, MODEL_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

logger.info("Configuration loaded successfully!")
logger.info(f"Grid: {GRID_CODE}, Model: {MODEL_TYPE}, Epochs: {EPOCHS}")
logger.info(f"Device: {device}")

################################################################################
#### Grid Time Series Creation
################################################################################

logger.info("="*80)
logger.info("GRID TIME SERIES CREATION")
logger.info("="*80)

# Generate unique identifier for this configuration
grid_id = f"GRID-{GRID_CODE}__MEAS_RATE-{MEASUREMENT_RATE}__ERROR-{ERROR_TYPE}__SEED-{SEED}"
grid_save_path = GRID_TS_DIR / grid_id
grid_ts = load_or_create_grid_ts(GRID_CODE, grid_save_path, MEASUREMENT_RATE)

# grid_ts.plot()
logger.info("Grid loaded successfully!")
logger.info(f"Buses: {len(grid_ts.net.bus)}")
logger.info(f"Lines: {len(grid_ts.net.line)}")
logger.info(f"Time steps: {len(grid_ts.profiles[('load', 'p_mw')])}")

################################################################################
#### Baseline State Estimation
################################################################################

# logger.info("="*80)
# logger.info("BASELINE STATE ESTIMATION")
# logger.info("="*80)

# # Load or create baseline state estimation
# baseline_save_path = BASELINE_SE_DIR / grid_id
# baseline_se = load_or_create_baseline_se(grid_save_path, baseline_save_path, n_jobs=18)
baseline_se = None
# logger.info("Baseline state estimation loaded!")
# logger.info(f"Time steps processed: {len(baseline_se.baseline_se_results_df)}")
# logger.info(f"Variables per time step: {len(baseline_se.baseline_se_results_df.columns)}")

################################################################################
#### Dataset Creation
################################################################################

logger.info("="*80)
logger.info("DATASET CREATION")
logger.info("="*80)

# Create datasets and data loaders
train_loader, val_loader, test_loader, normalization_params, train_data, val_data, test_data = create_datasets_and_loaders(
    grid_ts, BATCH_SIZE, device
)

################################################################################
#### Model Training
################################################################################

logger.info("="*80)
logger.info("MODEL TRAINING")
logger.info("="*80)

logger.info(f"Training {MODEL_TYPE.upper()} model...")
logger.info(f"Loss type: {LOSS_TYPE}")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Device: {device}")

# Train the model
trainer, model = train_se_methods(
    net=grid_ts.net,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    normalization_params=normalization_params,
    loss_kwargs=LOSS_KWARGS,
    model_str=MODEL_TYPE,
    epochs=EPOCHS,
    save_path=str(MODEL_DIR),
    loss_type=LOSS_TYPE
)

logger.info("Model training completed!")
logger.info(f"Model saved to: {MODEL_DIR / MODEL_TYPE}")

################################################################################
#### Model Evaluation and Visualization
################################################################################

# Run complete evaluation pipeline with detailed metrics
test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path = evaluate_model(
    trainer, model, test_loader, grid_ts, baseline_se, train_data, val_data,
    MODEL_TYPE, GRID_CODE, MEASUREMENT_RATE, PLOTS_DIR,
    device=device
)

# Calculate additional evaluation metrics using the created test datasets
metrics = calculate_evaluation_metrics(
    test_results_df, test_baseline, test_true
)

# Print detailed metrics summary
logger.info("="*80)
logger.info("DETAILED METRICS SUMMARY")
logger.info("="*80)
logger.info(f"Voltage Magnitude - RMSE: {detailed_metrics['rmse_v']:.6f}, MAE: {detailed_metrics['mae_v']:.6f}")
logger.info(f"Voltage Angle - RMSE: {detailed_metrics['rmse_th']:.6f}, MAE: {detailed_metrics['mae_th']:.6f}")
logger.info(f"Line Loading - RMSE: {detailed_metrics['rmse_loading']:.6f}, MAE: {detailed_metrics['mae_loading']:.6f}")
logger.info(f"Trafo Loading - RMSE: {detailed_metrics['rmse_loading_trafos']:.6f}, MAE: {detailed_metrics['mae_loading_trafos']:.6f}")
logger.info(f"Std Proportion - V: {detailed_metrics['prop_std_v']:.2f}%, Th: {detailed_metrics['prop_std_th']:.2f}%")

