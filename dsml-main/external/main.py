#!/usr/bin/env python3
import os
import warnings
import logging
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from external.utils import (
    load_or_create_grid_ts,
    load_or_create_baseline_se,
    load_or_create_datasets_and_loaders,
    train_se_methods,
    process_test_results
)
from external.plot_utils import plot_state_estimation_results

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
GRID_CODE = '1-LV-rural1--0-sw'
ERROR_TYPE = 'no_errors'
MEASUREMENT_RATE = 0.5
SEED = 15

# Model Parameters
MODEL_TYPE = 'gat_dsse'  # Options: 'gat_dsse', 'mlp_dsse', 'ensemble_gat_dsse'
EPOCHS = 20
BATCH_SIZE = 64

# Loss Configuration
LOSS_TYPE = 'combined'  # Options: 'gsp_wls', 'wls', 'physical', 'combined', 'mse'
LOSS_KWARGS = {
    'lambda_wls': 1.0,        # Weight for WLS loss component
    'lambda_physical': 1.0,   # Weight for physical constraint loss component
    'mu_v': 1e-1,            # Regularization coefficients
    'mu_theta': 1e-1,
    'lam_v': 1,
    'lam_p': 1,
    'lam_pf': 1,
    'lam_reg': 0.8,
}

# Directory Setup (relative to main.py location)
SCRIPT_DIR = Path(__file__).parent.resolve().parent.resolve()
BASE_DIR = SCRIPT_DIR.parent / "dsml-data"
GRID_TS_DIR = BASE_DIR / "grid-time-series"
BASELINE_SE_DIR = BASE_DIR / "baseline-state-estimation"
DATASET_DIR = BASE_DIR / "datasets"
MODEL_DIR = SCRIPT_DIR.parent / "dsml-models"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"

# Create directories
for directory in [GRID_TS_DIR, BASELINE_SE_DIR, DATASET_DIR, MODEL_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

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

logger.info("Grid loaded successfully!")
logger.info(f"Buses: {len(grid_ts.net.bus)}")
logger.info(f"Lines: {len(grid_ts.net.line)}")
logger.info(f"Time steps: {len(grid_ts.profiles[('load', 'p_mw')])}")

################################################################################
#### Baseline State Estimation
################################################################################

logger.info("="*80)
logger.info("BASELINE STATE ESTIMATION")
logger.info("="*80)

# Load or create baseline state estimation
baseline_save_path = BASELINE_SE_DIR / grid_id
baseline_se = load_or_create_baseline_se(grid_save_path, baseline_save_path, n_jobs=18)

logger.info("Baseline state estimation loaded!")
logger.info(f"Time steps processed: {len(baseline_se.baseline_se_results_df)}")
logger.info(f"Variables per time step: {len(baseline_se.baseline_se_results_df.columns)}")

################################################################################
#### Dataset Creation
################################################################################

logger.info("="*80)
logger.info("DATASET CREATION")
logger.info("="*80)

# Create datasets and data loaders
dataset_save_path = DATASET_DIR / grid_id
train_loader, val_loader, test_loader, normalization_params, train_data, val_data, test_data = load_or_create_datasets_and_loaders(
    grid_ts, baseline_se, BATCH_SIZE, device, dataset_save_path
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
    x_set_mean=normalization_params['x_set_mean'],
    x_set_std=normalization_params['x_set_std'],
    edge_attr_set_mean=normalization_params['edge_attr_set_mean'],
    edge_attr_set_std=normalization_params['edge_attr_set_std'],
    loss_kwargs=LOSS_KWARGS,
    model_str=MODEL_TYPE,
    epochs=EPOCHS,
    save_path=str(MODEL_DIR),
    loss_type=LOSS_TYPE
)

logger.info("Model training completed!")
logger.info(f"Model saved to: {MODEL_DIR / MODEL_TYPE}")

################################################################################
#### Model Evaluation
################################################################################

logger.info("="*80)
logger.info("MODEL EVALUATION")
logger.info("="*80)

logger.info("Running model evaluation on test set...")

# Run prediction on test data
test_results = trainer.predict(model, test_loader)

# Process results into DataFrame
test_results_df = process_test_results(test_results, grid_ts)

logger.info("Evaluation completed!")

################################################################################
#### Results Visualization
################################################################################

logger.info("="*80)
logger.info("RESULTS VISUALIZATION")
logger.info("="*80)

# Create state estimation results visualization
plot_path = plot_state_estimation_results(
    test_results_df, baseline_se, grid_ts, train_data, val_data,
    MODEL_TYPE, GRID_CODE, MEASUREMENT_RATE, PLOTS_DIR
)

################################################################################
#### Summary
################################################################################

logger.info("="*80)
logger.info("PIPELINE SUMMARY")
logger.info("="*80)

logger.info("DSML Pipeline completed successfully!")
logger.info("Summary:")
logger.info(f"Grid: {GRID_CODE}")
logger.info(f"Model: {MODEL_TYPE.upper()}")
logger.info(f"Loss type: {LOSS_TYPE}")
if LOSS_TYPE == 'combined':
    logger.info(f"Loss weights: WLS={LOSS_KWARGS['lambda_wls']}, Physical={LOSS_KWARGS['lambda_physical']}")
logger.info(f"Training epochs: {EPOCHS}")
logger.info(f"Measurement rate: {MEASUREMENT_RATE}")
logger.info(f"Test samples: {len(test_results_df)}")
logger.info("Output files:")
logger.info(f"Grid data: {grid_save_path}")
logger.info(f"Baseline SE: {baseline_save_path}")
logger.info(f"Datasets: {dataset_save_path}")
logger.info(f"Model: {MODEL_DIR / MODEL_TYPE}")
logger.info(f"Plot: {plot_path}")
logger.info("Features:")
logger.info(f"- PyTorch-style configurable loss functions")
logger.info(f"- Support for WLS, physical, combined, and MSE losses")
logger.info(f"- Configurable lambda weights for combined loss")
logger.info(f"- CPU-optimized execution")
logger.info(f"- Modular dataset caching and reuse")
logger.info("All steps completed successfully!")