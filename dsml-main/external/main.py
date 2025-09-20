#!/usr/bin/env python3
"""
DSML State Estimation Pipeline
==============================
Notebook-style execution with clear sections
"""

import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from external.utils import (
    no_errors,
    train_se_methods,
    load_or_create_baseline_se,
    load_or_create_grid_ts
)

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
MODEL_TYPE = 'gat_dsse'
EPOCHS = 100
BATCH_SIZE = 64

# Regularization Coefficients
REG_COEFS = {
    'mu_v': 1e-1,
    'mu_theta': 1e-1,
    'lam_v': 1,
    'lam_p': 1,
    'lam_pf': 1,
    'lam_reg': 0.8,
}

# Directory Setup
BASE_DIR = Path("../dsml-data")
GRID_TS_DIR = BASE_DIR / "grid-time-series"
BASELINE_SE_DIR = BASE_DIR / "baseline-state-estimation"
MODEL_DIR = Path("../dsml-models")
PLOTS_DIR = Path("../plots")

# Create directories
for directory in [GRID_TS_DIR, BASELINE_SE_DIR, MODEL_DIR, PLOTS_DIR]:
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

# Load or create grid time series
grid_ts = load_or_create_grid_ts(GRID_CODE, grid_save_path, MEASUREMENT_RATE)

# Display grid information
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

# Use same identifier for baseline
baseline_save_path = BASELINE_SE_DIR / grid_id

# Load or create baseline state estimation
baseline_se = load_or_create_baseline_se(grid_ts, baseline_save_path, n_jobs=18)

# Display baseline information
logger.info("Baseline state estimation loaded!")
logger.info(f"Time steps processed: {len(baseline_se.baseline_se_results_df)}")
logger.info(f"Variables per time step: {len(baseline_se.baseline_se_results_df.columns)}")

################################################################################
#### Dataset Creation
################################################################################

logger.info("="*80)
logger.info("DATASET CREATION")
logger.info("="*80)

logger.info("Creating PyTorch Geometric datasets...")

# Create datasets from grid and baseline data
datasets = grid_ts.create_pyg_data(baseline_se.baseline_se_results_df)
train_data, val_data, test_data = datasets[:3]
x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = datasets[3:]

# Move normalization parameters to device
normalization_params = {
    'x_set_mean': x_set_mean.to(device),
    'x_set_std': x_set_std.to(device),
    'edge_attr_set_mean': edge_attr_set_mean.to(device),
    'edge_attr_set_std': edge_attr_set_std.to(device)
}

logger.info("Datasets created successfully!")
logger.info(f"Training samples: {len(train_data)}")
logger.info(f"Validation samples: {len(val_data)}")
logger.info(f"Test samples: {len(test_data)}")

# Create data loaders
logger.info("Creating data loaders...")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

logger.info("Data loaders created!")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Training batches: {len(train_loader)}")
logger.info(f"Validation batches: {len(val_loader)}")

################################################################################
#### Model Training
################################################################################

logger.info("="*80)
logger.info("MODEL TRAINING")
logger.info("="*80)

logger.info(f"Training {MODEL_TYPE.upper()} model...")
logger.info(f"Epochs: {EPOCHS}")
logger.info(f"Device: {device}")
logger.info("This may take several minutes...")

# Train the model
trainer, model = train_se_methods(
    net=grid_ts.net,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    x_set_mean=normalization_params['x_set_mean'],
    x_set_std=normalization_params['x_set_std'],
    edge_attr_set_mean=normalization_params['edge_attr_set_mean'],
    edge_attr_set_std=normalization_params['edge_attr_set_std'],
    reg_coefs=REG_COEFS,
    model_str=MODEL_TYPE,
    epochs=EPOCHS,
    save_path=str(MODEL_DIR)
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
logger.info("Processing test results...")
test_results_df = pd.DataFrame()

for timestamp, (vm_pu_tensor, va_degree_tensor) in enumerate(test_results):
    vm_pu = vm_pu_tensor.squeeze().tolist()
    va_degree = va_degree_tensor.squeeze().tolist()

    for i in range(len(vm_pu)):
        bus_id = grid_ts.net.bus.index[i]
        test_results_df.loc[timestamp, f"bus_{bus_id}_vm_pu"] = vm_pu[i]
        test_results_df.loc[timestamp, f"bus_{bus_id}_va_degree"] = va_degree[i] * (180 / np.pi)

logger.info("Evaluation completed!")
logger.info(f"Test time steps: {len(test_results_df)}")
logger.info(f"Variables predicted: {len(test_results_df.columns)}")

################################################################################
#### Results Visualization
################################################################################

logger.info("="*80)
logger.info("RESULTS VISUALIZATION")
logger.info("="*80)

logger.info("Creating visualization...")

# Prepare data for visualization
test_start = len(train_data) + len(val_data)
test_baseline = baseline_se.baseline_se_results_df[test_start:]
test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
test_results_true = grid_ts.values_bus_ts_df[test_start:]

# Get bus indices for x-axis
x_values = grid_ts.net.bus.index

# Extract voltage magnitude data for the first test time step
y_measurements = [test_measurements.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
y_results_true = [test_results_true.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
y_baseline = [test_baseline.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
y_model = [test_results_df.loc[0, f'bus_{bus}_vm_pu'] for bus in x_values]

# Create the visualization
plt.figure(figsize=(14, 8))

# Plot different traces
plt.scatter(x_values, y_measurements, label='Measurements', marker='o',
           color='blue', alpha=0.7, s=50)
plt.plot(x_values, y_results_true, label='True Values',
         color='orange', linestyle='-', linewidth=2.5)
plt.plot(x_values, y_baseline, label='Baseline WLS-DSSE',
         color='green', linestyle='-', linewidth=2)
plt.plot(x_values, y_model, label=f'{MODEL_TYPE.upper()}',
         color='red', linestyle='-', linewidth=2.5)

# Customize the plot
plt.ylim([1.0125, 1.0275])
plt.xlabel('Bus Index', fontsize=14)
plt.ylabel('Voltage Magnitude (p.u.)', fontsize=14)
plt.title(f'State Estimation Results - {GRID_CODE}\\n'
          f'Model: {MODEL_TYPE.upper()}, Measurement Rate: {MEASUREMENT_RATE}',
          fontsize=16, pad=20)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plot_filename = f"{MODEL_TYPE}_results_{GRID_CODE.replace('-', '_')}.png"
plot_path = PLOTS_DIR / plot_filename
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

logger.info("Visualization created!")
logger.info(f"Plot saved to: {plot_path}")

# Display the plot
plt.show()

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
logger.info(f"Training epochs: {EPOCHS}")
logger.info(f"Measurement rate: {MEASUREMENT_RATE}")
logger.info(f"Test samples: {len(test_results_df)}")
logger.info("Output files:")
logger.info(f"Grid data: {grid_save_path}")
logger.info(f"Baseline SE: {baseline_save_path}")
logger.info(f"Model: {MODEL_DIR / MODEL_TYPE}")
logger.info(f"Plot: {plot_path}")
logger.info("All steps completed successfully!")