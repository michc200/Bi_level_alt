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
from external.evaluate import (
    load_model_from_cpkt_file,
    generate_test_dataframes,
    print_evaluation_metrics
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
GRID_CODE = "1-MV-urban--0-sw" # '1-MV-urban--0-sw' # 1-LV-rural1--0-sw
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
MODEL_DIR = SCRIPT_DIR.parent / "dsml-model"  # Save models inside dsml-main
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

# Get the grid_ts and test loader 
grid_id = f"GRID-{GRID_CODE}__MEAS_RATE-{MEASUREMENT_RATE}__ERROR-{ERROR_TYPE}__SEED-{SEED}"
grid_save_path = GRID_TS_DIR / grid_id

grid_ts = load_or_create_grid_ts(GRID_CODE, grid_save_path, MEASUREMENT_RATE)
_, _, test_loader, _, _, _, _ = create_datasets_and_loaders(grid_ts, BATCH_SIZE, device)

################################################################################
#### Model Evaluation and Visualization
################################################################################

init_model_path = r"/Users/itamarbarron/Bi_level_alt/dsml-model/gat_dsse/20250927_130350/init.ckpt"
final_model_path = r"/Users/itamarbarron/Bi_level_alt/dsml-model/gat_dsse/20250927_130350/final.ckpt"
# Run complete evaluation pipeline with detailed metrics


initial_model = load_model_from_cpkt_file(cpkt_path = final_model_path)
final_model = load_model_from_cpkt_file(cpkt_path = final_model_path)

test_results_initial , _ , _ , _ = generate_test_dataframes(test_loader = test_loader , model = initial_model , grid_ts = grid_ts , baseline_se = None)
test_results, test_baseline, test_measurements, test_true = generate_test_dataframes(test_loader = test_loader , model = final_model , grid_ts = grid_ts , baseline_se = None)

print_evaluation_metrics(test_results , test_results_initial , test_true)
