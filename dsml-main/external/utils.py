import os
import random
import logging
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.robusttest.core.SE.gat_dsse import GAT_DSSE_Lightning
from src.robusttest.core.SE.mlp_dsse import MLP_DSSE_Lightning
from src.robusttest.core.SE.gnn_dsse import GCN_DSSE_Lightning
from src.robusttest.core.SE.ensemble_gat_dsse import EnsembleGAT_DSSE
from src.robusttest.core.SE.baseline_state_estimation import BaselineStateEstimation

# Setup logger
logger = logging.getLogger("dsml_utils")

# Force CPU usage
device = torch.device('cpu')
accelerator = "cpu"


def no_errors(grid_ts_instance, rand_topology=False):
    """Process grid time series without adding any errors."""
    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))
    length = random.randint(2688, len(grid_ts_instance.profiles[('load', 'p_mw')].index))
    grid_ts_instance.adjust_profiles_start_time(offset, length)

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    return grid_ts_instance


def load_or_create_baseline_se(grid_ts, baseline_save_path, n_jobs=18):
    """
    Load existing baseline state estimation or create new one.

    Args:
        grid_ts: Grid time series instance
        baseline_save_path: Path to save/load baseline SE
        n_jobs: Number of parallel jobs for SE computation

    Returns:
        BaselineStateEstimation instance
    """
    baseline_save_path = Path(baseline_save_path)

    if baseline_save_path.exists():
        logger.info(f"Loading existing baseline SE from: {baseline_save_path}")
        baseline_se = BaselineStateEstimation.load(str(baseline_save_path))
    else:
        logger.info("Creating new baseline state estimation...")
        logger.info("This may take several minutes...")

        baseline_se = BaselineStateEstimation(grid_ts)
        baseline_se.run_parallel_state_estimation(n_jobs=n_jobs)

        # Save for future use
        baseline_se.save(str(baseline_save_path))
        logger.info(f"Baseline SE saved to: {baseline_save_path}")

    return baseline_se


def load_or_create_grid_ts(grid_code, grid_save_path, measurement_rate=0.5):
    """
    Load existing grid time series or create new one.

    Args:
        grid_code: Grid code identifier
        grid_save_path: Path to save/load grid time series
        measurement_rate: Bus measurement rate

    Returns:
        GridTimeSeries instance
    """
    from src.robusttest.core.grid_time_series import GridTimeSeries

    grid_save_path = Path(grid_save_path)

    if grid_save_path.exists():
        logger.info(f"Loading existing grid time series from: {grid_save_path}")
        grid_ts = GridTimeSeries.load(str(grid_save_path))
    else:
        logger.info(f"Creating new grid time series for: {grid_code}")
        grid_ts = GridTimeSeries(grid_code)

        # Apply error scenario (currently no errors)
        grid_ts = no_errors(grid_ts)

        # Create measurements with specified rate
        grid_ts.create_measurements_ts(bus_measurement_rate=measurement_rate)

        # Save for future use
        grid_ts.save(str(grid_save_path))
        logger.info(f"Grid time series saved to: {grid_save_path}")

    return grid_ts


def train_se_methods(net, train_dataloader, val_dataloader, x_set_mean, x_set_std,
                    edge_attr_set_mean, edge_attr_set_std, reg_coefs, model_str='gat_dsse',
                    epochs=50, save_path=''):
    """Train state estimation methods with specified parameters."""

    num_bus = len(net.bus)

    if model_str == 'gat_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense': 32,
            'gnn_layers': 5,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs, time_info=True
        )

    elif model_str == 'mlp_dsse_mse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,
            'num_nodes': num_bus,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'mlp_layers': 4,
            'dropout_rate': 0.3,
            'L': 5,
            'lr': 1e-2,
        }
        model = MLP_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            use_mse_loss=True, time_info=True
        )

    elif model_str == 'mlp_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,
            'num_nodes': num_bus,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'mlp_layers': 4,
            'dropout_rate': 0.3,
            'L': 5,
            'lr': 1e-2,
        }
        model = MLP_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs, time_info=True
        )

    elif model_str == 'gat_dsse_mse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense': 32,
            'gnn_layers': 5,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            use_mse_loss=True, time_info=True
        )

    elif model_str == 'ensemble_gat_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense': 32,
            'gnn_layers': 4,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
        model = EnsembleGAT_DSSE(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            train_dataloader.dataset, time_info=True, use_mse_loss=True
        )
        train_dataloader = model.train_dataloader()
        val_dataloader = DataLoader(val_dataloader.dataset[:30], batch_size=1, shuffle=False)

    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Create trainer
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save model
    os.makedirs(f"{save_path}/{model_str}", exist_ok=True)
    trainer.save_checkpoint(f"{save_path}/{model_str}/model.ckpt")

    return trainer, model