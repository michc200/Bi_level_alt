import os
import random
import logging
import subprocess
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
from src.robusttest.core.SE.pf_funcs import compute_wls_loss, compute_physical_loss

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


def load_or_create_baseline_se(grid_ts_path, baseline_save_path, n_jobs=18):
    """
    Load existing baseline state estimation or create new one using subprocess.

    Args:
        grid_ts_path: Path to grid time series file
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

        # Use subprocess to avoid multiprocessing issues
        baseline_utils_path = Path(__file__).parent / "baseline_utils.py"

        # Use the current Python interpreter (from venv)
        python_path = sys.executable
        cmd = [
            python_path, str(baseline_utils_path),
            str(grid_ts_path), str(baseline_save_path), str(n_jobs)
        ]

        try:
            # Run subprocess with real-time output streaming
            logger.info(f"Executing: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # Print subprocess output with prefix
                    print(f"[BASELINE] {output.strip()}")

            # Check return code
            return_code = process.poll()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)

            logger.info("Baseline SE creation completed via subprocess")
            baseline_se = BaselineStateEstimation.load(str(baseline_save_path))
        except subprocess.CalledProcessError as e:
            logger.error(f"Baseline SE creation failed with return code {e.returncode}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during baseline SE creation: {e}")
            raise

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


def get_model_config(model_str, num_bus):
    """Get hyperparameter configuration for different model types."""
    configs = {
        'gat_dsse': {
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
        },
        'gat_dsse_mse': {
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
        },
        'mlp_dsse': {
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
        },
        'mlp_dsse_mse': {
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
        },
        'ensemble_gat_dsse': {
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
    }
    return configs.get(model_str, configs['gat_dsse'])


def train_se_methods(net, train_dataloader, val_dataloader, x_set_mean, x_set_std,
                    edge_attr_set_mean, edge_attr_set_std, reg_coefs, model_str='gat_dsse',
                    epochs=50, save_path='', use_mse_loss=False):
    """
    Train state estimation methods with specified parameters.

    Args:
        net: Power network
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        x_set_mean, x_set_std: Node normalization parameters
        edge_attr_set_mean, edge_attr_set_std: Edge normalization parameters
        reg_coefs: Regularization coefficients
        model_str: Model type ('gat_dsse', 'mlp_dsse', etc.)
        epochs: Number of training epochs
        save_path: Path to save model
        use_mse_loss: Whether to use MSE loss instead of physics-based loss

    Returns:
        trainer, model: Trained PyTorch Lightning trainer and model
    """
    num_bus = len(net.bus)
    hyperparameters = get_model_config(model_str, num_bus)

    # Determine if MSE loss should be used
    use_mse = use_mse_loss or model_str.endswith('_mse')

    logger.info(f"Creating {model_str} model with {'MSE' if use_mse else 'Physics-based'} loss")

    if model_str.startswith('gat_dsse'):
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            time_info=True, use_mse_loss=use_mse
        )

    elif model_str.startswith('mlp_dsse'):
        model = MLP_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            use_mse_loss=use_mse, time_info=True
        )

    elif model_str == 'ensemble_gat_dsse':
        model = EnsembleGAT_DSSE(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, reg_coefs,
            train_dataloader.dataset, time_info=True, use_mse_loss=use_mse
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


def evaluate_loss_components(model, test_loader, x_set_mean, x_set_std,
                           edge_attr_set_mean, edge_attr_set_std, reg_coefs):
    """
    Evaluate separate WLS and physical loss components for model analysis.

    Args:
        model: Trained model
        test_loader: Test data loader
        x_set_mean, x_set_std: Node normalization parameters
        edge_attr_set_mean, edge_attr_set_std: Edge normalization parameters
        reg_coefs: Regularization coefficients

    Returns:
        dict: Dictionary containing loss components and metrics
    """
    model.eval()
    total_wls_loss = 0.0
    total_phys_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Forward pass
            output = model(batch.x, batch.edge_index, batch.edge_attr)

            # Extract batch components
            x = batch.x
            edge_attr = batch.edge_attr
            edge_index = batch.edge_index

            # Get node and edge parameters
            node_param = x[:, model.num_nfeat:model.num_nfeat+3]
            edge_param = edge_attr[:, model.num_efeat:]

            try:
                # Compute WLS loss
                wls_loss, v_i, theta_i = compute_wls_loss(
                    x[:, :model.num_nfeat], edge_attr[:, :model.num_efeat], output,
                    x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std,
                    edge_index, reg_coefs, node_param, edge_param
                )

                # Compute physical loss
                phys_loss = compute_physical_loss(
                    v_i, theta_i, edge_index, edge_param, node_param, reg_coefs
                )

                total_wls_loss += wls_loss.item()
                total_phys_loss += phys_loss.item()
                num_batches += 1

            except Exception as e:
                logger.warning(f"Error computing loss components: {e}")
                continue

    if num_batches == 0:
        return {'error': 'No valid batches processed'}

    return {
        'avg_wls_loss': total_wls_loss / num_batches,
        'avg_physical_loss': total_phys_loss / num_batches,
        'total_loss': (total_wls_loss + total_phys_loss) / num_batches,
        'num_batches': num_batches
    }