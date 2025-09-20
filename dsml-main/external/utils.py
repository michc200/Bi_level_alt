import os
import random
import logging
import subprocess
import pickle
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
from external.loss import wls_loss, physical_loss, wls_and_physical_loss

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
                    edge_attr_set_mean, edge_attr_set_std, loss_kwargs, model_str='gat_dsse',
                    epochs=50, save_path='', loss_type='gsp_wls'):
    """
    Train state estimation methods with specified parameters.

    Args:
        net: Power network
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        x_set_mean, x_set_std: Node normalization parameters
        edge_attr_set_mean, edge_attr_set_std: Edge normalization parameters
        loss_kwargs: Unified loss configuration containing both regularization coefficients and lambda weights
        model_str: Model type ('gat_dsse', 'mlp_dsse', etc.)
        epochs: Number of training epochs
        save_path: Path to save model
        loss_type: Type of loss function ('gsp_wls', 'wls', 'physical', 'combined', 'mse')

    Returns:
        trainer, model: Trained PyTorch Lightning trainer and model
    """
    num_bus = len(net.bus)
    hyperparameters = get_model_config(model_str, num_bus)

    logger.info(f"Creating {model_str} model with {loss_type} loss")

    if model_str.startswith('gat_dsse'):
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            time_info=True, loss_type=loss_type, loss_kwargs=loss_kwargs
        )

    elif model_str.startswith('mlp_dsse'):
        # MLP DSSE still uses the old use_mse_loss parameter (would need similar updates)
        use_mse = loss_type == 'mse'
        model = MLP_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            use_mse_loss=use_mse, time_info=True
        )

    elif model_str == 'ensemble_gat_dsse':
        # Ensemble GAT DSSE still uses the old use_mse_loss parameter (would need similar updates)
        use_mse = loss_type == 'mse'
        model = EnsembleGAT_DSSE(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
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
                           edge_attr_set_mean, edge_attr_set_std, loss_kwargs):
    """
    Evaluate separate WLS and physical loss components for model analysis.

    Args:
        model: Trained model
        test_loader: Test data loader
        x_set_mean, x_set_std: Node normalization parameters
        edge_attr_set_mean, edge_attr_set_std: Edge normalization parameters
        loss_kwargs: Unified loss configuration containing both regularization coefficients and lambda weights

    Returns:
        dict: Dictionary containing loss components and metrics
    """
    model.eval()
    total_wls_loss = 0.0
    total_phys_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            # Extract batch components
            x = batch.x
            edge_attr = batch.edge_attr
            edge_index = batch.edge_index

            # Process input like in training step
            x_nodes = x[:, :model.num_nfeat]
            x_nodes_gnn = x_nodes.clone()

            # Add time info if enabled
            if model.time_info:
                time_info = x[:, model.num_nfeat+3:]
                x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

            edge_input = edge_attr[:, :model.num_efeat]

            # Forward pass with processed input
            output = model(x_nodes_gnn, edge_index, edge_input)

            # Get node and edge parameters
            node_param = x[:, model.num_nfeat:model.num_nfeat+3]
            edge_param = edge_attr[:, model.num_efeat:]

            try:
                # Get num_samples like in training
                num_samples = batch.batch[-1] + 1 if hasattr(batch, 'batch') else 1

                # Use the model's calculate_loss method but override to get components
                if model.loss_type == 'combined' or model.loss_type == 'gsp_wls':
                    # For combined/gsp_wls, compute WLS and physical separately for analysis
                    wls_loss_val = wls_loss(
                        output, x_nodes, edge_input,
                        x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std,
                        edge_index, loss_kwargs, node_param, edge_param
                    )

                    phys_loss = physical_loss(
                        output, x_set_mean, x_set_std, edge_index, edge_param, node_param, loss_kwargs
                    )
                elif model.loss_type == 'wls':
                    # Only WLS loss
                    wls_loss_val = model.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
                    phys_loss = torch.tensor(0.0)
                elif model.loss_type == 'physical':
                    # Only physical loss
                    wls_loss_val = torch.tensor(0.0)
                    phys_loss = model.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
                else:
                    # For MSE or other losses, skip component analysis
                    logger.warning(f"Loss component analysis not supported for loss_type: {model.loss_type}")
                    continue

                total_wls_loss += wls_loss_val.item()
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


def load_or_create_datasets(grid_ts, baseline_se, dataset_save_path):
    """
    Load existing datasets or create new ones from grid and baseline data.

    Args:
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation instance
        dataset_save_path: Path to save/load dataset

    Returns:
        tuple: (train_data, val_data, test_data, normalization_params)
    """
    dataset_save_path = Path(dataset_save_path)

    if dataset_save_path.exists():
        logger.info(f"Loading existing datasets from: {dataset_save_path}")
        with open(dataset_save_path, 'rb') as f:
            dataset_dict = pickle.load(f)

        train_data = dataset_dict['train_data']
        val_data = dataset_dict['val_data']
        test_data = dataset_dict['test_data']
        normalization_params = dataset_dict['normalization_params']

        logger.info("Datasets loaded successfully!")
    else:
        logger.info("Creating new PyTorch Geometric datasets...")

        # Create datasets from grid and baseline data
        datasets = grid_ts.create_pyg_data(baseline_se.baseline_se_results_df)
        train_data, val_data, test_data = datasets[:3]
        x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = datasets[3:]

        # Prepare normalization parameters
        normalization_params = {
            'x_set_mean': x_set_mean,
            'x_set_std': x_set_std,
            'edge_attr_set_mean': edge_attr_set_mean,
            'edge_attr_set_std': edge_attr_set_std
        }

        # Save datasets for future use
        dataset_dict = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'normalization_params': normalization_params
        }

        with open(dataset_save_path, 'wb') as f:
            pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Datasets saved to: {dataset_save_path}")
        logger.info("Datasets created successfully!")

    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    logger.info(f"Test samples: {len(test_data)}")

    return train_data, val_data, test_data, normalization_params


def load_or_create_datasets_and_loaders(grid_ts, baseline_se, batch_size, device, dataset_save_path=None):
    """
    Load existing datasets or create PyTorch Geometric datasets and data loaders.

    Args:
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation instance
        batch_size: Batch size for data loaders
        device: Device to move tensors to
        dataset_save_path: Optional path to save/load dataset

    Returns:
        tuple: (train_loader, val_loader, test_loader, normalization_params, train_data, val_data, test_data)
    """
    if dataset_save_path:
        train_data, val_data, test_data, normalization_params = load_or_create_datasets(
            grid_ts, baseline_se, dataset_save_path
        )
    else:
        logger.info("Creating PyTorch Geometric datasets...")

        # Create datasets from grid and baseline data
        datasets = grid_ts.create_pyg_data(baseline_se.baseline_se_results_df)
        train_data, val_data, test_data = datasets[:3]
        x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = datasets[3:]

        # Prepare normalization parameters
        normalization_params = {
            'x_set_mean': x_set_mean,
            'x_set_std': x_set_std,
            'edge_attr_set_mean': edge_attr_set_mean,
            'edge_attr_set_std': edge_attr_set_std
        }

        logger.info("Datasets created successfully!")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        logger.info(f"Test samples: {len(test_data)}")

    # Move normalization parameters to device
    normalization_params = {
        'x_set_mean': normalization_params['x_set_mean'].to(device),
        'x_set_std': normalization_params['x_set_std'].to(device),
        'edge_attr_set_mean': normalization_params['edge_attr_set_mean'].to(device),
        'edge_attr_set_std': normalization_params['edge_attr_set_std'].to(device)
    }

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    logger.info("Data loaders created!")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader, test_loader, normalization_params, train_data, val_data, test_data


def process_test_results(test_results, grid_ts):
    """
    Process test results from model predictions into a DataFrame.

    Args:
        test_results: List of prediction results from trainer.predict()
        grid_ts: Grid time series instance for bus indexing

    Returns:
        pd.DataFrame: Processed test results with bus voltage magnitudes and angles
    """
    import pandas as pd
    import numpy as np

    logger.info("Processing test results...")
    test_results_df = pd.DataFrame()

    for timestamp, (vm_pu_tensor, va_degree_tensor) in enumerate(test_results):
        vm_pu = vm_pu_tensor.squeeze().tolist()
        va_degree = va_degree_tensor.squeeze().tolist()

        for i in range(len(vm_pu)):
            bus_id = grid_ts.net.bus.index[i]
            test_results_df.loc[timestamp, f"bus_{bus_id}_vm_pu"] = vm_pu[i]
            test_results_df.loc[timestamp, f"bus_{bus_id}_va_degree"] = va_degree[i] * (180 / np.pi)

    logger.info("Test results processing completed!")
    logger.info(f"Test time steps: {len(test_results_df)}")
    logger.info(f"Variables predicted: {len(test_results_df.columns)}")

    return test_results_df