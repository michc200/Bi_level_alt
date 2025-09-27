import os
import random
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from external.models.gat_dsse import GAT_DSSE_Lightning
from external.models.bi_level_gat_dsse import FAIR_GAT_BILEVEL_Lightning
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
        grid_ts.random_seed_measurements = None

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
        },
        'bi_level_gat_dsse': {
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
    }
    return configs.get(model_str, configs['gat_dsse'])


class CustomProgressCallback(Callback):
    """Custom progress callback to show epoch progress in desired format."""

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch to show both train and val losses."""
        current_epoch = trainer.current_epoch + 1

        # Get train metrics
        train_loss = trainer.logged_metrics.get('train_loss_epoch')
        train_wls = trainer.logged_metrics.get('train_wls_epoch')
        train_physical = trainer.logged_metrics.get('train_physical_epoch')

        # Get validation metrics
        val_loss = trainer.logged_metrics.get('val_loss')
        val_wls = trainer.logged_metrics.get('val_wls')
        val_physical = trainer.logged_metrics.get('val_physical')

        # Convert train tensors to floats if they exist
        if train_loss is not None:
            train_loss = float(train_loss.cpu()) if hasattr(train_loss, 'cpu') else float(train_loss)
        if train_wls is not None:
            train_wls = float(train_wls.cpu()) if hasattr(train_wls, 'cpu') else float(train_wls)
        if train_physical is not None:
            train_physical = float(train_physical.cpu()) if hasattr(train_physical, 'cpu') else float(train_physical)

        # Convert val tensors to floats if they exist
        if val_loss is not None:
            val_loss = float(val_loss.cpu()) if hasattr(val_loss, 'cpu') else float(val_loss)
        if val_wls is not None:
            val_wls = float(val_wls.cpu()) if hasattr(val_wls, 'cpu') else float(val_wls)
        if val_physical is not None:
            val_physical = float(val_physical.cpu()) if hasattr(val_physical, 'cpu') else float(val_physical)

        # Format train losses
        if train_loss is not None:
            if train_wls is not None and train_physical is not None:
                train_str = f"({train_loss:.6f}, {train_wls:.6f}, {train_physical:.6f})"
            else:
                train_str = f"({train_loss:.6f}, -, -)"
        else:
            train_str = "(-, -, -)"

        # Format val losses
        if val_loss is not None:
            if val_wls is not None and val_physical is not None:
                val_str = f"({val_loss:.6f}, {val_wls:.6f}, {val_physical:.6f})"
            else:
                val_str = f"({val_loss:.6f}, -, -)"
        else:
            val_str = "(-, -, -)"

        print(f"EPOCH {current_epoch}/{self.total_epochs} | TRAIN LOSS {train_str} | VAL LOSS {val_str}")


def train_se_methods(net, train_dataloader, val_dataloader, normalization_params,
                    loss_kwargs, model_str='gat_dsse', epochs=50, save_path='', loss_type='gsp_wls'):
    """
    Train state estimation methods with specified parameters.

    Args:
        net: Power network
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        normalization_params: Dictionary containing x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std
        loss_kwargs: Unified loss configuration containing both regularization coefficients and lambda weights
        model_str: Model type ('gat_dsse', 'mlp_dsse', etc.)
        epochs: Number of training epochs
        save_path: Path to save model
        loss_type: Type of loss function ('gsp_wls', 'wls', 'physical', 'wls_and_physical', 'mse')

    Returns:
        trainer, model: Trained PyTorch Lightning trainer and model
    """
    num_bus = len(net.bus)
    hyperparameters = get_model_config(model_str, num_bus)

    # Extract normalization parameters
    x_set_mean = normalization_params['x_set_mean']
    x_set_std = normalization_params['x_set_std']
    edge_attr_set_mean = normalization_params['edge_attr_set_mean']
    edge_attr_set_std = normalization_params['edge_attr_set_std']

    logger.info(f"Creating {model_str} model with {loss_type} loss")

    if model_str.startswith('gat_dsse'):
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            time_info=True, loss_type=loss_type, loss_kwargs=loss_kwargs
        )

    elif model_str == 'bi_level_gat_dsse':
        model = FAIR_GAT_BILEVEL_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            time_info=True, loss_type=loss_type, loss_kwargs=loss_kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Create timestamp-based directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_path) / f"{model_str}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models will be saved to: {run_dir}")

    # Setup best model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="model_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    # Create custom progress callback
    progress_callback = CustomProgressCallback(epochs)

    # Create trainer with checkpoint callback
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        enable_progress_bar=False,  # Disable default progress bar
        enable_model_summary=False,  # Disable model summary
        callbacks=[progress_callback, checkpoint_callback]  # Use custom progress and checkpoint callbacks
    )

    # Create a temporary trainer to save initial model
    temp_trainer = Trainer(accelerator=accelerator, max_epochs=1)
    # Attach model to trainer by doing a minimal setup
    temp_trainer.strategy.setup(model)
    temp_trainer.model = model

    # Save initial model (before training)
    initial_model_path = run_dir / "model_initial.ckpt"
    temp_trainer.save_checkpoint(str(initial_model_path))
    logger.info(f"Initial model saved to: {initial_model_path}")

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save final model (after training)
    final_model_path = run_dir / "model_final.ckpt"
    trainer.save_checkpoint(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")

    # Save model configuration and training info
    import json
    config_info = {
        "model_type": model_str,
        "loss_type": loss_type,
        "epochs": epochs,
        "hyperparameters": hyperparameters,
        "loss_kwargs": {k: float(v) if hasattr(v, 'item') else v for k, v in loss_kwargs.items()},
        "timestamp": timestamp,
        "best_model_path": str(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else None
    }

    with open(run_dir / "training_config.json", 'w') as f:
        json.dump(config_info, f, indent=2)

    logger.info(f"Training configuration saved to: {run_dir / 'training_config.json'}")

    return trainer, model


def create_datasets(grid_ts):
    """
    Create datasets from grid and baseline data.

    Args:
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation instance

    Returns:
        tuple: (train_data, val_data, test_data, normalization_params)
    """
    logger.info("Creating PyTorch Geometric datasets...")

    # Create datasets from grid and baseline data
    datasets = grid_ts.create_pyg_data(grid_ts.values_bus_ts_df)
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

    return train_data, val_data, test_data, normalization_params


def create_datasets_and_loaders(grid_ts, batch_size, device):
    """
    Create PyTorch Geometric datasets and data loaders.

    Args:
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation instance
        batch_size: Batch size for data loaders
        device: Device to move tensors to

    Returns:
        tuple: (train_loader, val_loader, test_loader, normalization_params, train_data, val_data, test_data)
    """
    # Create datasets
    train_data, val_data, test_data, normalization_params = create_datasets(grid_ts)

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


def load_saved_model(model_dir, model_name="model_best"):
    """
    Load a saved model from a timestamped directory.

    Args:
        model_dir: Path to the model directory (e.g., "gat_dsse_20241127_143022")
        model_name: Name of the model checkpoint ("model_best", "model_initial", or "model_final")

    Returns:
        tuple: (model, config_info, normalization_params)
    """
    import json

    model_dir = Path(model_dir)

    # Load training configuration
    config_path = model_dir / "training_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training configuration not found at: {config_path}")

    with open(config_path, 'r') as f:
        config_info = json.load(f)

    # Determine checkpoint path
    if model_name == "model_best" and config_info.get("best_model_path"):
        checkpoint_path = config_info["best_model_path"]
    else:
        checkpoint_path = model_dir / f"{model_name}.ckpt"

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading model from: {checkpoint_path}")

    # Create model instance based on saved configuration
    model_str = config_info["model_type"]
    hyperparameters = config_info["hyperparameters"]
    loss_kwargs = config_info["loss_kwargs"]
    loss_type = config_info.get("loss_type", "gsp_wls")

    # Note: For evaluation, normalization parameters need to be provided separately
    # or loaded from the grid_ts that was used for training
    logger.info(f"Model type: {model_str}")
    logger.info(f"Loss type: {loss_type}")
    logger.info(f"Training epochs: {config_info['epochs']}")
    logger.info(f"Timestamp: {config_info['timestamp']}")

    return checkpoint_path, config_info


def evaluate_saved_model(model_dir, grid_ts, test_loader, model_name="model_best"):
    """
    Evaluate a saved model on test data.

    Args:
        model_dir: Path to the model directory
        grid_ts: Grid time series instance (for normalization parameters)
        test_loader: Test data loader
        model_name: Name of the model checkpoint to evaluate

    Returns:
        tuple: (predictions, model_info)
    """
    # Load model configuration
    checkpoint_path, config_info = load_saved_model(model_dir, model_name)

    # Get normalization parameters from grid_ts
    # Recreate the same datasets to get normalization params
    datasets = grid_ts.create_pyg_data(grid_ts.values_bus_ts_df)
    x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = datasets[3:]

    # Move to device
    x_set_mean = x_set_mean.to(device)
    x_set_std = x_set_std.to(device)
    edge_attr_set_mean = edge_attr_set_mean.to(device)
    edge_attr_set_std = edge_attr_set_std.to(device)

    # Load the model
    model_str = config_info["model_type"]
    loss_kwargs = config_info["loss_kwargs"]
    loss_type = config_info.get("loss_type", "gsp_wls")

    if model_str.startswith('gat_dsse'):
        model = GAT_DSSE_Lightning.load_from_checkpoint(
            checkpoint_path,
            hyperparameters=config_info["hyperparameters"],
            x_set_mean=x_set_mean,
            x_set_std=x_set_std,
            edge_attr_set_mean=edge_attr_set_mean,
            edge_attr_set_std=edge_attr_set_std,
            loss_kwargs=loss_kwargs,
            time_info=True,
            loss_type=loss_type,
            map_location=device
        )
    elif model_str == 'bi_level_gat_dsse':
        model = FAIR_GAT_BILEVEL_Lightning.load_from_checkpoint(
            checkpoint_path,
            hyperparameters=config_info["hyperparameters"],
            x_set_mean=x_set_mean,
            x_set_std=x_set_std,
            edge_attr_set_mean=edge_attr_set_mean,
            edge_attr_set_std=edge_attr_set_std,
            loss_kwargs=loss_kwargs,
            time_info=True,
            loss_type=loss_type,
            map_location=device
        )
    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Create trainer for evaluation
    trainer = Trainer(accelerator=accelerator, enable_progress_bar=False)

    # Run predictions
    logger.info(f"Evaluating {model_name} model...")
    predictions = trainer.predict(model, test_loader)

    logger.info(f"Evaluation completed for {model_name} model from {config_info['timestamp']}")

    return predictions, config_info


def list_saved_models(models_base_dir):
    """
    List all saved models in the models directory.

    Args:
        models_base_dir: Base directory containing model subdirectories

    Returns:
        list: List of model directory paths with their information
    """
    import json

    models_base_dir = Path(models_base_dir)
    if not models_base_dir.exists():
        logger.warning(f"Models directory not found: {models_base_dir}")
        return []

    model_dirs = []
    for subdir in models_base_dir.iterdir():
        if subdir.is_dir():
            config_path = subdir / "training_config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)

                    # Check which models are available
                    available_models = []
                    for model_name in ["model_initial", "model_best", "model_final"]:
                        if model_name == "model_best" and config.get("best_model_path"):
                            if Path(config["best_model_path"]).exists():
                                available_models.append(model_name)
                        elif (subdir / f"{model_name}.ckpt").exists():
                            available_models.append(model_name)

                    model_info = {
                        "directory": str(subdir),
                        "model_type": config.get("model_type", "unknown"),
                        "loss_type": config.get("loss_type", "unknown"),
                        "epochs": config.get("epochs", "unknown"),
                        "timestamp": config.get("timestamp", "unknown"),
                        "available_models": available_models
                    }
                    model_dirs.append(model_info)

                except Exception as e:
                    logger.warning(f"Could not read config for {subdir}: {e}")

    # Sort by timestamp (newest first)
    model_dirs.sort(key=lambda x: x["timestamp"], reverse=True)

    return model_dirs


