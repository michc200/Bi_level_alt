#!/usr/bin/env python3
"""
Training utilities for DSML State Estimation

This module contains functions and classes for training state estimation models.
"""

import logging
from pathlib import Path
from datetime import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from external.models.gat_dsse import GAT_DSSE_Lightning
from external.models.bi_level_gat_dsse import FAIR_GAT_BILEVEL_Lightning_Stable

# Setup logger
logger = logging.getLogger("dsml_train")

# Force CPU usage
device = torch.device('cpu')
accelerator = "cpu"


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

    def on_validation_epoch_end(self, trainer, _pl_module):
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
        MyLightningModule = GAT_DSSE_Lightning
        model = GAT_DSSE_Lightning(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            time_info=True, loss_type=loss_type, loss_kwargs=loss_kwargs
        )

    elif model_str == 'bi_level_gat_dsse':
        MyLightningModule = FAIR_GAT_BILEVEL_Lightning_Stable
        model = FAIR_GAT_BILEVEL_Lightning_Stable(
            hyperparameters, x_set_mean, x_set_std,
            edge_attr_set_mean, edge_attr_set_std, loss_kwargs,
            time_info=True, loss_type=loss_type, loss_kwargs=loss_kwargs
        )

    else:
        raise ValueError(f"Unknown model type: {model_str}")

    # Create timestamp-based directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(save_path) / model_str / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models will be saved to: {run_dir}")

    # Setup best model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(run_dir),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    # Create custom progress callback
    progress_callback = CustomProgressCallback(epochs)

    # Step 1: Create trainer for initial checkpoint (0 epochs)
    initial_trainer = Trainer(
        max_epochs=0,
        accelerator=accelerator,
        enable_progress_bar=False,
        enable_model_summary=False
    )

    # Train for 0 epochs to create initial checkpoint
    logger.info("Creating initial model checkpoint...")
    initial_trainer.fit(model, train_dataloader, val_dataloader)

    # Save initial model checkpoint
    initial_model_path = run_dir / "init.ckpt"
    initial_trainer.save_checkpoint(str(initial_model_path))
    logger.info(f"Initial model saved to: {initial_model_path}")

    # Step 2: Load from initial checkpoint and train for full epochs
    logger.info("Loading from initial checkpoint and starting full training...")

    model = MyLightningModule.load_from_checkpoint(f"{initial_model_path}")


    # Create trainer for full training with checkpoint callback
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        enable_progress_bar=False,  # Disable default progress bar
        enable_model_summary=False,  # Disable model summary
        callbacks=[progress_callback, checkpoint_callback]  # Use custom progress and checkpoint callbacks
    )

    # Train the model for full epochs
    trainer.fit(model, train_dataloader, val_dataloader)

    # Save final model (after training)
    final_model_path = run_dir / "final.ckpt"
    trainer.save_checkpoint(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")


    return trainer, model, str(run_dir)






