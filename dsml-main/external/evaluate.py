#!/usr/bin/env python3
"""
Model Evaluation Module for DSML State Estimation

This module contains functions for evaluating trained state estimation models,
processing test results, and creating visualizations.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer


from external.models.gat_dsse import GAT_DSSE_Lightning
from external.models.bi_level_gat_dsse import FAIR_GAT_BILEVEL_Lightning

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.robusttest.core.SE.pf_funcs import get_pflow

# Setup logger
logger = logging.getLogger("dsml_evaluate")


def generate_test_dataframes(test_loader, model, grid_ts, baseline_se=None):

    dummy_trainer =  Trainer(
        max_epochs = 0,
        accelerator= "cpu",
        enable_progress_bar=False,
        enable_model_summary=False
    )

    dummy_trainer.fit(model, test_loader, test_loader)
    test_results = dummy_trainer.predict(model, test_loader)

    test_results_df = pd.DataFrame(
        [
            (timestamp, bus_id, vm_pu_tensor.squeeze().tolist()[i], va_degree_tensor.squeeze().tolist()[i] * (180 / np.pi))
            for timestamp, (vm_pu_tensor, va_degree_tensor) in enumerate(test_results)
            for i, bus_id in enumerate(grid_ts.net.bus.index)
        ],
        columns=["timestamp", "bus_idx", "vm_pu", "va_degree"]
    )
    test_results_df["timestamp"] = test_results_df["timestamp"] + grid_ts.test_idx

    test_true_df = transform_datasets_format(grid_ts.values_bus_ts_df[grid_ts.test_idx : ])
    test_measurements_df = transform_datasets_format(grid_ts.measurements_bus_ts_df[grid_ts.test_idx : ])
    test_baseline_df = transform_datasets_format(baseline_se.baseline_se_results_df[grid_ts.test_idx : ] if baseline_se is not None else None)

    return test_results_df, test_baseline_df, test_measurements_df, test_true_df

def transform_datasets_format(df):
    if df is None:
        return None
    
    columns_names = list(df.columns)
    data_names = [col.split('_')[-2] + '_' + col.split('_')[-1] for col in columns_names]
    data_names = list(set(data_names))

    # Initialize an empty list to store rows
    rows = []
    # Iterate over each timestamp (index) in the DataFrame
    for timestamp in df.index:
        # Iterate over each bus index
        for bus_id in set(int(col.split('_')[1]) for col in df.columns if col.startswith('bus_')):
            row = {'timestamp': timestamp, 'bus_idx': bus_id}
            # Extract values for each data name
            for data_name in data_names:
                col_name = f'bus_{bus_id}_{data_name}'
                if col_name in df.columns:
                    row[data_name] = df.loc[timestamp, col_name]
            rows.append(row)
    # Create DataFrame from rows
    return pd.DataFrame(rows, columns=['timestamp', 'bus_idx'] + data_names)

def load_model_from_cpkt_file(cpkt_path):
    
    cpkt_path = Path(cpkt_path)

    if cpkt_path.parent.parent.name == "gat_dsse":
        LightningModule = GAT_DSSE_Lightning
    elif cpkt_path.parent.parent.name == "bi_level_gat_dsse":
        LightningModule = FAIR_GAT_BILEVEL_Lightning
    else:
        raise ValueError(f"Unknown model type: {cpkt_path.parent.name}")
    
    model = LightningModule.load_from_checkpoint(str(cpkt_path))

    return model 

def angular_rmse(pred_angles, true_angles):
    """Calculate RMSE for angles accounting for 360° wrapping."""
    angle_diff = pred_angles - true_angles
    angle_diff = torch.where(angle_diff > torch.pi, angle_diff - 2*torch.pi, angle_diff)
    angle_diff = torch.where(angle_diff < -torch.pi, angle_diff + 2*torch.pi, angle_diff)
    return torch.sqrt(F.mse_loss(pred_angles, true_angles, reduction='none').mean())


def angular_mae(pred_angles, true_angles):
    """Calculate MAE for angles accounting for 360° wrapping."""
    angle_diff = pred_angles - true_angles
    angle_diff = torch.where(angle_diff > torch.pi, angle_diff - 2*torch.pi, angle_diff)
    angle_diff = torch.where(angle_diff < -torch.pi, angle_diff + 2*torch.pi, angle_diff)
    return torch.abs(angle_diff).mean()


def mae(pred, true):
    """Calculate Mean Absolute Error."""
    return torch.abs(pred - true).mean()


def calculate_detailed_metrics(model, test_loader, x_mean, x_std, num_node_features, num_edge_features, device):
    """
    Calculate comprehensive evaluation metrics including power flow metrics.

    Args:
        model: Trained model
        test_loader: Test data loader
        x_mean: Normalization mean for nodes
        x_std: Normalization std for nodes
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        device: Device to run computations on

    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    model.eval()

    # Try to auto-detect model requirements from first batch
    first_data = next(iter(test_loader))
    logger.info(f"Input data shape: x={first_data.x.shape}, edge_attr={first_data.edge_attr.shape}")

    # Check if model expects time info
    has_time_info = hasattr(model, 'time_info') and model.time_info
    logger.info(f"Model time_info setting: {has_time_info}")

    # Initialize metric accumulators
    total_rmse_v = 0.0
    total_mae_v = 0.0
    total_rmse_th = 0.0
    total_mae_th = 0.0
    total_rmse_loading = 0.0
    total_mae_loading = 0.0
    total_rmse_loading_trafos = 0.0
    total_mae_loading_trafos = 0.0
    total_prop_std_v = 0.0
    total_prop_std_th = 0.0

    num_test_batches = len(test_loader)

    logger.info(f"Calculating detailed metrics over {num_test_batches} test batches...")

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)

            # Replicate exact input processing from model's predict_step
            x = data.x.clone()
            edge_index = data.edge_index
            edge_attr = data.edge_attr

            # Get model's num_nfeat and num_efeat
            model_num_nfeat = getattr(model, 'num_nfeat', num_node_features)
            model_num_efeat = getattr(model, 'num_efeat', num_edge_features)

            node_param = x[:, model_num_nfeat:model_num_nfeat+3]
            x_nodes = x[:, :model_num_nfeat]
            x_nodes_gnn = x_nodes.clone()

            if has_time_info:
                time_info = x[:, model_num_nfeat+3:]
                x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

            edge_input = edge_attr[:, :model_num_efeat]

            if batch_idx == 0:  # Log shapes for first batch only
                logger.info(f"Model expects - num_nfeat: {model_num_nfeat}, num_efeat: {model_num_efeat}")
                logger.info(f"Input shapes - x_nodes_gnn: {x_nodes_gnn.shape}, edge_input: {edge_input.shape}")

            try:
                output = model(x_nodes_gnn, edge_index, edge_input)
            except RuntimeError as e:
                logger.error(f"Model forward pass failed on batch {batch_idx}")
                logger.error(f"Input shapes: x_nodes_gnn={x_nodes_gnn.shape}, edge_input={edge_input.shape}")
                logger.error(f"Error: {e}")
                raise

            # Denormalize using model's normalization parameters (like in predict_step)
            v_i = output[:, 0:1] * model.x_std[:1] + model.x_mean[:1]
            theta_i = output[:, 1:] * model.x_std[2:3] + model.x_mean[2:3]
            theta_i *= (1. - node_param[:, 1:2])  # Apply slack bus constraint

            # Combine denormalized outputs
            out = torch.cat([v_i, theta_i], dim=1)

            # Voltage magnitude metrics
            total_rmse_v += torch.sqrt(F.mse_loss(out[:, :1], data.y[:, :1])).item()
            total_mae_v += mae(out[:, :1], data.y[:, :1]).item()

            # Voltage angle metrics - using proper angular distance
            total_rmse_th += angular_rmse(out[:, 1:], data.y[:, 1:]).item()
            total_mae_th += angular_mae(out[:, 1:], data.y[:, 1:]).item()

            # Power flow calculations
            try:
                # Get node and edge parameters using correct indices
                edge_param = edge_attr[:, model_num_efeat:]

                # True power flow
                true_loading_lines, true_loading_trafos = get_pflow(
                    data.y,
                    data.edge_index,
                    node_param=node_param,
                    edge_param=edge_param,
                    phase_shift=False
                )[:2]

                # Predicted power flow
                out_loading_lines, out_loading_trafos = get_pflow(
                    out,
                    data.edge_index,
                    node_param=node_param,
                    edge_param=edge_param,
                    phase_shift=False
                )[:2]

                # Filter non-zero for lines
                if true_loading_lines.numel() > 0:
                    nonzero_mask_lines = true_loading_lines != 0
                    if nonzero_mask_lines.any():
                        true_loading_lines_nonzero = true_loading_lines[nonzero_mask_lines]
                        out_loading_lines_filtered = out_loading_lines[nonzero_mask_lines]

                        total_rmse_loading += torch.sqrt(F.mse_loss(out_loading_lines_filtered, true_loading_lines_nonzero)).item()
                        total_mae_loading += mae(out_loading_lines_filtered, true_loading_lines_nonzero).item()

                # Filter non-zero for trafos
                if true_loading_trafos.numel() > 0:
                    nonzero_mask_trafos = true_loading_trafos != 0
                    if nonzero_mask_trafos.any():
                        true_loading_trafos_nonzero = true_loading_trafos[nonzero_mask_trafos]
                        out_loading_trafos_filtered = out_loading_trafos[nonzero_mask_trafos]

                        total_rmse_loading_trafos += torch.sqrt(F.mse_loss(out_loading_trafos_filtered, true_loading_trafos_nonzero)).item()
                        total_mae_loading_trafos += mae(out_loading_trafos_filtered, true_loading_trafos_nonzero).item()

            except Exception as e:
                logger.warning(f"Power flow calculation failed for batch {batch_idx}: {e}")

            # Proportion of std
            if data.y.std(dim=0)[0] != 0 and data.y.std(dim=0)[1] != 0:
                std_ratios = (out.std(dim=0) / data.y.std(dim=0)) * 100
                total_prop_std_v += std_ratios[0].item()
                total_prop_std_th += std_ratios[1].item()

    # Average metrics
    metrics = {
        'rmse_v': total_rmse_v / num_test_batches,
        'mae_v': total_mae_v / num_test_batches,
        'rmse_th': total_rmse_th / num_test_batches,
        'mae_th': total_mae_th / num_test_batches,
        'rmse_loading': total_rmse_loading / num_test_batches,
        'mae_loading': total_mae_loading / num_test_batches,
        'rmse_loading_trafos': total_rmse_loading_trafos / num_test_batches,
        'mae_loading_trafos': total_mae_loading_trafos / num_test_batches,
        'prop_std_v': total_prop_std_v / num_test_batches,
        'prop_std_th': total_prop_std_th / num_test_batches
    }

    logger.info("Detailed metrics calculated:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.6f}")

    return metrics


def process_test_results(test_results, grid_ts):
    logger.info("Processing test results...")
    test_results_df = pd.DataFrame()

    test_results_df = pd.DataFrame(
        [
            (timestamp, bus_id, vm_pu_tensor.squeeze().tolist()[i], va_degree_tensor.squeeze().tolist()[i] * (180 / np.pi))
            for timestamp, (vm_pu_tensor, va_degree_tensor) in enumerate(test_results)
            for i, bus_id in enumerate(grid_ts.net.bus.index)
        ],
        columns=["timestamp", "bus_idx", "vm_pu", "va_degree"]
    )
    

    logger.info("Test results processing completed!")
    logger.info(f"Test time steps: {len(test_results_df)}")
    logger.info(f"Variables predicted: {len(test_results_df.columns)}")
    return test_results_df


def plot_state_estimation_results(test_results_df, test_baseline, test_measurements, test_true,
                                 model_type, grid_code, measurement_rate, plots_dir):
    """
    Create and save state estimation results visualization with voltage magnitudes and angles.

    Args:
        test_results_df: DataFrame with model predictions
        test_baseline: Pre-created baseline test dataset (can be None)
        test_measurements: Pre-created measurements test dataset
        test_true: Pre-created true values test dataset
        model_type: Model type string for labeling
        grid_code: Grid code for title and filename
        measurement_rate: Measurement rate for title
        plots_dir: Directory to save plots

    Returns:
        Path: Path to saved plot file
    """
    logger.info("Creating visualization...")

    # Use the first timestep for visualization
    first_timestep = test_true.index[0]

    # Extract bus indices from column names
    vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
    bus_indices = [int(col.split('_')[1]) for col in vm_cols]
    bus_indices.sort()
    print("Bus indices:", bus_indices)

    # Create continuous x-axis mapping
    x_continuous = list(range(len(bus_indices)))  # [0, 1, 2, ..., n-1]
    bus_to_continuous = {bus: i for i, bus in enumerate(bus_indices)}

    # Extract voltage magnitude data for the first test time step
    y_vm_measurements = []
    x_vm_measurements = []
    y_vm_results_true = [test_true.loc[first_timestep, f'bus_{bus}_vm_pu'] for bus in bus_indices]
    y_vm_baseline = [test_baseline.loc[first_timestep, f'bus_{bus}_vm_pu'] for bus in bus_indices] if test_baseline is not None else None
    y_vm_model = [test_results_df.loc[0, f'bus_{bus}_vm_pu'] for bus in bus_indices]

    # Filter out zero measurements (missing measurements) for voltage magnitude
    for bus in bus_indices:
        measurement = test_measurements.loc[first_timestep, f'bus_{bus}_vm_pu']
        if measurement != 0:  # Only include non-zero measurements
            y_vm_measurements.append(measurement)
            x_vm_measurements.append(bus_to_continuous[bus])  # Use continuous mapping

    # Extract voltage angle data for the first test time step
    y_va_measurements = []
    x_va_measurements = []
    y_va_results_true = [test_true.loc[first_timestep, f'bus_{bus}_va_degree'] for bus in bus_indices]
    y_va_baseline = [test_baseline.loc[first_timestep, f'bus_{bus}_va_degree'] for bus in bus_indices] if test_baseline is not None else None
    y_va_model = [test_results_df.loc[0, f'bus_{bus}_va_degree'] for bus in bus_indices]

    # Filter out zero measurements (missing measurements) for voltage angle
    for bus in bus_indices:
        measurement = test_measurements.loc[first_timestep, f'bus_{bus}_va_degree']
        if measurement != 0:  # Only include non-zero measurements
            y_va_measurements.append(measurement)
            x_va_measurements.append(bus_to_continuous[bus])  # Use continuous mapping

    # Create the visualization with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Voltage Magnitude Plot
    ax1.scatter(x_vm_measurements, y_vm_measurements, label='Measurements', marker='o',
               color='blue', alpha=0.8, s=60)
    ax1.plot(x_continuous, y_vm_results_true, label='True Values', marker='o',
             color='orange', linestyle='-', linewidth=2, alpha=0.7, markersize=4)
    if y_vm_baseline is not None:
        ax1.plot(x_continuous, y_vm_baseline, label='Baseline WLS-DSSE', marker='s',
                 color='green', linestyle='-', linewidth=2, alpha=0.7, markersize=4)
    ax1.plot(x_continuous, y_vm_model, label=f'{model_type.upper()}', marker='^',
             color='red', linestyle='-', linewidth=2, alpha=0.7, markersize=4)

    ax1.set_xlabel('Bus Index', fontsize=14)
    ax1.set_ylabel('Voltage Magnitude (p.u.)', fontsize=14)
    ax1.set_title(f'Voltage Magnitude - {grid_code}', fontsize=14)
    ax1.set_xticks(x_continuous)
    ax1.set_xticklabels(bus_indices)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Voltage Angle Plot
    ax2.scatter(x_va_measurements, y_va_measurements, label='Measurements', marker='o',
               color='blue', alpha=0.8, s=60)
    ax2.plot(x_continuous, y_va_results_true, label='True Values', marker='o',
             color='orange', linestyle='-', linewidth=2, alpha=0.7, markersize=4)
    if y_va_baseline is not None:
        ax2.plot(x_continuous, y_va_baseline, label='Baseline WLS-DSSE', marker='s',
                 color='green', linestyle='-', linewidth=2, alpha=0.7, markersize=4)
    ax2.plot(x_continuous, y_va_model, label=f'{model_type.upper()}', marker='^',
             color='red', linestyle='-', linewidth=2, alpha=0.7, markersize=4)

    ax2.set_xlabel('Bus Index', fontsize=14)
    ax2.set_ylabel('Voltage Angle (degrees)', fontsize=14)
    ax2.set_title(f'Voltage Angle - {grid_code}', fontsize=14)
    ax2.set_xticks(x_continuous)
    ax2.set_xticklabels(bus_indices)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle(f'State Estimation Results - {grid_code}\\n'
                 f'Model: {model_type.upper()}, Measurement Rate: {measurement_rate}',
                 fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for suptitle

    # Save the plot
    plot_filename = f"{model_type}_results_{grid_code.replace('-', '_')}.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info("Visualization created!")
    logger.info(f"Plot saved to: {plot_path}")

    # Display the plot
    plt.show()

    return plot_path


def evaluate_model(model_dir, test_loader, grid_ts, baseline_se, train_data, val_data,
                  grid_code, measurement_rate, plots_dir,
                  model_name="best", device='cpu', skip_detailed_metrics=False):
    """
    Complete model evaluation pipeline by loading model from directory and evaluating.

    Args:
        model_dir: Path to the model directory containing checkpoints
        test_loader: Test data loader
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation results
        train_data: Training dataset (for indexing)
        val_data: Validation dataset (for indexing)
        grid_code: Grid code string
        measurement_rate: Measurement rate
        plots_dir: Directory to save plots
        model_name: Which checkpoint to load ("init", "best", "final")
        device: Device to run computations on
        skip_detailed_metrics: Whether to skip detailed evaluation metrics

    Returns:
        tuple: (test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path)
    """
    from external.train import evaluate_saved_model, load_saved_model

    logger.info("="*80)
    logger.info("MODEL EVALUATION FROM DIRECTORY")
    logger.info("="*80)

    # Load model configuration and get model type
    _, config_info = load_saved_model(model_dir, model_name)
    model_type = config_info['model_type']

    logger.info(f"Loading {model_name} model from: {model_dir}")
    logger.info(f"Model type: {model_type}")

    # Evaluate the saved model
    test_results, _ = evaluate_saved_model(model_dir, grid_ts, test_loader, model_name)

    # Process results into DataFrame
    test_results_df = process_test_results(test_results, grid_ts)

    logger.info("Creating test datasets...")

    # Create test datasets
    test_start = len(train_data) + len(val_data)
    test_baseline = baseline_se.baseline_se_results_df[test_start:] if baseline_se is not None else None
    test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
    test_true = grid_ts.values_bus_ts_df[test_start:]

    logger.info(f"Test datasets created with {len(test_true)} timesteps")

    # Calculate detailed metrics if requested
    detailed_metrics = {}
    if not skip_detailed_metrics:
        logger.info("Calculating detailed evaluation metrics...")
        detailed_metrics = calculate_detailed_metrics(test_loader, grid_ts, test_results, device)

    # Create visualization plots
    logger.info("Creating visualization plots...")

    # Main state estimation results plot
    from external.plot_utils import plot_state_estimation_results
    plot_path = plot_state_estimation_results(
        test_results_df, test_baseline, grid_ts, train_data, val_data,
        model_type, grid_code, measurement_rate, plots_dir
    )

    # RMSE comparison plot
    # rmse_plot_path = create_rmse_comparison_plot(
    #     test_results_df, test_baseline, test_true, model_type, plots_dir
    # )

    logger.info("Model evaluation completed!")

    return test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path
#         trainer: PyTorch Lightning trainer instance
#         model: Trained model
#         test_loader: Test data loader
#         grid_ts: Grid time series instance
#         baseline_se: Baseline state estimation results
#         train_data: Training dataset (for indexing)
#         val_data: Validation dataset (for indexing)
#         model_type: Model type string
#         grid_code: Grid code string
#         measurement_rate: Measurement rate
#         plots_dir: Directory to save plots
#         x_mean: Normalization mean (optional, will extract from model if not provided)
#         x_std: Normalization std (optional, will extract from model if not provided)
#         num_node_features: Number of node features
#         num_edge_features: Number of edge features
#         device: Device to run computations on

#     Returns:
#         tuple: (test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path)
#     """
#     logger.info("="*80)
#     logger.info("MODEL EVALUATION")
#     logger.info("="*80)

#     logger.info("Creating test datasets...")

#     # Create test datasets first
#     test_start = len(train_data) + len(val_data)
#     test_baseline = baseline_se.baseline_se_results_df[test_start:] if baseline_se is not None else None
#     test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
#     test_true = grid_ts.values_bus_ts_df[test_start:]

#     logger.info(f"Test datasets created with {len(test_true)} timesteps")

#     logger.info("Running model evaluation on test set...")

#     # Run prediction on test data
#     test_results = trainer.predict(model, test_loader)

#     # Process results into DataFrame
#     test_results_df = process_test_results(test_results, grid_ts)

#     logger.info("Evaluation completed!")

#     # Extract normalization parameters from model if not provided
#     if x_mean is None or x_std is None:
#         try:
#             x_mean = model.x_mean if hasattr(model, 'x_mean') else torch.zeros(2)
#             x_std = model.x_std if hasattr(model, 'x_std') else torch.ones(2)
#         except:
#             logger.warning("Could not extract normalization parameters from model. Using defaults.")
#             x_mean = torch.zeros(2)
#             x_std = torch.ones(2)

#     # Calculate detailed metrics
#     logger.info("="*80)
#     logger.info("DETAILED METRICS CALCULATION")
#     logger.info("="*80)

#     detailed_metrics = calculate_detailed_metrics(
#         model, test_loader, x_mean, x_std,
#         num_node_features, num_edge_features, device
#     )

#     # Create visualization
#     logger.info("="*80)
#     logger.info("RESULTS VISUALIZATION")
#     logger.info("="*80)

#     # Generate initial (untrained) model predictions for comparison
#     logger.info("Generating initial model predictions...")
#     initial_model_predictions = None
#     try:
#         # Create a fresh instance of the same model type without loading weights
#         from external.models.gat_dsse import GAT_DSSE_Lightning
#         from external.utils import get_model_config

#         # Get model configuration
#         num_bus = len(grid_ts.net.bus)
#         hyperparameters = get_model_config(model_type, num_bus)

#         # Create fresh model with same architecture
#         initial_model = GAT_DSSE_Lightning(
#             hyperparameters, model.x_mean, model.x_std,
#             model.edge_mean, model.edge_std, {},
#             time_info=True, loss_type='mse'
#         )
#         initial_model.eval()

#         # Run predictions on test set with untrained model
#         initial_results = []
#         with torch.no_grad():
#             for batch in test_loader:
#                 batch = batch.to(device)

#                 # Use same input processing as the trained model
#                 x = batch.x.clone()
#                 node_param = x[:, initial_model.num_nfeat:initial_model.num_nfeat+3]
#                 x_nodes = x[:, :initial_model.num_nfeat]
#                 x_nodes_gnn = x_nodes.clone()

#                 if initial_model.time_info:
#                     time_info = x[:, initial_model.num_nfeat+3:]
#                     x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

#                 edge_input = batch.edge_attr[:, :initial_model.num_efeat]

#                 # Forward pass
#                 output = initial_model(x_nodes_gnn, batch.edge_index, edge_input)

#                 # Denormalize like in predict_step
#                 v_i = output[:, 0:1] * initial_model.x_std[:1] + initial_model.x_mean[:1]
#                 theta_i = output[:, 1:] * initial_model.x_std[2:3] + initial_model.x_mean[2:3]
#                 theta_i *= (1. - node_param[:, 1:2])

#                 initial_results.append((v_i, theta_i))

#         # Process initial results into DataFrame
#         initial_model_predictions = process_test_results(initial_results, grid_ts)
#         logger.info("Initial model predictions generated successfully!")

#     except Exception as e:
#         logger.warning(f"Failed to generate initial model predictions: {e}")
#         initial_model_predictions = None

#     # Create state estimation results visualization (using first timestep for visualization)
#     plot_path = plot_state_estimation_results(
#         test_results_df, test_baseline, test_measurements, test_true,
#         model_type, grid_code, measurement_rate, plots_dir
#     )

#     # Create RMSE histograms (using all timesteps)
#     rmse_plot_path = plot_rmse_histograms(
#         test_results_df, test_baseline, test_true, grid_ts,
#         model_type, grid_code, plots_dir, initial_model_predictions
#     )

#     return test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path


# def plot_baseline_distribution(test_baseline, test_true, grid_code, plots_dir):
#     """
#     Create a separate distribution analysis plot specifically for baseline results.

#     Args:
#         test_baseline: Pre-created baseline test dataset
#         test_true: Pre-created true values test dataset
#         grid_code: Grid code string
#         plots_dir: Directory to save plots

#     Returns:
#         Path: Path to saved plot file
#     """
#     logger.info("Creating baseline-specific distribution analysis...")

#     # Extract bus indices from column names
#     vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
#     bus_indices = [int(col.split('_')[1]) for col in vm_cols]
#     bus_indices.sort()

#     # Calculate baseline errors
#     vm_errors_baseline = []
#     va_errors_baseline = []

#     min_len = min(len(test_baseline), len(test_true))

#     for bus_id in bus_indices:
#         vm_col = f'bus_{bus_id}_vm_pu'
#         va_col = f'bus_{bus_id}_va_degree'

#         if vm_col in test_baseline.columns and vm_col in test_true.columns:
#             # Baseline vs true - voltage magnitude
#             vm_baseline = test_baseline[vm_col].values[:min_len]
#             vm_true = test_true[vm_col].values[:min_len]
#             vm_errors_baseline.extend((vm_baseline - vm_true) ** 2)

#         if va_col in test_baseline.columns and va_col in test_true.columns:
#             # Baseline vs true - voltage angle with 360° wrapping
#             va_baseline = test_baseline[va_col].values[:min_len]
#             va_true = test_true[va_col].values[:min_len]

#             # Calculate angle difference accounting for 360° wrap-around
#             angle_diff_baseline = va_baseline - va_true
#             angle_diff_baseline = np.where(angle_diff_baseline > 180, angle_diff_baseline - 360, angle_diff_baseline)
#             angle_diff_baseline = np.where(angle_diff_baseline < -180, angle_diff_baseline + 360, angle_diff_baseline)
#             va_errors_baseline.extend(angle_diff_baseline ** 2)

#     # Calculate RMSE values
#     vm_rmse_baseline = np.sqrt(np.mean(vm_errors_baseline)) if vm_errors_baseline else 0
#     va_rmse_baseline = np.sqrt(np.mean(va_errors_baseline)) if va_errors_baseline else 0

#     # Create error distributions for histograms
#     vm_errors_baseline = np.sqrt(vm_errors_baseline) if vm_errors_baseline else np.array([])
#     va_errors_baseline = np.sqrt(va_errors_baseline) if va_errors_baseline else np.array([])

#     # Create 1x2 subplot for baseline analysis
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

#     # Baseline Voltage Magnitude Error Distribution
#     if len(vm_errors_baseline) > 0:
#         ax1.hist(vm_errors_baseline, bins=50, alpha=0.7, color='green', edgecolor='black')
#         ax1.axvline(vm_rmse_baseline, color='darkgreen', linestyle='--', linewidth=2,
#                     label=f'RMSE: {vm_rmse_baseline:.4f}')
#         ax1.legend()
#     else:
#         ax1.text(0.5, 0.5, 'No Baseline Data Available', transform=ax1.transAxes,
#                 ha='center', va='center', fontsize=14, color='gray')

#     ax1.set_xlabel('Error (p.u.)', fontsize=14)
#     ax1.set_ylabel('Frequency', fontsize=14)
#     ax1.set_title('Baseline WLS-DSSE - Voltage Magnitude Error Distribution', fontsize=14)
#     ax1.grid(True, alpha=0.3)

#     # Baseline Voltage Angle Error Distribution
#     if len(va_errors_baseline) > 0:
#         ax2.hist(va_errors_baseline, bins=50, alpha=0.7, color='green', edgecolor='black')
#         ax2.axvline(va_rmse_baseline, color='darkgreen', linestyle='--', linewidth=2,
#                     label=f'RMSE: {va_rmse_baseline:.4f}')
#         ax2.legend()
#     else:
#         ax2.text(0.5, 0.5, 'No Baseline Data Available', transform=ax2.transAxes,
#                 ha='center', va='center', fontsize=14, color='gray')

#     ax2.set_xlabel('Error (degrees)', fontsize=14)
#     ax2.set_ylabel('Frequency', fontsize=14)
#     ax2.set_title('Baseline WLS-DSSE - Voltage Angle Error Distribution', fontsize=14)
#     ax2.grid(True, alpha=0.3)

#     # Overall title
#     fig.suptitle(f'Baseline Distribution Analysis - {grid_code}', fontsize=16, y=0.98)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.88)  # Make room for suptitle

#     # Save the plot
#     plot_filename = f"baseline_distribution_{grid_code.replace('-', '_')}.png"
#     plot_path = Path(plots_dir) / plot_filename
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     logger.info("Baseline distribution analysis created!")
#     logger.info(f"Plot saved to: {plot_path}")
#     logger.info(f"Baseline VM RMSE: {vm_rmse_baseline:.4f}")
#     logger.info(f"Baseline VA RMSE: {va_rmse_baseline:.4f}")

#     # Display the plot
#     plt.show()

#     return plot_path


# def calculate_evaluation_metrics(model_dir, grid_ts, test_loader, baseline_se, train_data, val_data, model_name="best"):
#     """
#     Calculate evaluation metrics by loading model from directory and comparing predictions to true values and baseline.

#     Args:
#         model_dir: Path to the model directory containing checkpoints
#         grid_ts: Grid time series instance
#         test_loader: Test data loader
#         baseline_se: Baseline state estimation results
#         train_data: Training dataset (for indexing)
#         val_data: Validation dataset (for indexing)
#         model_name: Which checkpoint to load ("init", "best", "final")

#     Returns:
#         dict: Dictionary containing various evaluation metrics
#     """
#     from external.train import evaluate_saved_model

#     # Evaluate the saved model to get predictions
#     test_results, _ = evaluate_saved_model(model_dir, grid_ts, test_loader, model_name)

#     # Process results into DataFrame
#     test_results_df = process_test_results(test_results, grid_ts)

#     # Create test datasets
#     test_start = len(train_data) + len(val_data)
#     test_baseline = baseline_se.baseline_se_results_df[test_start:] if baseline_se is not None else None
#     test_true = grid_ts.values_bus_ts_df[test_start:]
#     logger.info("Calculating evaluation metrics...")

#     metrics = {}

#     # Calculate RMSE for voltage magnitudes and angles
#     # Extract bus indices from column names
#     vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
#     bus_indices = [int(col.split('_')[1]) for col in vm_cols]
#     bus_indices.sort()

#     vm_errors_model = []
#     va_errors_model = []
#     vm_errors_baseline = []
#     va_errors_baseline = []

#     for bus_id in bus_indices:
#         vm_col = f'bus_{bus_id}_vm_pu'
#         va_col = f'bus_{bus_id}_va_degree'

#         if vm_col in test_results_df.columns and vm_col in test_true.columns:
#             # Model vs true
#             vm_pred = test_results_df[vm_col].values
#             vm_true = test_true[vm_col].values[:len(vm_pred)]
#             vm_errors_model.extend((vm_pred - vm_true) ** 2)

#             # Baseline vs true
#             if test_baseline is not None:
#                 vm_baseline = test_baseline[vm_col].values[:len(vm_pred)]
#                 vm_errors_baseline.extend((vm_baseline - vm_true) ** 2)

#         if va_col in test_results_df.columns and va_col in test_true.columns:
#             # Model vs true (accounting for 360° wrapping)
#             va_pred = test_results_df[va_col].values
#             va_true = test_true[va_col].values[:len(va_pred)]

#             # Calculate angle difference accounting for 360° wrap-around
#             angle_diff_model = va_pred - va_true
#             angle_diff_model = np.where(angle_diff_model > 180, angle_diff_model - 360, angle_diff_model)
#             angle_diff_model = np.where(angle_diff_model < -180, angle_diff_model + 360, angle_diff_model)
#             va_errors_model.extend(angle_diff_model ** 2)

#             # Baseline vs true (accounting for 360° wrapping)
#             if test_baseline is not None:
#                 va_baseline = test_baseline[va_col].values[:len(va_pred)]
#                 angle_diff_baseline = va_baseline - va_true
#                 angle_diff_baseline = np.where(angle_diff_baseline > 180, angle_diff_baseline - 360, angle_diff_baseline)
#                 angle_diff_baseline = np.where(angle_diff_baseline < -180, angle_diff_baseline + 360, angle_diff_baseline)
#                 va_errors_baseline.extend(angle_diff_baseline ** 2)

#     # Calculate RMSE values
#     metrics['model_vm_rmse'] = np.sqrt(np.mean(vm_errors_model))
#     metrics['model_va_rmse'] = np.sqrt(np.mean(va_errors_model))
#     metrics['baseline_vm_rmse'] = np.sqrt(np.mean(vm_errors_baseline)) if vm_errors_baseline else 0
#     metrics['baseline_va_rmse'] = np.sqrt(np.mean(va_errors_baseline)) if va_errors_baseline else 0

#     # Calculate improvement ratios
#     metrics['vm_improvement_ratio'] = metrics['baseline_vm_rmse'] / metrics['model_vm_rmse'] if metrics['model_vm_rmse'] > 0 and metrics['baseline_vm_rmse'] > 0 else 1.0
#     metrics['va_improvement_ratio'] = metrics['baseline_va_rmse'] / metrics['model_va_rmse'] if metrics['model_va_rmse'] > 0 and metrics['baseline_va_rmse'] > 0 else 1.0

#     logger.info("Evaluation metrics calculated:")
#     logger.info(f"Model VM RMSE: {metrics['model_vm_rmse']:.6f}")
#     logger.info(f"Model VA RMSE: {metrics['model_va_rmse']:.6f}")
#     logger.info(f"Baseline VM RMSE: {metrics['baseline_vm_rmse']:.6f}")
#     logger.info(f"Baseline VA RMSE: {metrics['baseline_va_rmse']:.6f}")
#     logger.info(f"VM Improvement: {metrics['vm_improvement_ratio']:.2f}x")
#     logger.info(f"VA Improvement: {metrics['va_improvement_ratio']:.2f}x")

#     return metrics


# def plot_rmse_histograms(test_results_df, test_baseline, test_true, grid_ts,
#                         model_type, grid_code, plots_dir, initial_model_predictions=None):
#     """
#     Create histograms of RMSE values for voltage magnitude and angle across all buses using all timesteps,
#     including comparisons with initial model and baseline.

#     Args:
#         test_results_df: Model predictions DataFrame
#         test_baseline: Pre-created baseline test dataset (can be None)
#         test_true: Pre-created true values test dataset
#         grid_ts: Grid time series instance (for bus indices)
#         model_type: Model type string
#         grid_code: Grid code string
#         plots_dir: Directory to save plots
#         initial_model_predictions: DataFrame with untrained model predictions (optional)

#     Returns:
#         Path: Path to saved plot file
#     """
#     logger.info("Creating RMSE histograms using all timesteps...")

#     # Extract bus indices from column names
#     vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
#     bus_indices = [int(col.split('_')[1]) for col in vm_cols]
#     bus_indices.sort()

#     # Calculate RMSE across all timesteps for all buses (not per bus)
#     vm_errors_model = []
#     va_errors_model = []
#     vm_errors_baseline = []
#     va_errors_baseline = []

#     # Get aligned timesteps (minimum length across all datasets)
#     if test_baseline is not None:
#         min_len = min(len(test_results_df), len(test_baseline), len(test_true))
#     else:
#         min_len = min(len(test_results_df), len(test_true))
#     logger.info(f"Using {min_len} timesteps for RMSE calculation")

#     for bus_id in bus_indices:
#         vm_col = f'bus_{bus_id}_vm_pu'
#         va_col = f'bus_{bus_id}_va_degree'

#         if vm_col in test_results_df.columns and vm_col in test_true.columns:
#             # Model vs true - voltage magnitude (all timesteps)
#             vm_pred = test_results_df[vm_col].values[:min_len]
#             vm_true = test_true[vm_col].values[:min_len]
#             vm_errors_model.extend((vm_pred - vm_true) ** 2)

#             # Baseline vs true - voltage magnitude (all timesteps)
#             if test_baseline is not None:
#                 vm_baseline = test_baseline[vm_col].values[:min_len]
#                 vm_errors_baseline.extend((vm_baseline - vm_true) ** 2)

#         if va_col in test_results_df.columns and va_col in test_true.columns:
#             # Model vs true - voltage angle with 360° wrapping (all timesteps)
#             va_pred = test_results_df[va_col].values[:min_len]
#             va_true = test_true[va_col].values[:min_len]

#             # Calculate angle difference accounting for 360° wrap-around
#             angle_diff_model = va_pred - va_true
#             angle_diff_model = np.where(angle_diff_model > 180, angle_diff_model - 360, angle_diff_model)
#             angle_diff_model = np.where(angle_diff_model < -180, angle_diff_model + 360, angle_diff_model)
#             va_errors_model.extend(angle_diff_model ** 2)

#             # Baseline vs true - voltage angle with 360° wrapping (all timesteps)
#             if test_baseline is not None:
#                 va_baseline = test_baseline[va_col].values[:min_len]
#                 angle_diff_baseline = va_baseline - va_true
#                 angle_diff_baseline = np.where(angle_diff_baseline > 180, angle_diff_baseline - 360, angle_diff_baseline)
#                 angle_diff_baseline = np.where(angle_diff_baseline < -180, angle_diff_baseline + 360, angle_diff_baseline)
#                 va_errors_baseline.extend(angle_diff_baseline ** 2)

#     # Calculate initial model errors if available
#     vm_errors_initial = []
#     va_errors_initial = []

#     if initial_model_predictions is not None:
#         for bus_id in bus_indices:
#             vm_col = f'bus_{bus_id}_vm_pu'
#             va_col = f'bus_{bus_id}_va_degree'

#             if vm_col in initial_model_predictions.columns and vm_col in test_true.columns:
#                 vm_pred_initial = initial_model_predictions[vm_col].values[:min_len]
#                 vm_true = test_true[vm_col].values[:min_len]
#                 vm_errors_initial.extend((vm_pred_initial - vm_true) ** 2)

#             if va_col in initial_model_predictions.columns and va_col in test_true.columns:
#                 va_pred_initial = initial_model_predictions[va_col].values[:min_len]
#                 va_true = test_true[va_col].values[:min_len]

#                 # Angular difference with 360° wrapping
#                 angle_diff_initial = va_pred_initial - va_true
#                 angle_diff_initial = np.where(angle_diff_initial > 180, angle_diff_initial - 360, angle_diff_initial)
#                 angle_diff_initial = np.where(angle_diff_initial < -180, angle_diff_initial + 360, angle_diff_initial)
#                 va_errors_initial.extend(angle_diff_initial ** 2)

#     # Calculate overall RMSE values
#     vm_rmse_model = np.sqrt(np.mean(vm_errors_model))
#     va_rmse_model = np.sqrt(np.mean(va_errors_model))
#     vm_rmse_initial = np.sqrt(np.mean(vm_errors_initial)) if vm_errors_initial else 0
#     va_rmse_initial = np.sqrt(np.mean(va_errors_initial)) if va_errors_initial else 0
#     vm_rmse_baseline = np.sqrt(np.mean(vm_errors_baseline)) if vm_errors_baseline else 0
#     va_rmse_baseline = np.sqrt(np.mean(va_errors_baseline)) if va_errors_baseline else 0

#     # Create error distributions for histograms (taking square root for RMSE-like values)
#     vm_errors_model = np.sqrt(vm_errors_model)
#     va_errors_model = np.sqrt(va_errors_model)
#     vm_errors_initial = np.sqrt(vm_errors_initial) if vm_errors_initial else np.array([])
#     va_errors_initial = np.sqrt(va_errors_initial) if va_errors_initial else np.array([])
#     vm_errors_baseline = np.sqrt(vm_errors_baseline) if vm_errors_baseline else np.array([])
#     va_errors_baseline = np.sqrt(va_errors_baseline) if va_errors_baseline else np.array([])

#     # Create the visualization with 2x2 subplots (Model and Initial Model only)
#     fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

#     # Row 1: Voltage Magnitude Error Distributions
#     # Trained Model
#     ax1.hist(vm_errors_model, bins=50, alpha=0.7, color='red', edgecolor='black')
#     ax1.set_xlabel('Error (p.u.)', fontsize=14)
#     ax1.set_ylabel('Frequency', fontsize=14)
#     ax1.set_title(f'{model_type.upper()} - VM Error Distribution', fontsize=14)
#     ax1.grid(True, alpha=0.3)
#     ax1.axvline(vm_rmse_model, color='darkred', linestyle='--', linewidth=2,
#                 label=f'RMSE: {vm_rmse_model:.4f}')
#     ax1.legend()

#     # Initial Model
#     if len(vm_errors_initial) > 0:
#         ax2.hist(vm_errors_initial, bins=50, alpha=0.7, color='orange', edgecolor='black')
#         ax2.axvline(vm_rmse_initial, color='darkorange', linestyle='--', linewidth=2,
#                     label=f'RMSE: {vm_rmse_initial:.4f}')
#         ax2.legend()
#     else:
#         ax2.text(0.5, 0.5, 'No Initial Model Data', transform=ax2.transAxes,
#                 ha='center', va='center', fontsize=14, color='gray')
#     ax2.set_xlabel('Error (p.u.)', fontsize=14)
#     ax2.set_ylabel('Frequency', fontsize=14)
#     ax2.set_title('Initial Model - VM Error Distribution', fontsize=14)
#     ax2.grid(True, alpha=0.3)

#     # Row 2: Voltage Angle Error Distributions
#     # Trained Model
#     ax3.hist(va_errors_model, bins=50, alpha=0.7, color='red', edgecolor='black')
#     ax3.set_xlabel('Error (degrees)', fontsize=14)
#     ax3.set_ylabel('Frequency', fontsize=14)
#     ax3.set_title(f'{model_type.upper()} - VA Error Distribution', fontsize=14)
#     ax3.grid(True, alpha=0.3)
#     ax3.axvline(va_rmse_model, color='darkred', linestyle='--', linewidth=2,
#                 label=f'RMSE: {va_rmse_model:.4f}')
#     ax3.legend()

#     # Initial Model
#     if len(va_errors_initial) > 0:
#         ax4.hist(va_errors_initial, bins=50, alpha=0.7, color='orange', edgecolor='black')
#         ax4.axvline(va_rmse_initial, color='darkorange', linestyle='--', linewidth=2,
#                     label=f'RMSE: {va_rmse_initial:.4f}')
#         ax4.legend()
#     else:
#         ax4.text(0.5, 0.5, 'No Initial Model Data', transform=ax4.transAxes,
#                 ha='center', va='center', fontsize=14, color='gray')
#     ax4.set_xlabel('Error (degrees)', fontsize=14)
#     ax4.set_ylabel('Frequency', fontsize=14)
#     ax4.set_title('Initial Model - VA Error Distribution', fontsize=14)
#     ax4.grid(True, alpha=0.3)

#     # Overall title
#     fig.suptitle(f'RMSE Distribution Analysis - {grid_code}\n'
#                  f'Model: {model_type.upper()}',
#                  fontsize=16, y=0.98)

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.90)

#     # Save the plot
#     plot_filename = f"{model_type}_rmse_histograms_{grid_code.replace('-', '_')}.png"
#     plot_path = Path(plots_dir) / plot_filename
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')

#     logger.info("RMSE histograms created!")
#     logger.info(f"Plot saved to: {plot_path}")
#     logger.info(f"VM RMSE - Model: {vm_rmse_model:.4f}, Initial: {vm_rmse_initial:.4f}, Baseline: {vm_rmse_baseline:.4f}")
#     logger.info(f"VA RMSE - Model: {va_rmse_model:.4f}, Initial: {va_rmse_initial:.4f}, Baseline: {va_rmse_baseline:.4f}")

#     # Display the plot
#     plt.show()

#     return plot_path


def print_evaluation_metrics(test_results, initial_test_results, test_true):
    """
    Calculate and print RMSE and MSE metrics comparing models to true values.

    Args:
        test_results: DataFrame with final model predictions (columns: timestamp, bus_idx, vm_pu, va_degree, etc.)
        initial_test_results: DataFrame with initial model predictions (columns: timestamp, bus_idx, vm_pu, va_degree, etc.)
        test_true: DataFrame with true values (columns: timestamp, bus_idx, vm_pu, va_degree, etc.)
    """
    logger.info("="*80)
    logger.info("EVALUATION METRICS")
    logger.info("="*80)

    # Ensure all dataframes have the same length and order
    logger.info(f"Data shapes - True: {test_true.shape}, Final: {test_results.shape}, Initial: {initial_test_results.shape}")

    # Extract aligned data directly
    vm_true = test_true['vm_pu']
    vm_final = test_results['vm_pu']
    vm_initial = initial_test_results['vm_pu']

    va_true = test_true['va_degree']
    va_final = test_results['va_degree']
    va_initial = initial_test_results['va_degree']

    # Calculate voltage magnitude errors
    final_vm_squared_errors = (vm_final - vm_true) ** 2
    initial_vm_squared_errors = (vm_initial - vm_true) ** 2

    # Calculate voltage angle errors (with proper angular difference)

    def angular_difference(pred, true):
        diff = pred - true
        diff = np.where(diff > 180, diff - 360, diff)
        diff = np.where(diff < -180, diff + 360, diff)
        return diff

    final_angle_diff = angular_difference(va_final, va_true)
    initial_angle_diff = angular_difference(va_initial, va_true)

    final_va_squared_errors = final_angle_diff ** 2
    initial_va_squared_errors = initial_angle_diff ** 2

    # Calculate metrics
    final_vm_mse = np.mean(final_vm_squared_errors)
    final_vm_rmse = np.sqrt(final_vm_mse)
    initial_vm_mse = np.mean(initial_vm_squared_errors)
    initial_vm_rmse = np.sqrt(initial_vm_mse)

    final_va_mse = np.mean(final_va_squared_errors)
    final_va_rmse = np.sqrt(final_va_mse)
    initial_va_mse = np.mean(initial_va_squared_errors)
    initial_va_rmse = np.sqrt(initial_va_mse)

    # Print results
    logger.info("VOLTAGE MAGNITUDE METRICS (p.u.):")
    logger.info(f"  Initial Model  - MSE: {initial_vm_mse:.8f},   RMSE: {initial_vm_rmse:.8f}")
    logger.info(f"  Final Model    - MSE: {final_vm_mse:.8f},   RMSE: {final_vm_rmse:.8f}")

    logger.info("")
    logger.info("VOLTAGE ANGLE METRICS (degrees):")
    logger.info(f"  Initial Model  - MSE: {initial_va_mse:.8f},   RMSE: {initial_va_rmse:.8f}")
    logger.info(f"  Final Model    - MSE: {final_va_mse:.8f},   RMSE: {final_va_rmse:.8f}")

    logger.info("="*80)

    # Create error histogram plots
    final_vm_errors = np.sqrt(final_vm_squared_errors)
    initial_vm_errors = np.sqrt(initial_vm_squared_errors)
    final_va_errors = np.sqrt(final_va_squared_errors)
    initial_va_errors = np.sqrt(initial_va_squared_errors)

    create_error_histograms(
        final_vm_errors, initial_vm_errors,
        final_va_errors, initial_va_errors
    )


def create_error_histograms(final_vm_errors, initial_vm_errors, final_va_errors, initial_va_errors):
    """
    Create histogram plots showing error distributions with MSE and RMSE metrics.

    Args:
        final_vm_errors: Final model voltage magnitude errors
        initial_vm_errors: Initial model voltage magnitude errors
        final_va_errors: Final model voltage angle errors
        initial_va_errors: Initial model voltage angle errors
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

    # Calculate metrics from errors
    final_vm_mae = np.mean(final_vm_errors)
    final_vm_rmse = np.sqrt(np.mean(final_vm_errors ** 2))
    initial_vm_mae = np.mean(initial_vm_errors)
    initial_vm_rmse = np.sqrt(np.mean(initial_vm_errors ** 2))

    final_va_mae = np.mean(final_va_errors)
    final_va_rmse = np.sqrt(np.mean(final_va_errors ** 2))
    initial_va_mae = np.mean(initial_va_errors)
    initial_va_rmse = np.sqrt(np.mean(initial_va_errors ** 2))

    # Determine common x-axis ranges
    vm_x_max = max(np.max(final_vm_errors), np.max(initial_vm_errors))
    va_x_max = max(np.max(final_va_errors), np.max(initial_va_errors))

    # Create common bins for each metric type
    vm_bins = np.linspace(0, vm_x_max, 50)
    va_bins = np.linspace(0, va_x_max, 50)

    # Subplot 1: Voltage Magnitude Errors - Final Model
    n1, _, _ = ax1.hist(final_vm_errors, bins=vm_bins, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(final_vm_rmse, color='darkred', linestyle='--', linewidth=2,
                label=f'RMSE: {final_vm_rmse:.6f}')
    ax1.axvline(final_vm_mae, color='red', linestyle=':', linewidth=2,
                label=f'MAE: {final_vm_mae:.6f}')
    ax1.set_xlabel('Error (p.u.)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Final Model - VM Error Distribution', fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Voltage Magnitude Errors - Initial Model
    n2, _, _ = ax2.hist(initial_vm_errors, bins=vm_bins, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(initial_vm_rmse, color='darkorange', linestyle='--', linewidth=2,
                label=f'RMSE: {initial_vm_rmse:.6f}')
    ax2.axvline(initial_vm_mae, color='orange', linestyle=':', linewidth=2,
                label=f'MAE: {initial_vm_mae:.6f}')
    ax2.set_xlabel('Error (p.u.)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Initial Model - VM Error Distribution', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Voltage Angle Errors - Final Model
    n3, _, _ = ax3.hist(final_va_errors, bins=va_bins, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(final_va_rmse, color='darkblue', linestyle='--', linewidth=2,
                label=f'RMSE: {final_va_rmse:.6f}')
    ax3.axvline(final_va_mae, color='blue', linestyle=':', linewidth=2,
                label=f'MAE: {final_va_mae:.6f}')
    ax3.set_xlabel('Error (degrees)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Final Model - VA Error Distribution', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Voltage Angle Errors - Initial Model
    n4, _, _ = ax4.hist(initial_va_errors, bins=va_bins, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(initial_va_rmse, color='darkgreen', linestyle='--', linewidth=2,
                label=f'RMSE: {initial_va_rmse:.6f}')
    ax4.axvline(initial_va_mae, color='green', linestyle=':', linewidth=2,
                label=f'MAE: {initial_va_mae:.6f}')
    ax4.set_xlabel('Error (degrees)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Initial Model - VA Error Distribution', fontsize=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Normalize axes
    # Set same x-axis limits for voltage magnitude plots
    ax1.set_xlim(0, vm_x_max)
    ax2.set_xlim(0, vm_x_max)

    # Set same x-axis limits for voltage angle plots
    ax3.set_xlim(0, va_x_max)
    ax4.set_xlim(0, va_x_max)

    # Set same y-axis limits for voltage magnitude plots
    vm_y_max = max(np.max(n1), np.max(n2))
    ax1.set_ylim(0, vm_y_max * 1.1)
    ax2.set_ylim(0, vm_y_max * 1.1)

    # Set same y-axis limits for voltage angle plots
    va_y_max = max(np.max(n3), np.max(n4))
    ax3.set_ylim(0, va_y_max * 1.1)
    ax4.set_ylim(0, va_y_max * 1.1)

    # Overall title and layout
    fig.suptitle('Model Evaluation - Error Distributions', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Show the plot
    plt.show()

    logger.info("Error histogram plots created and displayed!")


def plot_error_histogram():
    """
    Placeholder function for plotting error histograms.
    This function should be implemented with specific plotting logic.
    """
    logger.info("plot_error_histogram function called - implementation needed")
    pass


# if __name__ == "__main__":
#     print("DSML Model Evaluation")
#     print("Available functions:")
#     print("- process_test_results: Process model predictions into DataFrame")
#     print("- calculate_detailed_metrics: Calculate comprehensive metrics including power flow")
#     print("- plot_state_estimation_results: Create state estimation visualization")
#     print("- plot_rmse_histograms: Create RMSE distribution histograms")
#     print("- evaluate_model: Complete evaluation pipeline with detailed metrics")
#     print("- calculate_evaluation_metrics: Calculate performance metrics")