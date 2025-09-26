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

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.robusttest.core.SE.pf_funcs import get_pflow

# Setup logger
logger = logging.getLogger("dsml_evaluate")


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
    """
    Process test results from model predictions into a DataFrame.

    Args:
        test_results: List of prediction results from trainer.predict()
        grid_ts: Grid time series instance for bus indexing

    Returns:
        pd.DataFrame: Processed test results with bus voltage magnitudes and angles
    """
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


def plot_state_estimation_results(test_results_df, test_baseline, test_measurements, test_true,
                                 model_type, grid_code, measurement_rate, plots_dir):
    """
    Create and save state estimation results visualization with voltage magnitudes and angles.

    Args:
        test_results_df: DataFrame with model predictions
        test_baseline: Pre-created baseline test dataset
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
    y_vm_baseline = [test_baseline.loc[first_timestep, f'bus_{bus}_vm_pu'] for bus in bus_indices]
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
    y_va_baseline = [test_baseline.loc[first_timestep, f'bus_{bus}_va_degree'] for bus in bus_indices]
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


def evaluate_model(trainer, model, test_loader, grid_ts, baseline_se, train_data, val_data,
                  model_type, grid_code, measurement_rate, plots_dir,
                  x_mean=None, x_std=None, num_node_features=8, num_edge_features=6, device='cpu',
                  skip_detailed_metrics=False):
    """
    Complete model evaluation pipeline including prediction, processing, and visualization.

    Args:
        trainer: PyTorch Lightning trainer instance
        model: Trained model
        test_loader: Test data loader
        grid_ts: Grid time series instance
        baseline_se: Baseline state estimation results
        train_data: Training dataset (for indexing)
        val_data: Validation dataset (for indexing)
        model_type: Model type string
        grid_code: Grid code string
        measurement_rate: Measurement rate
        plots_dir: Directory to save plots
        x_mean: Normalization mean (optional, will extract from model if not provided)
        x_std: Normalization std (optional, will extract from model if not provided)
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        device: Device to run computations on

    Returns:
        tuple: (test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path)
    """
    logger.info("="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)

    logger.info("Creating test datasets...")

    # Create test datasets first
    test_start = len(train_data) + len(val_data)
    test_baseline = baseline_se.baseline_se_results_df[test_start:]
    test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
    test_true = grid_ts.values_bus_ts_df[test_start:]

    logger.info(f"Test datasets created with {len(test_true)} timesteps")

    logger.info("Running model evaluation on test set...")

    # Run prediction on test data
    test_results = trainer.predict(model, test_loader)

    # Process results into DataFrame
    test_results_df = process_test_results(test_results, grid_ts)

    logger.info("Evaluation completed!")

    # Extract normalization parameters from model if not provided
    if x_mean is None or x_std is None:
        try:
            x_mean = model.x_mean if hasattr(model, 'x_mean') else torch.zeros(2)
            x_std = model.x_std if hasattr(model, 'x_std') else torch.ones(2)
        except:
            logger.warning("Could not extract normalization parameters from model. Using defaults.")
            x_mean = torch.zeros(2)
            x_std = torch.ones(2)

    # Calculate detailed metrics
    logger.info("="*80)
    logger.info("DETAILED METRICS CALCULATION")
    logger.info("="*80)

    detailed_metrics = calculate_detailed_metrics(
        model, test_loader, x_mean, x_std,
        num_node_features, num_edge_features, device
    )

    # Create visualization
    logger.info("="*80)
    logger.info("RESULTS VISUALIZATION")
    logger.info("="*80)

    # Create state estimation results visualization (using first timestep for visualization)
    plot_path = plot_state_estimation_results(
        test_results_df, test_baseline, test_measurements, test_true,
        model_type, grid_code, measurement_rate, plots_dir
    )

    # Create RMSE histograms (using all timesteps)
    rmse_plot_path = plot_rmse_histograms(
        test_results_df, test_baseline, test_true, grid_ts,
        model_type, grid_code, plots_dir
    )

    return test_results_df, test_baseline, test_measurements, test_true, detailed_metrics, plot_path, rmse_plot_path


def calculate_evaluation_metrics(test_results_df, test_baseline, test_true):
    """
    Calculate evaluation metrics comparing model predictions to true values and baseline.

    Args:
        test_results_df: Model predictions DataFrame
        test_baseline: Pre-created baseline test dataset
        test_true: Pre-created true values test dataset

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    logger.info("Calculating evaluation metrics...")

    metrics = {}

    # Calculate RMSE for voltage magnitudes and angles
    # Extract bus indices from column names
    vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
    bus_indices = [int(col.split('_')[1]) for col in vm_cols]
    bus_indices.sort()

    vm_errors_model = []
    va_errors_model = []
    vm_errors_baseline = []
    va_errors_baseline = []

    for bus_id in bus_indices:
        vm_col = f'bus_{bus_id}_vm_pu'
        va_col = f'bus_{bus_id}_va_degree'

        if vm_col in test_results_df.columns and vm_col in test_true.columns:
            # Model vs true
            vm_pred = test_results_df[vm_col].values
            vm_true = test_true[vm_col].values[:len(vm_pred)]
            vm_errors_model.extend((vm_pred - vm_true) ** 2)

            # Baseline vs true
            vm_baseline = test_baseline[vm_col].values[:len(vm_pred)]
            vm_errors_baseline.extend((vm_baseline - vm_true) ** 2)

        if va_col in test_results_df.columns and va_col in test_true.columns:
            # Model vs true (accounting for 360° wrapping)
            va_pred = test_results_df[va_col].values
            va_true = test_true[va_col].values[:len(va_pred)]

            # Calculate angle difference accounting for 360° wrap-around
            angle_diff_model = va_pred - va_true
            angle_diff_model = np.where(angle_diff_model > 180, angle_diff_model - 360, angle_diff_model)
            angle_diff_model = np.where(angle_diff_model < -180, angle_diff_model + 360, angle_diff_model)
            va_errors_model.extend(angle_diff_model ** 2)

            # Baseline vs true (accounting for 360° wrapping)
            va_baseline = test_baseline[va_col].values[:len(va_pred)]
            angle_diff_baseline = va_baseline - va_true
            angle_diff_baseline = np.where(angle_diff_baseline > 180, angle_diff_baseline - 360, angle_diff_baseline)
            angle_diff_baseline = np.where(angle_diff_baseline < -180, angle_diff_baseline + 360, angle_diff_baseline)
            va_errors_baseline.extend(angle_diff_baseline ** 2)

    # Calculate RMSE values
    metrics['model_vm_rmse'] = np.sqrt(np.mean(vm_errors_model))
    metrics['model_va_rmse'] = np.sqrt(np.mean(va_errors_model))
    metrics['baseline_vm_rmse'] = np.sqrt(np.mean(vm_errors_baseline))
    metrics['baseline_va_rmse'] = np.sqrt(np.mean(va_errors_baseline))

    # Calculate improvement ratios
    metrics['vm_improvement_ratio'] = metrics['baseline_vm_rmse'] / metrics['model_vm_rmse']
    metrics['va_improvement_ratio'] = metrics['baseline_va_rmse'] / metrics['model_va_rmse']

    logger.info("Evaluation metrics calculated:")
    logger.info(f"Model VM RMSE: {metrics['model_vm_rmse']:.6f}")
    logger.info(f"Model VA RMSE: {metrics['model_va_rmse']:.6f}")
    logger.info(f"Baseline VM RMSE: {metrics['baseline_vm_rmse']:.6f}")
    logger.info(f"Baseline VA RMSE: {metrics['baseline_va_rmse']:.6f}")
    logger.info(f"VM Improvement: {metrics['vm_improvement_ratio']:.2f}x")
    logger.info(f"VA Improvement: {metrics['va_improvement_ratio']:.2f}x")

    return metrics


def plot_rmse_histograms(test_results_df, test_baseline, test_true, grid_ts,
                        model_type, grid_code, plots_dir):
    """
    Create histograms of RMSE values for voltage magnitude and angle across all buses using all timesteps.

    Args:
        test_results_df: Model predictions DataFrame
        test_baseline: Pre-created baseline test dataset
        test_true: Pre-created true values test dataset
        grid_ts: Grid time series instance (for bus indices)
        model_type: Model type string
        grid_code: Grid code string
        plots_dir: Directory to save plots

    Returns:
        Path: Path to saved plot file
    """
    logger.info("Creating RMSE histograms using all timesteps...")

    # Extract bus indices from column names
    vm_cols = [col for col in test_true.columns if col.endswith('_vm_pu')]
    bus_indices = [int(col.split('_')[1]) for col in vm_cols]
    bus_indices.sort()

    # Calculate RMSE across all timesteps for all buses (not per bus)
    vm_errors_model = []
    va_errors_model = []
    vm_errors_baseline = []
    va_errors_baseline = []

    # Get aligned timesteps (minimum length across all datasets)
    min_len = min(len(test_results_df), len(test_baseline), len(test_true))
    logger.info(f"Using {min_len} timesteps for RMSE calculation")

    for bus_id in bus_indices:
        vm_col = f'bus_{bus_id}_vm_pu'
        va_col = f'bus_{bus_id}_va_degree'

        if vm_col in test_results_df.columns and vm_col in test_true.columns:
            # Model vs true - voltage magnitude (all timesteps)
            vm_pred = test_results_df[vm_col].values[:min_len]
            vm_true = test_true[vm_col].values[:min_len]
            vm_errors_model.extend((vm_pred - vm_true) ** 2)

            # Baseline vs true - voltage magnitude (all timesteps)
            vm_baseline = test_baseline[vm_col].values[:min_len]
            vm_errors_baseline.extend((vm_baseline - vm_true) ** 2)

        if va_col in test_results_df.columns and va_col in test_true.columns:
            # Model vs true - voltage angle with 360° wrapping (all timesteps)
            va_pred = test_results_df[va_col].values[:min_len]
            va_true = test_true[va_col].values[:min_len]

            # Calculate angle difference accounting for 360° wrap-around
            angle_diff_model = va_pred - va_true
            angle_diff_model = np.where(angle_diff_model > 180, angle_diff_model - 360, angle_diff_model)
            angle_diff_model = np.where(angle_diff_model < -180, angle_diff_model + 360, angle_diff_model)
            va_errors_model.extend(angle_diff_model ** 2)

            # Baseline vs true - voltage angle with 360° wrapping (all timesteps)
            va_baseline = test_baseline[va_col].values[:min_len]
            angle_diff_baseline = va_baseline - va_true
            angle_diff_baseline = np.where(angle_diff_baseline > 180, angle_diff_baseline - 360, angle_diff_baseline)
            angle_diff_baseline = np.where(angle_diff_baseline < -180, angle_diff_baseline + 360, angle_diff_baseline)
            va_errors_baseline.extend(angle_diff_baseline ** 2)

    # Calculate overall RMSE values
    vm_rmse_model = np.sqrt(np.mean(vm_errors_model))
    va_rmse_model = np.sqrt(np.mean(va_errors_model))
    vm_rmse_baseline = np.sqrt(np.mean(vm_errors_baseline))
    va_rmse_baseline = np.sqrt(np.mean(va_errors_baseline))

    # Create error distributions for histograms (taking square root for RMSE-like values)
    vm_errors_model = np.sqrt(vm_errors_model)
    va_errors_model = np.sqrt(va_errors_model)
    vm_errors_baseline = np.sqrt(vm_errors_baseline)
    va_errors_baseline = np.sqrt(va_errors_baseline)

    # Create the visualization with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Voltage Magnitude Error Distribution - Model
    ax1.hist(vm_errors_model, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.set_xlabel('Error (p.u.)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'{model_type.upper()} - Voltage Magnitude Error Distribution', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(vm_rmse_model, color='darkred', linestyle='--', linewidth=2,
                label=f'RMSE: {vm_rmse_model:.4f}')
    ax1.legend()

    # Voltage Magnitude Error Distribution - Baseline
    ax2.hist(vm_errors_baseline, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Error (p.u.)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Baseline WLS-DSSE - Voltage Magnitude Error Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(vm_rmse_baseline, color='darkgreen', linestyle='--', linewidth=2,
                label=f'RMSE: {vm_rmse_baseline:.4f}')
    ax2.legend()

    # Voltage Angle Error Distribution - Model
    ax3.hist(va_errors_model, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.set_xlabel('Error (degrees)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'{model_type.upper()} - Voltage Angle Error Distribution', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(va_rmse_model, color='darkred', linestyle='--', linewidth=2,
                label=f'RMSE: {va_rmse_model:.4f}')
    ax3.legend()

    # Voltage Angle Error Distribution - Baseline
    ax4.hist(va_errors_baseline, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Error (degrees)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Baseline WLS-DSSE - Voltage Angle Error Distribution', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(va_rmse_baseline, color='darkgreen', linestyle='--', linewidth=2,
                label=f'RMSE: {va_rmse_baseline:.4f}')
    ax4.legend()

    # Overall title
    fig.suptitle(f'RMSE Distribution Analysis - {grid_code}\n'
                 f'Model: {model_type.upper()}',
                 fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # Save the plot
    plot_filename = f"{model_type}_rmse_histograms_{grid_code.replace('-', '_')}.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info("RMSE histograms created!")
    logger.info(f"Plot saved to: {plot_path}")
    logger.info(f"VM RMSE - Model Mean: {np.mean(vm_rmse_model):.4f}, Baseline Mean: {np.mean(vm_rmse_baseline):.4f}")
    logger.info(f"VA RMSE - Model Mean: {np.mean(va_rmse_model):.4f}, Baseline Mean: {np.mean(va_rmse_baseline):.4f}")

    # Display the plot
    plt.show()

    return plot_path


if __name__ == "__main__":
    print("DSML Model Evaluation")
    print("Available functions:")
    print("- process_test_results: Process model predictions into DataFrame")
    print("- calculate_detailed_metrics: Calculate comprehensive metrics including power flow")
    print("- plot_state_estimation_results: Create state estimation visualization")
    print("- plot_rmse_histograms: Create RMSE distribution histograms")
    print("- evaluate_model: Complete evaluation pipeline with detailed metrics")
    print("- calculate_evaluation_metrics: Calculate performance metrics")