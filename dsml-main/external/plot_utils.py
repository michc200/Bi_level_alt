#!/usr/bin/env python3
"""
Plot Utilities for DSML State Estimation

This module contains plotting functions for visualizing state estimation results.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Setup logger
logger = logging.getLogger("plot_utils")

# Globals for live plotting
fig = None
axs = None
lines = []
metric_keys = []
history = []
epochs_hist = []

def plot_state_estimation_results(test_results_df, baseline_se, grid_ts, train_data, val_data,
                                 model_type, grid_code, measurement_rate, plots_dir):
    """
    Create and save state estimation results visualization with voltage magnitudes, angles,
    and RMSE/MAE comparison for model and baseline over the entire test set.
    """

    logger.info("Creating visualization...")

    # Prepare data for visualization
    test_start = len(train_data) + len(val_data)
    test_baseline = baseline_se.baseline_se_results_df[test_start:]
    test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
    test_results_true = grid_ts.values_bus_ts_df[test_start:]

    # Get bus indices
    x_values = grid_ts.net.bus.index

    # --- Convert full series for metrics (all timesteps and buses) ---
    true_vm = test_results_true[[f'bus_{bus}_vm_pu' for bus in x_values]].values.flatten()
    baseline_vm = test_baseline[[f'bus_{bus}_vm_pu' for bus in x_values]].values.flatten()
    model_vm = test_results_df[[f'bus_{bus}_vm_pu' for bus in x_values]].values.flatten()

    true_va = test_results_true[[f'bus_{bus}_va_degree' for bus in x_values]].values.flatten()
    baseline_va = test_baseline[[f'bus_{bus}_va_degree' for bus in x_values]].values.flatten()
    model_va = test_results_df[[f'bus_{bus}_va_degree' for bus in x_values]].values.flatten()

    # --- RMSE and MAE functions ---
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    def mae(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

    # --- Compute metrics over all timesteps ---
    metrics = {
        "RMSE_vm_model": rmse(true_vm, model_vm),
        "RMSE_vm_baseline": rmse(true_vm, baseline_vm),
        "RMSE_va_model": rmse(true_va, model_va),
        "RMSE_va_baseline": rmse(true_va, baseline_va),
        "MAE_vm_model": mae(true_vm, model_vm),
        "MAE_vm_baseline": mae(true_vm, baseline_vm),
        "MAE_va_model": mae(true_va, model_va),
        "MAE_va_baseline": mae(true_va, baseline_va),
    }

    logger.info("Full test set metrics:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.6f}")

    # --- Pick first row consistently for snapshot visualization ---
    t0_true = test_results_true.index[0]   # first timestamp in true data
    t0_model = test_results_df.index[0]    # usually 0 in model predictions

    y_vm_measurements = [test_measurements.loc[t0_true, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_vm_results_true = [test_results_true.loc[t0_true, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_vm_baseline = [test_baseline.loc[t0_true, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_vm_model = [test_results_df.loc[t0_model, f'bus_{bus}_vm_pu'] for bus in x_values]

    y_va_measurements = [test_measurements.loc[t0_true, f'bus_{bus}_va_degree'] for bus in x_values]
    y_va_results_true = [test_results_true.loc[t0_true, f'bus_{bus}_va_degree'] for bus in x_values]
    y_va_baseline = [test_baseline.loc[t0_true, f'bus_{bus}_va_degree'] for bus in x_values]
    y_va_model = [test_results_df.loc[t0_model, f'bus_{bus}_va_degree'] for bus in x_values]

    # --- Create figure with 3 plots ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 18))

    # Voltage Magnitude Plot
    ax1.scatter(x_values, y_vm_measurements, label='Measurements', marker='o',
                color='blue', alpha=0.7, s=50)
    ax1.plot(x_values, y_vm_results_true, label='True Values',
             color='orange', linestyle='-', linewidth=2.5)
    ax1.plot(x_values, y_vm_baseline, label='Baseline WLS-DSSE',
             color='green', linestyle='-', linewidth=2)
    ax1.plot(x_values, y_vm_model, label=f'{model_type.upper()}',
             color='red', linestyle='-', linewidth=2.5)

    ax1.set_xlabel('Bus Index', fontsize=14)
    ax1.set_ylabel('Voltage Magnitude (p.u.)', fontsize=14)
    ax1.set_title(f'Voltage Magnitude - {grid_code}', fontsize=14)
    ax1.legend(fontsize=12, loc='best')
    ax1.grid(True, alpha=0.3)

    # Voltage Angle Plot
    ax2.scatter(x_values, y_va_measurements, label='Measurements', marker='o',
                color='blue', alpha=0.7, s=50)
    ax2.plot(x_values, y_va_results_true, label='True Values',
             color='orange', linestyle='-', linewidth=2.5)
    ax2.plot(x_values, y_va_baseline, label='Baseline WLS-DSSE',
             color='green', linestyle='-', linewidth=2)
    ax2.plot(x_values, y_va_model, label=f'{model_type.upper()}',
             color='red', linestyle='-', linewidth=2.5)

    ax2.set_xlabel('Bus Index', fontsize=14)
    ax2.set_ylabel('Voltage Angle (degrees)', fontsize=14)
    ax2.set_title(f'Voltage Angle - {grid_code}', fontsize=14)
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(True, alpha=0.3)

    # --- RMSE & MAE Bar Plot (full test set) ---
    categories = ['Voltage Magnitude', 'Voltage Angle']
    model_rmse = [metrics['RMSE_vm_model'], metrics['RMSE_va_model']]
    baseline_rmse = [metrics['RMSE_vm_baseline'], metrics['RMSE_va_baseline']]
    model_mae = [metrics['MAE_vm_model'], metrics['MAE_va_model']]
    baseline_mae = [metrics['MAE_vm_baseline'], metrics['MAE_va_baseline']]

    width = 0.18
    x = np.arange(len(categories))

    ax3.bar(x - 1.5*width, model_rmse, width, label=f'{model_type.upper()} RMSE', color='red')
    ax3.bar(x - 0.5*width, baseline_rmse, width, label='Baseline RMSE', color='green')
    ax3.bar(x + 0.5*width, model_mae, width, label=f'{model_type.upper()} MAE', color='salmon')
    ax3.bar(x + 1.5*width, baseline_mae, width, label='Baseline MAE', color='limegreen')

    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=13)
    ax3.set_ylabel('Error Value', fontsize=14)
    ax3.set_title('RMSE and MAE over Entire Test Set', fontsize=14)
    ax3.legend(fontsize=12, loc='best')
    ax3.grid(axis='y', alpha=0.3)

    # Add numeric labels above bars
    for bars in ax3.containers:
        ax3.bar_label(bars, fmt='%.4f', fontsize=10, padding=3)

    # --- Figure title and save ---
    fig.suptitle(f'State Estimation Results - {grid_code}\n'
                 f'Model: {model_type.upper()}, Measurement Rate: {measurement_rate}',
                 fontsize=18, y=0.99)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot
    plot_filename = f"{model_type}_results_{grid_code.replace('-', '_')}.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info("Visualization created!")
    logger.info(f"Plot saved to: {plot_path}")

    plt.show()

    return plot_path

def create_loss_comparison_plot(loss_history, model_type, plots_dir):
    """
    Create a plot comparing training and validation loss over epochs.

    Args:
        loss_history: Dictionary with 'train' and 'val' loss arrays
        model_type: Model type string for filename
        plots_dir: Directory to save plots

    Returns:
        Path: Path to saved plot file
    """
    logger.info("Creating loss comparison plot...")

    plt.figure(figsize=(10, 6))

    epochs = range(1, len(loss_history['train']) + 1)
    plt.plot(epochs, loss_history['train'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(epochs, loss_history['val'], label='Validation Loss', color='red', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_type.upper()} - Training vs Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_filename = f"{model_type}_loss_comparison.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info(f"Loss comparison plot saved to: {plot_path}")
    plt.show()

    return plot_path


def create_error_distribution_plot(predictions, targets, model_type, plots_dir):
    """
    Create error distribution plots for model predictions.

    Args:
        predictions: Model predictions array
        targets: True values array
        model_type: Model type string for filename
        plots_dir: Directory to save plots

    Returns:
        Path: Path to saved plot file
    """
    import numpy as np

    logger.info("Creating error distribution plot...")

    errors = predictions - targets

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Error histogram
    ax1.hist(errors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Prediction Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Error Distribution', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Error vs true values scatter
    ax2.scatter(targets.flatten(), errors.flatten(), alpha=0.5, s=10)
    ax2.set_xlabel('True Values', fontsize=12)
    ax2.set_ylabel('Prediction Error', fontsize=12)
    ax2.set_title('Error vs True Values', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot
    plot_filename = f"{model_type}_error_distribution.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info(f"Error distribution plot saved to: {plot_path}")
    plt.show()

    return plot_path


def init_live_plot(metrics_list):
    """
    Initialize live plots.

    Args:
        metrics_list (list[dict]): Structure defining subplots and their labels.
                                   Only keys are used for initialization.
    """
    global fig, axs, lines, metric_keys, history, epochs_hist

    plt.ion()
    num_plots = len(metrics_list)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 3*num_plots), sharex=True)

    if num_plots == 1:
        axs = [axs]  # ensure iterable

    lines = []
    metric_keys = []
    history = [defaultdict(list) for _ in range(num_plots)]
    epochs_hist = []

    for ax, metrics in zip(axs, metrics_list):
        subplot_lines = {}
        subplot_keys = list(metrics.keys())

        for label in subplot_keys:
            line, = ax.plot([], [], label=label)
            subplot_lines[label] = line

        ax.set_title(" / ".join(subplot_keys))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        lines.append(subplot_lines)
        metric_keys.append(subplot_keys)

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()


def update_live_plot(epoch, metrics_list):
    """
    Update plots with one new epoch of data.

    Args:
        epoch (int): New epoch index.
        metrics_list (list[dict]): Each dict corresponds to one subplot.
                                   Keys = label names, values = scalar values (not arrays).
    """
    global epochs_hist, history

    if fig is None:
        raise ValueError("Live plot not initialized. Call init_live_plot first.")

    # Save new data
    epochs_hist.append(epoch)
    for subplot_idx, metrics in enumerate(metrics_list):
        for label, value in metrics.items():
            history[subplot_idx][label].append(value)

    # Update plots with raw values (skip first 50 epochs)
    skip_epochs = 0

    # Only update plots if we have more than skip_epochs
    if len(epochs_hist) > skip_epochs:
        display_epochs = epochs_hist[skip_epochs:]

        for ax, subplot_lines, hist in zip(axs, lines, history):
            for label, values in hist.items():
                values = np.array(values)
                display_values = values[skip_epochs:]  # Skip first 50 values
                subplot_lines[label].set_data(display_epochs, display_values)

            ax.relim()               # Recalculate limits
            ax.autoscale_view(True, True, True)  # Autoscale both axes

    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.pause(0.01)


def finalize_live_plot():
    """Finalize live plots by turning off interactive mode."""
    plt.ioff()


if __name__ == "__main__":
    print("DSML Plot Utilities")
    print("Available functions:")
    print("- plot_state_estimation_results: Main state estimation visualization")
    print("- create_loss_comparison_plot: Training vs validation loss")
    print("- create_error_distribution_plot: Prediction error analysis")
    print("- init_live_plot: Initialize live plotting")
    print("- update_live_plot: Update live plots with new epoch data")
    print("- finalize_live_plot: Finalize live plots")