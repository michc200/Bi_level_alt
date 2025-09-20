#!/usr/bin/env python3
"""
Plot Utilities for DSML State Estimation

This module contains plotting functions for visualizing state estimation results.
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Setup logger
logger = logging.getLogger("plot_utils")


def plot_state_estimation_results(test_results_df, baseline_se, grid_ts, train_data, val_data,
                                 model_type, grid_code, measurement_rate, plots_dir):
    """
    Create and save state estimation results visualization.

    Args:
        test_results_df: DataFrame with model predictions
        baseline_se: Baseline state estimation results
        grid_ts: Grid time series instance
        train_data: Training dataset for indexing
        val_data: Validation dataset for indexing
        model_type: Model type string for labeling
        grid_code: Grid code for title and filename
        measurement_rate: Measurement rate for title
        plots_dir: Directory to save plots

    Returns:
        Path: Path to saved plot file
    """
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
    plt.plot(x_values, y_model, label=f'{model_type.upper()}',
             color='red', linestyle='-', linewidth=2.5)

    # Customize the plot
    plt.ylim([1.0125, 1.0275])
    plt.xlabel('Bus Index', fontsize=14)
    plt.ylabel('Voltage Magnitude (p.u.)', fontsize=14)
    plt.title(f'State Estimation Results - {grid_code}\\n'
              f'Model: {model_type.upper()}, Measurement Rate: {measurement_rate}',
              fontsize=16, pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the plot
    plot_filename = f"{model_type}_results_{grid_code.replace('-', '_')}.png"
    plot_path = Path(plots_dir) / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    logger.info("Visualization created!")
    logger.info(f"Plot saved to: {plot_path}")

    # Display the plot
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


if __name__ == "__main__":
    print("DSML Plot Utilities")
    print("Available functions:")
    print("- plot_state_estimation_results: Main state estimation visualization")
    print("- create_loss_comparison_plot: Training vs validation loss")
    print("- create_error_distribution_plot: Prediction error analysis")