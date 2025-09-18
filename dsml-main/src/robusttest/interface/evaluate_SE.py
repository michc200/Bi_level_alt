import os
import pandas as pd
import numpy as np
import logging
from robusttest.core.grid_time_series import GridTimeSeries
import pandapower as pp  # Importing pandapower for line loading calculations
from robusttest.interface.SE_grid_TS import no_errors, measurement_loss, switching, parameter_errors, grid_uncertainty
from robusttest.core.SE.baseline_state_estimation import BaselineStateEstimation
from robusttest.core.grid_utils.perturbe_topology import perturb_topology
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.lines import Line2D


logger = logging.getLogger("evaluate_SE")



def calculate_metrics(true_df, predicted_df):
    """
    Compute Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) between true and predicted values.

    Args:
        true_df (pd.DataFrame): DataFrame containing true values.
        predicted_df (pd.DataFrame): DataFrame containing predicted values.

    Returns:
        tuple: (rmse, mae, missing_data)
            - rmse (float): Root Mean Squared Error.
            - mae (float): Mean Absolute Error.
            - missing_data (int): Number of missing (NaN) values in the input data.

    Notes:
        - NaN values are ignored in the calculation.
        - If all values are NaN, returns NaN for both RMSE and MAE.
    """
    # Remove NaN values
    mask = ~np.isnan(true_df.values) & ~np.isnan(predicted_df.values)
    true_values = true_df.values[mask]
    predicted_values = predicted_df.values[mask]

    missing_data = np.isnan(true_df.values).sum() + np.isnan(predicted_df.values).sum()
    
    if len(true_values) == 0:
        return np.nan, np.nan  # Return NaN if all values are NaN
    
    rmse = np.sqrt(((true_values - predicted_values) ** 2).mean())
    mae = np.abs(true_values - predicted_values).mean()
    return rmse, mae, missing_data


def calculate_line_loadings_for_time_steps(pp_grid, vm_pu, va_degree):
    """
    Calculate line loadings for multiple time steps using a power flow analysis.

    Args:
        pp_grid (pandapowerNet): The pandapower grid network.
        vm_pu (pd.DataFrame): Voltage magnitudes (pu) for each bus at different time steps.
        va_degree (pd.DataFrame): Voltage angles (degrees) for each bus at different time steps.

    Returns:
        pd.DataFrame: Line loadings (%) for each line at each time step.

    Raises:
        ValueError: If vm_pu and va_degree do not have the same shape.

    Notes:
        - Uses the Pandapower power flow solver to compute line loadings.
        - Each row corresponds to a time step, and each column corresponds to a power line.
    """
    # Ensure the shapes of vm_pu and va_degree match
    if vm_pu.shape != va_degree.shape:
        raise ValueError("vm_pu and va_degree must have the same shape.")
    
    # Get the number of time steps
    time_steps = vm_pu.index
    line_loadings = []
    
    # Iterate over time steps
    for t in time_steps:
        # Assign voltage magnitude and angle to the grid for the current time step
        pp_grid.res_bus['vm_pu'] = pd.Series(vm_pu.loc[t].values, index=pp_grid.bus.index)
        pp_grid.res_bus['va_degree'] = pd.Series(va_degree.loc[t].values, index=pp_grid.bus.index)
        
        # Run power flow to compute line loadings
        pp.runpp(pp_grid, calculate_voltage_angles=True)
        
        # Store the line loadings for the current time step
        line_loadings.append(pp_grid.res_line.loading_percent.values)
    
    # Convert results to a DataFrame
    line_loadings_df = pd.DataFrame(line_loadings, index=time_steps, columns=pp_grid.line.index)
    
    return line_loadings_df


def load_results_from_folder(folder_path):
    """
    Load state estimation results from a given folder.

    Args:
        folder_path (str): Path to the folder containing result files.

    Returns:
        list: A list of dictionaries containing results for different models.

    Notes:
        - Extracts the measurement rate from the folder name.
        - Loads the true values and predicted values from different models.
        - Supports multiple models including baseline and deep learning-based approaches.
    """
    results_dfs_list = []

    # Path to the grid_hyperparams.json file
    grid_hyperparams_file = os.path.join(folder_path, "grid_hyperparams.json")
    print(grid_hyperparams_file)
    if os.path.exists(grid_hyperparams_file):
        print('hi')
        # Load the grid_hyperparams.json and true values DataFrame
        grid_true = GridTimeSeries(save_path=folder_path, load_from_file=grid_hyperparams_file)
        true_df = grid_true.values_bus_ts_df[int(0.9 * len(grid_true.values_bus_ts_df)):]
        results_dfs_list.append({"type": "true", "data": true_df})
        
        # Extract the measurement rate from the folder name
        measurement_rate = folder_path.split("_")[-1]
        
        # Iterate over the specified models and load their results
        for model in ["baseline", "gat_dsse", "gat_dsse_mse", "mlp_dsse", "mlp_dsse_mse", "ensemble_gat_dsse"]:
            result_file = f"{model}_se_results.csv"
            result_path = os.path.join(folder_path, result_file)
            
            # Check if the result file exists
            if os.path.exists(result_path):
                predicted_df = pd.read_csv(result_path)
                
                # Slice the last 10% for baseline
                if model == 'baseline':
                    predicted_df = predicted_df[int(0.9 * len(predicted_df)):]
                
                results_dfs_list.append({
                    "type": "predicted",
                    "model": model,
                    "measurement_rate": measurement_rate,
                    "data": predicted_df
                })
    
    return results_dfs_list




def analyze_results(root_dir):
    """
    Analyze state estimation results across multiple experiments.

    Args:
        root_dir (str): Path to the root directory containing experiment folders.

    Returns:
        pd.DataFrame: A DataFrame containing performance metrics for different models.

    Notes:
        - Iterates through experiment folders and extracts results.
        - Computes RMSE and MAE for voltage magnitudes and angles.
        - Handles multiple scenarios including grid uncertainty, switching, and measurement loss.
    """
    results = []
    
    for root, dirs, files in os.walk(root_dir):
        if "grid_hyperparams.json" in files:
            file_path = os.path.join(root, "grid_hyperparams.json")
            grid_true = GridTimeSeries(save_path=root, load_from_file=file_path)
            true_df = grid_true.values_bus_ts_df[int(0.9*len(grid_true.values_bus_ts_df)):]
            pp_grid = grid_true.net

            if 'switching' in root:
                case = 'switching'
            elif 'grid_uncertainty' in root:
                case = 'grid_uncertainty'
            elif 'parameter_errors' in root:
                case = 'parameter_errors'
            elif 'no_errors' in root:
                case = 'no_errors'
            elif 'measurement_loss' in root:
                case = 'measurement_loss'

            if 'random_topology' in root:
                grid = 'random_topology'
            else:
                grid = 'simbench_topology'

            
            if '--1-sw' in root:
                load = 1
            elif '--2-sw' in root:
                load = 2
            else:
                load = 0


            for dir in dirs:
                if dir.startswith("measurement_rate_"):
                    measurement_rate = float(dir.split("_")[-1])

                    for model in ["baseline", "gat_dsse", "gat_dsse_mse", "mlp_dsse", 'mlp_dsse_mse', "ensemble_gat_dsse"]:
                        result_file = f"{model}_se_results.csv"
                        result_path = os.path.join(root, dir, result_file)
                        
                        if os.path.exists(result_path):
                            predicted_df = pd.read_csv(result_path)
                            if model == 'baseline':
                                predicted_df = predicted_df[int(0.9*len(predicted_df)):]
                            
                            vm_pu_true = true_df[[col for col in true_df.columns if col.endswith("_vm_pu")]]
                            vm_pu_pred = predicted_df[[col for col in predicted_df.columns if col.endswith("_vm_pu")]]
                            
                            va_degree_true = true_df[[col for col in true_df.columns if col.endswith("_va_degree")]]
                            va_degree_pred = predicted_df[[col for col in predicted_df.columns if col.endswith("_va_degree")]]
                            
                            rmse_vm, mae_vm, missing_data = calculate_metrics(vm_pu_true, vm_pu_pred)
                            rmse_va, mae_va, _ = calculate_metrics(va_degree_true, va_degree_pred)

                            # Calculate true and predicted line loadings
                            # true_line_loadings = calculate_line_loadings_for_time_steps(pp_grid, vm_pu_true, va_degree_true)
                            # predicted_line_loadings = calculate_line_loadings_for_time_steps(pp_grid, vm_pu_pred, va_degree_pred)
                            
                            # Compute metrics for line loadings
                            # rmse_line, mae_line = calculate_metrics(
                            #     pd.DataFrame(true_line_loadings),
                            #     pd.DataFrame(predicted_line_loadings)
                            # )
                            
                            results.append({
                                "Case": case,
                                "grid": grid,
                                "load_scenario": load,
                                "Measurement Rate": measurement_rate,
                                "Model": model,
                                "VM_RMSE": rmse_vm,
                                "VM_MAE": mae_vm,
                                "VA_RMSE": rmse_va,
                                "VA_MAE": mae_va,
                                "Missing_data": (missing_data/ len(pp_grid.bus))/len(predicted_df),
                                # "Line_RMSE": rmse_line,
                                # "Line_MAE": mae_line
                            })

    return pd.DataFrame(results)


def plot_vm_rmse(ana_tot, output_filename="VM_RMSE_PJF_plot.pdf"):
    """
    Generates and saves a box plot for VM_RMSE with customized styling.

    Args:
        ana_tot (pd.DataFrame): DataFrame containing analysis results.
        output_filename (str): Name of the output PDF file.

    Returns:
        None
    """
    # Convert units
    ana_tot['VM_MAE'] = ana_tot['VM_MAE'] * 1000
    ana_tot['VM_RMSE'] = ana_tot['VM_RMSE'] * 1000

    # Define a custom color palette
    custom_palette = [
        "#1F4E79", "#F7D507", "#3795D5", "#7A1C1C", "#B77201", "#41641A",
        "#794B01", "#AB2626", "#639729", "#7E6D04", "#DD2525", "#92D050",
        "#BFA405", "#EC9302", "#8AB5E1", "#FBE875", "#356CA5", "#EB8585",
        "#B9E38A", "#FEC05C", "#D7E6F5", "#FEF2B0", "#F5D7D7", "#D8EFBF",
        "#515151", "#777777", "#A3A3A3", "#C4C4C4", "#E0E0E0"
    ]

    # Mapping of model names for legend
    custom_legend_labels = {
        "baseline": "WLS",
        "mlp_dsse": "MLP$_{md}$",
        "gat_dsse": "GNN$_{md}$",
        "mlp_dsse_mse": "MLP",
        "gat_dsse_mse": "GNN",
        "ensemble_gat_dsse": "Ensemble"
    }

    # Set the plot style and font
    sns.set(style="whitegrid")
    rcParams["font.family"] = "Arial"
    rcParams["font.size"] = 8

    # Create the box plot
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax = sns.boxplot(
        data=ana_tot[ana_tot["Case"] == 'no_errors'],
        x="Measurement Rate",
        y="VM_RMSE",
        hue="Model",
        palette=custom_palette[:len(ana_tot["Model"].unique())],
        showfliers=False
    )

    # Update legend labels
    handles, labels = ax.get_legend_handles_labels()
    updated_labels = [custom_legend_labels.get(label, label) for label in labels]
    ax.legend(
        handles,
        updated_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),
        fontsize=7,
        title_fontsize=8,
        ncol=3,
        frameon=False
    )

    # Customize axis labels
    plt.xlabel("Measurement Rate", fontsize=8)
    plt.ylabel("RMSE of V in mp.u.", fontsize=8)
    plt.ylim(0, 5)

    # Adjust layout for clarity
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

def plot_radar_chart(ana_tot, custom_palette, save_path="radar_chart.pdf", meas_rate=0.9):
    """
    Generates a radar chart comparing VM_RMSE medians for different models under various conditions.

    Args:
        ana_tot (pd.DataFrame): Dataframe containing results with columns ['Model', 'Measurement Rate', 'Case', 'VM_RMSE'].
        custom_palette (list): List of colors for the plot.
        save_path (str): Path to save the radar chart PDF.
    """
    
    # Filter data for different models
    wls_df = ana_tot[ana_tot.Model == 'baseline']
    gnn_df = ana_tot[ana_tot.Model == 'gat_dsse']
    mlp_df = ana_tot[ana_tot.Model == 'mlp_dsse']
    gnn_mse_df = ana_tot[ana_tot.Model == 'gat_dsse_mse']
    mlp_mse_df = ana_tot[ana_tot.Model == 'mlp_dsse_mse']
    
    # Define categories and conditions
    categories = ['No Errors', 'Missing Measurements',
              'Parameter Errors', 'Line \n Conncetion Errors', 'Switching', ]
    n_categories = len(categories)
    
    # Helper function to extract median VM_RMSE for each model
    def get_values(df):
        return [
                    df[(df['Case'] == 'no_errors') & (df['Measurement Rate'] == meas_rate)].VM_RMSE.median(),
                    df[(df['Case'] == 'measurement_loss') & (df['Measurement Rate'] == meas_rate)].VM_RMSE.median(),
                    df[(df['Case'] == 'parameter_errors') & (df['Measurement Rate'] == meas_rate)].VM_RMSE.median(),
                    df[(df['Case'] == 'grid_uncertainty') & (df['Measurement Rate'] == meas_rate)].VM_RMSE.median(),
                    df[(df['Case'] == 'switching') & (df['Measurement Rate'] == meas_rate)].VM_RMSE.median(), ]
    
    # Extract values for all models
    benchmark_values = get_values(wls_df)
    mlp_values = get_values(mlp_df)
    gnn_values = get_values(gnn_df)
    mlp_values_mse = get_values(mlp_mse_df)
    gnn_values_mse = get_values(gnn_mse_df)
    
    # Convert data to angles
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Close the loop for each dataset
    benchmark_values += benchmark_values[:1]
    mlp_values += mlp_values[:1]
    gnn_values += gnn_values[:1]
    mlp_values_mse += mlp_values_mse[:1]
    gnn_values_mse += gnn_values_mse[:1]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(4, 3.5), subplot_kw={'projection': 'polar'})
    
    # Plot the data
    ax.plot(angles, benchmark_values, linestyle='--', linewidth=1.5, label='WLS', color=custom_palette[0])
    ax.plot(angles, mlp_values, linestyle='-', linewidth=1.5, label='MLP$_{md}$', color=custom_palette[1])
    ax.plot(angles, gnn_values, linestyle='-', linewidth=1.5, label='GNN$_{md}$', color=custom_palette[3])
    ax.plot(angles, mlp_values_mse, linestyle='-', linewidth=1.5, label='MLP', color=custom_palette[4])
    ax.plot(angles, gnn_values_mse, linestyle='-', linewidth=1.5, label='GNN', color=custom_palette[2])
    
    # Add category labels
    ax.set_yticks([])  # Remove radial labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    
    # Add custom legend
    legend_elements = [
        Line2D([0], [0], color=custom_palette[0], linestyle='--', lw=1.5, label='WLS'),
        Line2D([0], [0], color=custom_palette[1], linestyle='-', lw=1.5, label='MLP$_{md}$'),
        Line2D([0], [0], color=custom_palette[3], linestyle='-', lw=1.5, label='GNN$_{md}$'),
        Line2D([0], [0], color=custom_palette[4], linestyle='-', lw=1.5, label='MLP'),
        Line2D([0], [0], color=custom_palette[2], linestyle='-', lw=1.5, label='GNN')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8, title="Model", title_fontsize=9)
    
    # Save and show plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()


def generate_data_loaders(voltage_level='LV', grid="1-LV-rural1--0-sw", case='no_errors', random_topo=-1, 
                          measurement_rate=0.5, load_grid_from_file = False, ts_exists=False, wls_results=False, save_measurement_dataframes = False):
    """
    Generate PyTorch DataLoaders for training state estimation models.

    Args:
        voltage_level (str): Voltage level (e.g., 'LV', 'MV').
        grid (str): Grid identifier.
        case (str): Experiment case ('no_errors', 'switching', 'measurement_loss', 'parameter_errors', 'grid_uncertainty').
        random_topo (int): Random topology seed (-1 for default).
        measurement_rate (float): Measurement rate percentage.
        load_grid_from_file (bool): Load grid from a before saved.
        ts_exists (bool): Whether time-series data exists.
        wls_results (bool): Whether to compute weighted least squares (WLS) results.

    Returns:
        tuple: Training, validation, and test datasets along with normalization parameters.
    """
    logger.info("Starting data loader generation...")
    
    # Determine whether we're using random topology or default
    rand = random_topo >= 0
    if rand:
        save_path = f"./{voltage_level}/{grid}/{case}/random_topology_{random_topo}"
        logger.info(f"Random topology enabled. Save path: {save_path}")
    else:
        save_path = f"./{voltage_level}/{grid}/{case}"
        logger.info(f"Default topology. Save path: {save_path}")

    folder_path = f"{save_path}/measurement_rate_{measurement_rate}/"
    file_path = f"measurement_rate_{measurement_rate}"
    logger.info(f"Folder path set to: {folder_path}")
    
    # Initialize GridTimeSeries without assuming pre-existing data
    
    if not load_grid_from_file:
        grid_ts_instance = GridTimeSeries(grid, save_path=save_path)
        logger.info("Initialized GridTimeSeries instance.")

        if rand:
            # Apply random topology perturbations
            np.random.seed(grid_ts_instance.random_seed)
            num_lines_to_remove = np.random.randint(1, len(grid_ts_instance.net.line))
            num_lines_to_add = np.random.randint(1, len(grid_ts_instance.net.line))
            logger.info(f"Perturbing topology: removing {num_lines_to_remove} lines and adding {num_lines_to_add} lines.")
            _, net_perturbed = perturb_topology(grid_ts_instance.net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add)
            grid_ts_instance.net = net_perturbed
            logger.info("Topology perturbation applied.")

        grid_ts_instance.save_state()

        # Apply case-specific transformations
        case_functions = {
            "no_errors": no_errors,
            "switching": switching,
            "measurement_loss": measurement_loss,
            "parameter_errors": parameter_errors,
            "grid_uncertainty": grid_uncertainty
        }

        if case in case_functions:
            logger.info(f"Applying case-specific transformation: {case}.")
            grid_ts_instance = case_functions[case](grid_ts_instance)
        else:
            logger.error(f"Unknown case: {case}")
            raise ValueError(f"Unknown case: {case}")
    else:
        if ts_exists:
            grid_ts_instance = GridTimeSeries(grid, save_path=save_path, load_from_file=True, filepath=file_path)
            logger.info("Loaded Grid Time Series from existing files.")
        else:
            grid_ts_instance = GridTimeSeries(grid, save_path=save_path, load_from_file=True, filepath=file_path)
            logger.info("Loaded Grid Time Series from existing files.")
            # Apply case-specific transformations
            case_functions = {
                "no_errors": no_errors,
                "switching": switching,
                "measurement_loss": measurement_loss,
                "parameter_errors": parameter_errors,
                "grid_uncertainty": grid_uncertainty
            }

            if case in case_functions:
                logger.info(f"Applying case-specific transformation: {case}.")
                grid_ts_instance = case_functions[case](grid_ts_instance, grid_loaded = True)
            else:
                logger.error(f"Unknown case: {case}")
                raise ValueError(f"Unknown case: {case}")


    # Generate measurements
    IMSys_loss_rate = 0.1 if case == 'measurement_loss' else 0
    logger.info(f"Creating measurements with bus measurement rate: {measurement_rate} and IMSys_loss_rate: {IMSys_loss_rate}.")
    grid_ts_instance.create_measurements_ts(bus_measurement_rate=measurement_rate, IMSys_loss_rate=IMSys_loss_rate)

    if save_measurement_dataframes:
        grid_ts_instance.save_measurement_dataframes(file_path)

    if not load_grid_from_file:
        grid_ts_instance.save_state(filepath= file_path)


    # Compute WLS results if required
    if wls_results:
        logger.info("Computing WLS baseline state estimation.")
        baseline_se = BaselineStateEstimation(grid_ts_instance)
        baseline_se_result_df = baseline_se.run_parallel_state_estimation(n_jobs=18)
        baseline_filepath = f"{folder_path}/baseline_se_results.csv"
        os.makedirs(os.path.dirname(baseline_filepath), exist_ok=True)
        baseline_se_result_df.to_csv(baseline_filepath, index=False)
        logger.info(f"WLS results saved to {baseline_filepath}.")
        
        train_data, val_data, test_data, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = \
            grid_ts_instance.create_pyg_data(baseline_se_result_df)
    else:
        logger.info("Creating PyG data using bus time-series values.")
        train_data, val_data, test_data, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = \
            grid_ts_instance.create_pyg_data(grid_ts_instance.values_bus_ts_df)

    logger.info("Data loader generation complete.")
    return grid_ts_instance, train_data, val_data, test_data, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std
