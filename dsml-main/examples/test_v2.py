import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from robusttest.core.grid_time_series import GridTimeSeries
from robusttest.core.grid_utils.grid_uncertainty import GridUncertainty
from robusttest.core.SE.baseline_state_estimation import BaselineStateEstimation
from robusttest.core.grid_utils.switching_profiles import SwitchingProfiles
from robusttest.core.grid_utils.perturbe_topology import perturb_topology
from robusttest.core.SE.gat_dsse import GAT_DSSE_Lightning
from robusttest.core.SE.mlp_dsse import MLP_DSSE_Lightning
from robusttest.core.SE.gnn_dsse import GCN_DSSE_Lightning
from robusttest.core.SE.ensemble_gat_dsse import EnsembleGAT_DSSE
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import logging
import os
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("generate_LV_TS")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU (rare on macOS)
    accelerator="cuda"
else:
    device = torch.device("cpu")   # Fallback
    accelerator="cpu"

torch.set_float32_matmul_precision('medium')

# Set seeds for reproducibility
SEED = 15
np.random.seed(SEED)
random.seed(SEED)


def no_errors(grid_ts_instance, rand_topology = False): # TODO: start by running this case only
    """Case 1: Add random errors to line parameters."""

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    length = random.randint(2688, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random length for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset, length) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    return grid_ts_instance


def parameter_errors(grid_ts_instance, rand_topology = False):
    """Case 1: Add random errors to line parameters."""

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    length = random.randint(2688, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random length for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset, length) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    grid_uncertainty_instance = GridUncertainty(grid_ts_instance.net)
    grid_uncertainty_instance.add_line_parameter_random_errors(grid_ts_instance.net.line.index)
    grid_ts_instance.net = grid_uncertainty_instance.net

    return grid_ts_instance


def grid_uncertainty(grid_ts_instance, rand_topology = False):
    """Case 2: Remove lines between buses and recover unsupplied buses."""

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    length = random.randint(2688, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random length for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset, length) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    bus_not_trafo = grid_ts_instance.net.bus[
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.lv_bus.values) &
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.hv_bus.values)
    ].index
    
    num_buses = np.random.randint(int(len(bus_not_trafo)/4), len(bus_not_trafo))
    remove_line_from_bus = random.sample(list(bus_not_trafo), num_buses)
    grid_uncertainty_instance = GridUncertainty(grid_ts_instance.net)
    grid_uncertainty_instance.remove_lines_between_buses(remove_line_from_bus)
    if rand_topology:
        grid_uncertainty_instance.recover_unsupplied_buses(mode='random')
    else:
        grid_uncertainty_instance.recover_unsupplied_buses(mode='location')
        
    grid_uncertainty_instance.add_line_parameter_random_errors(grid_ts_instance.net.line.index)
    grid_ts_instance.net = grid_uncertainty_instance.net

    return grid_ts_instance


def switching(grid_ts_instance, rand_topology = False, voltage_level = 'LV'):
    """Case 3: Add random lines and generate switching profiles."""
    switching = SwitchingProfiles(grid_ts_instance)
    bus_not_trafo = grid_ts_instance.net.bus[
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.lv_bus.values) &
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.hv_bus.values)
    ].index
    np.random.seed(grid_ts_instance.random_seed)

    if voltage_level == 'MV':
        num_switch_buses  = np.random.randint(1, 11)
        for _ in range(num_switch_buses):
            chosen_buses = random.sample(list(bus_not_trafo), 1)
            lines = switching.find_redundend_lines(chosen_buses[0])
            if len(lines) > 1:
                line_pairs.append((lines[0], lines[-1]))
    else:
        num_added_lines = np.random.randint(1, 11)  # Random, but reproducible with the set seed
        line_pairs = []
        for _ in range(num_added_lines):
            chosen_buses = random.sample(list(bus_not_trafo), 2)  # Random sampling, but reproducible
            grid_ts_instance.add_line_between_buses(chosen_buses[0], chosen_buses[1])
            lines = switching.find_redundend_lines(chosen_buses[0])
            if len(lines) > 1:
                line_pairs.append((lines[0], lines[-1]))
    

    # Generate random switching profiles
    switching.generate_random_switching_profile(line_pairs)

    switching_profile = switching.profiles[('switch', 'closed')]
    grid_ts_instance.profiles[('switch', 'closed')] = switching_profile

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    length = random.randint(2688, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random length for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset, length) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables) 
    grid_ts_instance.read_time_series_data()

    return grid_ts_instance



                
def train_se_methods(net, train_dataloader, val_dataloader,  x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, model_str = 'gat_dsse', epochs = "50" , save_path = ''):
    """Train the state estimation methods."""
    
    num_bus = len(net.bus)
    if model_str == 'gat_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,    # Example value
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense' : 32,
            'gnn_layers': 5,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
        model = GAT_DSSE_Lightning(hyperparameters, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, time_info=True)

    elif model_str == 'mlp_dsse_mse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,    # Example value
            'num_nodes': num_bus,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'mlp_layers': 4,
            'dropout_rate': 0.3,
            'L': 5,
            'lr': 1e-2,
        }

        model = MLP_DSSE_Lightning(hyperparameters, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, use_mse_loss=True, time_info=True)

    elif model_str == 'mlp_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,    # Example value
            'num_nodes': num_bus,
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'mlp_layers': 4,
            'dropout_rate': 0.3,
            'L': 5,
            'lr': 1e-2,
        }

        model = MLP_DSSE_Lightning(hyperparameters, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, time_info=True)

    elif model_str == 'gat_dsse_mse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,    # Example value
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense' : 32,
            'gnn_layers': 5,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
        model = GAT_DSSE_Lightning(hyperparameters, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, use_mse_loss=True, time_info=True)

    elif model_str == 'ensemble_gat_dsse':
        hyperparameters = {
            'num_nfeat': 8,
            'dim_nodes': 11,    # Example value
            'dim_lines': 6,
            'dim_out': 2,
            'dim_hid': 32,
            'dim_dense' : 32,
            'gnn_layers': 4,
            'heads': 1,
            'K': 2,
            'dropout_rate': 0.0,
            'L': 5,
            'lr': 1e-2,
        }
   
        model = EnsembleGAT_DSSE(hyperparameters, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, reg_coefs, train_dataloader.dataset, time_info=True, use_mse_loss = True)
        train_dataloader = model.train_dataloader()
        val_dataloader = DataLoader(val_dataloader.dataset[:30], batch_size=1, shuffle=False)

    # Use the custom callback in the trainer
    trainer = Trainer( # TODO: we will need to use a custom trainer
        max_epochs = epochs,
        accelerator = accelerator # TODO: revert back to 150
        # callbacks=[early_stopping_callback],
    )
    trainer.fit(model, train_dataloader, val_dataloader)

    os.makedirs(f"{save_path}/{model_str}", exist_ok=True)
    trainer.save_checkpoint(f"{save_path}/{model_str}/model.ckpt")

    return trainer, model


if __name__ == "__main__":
    # Collect all LV grid codes
    logger.info(f'Device: {device}')

    GRID = '1-LV-rural1--0-sw' # '1-MV-urban--0-sw'
    ERRORS = 'no_errors'
    MEAS_RATE = 0.5

    MODEL = 'gat_dsse'
    REG_COEFS = {
        'mu_v':     1e-1,
        'mu_theta': 1e-1,
        'lam_v':    1, #1e-4,
        'lam_p':    1, #1e-8,
        'lam_pf':   1, #1e-6,
        'lam_reg':  0.8,
    } 

    EPOCHS = 100
    
    # Grid and Baseline creation
    grid_time_series_folder = "dsml-data/grid-time-series"
    baseline_state_estimation_folder = "dsml-data/basline-state-estimation"
    model_folder = "dsml-model/"
    
    if not os.path.exists(grid_time_series_folder):
        os.makedirs(grid_time_series_folder)
    
    if not os.path.exists(baseline_state_estimation_folder):
        os.makedirs(baseline_state_estimation_folder)

    grid_id = f"GRID-{GRID}__MEAS_RATE-{MEAS_RATE}__ERROR-{ERRORS}__SEED-{SEED}"
    grid_save_path = f"{grid_time_series_folder}/{grid_id}"
    if not os.path.exists(grid_save_path):
        grid_ts = GridTimeSeries(GRID)
        grid_ts = no_errors(grid_ts)
        grid_ts.create_measurements_ts(bus_measurement_rate = MEAS_RATE)
        grid_ts.save(grid_save_path)
    
    else:
        grid_ts = GridTimeSeries.load(grid_save_path)
    
    baseline_se_id = f"GRID-{GRID}__MEAS_RATE-{MEAS_RATE}__ERROR-{ERRORS}__SEED-{SEED}"
    baseline_se_save_path = f"{baseline_state_estimation_folder}/{baseline_se_id}"
    if not os.path.exists(baseline_se_save_path):
        baseline_se = BaselineStateEstimation(grid_ts)
        baseline_se.run_parallel_state_estimation(n_jobs=18)
        baseline_se.save(baseline_se_save_path)

    else:
        baseline_se = BaselineStateEstimation.load(baseline_se_save_path)

    # Datasets
    train_data, val_data, test_data, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = grid_ts.create_pyg_data(baseline_se.baseline_se_results_df)
    x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = (x_set_mean.to(device), x_set_std.to(device), edge_attr_set_mean.to(device), edge_attr_set_std.to(device))
    
    test_start = int(len(train_data) + len(val_data))
    test_baseline = baseline_se.baseline_se_results_df[test_start:]
    test_measurements = grid_ts.measurements_bus_ts_df[test_start:]
    test_results_true = grid_ts.values_bus_ts_df[test_start:]
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Train
    trainer, model = train_se_methods(grid_ts.net, train_loader, val_loader, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std, REG_COEFS, MODEL, EPOCHS, f"{model_folder}")
    
    # Evaluate
    test_results = trainer.predict(model, test_loader)
    test_results_df = pd.DataFrame() 
    for timestamp, (vm_pu_tensor, va_degree_tensor) in enumerate(test_results):
        vm_pu = vm_pu_tensor.squeeze().tolist()
        va_degree = va_degree_tensor.squeeze().tolist()
        
        # For each bus, add the data to the DataFrame
        for i in range(len(vm_pu)):
            bus_id = grid_ts.net.bus.index[i]
            test_results_df.loc[timestamp, f"bus_{bus_id}_vm_pu"] = vm_pu[i]
            test_results_df.loc[timestamp, f"bus_{bus_id}_va_degree"] = va_degree[i] * (180 / np.pi)

    x_values = grid_ts.net.bus.index

    # Extract y values for each trace
    y_measurements = [test_measurements.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_results_true = [test_results_true.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_baseline = [test_baseline.loc[test_start, f'bus_{bus}_vm_pu'] for bus in x_values]
    y_model = [test_results_df.loc[0, f'bus_{bus}_vm_pu'] for bus in x_values]

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each trace
    ax.scatter(x_values, y_measurements, label='measurements', marker='o', color='blue')
    ax.plot(x_values, y_results_true, label='results', color='orange', linestyle='-')
    ax.plot(x_values, y_baseline, label='wls_dsse', color='green', linestyle='-')
    ax.plot(x_values, y_model, label= MODEL , color='red', linestyle='-')

    # Set y-axis limits
    ax.set_ylim([1.0125, 1.0275])

    # Add labels, legend, and title
    ax.set_xlabel('Bus Index')
    ax.set_ylabel('Voltage Magnitude (pu)')
    ax.set_title('State Estimation Results')
    ax.legend()
    plt.show()
    

