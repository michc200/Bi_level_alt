import simbench as sb
import numpy as np
import random
import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from robusttest.core.grid_time_series import GridTimeSeries
from robusttest.core.grid_uncertainty import GridUncertainty
from robusttest.core.SE.baseline_state_estimation import BaselineStateEstimation
from robusttest.core.switching_profiles import SwitchingProfiles
import logging

logger = logging.getLogger("generate_LV_TS")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def handle_case_1(lv_code, save_path):
    """Case 1: Add random errors to line parameters."""
    grid_ts_instance = GridTimeSeries(lv_code, save_path=save_path)

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    grid_uncertainty_instance = GridUncertainty(grid_ts_instance.net)
    grid_uncertainty_instance.add_line_parameter_random_errors(grid_ts_instance.net.line.index)
    grid_ts_instance.net = grid_uncertainty_instance.net

    return grid_ts_instance


def handle_case_2(lv_code, save_path):
    """Case 2: Remove lines between buses and recover unsupplied buses."""
    grid_ts_instance = GridTimeSeries(lv_code, save_path=save_path)

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset) # Adjust the start time of the profiles

    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    bus_not_trafo = grid_ts_instance.net.bus[
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.lv_bus.values) &
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.hv_bus.values)
    ].index

    grid_uncertainty_instance = GridUncertainty(grid_ts_instance.net)
    grid_ts_instance = grid_uncertainty_instance.remove_lines_between_buses(bus_not_trafo)
    grid_uncertainty_instance.recover_unsupplied_buses()
    grid_ts_instance.net = grid_uncertainty_instance.net

    return grid_ts_instance


def handle_case_3(lv_code, save_path):
    """Case 3: Add random lines and generate switching profiles."""
    grid_ts_instance = GridTimeSeries(lv_code, save_path=save_path)
    switching = SwitchingProfiles(grid_ts_instance)
    bus_not_trafo = grid_ts_instance.net.bus[
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.lv_bus.values) &
        ~grid_ts_instance.net.bus.index.isin(grid_ts_instance.net.trafo.hv_bus.values)
    ].index

    num_added_lines = np.random.randint(1, 11)  # Random, but reproducible with the set seed
    line_pairs = []
    for _ in range(num_added_lines):
        chosen_buses = random.sample(list(bus_not_trafo), 2)  # Random sampling, but reproducible
        grid_ts_instance.add_line_between_buses(chosen_buses[0], chosen_buses[1])
        line_pairs.append(switching.find_redundend_lines(chosen_buses[0]))

    # Generate random switching profiles
    switching_profile = None
    for lines in line_pairs:
        switching_profile = switching.generate_random_switching_profile(lines)

    grid_ts_instance.profiles[('switch', 'closed')] = switching_profile

    offset = random.randint(0, len(grid_ts_instance.profiles[('load', 'p_mw')].index))  # Random offset for reproducibility
    grid_ts_instance.adjust_profiles_start_time(offset) # Adjust the start time of the profiles
    variables = list(grid_ts_instance.profiles.keys())
    grid_ts_instance.run_timeseries(variables)
    grid_ts_instance.read_time_series_data()

    return grid_ts_instance


def process_measurement_rates(grid_ts_instance, save_path):
    """Process measurement rates and save results."""
    for measurement_rate in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1]:
        grid_ts_instance.create_measurements_ts(measurement_rate=measurement_rate)
        save_path_meas = f"{save_path}/measurement_rate_{measurement_rate}"
        grid_ts_instance.save_measurement_dataframes(save_path_meas)

        data_list, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std = grid_ts_instance.create_pyg_data()

        # Split data_list into training and testing sets (reproducible split)
        train_data_list, test_data_list = train_test_split(data_list, test_size=0.1, random_state=SEED)

        # Create PyTorch Geometric DataLoaders
        train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

        # Save variables into a pickle file
        with open(f"{save_path_meas}/data_loader.pkl", 'wb') as f:
            pickle.dump((train_loader, test_loader, x_set_mean, x_set_std, edge_attr_set_mean, edge_attr_set_std), f)

        # Create a baseline state estimation for these measurements
        baseline_se = BaselineStateEstimation(grid_ts_instance)
        baseline_se_result_df = baseline_se.run_parallel_state_estimation(n_jobs=12)
        baseline_se_result_df.to_csv(f"{save_path_meas}/baseline/baseline_results.csv", index=False) 


def generate_LV_TS():
    lv_codes = [
        code for code in sb.collect_all_simbench_codes(lv_level="", all_data=False)
        if code.split('-')[1] == 'LV' and code.split('-')[-1] == 'sw'
    ]

    for lv_code in lv_codes:
        save_path = f"LV/{lv_code}"
        logger.info(f"Processing LV grid: {lv_code}")

        # Run the three cases
        for case_fn in [handle_case_1, handle_case_2, handle_case_3]:
            grid_ts_instance = case_fn(lv_code, save_path)
            process_measurement_rates(grid_ts_instance, save_path)
