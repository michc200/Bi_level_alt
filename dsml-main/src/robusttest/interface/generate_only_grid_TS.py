import simbench as sb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import multiprocessing
import torch
from robusttest.core.grid_time_series import GridTimeSeries
from robusttest.core.grid_utils.perturbe_topology import perturb_topology
from robusttest.interface.SE_grid_TS import no_errors, measurement_loss, switching, parameter_errors, grid_uncertainty
import logging
import os
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger("generate_LV_TS")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

# Set seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)



def generate_measurements(grid_ts_instance, save_measurement_dataframes = False):
    """Process measurement rates and save results."""
    for measurement_rate, lam_p, lam_pf in zip([0.05, 0.1, 0.2, 0.5, 0.9], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]): # 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.9, [3, 5, 10, 20, 40], [100, 100, 100, 100, 100]
        if 'measurement_loss'in grid_ts_instance.save_path:
            grid_ts_instance.create_measurements_ts(bus_measurement_rate=measurement_rate, IMSys_loss_rate = 0.1)
        else:
            grid_ts_instance.create_measurements_ts(bus_measurement_rate=measurement_rate)

        save_path_meas = f"measurement_rate_{measurement_rate}"
        if save_measurement_dataframes:
            grid_ts_instance.save_measurement_dataframes(save_path_meas)

        grid_ts_instance.save_state(filepath= save_path_meas)

        grid_ts_instance.random_seed_measurements = None


if __name__ == "__main__":
    # Collect all LV grid codes
    logger.info(f'Device: {device}')
    multiprocessing.freeze_support()
    voltage_level = 'LV'
    grid_codes = [code for code in sb.collect_all_simbench_codes(lv_level="", all_data=False)
        if code.split('-')[1] == voltage_level and code.split('-')[-1] == 'sw']
    num_random_topolgies = 2
    j = 0

    for code in grid_codes[:]:
        save_path = f"{voltage_level}/{code}"
        logger.info(f"Processing {voltage_level} grid: {code}")

        # Run the three cases
        for case_fn in [no_errors, switching, measurement_loss, parameter_errors, grid_uncertainty,]: # switching,

            models = ['gat_dsse', 'gat_dsse_mse', 'mlp_dsse', 'mlp_dsse_mse'] #, 'gcn_dsse','gat_dsse', 'gat_dsse_mse', 'mlp_dsse', 
            # if case_fn.__name__ == 'switching':
            #     models.append('ensemble_gat_dsse')
                    
            logger.info(f"Started processing {voltage_level} grid: {code} in case {case_fn.__name__}")
            save_path = f"{voltage_level}/{code}"
            save_path = f"{save_path}/{case_fn.__name__}"
            grid_ts_instance = GridTimeSeries(code, save_path=save_path)
            grid_ts_instance.save_state()
            grid_ts_instance = case_fn(grid_ts_instance)
            generate_measurements(grid_ts_instance, save_measurement_dataframes= False)

            logger.info(f"Finished processing {voltage_level} grid: {code} in case {case_fn.__name__}")

            for i in range(num_random_topolgies):
                logger.info(f"Started processing {voltage_level} random_topology_{i} grid: {code} in case {case_fn.__name__}")
                save_path = f"{voltage_level}/{code}/{case_fn.__name__}/random_topology_{i}"
                grid_ts_instance = GridTimeSeries(code, save_path=save_path)
                np.random.seed(grid_ts_instance.random_seed)
                num_lines_to_remove = np.random.randint(1, len(grid_ts_instance.net.line))
                num_lines_to_add = np.random.randint(1, len(grid_ts_instance.net.line)) 
                _, net_perturbed = perturb_topology(grid_ts_instance.net, num_lines_to_remove=num_lines_to_remove, num_lines_to_add=num_lines_to_add)
                grid_ts_instance.net = net_perturbed
                grid_ts_instance.save_state()
                grid_ts_instance = case_fn(grid_ts_instance, rand_topology = True)
                generate_measurements(grid_ts_instance, save_measurement_dataframes= False)
                logger.info(f"Finished processing {voltage_level} random_topology_{i} grid: {code} in case {case_fn.__name__}")