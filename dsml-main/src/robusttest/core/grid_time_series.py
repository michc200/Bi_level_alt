import pandas as pd
import numpy as np
import logging
import simbench as sb
import pandapower as pp
import scipy.sparse as sp
from scipy.spatial.distance import euclidean
import json
from torch_geometric.data import Data
import torch
import pandapower.timeseries as ts
import pandapower.toolbox as tb
from pandapower.control.controller.const_control import ConstControl
from pandapower.timeseries.data_sources.frame_data import DFData
from demandlib import bdew
from typing import Tuple, List
import pvlib
import os
import pickle


# Configure logging
logger = logging.getLogger("GridTimeSeries")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')


class GridTimeSeries:
    """
    Transforms a SimBench grid with profiles to a time series of measurements.
    """

    def __init__(self, sb_grid_code: str = None, save_path: str = None, load_from_file: bool = False, filepath: str = None):
        self.save_path = save_path

        if load_from_file and save_path:
            self.load_state(filepath)
        else:
            if not sb_grid_code:
                raise ValueError("sb_grid_code must be provided when not loading from file.")
            self.sb_grid_code = sb_grid_code
            self.net = sb.get_simbench_net(sb_grid_code)
            logger.info(f"Loading profiles from {sb_grid_code}")
            self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
            self.profiles = {k: v for k, v in self.profiles.items() if len(v) > 0}
            logger.info(f"Profile keys: {list(self.profiles.keys())}")
            self.fix_geodata()
            logger.info("Geodata fixed.")

            self.edge_param = self.get_edge_param()
            self.variables = None
            self.errors = None
            self.start_offset = 0
            self.length = len(self.profiles[('load', 'p_mw')])
            self.bus_measurement_rate = None
            self.line_measurement_rate = None
            self.random_seed = np.random.randint(0, 1000)
            self.random_seed_measurements = None

        if load_from_file and os.path.exists(os.path.join(self.save_path, "res_bus/vm_pu.csv")):
            self.read_time_series_data()

    def load_state(self, filepath: str = None) -> None:
        try:
            full_path = self.save_path
            if filepath:
                full_path = os.path.join(self.save_path, filepath)

            with open(os.path.join(full_path, "grid_hyperparams.json"), 'r') as f:
                state = json.load(f)
            
            self.start_offset = state['start_offset']
            self.length = state['length']
            self.random_seed = state['random_seed']
            if 'random_seed_measurements' in state.keys():
                self.random_seed_measurements = state['random_seed_measurements']
            self.variables = state['variables']
            self.errors = state['errors']
            self.bus_measurement_rate = state['bus_measurement_rate']
            self.line_measurement_rate = state['line_measurement_rate']
            self.save_path = state['save_path']

            sb_grid_code = state['save_path'].split('/')[2]
            self.net = sb.get_simbench_net(sb_grid_code)

            self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)


            self.net = pp.from_json(f"{full_path}/grid.json")

            logger.info(f"Profile keys: {list(self.profiles.keys())}")

            self.adjust_profiles_start_time(self.start_offset, self.length)

            logger.info(f"State loaded from {full_path}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            raise

    def fix_geodata(self) -> None:
        """Fix missing geodata for the network buses."""
        if not all(self.net.bus_geodata.x):
            for i in self.net.bus_geodata.index:
                self.net.bus_geodata.coords[i] = (self.net.bus_geodata.x[i], self.net.bus_geodata.y[i])

    def set_variables(self, variables: List[Tuple[str, str]]) -> None:
        """Set variables for time-series control."""
        self.variables = variables

    def plot(self) -> None:
        """Plot network results."""
        try:
            pp.plotting.plotly.simple_plotly(self.net)
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

    def save_state(self, filepath: str = None, additional_params: dict = None) -> None:
        """
        Save important hyperparameters, random seed, and a version of the grid for replicability.

        Args:
            filename (str): The name of the file to save the state to (without extension).
            additional_params (dict, optional): Additional parameters to save.

        """
        state = {
            'start_offset': self.start_offset,
            'length': self.length,
            'random_seed': self.random_seed,
            'random_seed_measurements': self.random_seed_measurements,
            'variables': self.variables,
            'errors': self.errors,
            'bus_measurement_rate': self.bus_measurement_rate,
            'line_measurement_rate': self.line_measurement_rate,
            'save_path': self.save_path,
            # 'grid': self.net.copy(),
        }

        # Add additional parameters if provided
        if additional_params:
            state.update(additional_params)

        # Ensure the directory exists
        full_path = self.save_path
        if filepath:
            full_path = os.path.join(self.save_path, filepath)
        
        os.makedirs(os.path.dirname(os.path.join(full_path, f"grid_hyperparams.json")), exist_ok=True)

        try:
            # Save hyperparameters and other important information as JSON
            with open(os.path.join(full_path, f"grid_hyperparams.json"), 'w') as f:
                json.dump(state, f, indent=4)

            # Create a copy of the network and remove results
            net_copy = self.net.deepcopy()
            tb.clear_result_tables(net_copy)
            net_copy.profiles.clear()
            if 'controller' in net_copy.keys():
                net_copy.controller.drop(net_copy.controller.index, inplace=True)
            if 'output_writer' in net_copy.keys():
                net_copy.output_writer.drop(net_copy.output_writer.index, inplace=True)

            # Save the grid as a JSON file
            pp.to_json(net_copy, os.path.join(full_path, "grid.json"))

            logger.info(f"State saved to {full_path}/grid_hyperparams.json and {full_path}/grid.json.")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def save(self, path: str):
        pickle.dump(self, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))
    
    def add_line_between_buses(self, from_bus: int, to_bus: int) -> None:
        """
        Add a line between two buses and corresponding switches.

        Args:
            from_bus: Source bus ID.
            to_bus: Destination bus ID.
        """
        try:
            from_coords = self.net.bus_geodata.loc[from_bus, ['x', 'y']]
            to_coords = self.net.bus_geodata.loc[to_bus, ['x', 'y']]
            length_km = euclidean(from_coords, to_coords) * abs(np.random.normal(40,20))  # Approx. distance in km
            copied_line = self.net.line.iloc[np.random.choice(len(self.net.line))]
            
            line = pp.create_line_from_parameters(
                self.net,
                from_bus,
                to_bus,
                length_km,
                copied_line['r_ohm_per_km'].item(), 
                copied_line['x_ohm_per_km'].item(), 
                copied_line['c_nf_per_km'].item(), 
                copied_line['max_i_ka'].item(),
                name=f"Line {from_bus}-{to_bus}",
                in_service=True,
            )

            pp.create_switch(self.net, bus=from_bus, element=line, et='l', closed=True)
            pp.create_switch(self.net, bus=to_bus, element=line, et='l', closed=True)
            logger.info(f"Added line and switches between buses {from_bus} and {to_bus}.")
        except Exception as e:
            logger.error(f"Error adding line between buses: {e}")

    def adjust_profiles_start_time(self, start_offset: int, length: int):
        """
        Adjusts profiles to start at a specific offset and appends the left-out portion to the end.
        
        Parameters:
        - start_offset (int): A single time offset (in terms of rows/indices) by which to shift all profiles.
        - length (int): The desired length of the profiles after adjustment.
        """

        for profile_name in self.profiles.keys():
            profile_data = self.profiles[profile_name]

            if not isinstance(profile_data, pd.Series) and not isinstance(profile_data, pd.DataFrame):
                raise ValueError(f"Profile {profile_name} must be a pandas Series or DataFrame.")
            
            # Shift the profile using pd.concat
            shifted_profile = pd.concat([profile_data.iloc[start_offset:], profile_data.iloc[:start_offset]]).reset_index(drop=True)

            self.profiles[profile_name] = shifted_profile[:length]

        self.start_offset = start_offset
        self.length = length

        logger.info(f"Profiles adjusted to start at offset {self.start_offset} with length {length}.")

    def run_timeseries(self, variables: List[Tuple[str, str]]) -> None:
        """Run time-series simulation with specified variables."""
        self.variables = variables

        try:
            logger.info(f"Variables for ConstControl: {variables}")
            for var in variables:
                ds = DFData(self.profiles[var])
                ConstControl(
                    self.net, var[0], var[1],
                    element_index=self.net[var[0]].index,
                    profile_name=self.profiles[var].columns,
                    data_source=ds
                )

            ow = ts.OutputWriter(self.net, output_path=f"{self.save_path}", output_file_type=".csv")
            self.configure_output_writer(ow)
            ts.run_timeseries(self.net)
            logger.info("Time-series simulation completed.")
        except Exception as e:
            logger.error(f"Error running time series: {e}")

    def configure_output_writer(self, ow: ts.OutputWriter) -> None:
        """Configure output writer variables."""
        variables_to_log = [
            ('res_trafo', 'p_hv_mw'), ('res_trafo', 'q_hv_mvar'),
            ('res_bus', 'vm_pu'), ('res_bus', 'va_degree'),
            ('res_bus', 'p_mw'), ('res_bus', 'q_mvar'),
            ('res_line', 'loading_percent'), ('res_line', 'p_from_mw'),
            ('res_line', 'q_from_mvar'), ('switch', 'closed'),
        ]
        for var in variables_to_log:
            ow.log_variable(*var)
        ow.init_log_variables(self.net)

    def read_time_series_data(self) -> None:
        """
        Read all time series data from the simulation results and organize them into dataframes.
        """
        try:

            # Load time series data
            res_bus_v_df = pd.read_csv(f'{self.save_path}/res_bus/vm_pu.csv', sep=';').add_prefix('bus_').add_suffix('_vm_pu')
            res_bus_va_df = pd.read_csv(f'{self.save_path}/res_bus/va_degree.csv', sep=';').add_prefix('bus_').add_suffix('_va_degree')
            res_bus_p_df = pd.read_csv(f'{self.save_path}/res_bus/p_mw.csv', sep=';').add_prefix('bus_').add_suffix('_p_mw')
            res_bus_q_df = pd.read_csv(f'{self.save_path}/res_bus/q_mvar.csv', sep=';').add_prefix('bus_').add_suffix('_q_mvar')
            res_line_loading_df = pd.read_csv(f'{self.save_path}/res_line/loading_percent.csv', sep=';').add_prefix('line_').add_suffix('_loading_percent')
            res_line_p_df = pd.read_csv(f'{self.save_path}/res_line/p_from_mw.csv', sep=';').add_prefix('line_').add_suffix('_p_from_mw')
            res_line_q_df = pd.read_csv(f'{self.save_path}/res_line/q_from_mvar.csv', sep=';').add_prefix('line_').add_suffix('_q_from_mvar')
            res_trafo_p_df = pd.read_csv(f'{self.save_path}/res_trafo/p_hv_mw.csv', sep=';').add_prefix('trafo_').add_suffix('_p_hv_mw')
            res_trafo_q_df = pd.read_csv(f'{self.save_path}/res_trafo/q_hv_mvar.csv', sep=';').add_prefix('trafo_').add_suffix('_q_hv_mvar')
            res_switch_closed_df = pd.read_csv(f'{self.save_path}/switch/closed.csv', sep=';').add_prefix('switch_').add_suffix('_closed')

            # Concatenate dataframes for bus, line, and results
            values_bus_ts_df = pd.concat([res_bus_v_df, res_bus_va_df, res_bus_p_df, res_bus_q_df], axis=1)
            values_line_ts_df = pd.concat([res_line_p_df, res_line_q_df, res_trafo_p_df, res_trafo_q_df], axis=1)
            # results_ts_df = pd.concat([res_bus_v_df, res_bus_va_df, res_line_loading_df], axis=1)

            # Drop unnamed columns if present
            values_bus_ts_df = self._drop_unnamed_columns(values_bus_ts_df)
            values_line_ts_df = self._drop_unnamed_columns(values_line_ts_df)
            # results_ts_df = self._drop_unnamed_columns(results_ts_df)
            self.res_switch_closed_df = self._drop_unnamed_columns(res_switch_closed_df)

            # Optionally sort columns
            self.values_bus_ts_df = self._sort_bus_columns(values_bus_ts_df)
            self.values_line_ts_df = self._sort_edge_columns(values_line_ts_df)

            logger.info("Successfully loaded and organized time series data.")
        except Exception as e:
            logger.error(f"Error reading time series data: {e}")

    def create_measurements_ts(self, bus_measurement_rate=0.1, line_measurement_rate=0.0, trafo_measured=True,
                                   IMSys_v_error=0.005, IMSys_va_error=0.01, IMSys_p_error=0.01, IMSys_q_error=0.02, IMSys_loss_rate = 0,
                                   ONS_v_error=0.002, ONS_va_error=0.005, ONS_p_error=0.005, ONS_q_error=0.01,
                                   IMSys_measurement_buses=None, IMSys_measurement_lines=None) -> pd.DataFrame:
        """
        Create a single time series DataFrame containing both measurements and their standard deviations.
        Small errors are applied to a randomly selected 10% of the buses, while large errors are applied to the rest.
        Transformer measurements receive smaller errors.

        Args:
            bus_measurement_rate (float): The percentage of buses to have small measurements.
            line_measurement_rate (float): The percentage of lines to have small measurements.
            trafo_measured (bool): Whether transformer measurements receive smaller errors.
            IMSys_v_error (float): The relative error for voltage measurements in the IMSys error group.
            IMSys_p_error (float): The relative error for active power measurements in the IMSys error group.
            IMSys_q_error (float): The relative error for reactive power measurements in the IMSys error group.
            ONS_v_error (float): The relative error for voltage measurements in the ONS error group.
            ONS_p_error (float): The relative error for active power measurements in the ONS error group.
            ONS_q_error (float): The relative error for reactive power measurements in the ONS error group.
            IMSys_measurement_buses (list[int]): List of bus indices to have small measurements (optional).
            IMSys_measurement_lines (list[int]): List of line indices to have small measurements (optional).

        Returns:
            pd.DataFrame: The combined time series DataFrame with measurements and standard deviations.

        Raises:
            ValueError: If no time series is generated yet to create measurements.
        """
            
        if len(self.values_bus_ts_df) == 0:
            raise ValueError("No time series is generated yet to create measurements.")

        self.get_bus_param()  # Ensure edge topology is loaded

        self.get_edge_param()  # Ensure edge topology is loaded

        # Define error settings
        self.errors = {
            'IMSys': {'vm': IMSys_v_error, 'va': IMSys_va_error, 'p': IMSys_p_error, 'q': IMSys_q_error},
            'ONS': {'vm': ONS_v_error, 'va': ONS_va_error, 'p': ONS_p_error, 'q': ONS_q_error},
        }

        if self.bus_measurement_rate is None:
            self.bus_measurement_rate = bus_measurement_rate

        if self.line_measurement_rate is None:
            self.line_measurement_rate = line_measurement_rate

        # Randomly assign small errors to a subset of buses if no list is provided
        if self.random_seed_measurements is None:
            self.random_seed_measurements = np.random.randint(0, 1000)
        np.random.seed(self.random_seed_measurements)  # Set seed for reproducibility
        if IMSys_measurement_buses is None:
            IMSys_measurement_buses = np.random.choice(self.net.bus.index, size=int(bus_measurement_rate * len(self.net.bus)), replace=False)

        trafo_buses = set(self.net.trafo.hv_bus).union(set(self.net.trafo.lv_bus))  # Combine HV and LV trafo buses into a set

        # Map each bus index to its error type
        bus_error_mapping = {
            bus: 'IMSys' if bus in IMSys_measurement_buses
            else 'ONS' if bus in trafo_buses
            else 'large'
            for bus in self.net.bus.index
        }

        # DataFrame for perturbed values and standard deviations
        combined_ts_df = pd.DataFrame(index=self.values_bus_ts_df.index)

        self.get_SLP_profiles()

        # Apply errors to each column
        for col in self.values_bus_ts_df.columns:
            # Extract the bus index from the column name
            bus_idx = int(col.split('_')[1])  # Assuming column names like 'bus_<index>_...'
            meas_type = col.split('_')[2]  # Assuming column names like 'bus_<index>_<meas_type>'
            error_type = bus_error_mapping.get(bus_idx, 'large')

            # Apply errors to the column
            if error_type == 'large' and self.node_param.loc[bus_idx, 'bool_zero_inj'] == 0:
                if meas_type == 'vm' or meas_type == 'va':
                    perturbed = [0] * len(self.values_bus_ts_df)
                    std = [0] * len(self.values_bus_ts_df)
                else:
                    slp_load_total, slp_sgen_total = self.compute_slp_for_bus(bus_idx)
                    # Combine the perturbed load and sgen into the bus column
                    if meas_type == 'p':
                        perturbed = slp_load_total - slp_sgen_total
                    elif meas_type == 'q':
                        perturbed = slp_load_total * np.tan(np.arccos(0.93)) - slp_sgen_total * np.tan(np.arccos(1))

                    std = [(self.values_bus_ts_df[col] - perturbed).std()] * len(self.values_bus_ts_df)

            elif self.node_param.loc[bus_idx, 'bool_zero_inj'] == 1:
                if (meas_type == 'vm' or meas_type == 'va') and error_type == 'large':
                    perturbed = [0] * len(self.values_bus_ts_df)
                    std = [0] * len(self.values_bus_ts_df)
                else:
                    rel_error = self.errors['IMSys'][meas_type]
                    abs_error = 1e-6
                    perturbed, std = zip(*self.values_bus_ts_df[col].apply(lambda x: self._apply_error(x, rel_error, abs_error)))
            
            elif (error_type == 'ONS' and trafo_measured) or error_type == 'IMSys':
                rel_error = self.errors[error_type][meas_type]
                abs_error = 0 # self.errors[error_type]['abs_error]
                perturbed, std = zip(*self.values_bus_ts_df[col].apply(lambda x: self._apply_error(x, rel_error, abs_error)))

            else:
                perturbed = [0] * len(self.values_bus_ts_df)
                std = [0] * len(self.values_bus_ts_df)

                # Add to the combined DataFrame
            combined_ts_df[col] = perturbed
            combined_ts_df[f"{col}_std"] = std

        if IMSys_loss_rate > 0:
            for bus in IMSys_measurement_buses:
                # Find all columns corresponding to this bus
                bus_cols = [col for col in combined_ts_df.columns if f"bus_{bus}_" in col]
                
                # Generate loss rates using a normal distribution centered at `avg_loss_rate`
                loss_rate = np.random.normal(loc=IMSys_loss_rate, scale=IMSys_loss_rate / 2)

                # Ensure loss rates are within [0, 1]
                loss_rate = np.clip(loss_rate, 0, 1)

                # Randomly lose a percentage of IMSys measurements for this bus
                loss_mask = np.random.choice([True, False], size=len(combined_ts_df), p=[loss_rate, 1 - loss_rate])

                slp_load_total, slp_sgen_total = self.compute_slp_for_bus(bus_idx)

                # Combine the load and sgen profiles for active/reactive power replacement
                for col in bus_cols:
                    meas_type = col.split('_')[2]  # Extract measurement type (e.g., 'p', 'q', etc.)
                    if meas_type == 'p':  # Active power
                        slp_replacement = slp_load_total - slp_sgen_total
                    elif meas_type == 'q':  # Reactive power
                        slp_replacement = slp_load_total * np.tan(np.arccos(0.93)) - slp_sgen_total * np.tan(np.arccos(1))
                    else:
                        slp_replacement = np.zeros(len(combined_ts_df))  # Default for other types (e.g., vm, va)

                    # Apply the loss mask and replace lost measurements with SLPs
                    combined_ts_df[col] = np.where(loss_mask, slp_replacement, combined_ts_df[col])

        trans_node_param_df = self._transform_node_dataframe(self.node_param)
        node_param_repeated = pd.DataFrame(np.tile(trans_node_param_df.values, (len(combined_ts_df), 1)), columns=trans_node_param_df.columns)
        combined_ts_df = pd.concat([combined_ts_df, node_param_repeated.reset_index(drop=True)], axis=1)

        logger.info(f"Measurements of nodes created.")

        combined_lines_ts_df = pd.DataFrame(index=self.values_line_ts_df.index)

        # Randomly assign small errors to a subset of lines if no list is provided
        if IMSys_measurement_lines is None:
            IMSys_measurement_lines = np.random.choice(self.net.line.index, size=int(line_measurement_rate * len(self.net.line.index)), replace=False)

        # Map each line index to its error type
        line_error_mapping = {
            line: 'IMSys' if line in IMSys_measurement_lines
            else 'line'
            for line in self.net.line.index
        }

        lines_in_ts = []

        for col in self.values_line_ts_df.columns:
            # Extract the line index from the column name
            element_type = col.split('_')[0]
            line_idx = int(col.split('_')[1])
            meas_type = col.split('_')[2] 
            if element_type == 'trafo':
                error_type = 'ONS'
            else:
                error_type = line_error_mapping.get(line_idx, 'line')
                lines_in_ts.append(line_idx)
                if line_idx not in self.net.line.index:
                    continue

            # Apply errors to the column
            if error_type == 'line':
                perturbed = [0] * len(self.values_line_ts_df)
                std = [0] * len(self.values_line_ts_df)
            else:
                if (error_type == 'ONS' and trafo_measured):
                    rel_error = self.errors[error_type][meas_type]
                    abs_error = 0 # self.errors[error_type]['abs_error]
                    perturbed, std = zip(*self.values_line_ts_df[col].apply(lambda x: self._apply_error(x, rel_error, abs_error)))
                elif error_type == 'IMSys':
                    rel_error = self.errors[error_type][meas_type]
                    abs_error = 0 # self.errors[error_type]['abs_error]
                    perturbed, std = zip(*self.values_bus_ts_df[col].apply(lambda x: self._apply_error(x, rel_error, abs_error)))
                else:
                    perturbed = [0] * len(self.values_line_ts_df)
                    std = [0] * len(self.values_line_ts_df)

            # Add to the combined DataFrame
            combined_lines_ts_df[col] = perturbed
            combined_lines_ts_df[f"{col}_std"] = std

        # Add lines that are not in the columns of values_line_ts_df
        for line_idx in self.net.line.index:
            if line_idx not in lines_in_ts:
                col_p = f"line_{line_idx}_p_from_mw"
                col_q = f"line_{line_idx}_q_from_mvar"
                combined_lines_ts_df[col_p] = [0] * len(self.values_line_ts_df)
                combined_lines_ts_df[col_q] = [0] * len(self.values_line_ts_df)
                combined_lines_ts_df[f"{col_p}_std"] = [0] * len(self.values_line_ts_df)
                combined_lines_ts_df[f"{col_q}_std"] = [0] * len(self.values_line_ts_df)


        logger.info(f"Measurements of lines and trafo created.")

        line_param_repeated = self.update_edge_params_with_operational_states()

        combined_lines_ts_df = pd.concat([combined_lines_ts_df, line_param_repeated.reset_index(drop=True)], axis=1)

        self.measurements_bus_ts_df = self._sort_bus_columns(combined_ts_df)
        self.measurements_line_ts_df = self._sort_edge_columns(combined_lines_ts_df)

        logger.info(f"Combined time series with measurements and standard deviations created.")

    def save_measurement_dataframes(self, filepath: str) -> None:
        """
        Save the measurement dataframes to CSV files.

        Args:
            bus_filepath (str): The file path to save the bus measurement dataframe.
            line_filepath (str): The file path to save the line measurement dataframe.
        """
        if not hasattr(self, "measurements_bus_ts_df") or not hasattr(self, "measurements_line_ts_df"):
            raise ValueError("Measurement dataframes have not been created yet.")
        
        bus_filepath = f"{self.save_path}/{filepath}/measurements_bus_ts.csv"
        line_filepath = f"{self.save_path}/{filepath}/measurements_line_ts.csv"

        os.makedirs(os.path.dirname(bus_filepath), exist_ok=True)
        os.makedirs(os.path.dirname(line_filepath), exist_ok=True)

        self.measurements_bus_ts_df.to_csv(bus_filepath, index=True)
        self.measurements_line_ts_df.to_csv(line_filepath, index=True)

    def load_measurement_dataframes(self, filepath: str) -> None:
        """
        Load the measurement dataframes from CSV files.

        Args:
            filepath (str): The file path where the measurement dataframes are located.
        """
        bus_filepath = f"{self.save_path}/{filepath}/measurements_bus_ts.csv"
        line_filepath = f"{self.save_path}/{filepath}/measurements_line_ts.csv"

        if not os.path.exists(bus_filepath) or not os.path.exists(line_filepath):
            raise FileNotFoundError("Measurement dataframes not found.")

        self.measurements_bus_ts_df = pd.read_csv(bus_filepath, index_col=0)
        self.measurements_line_ts_df = pd.read_csv(line_filepath, index_col=0)
    
    def create_pyg_data(self, results_df = None):
        """
        Converts the measurements_ts_df into a PyTorch Geometric Data list using the grid topology.
        Computes mean and std for node and edge features, normalizes them, and returns the normalized data list.

        Returns:
            data_list: A list of PyTorch Geometric Data objects.
            x_set_mean: Mean of node features (excluding params).
            x_set_std: Standard deviation of node features (excluding params).
            edge_attr_set_mean: Mean of edge attributes (excluding params).
            edge_attr_set_std: Standard deviation of edge attributes (excluding params).
        """
        if not hasattr(self, "measurements_bus_ts_df") or not hasattr(self, "measurements_line_ts_df"):
            raise ValueError("No measurements_ts_df found. Please create it first.")

        num_buses = self.net.bus.index.size
        num_edges = len(self.edge_param)

        # Extract edge parameters
        num_bus_features = int(len(self.measurements_bus_ts_df.columns) / num_buses) 
        num_edge_features = int(len(self.measurements_line_ts_df.columns) / num_edges)

        for bus in self.net.bus.index:
            self.measurements_bus_ts_df[f'bus_{bus}_va_degree'] = ((self.measurements_bus_ts_df[f'bus_{bus}_va_degree']) * (np.pi / 180)) #% (2 * np.pi)
            self.measurements_bus_ts_df[f'bus_{bus}_va_degree_std'] = ((self.measurements_bus_ts_df[f'bus_{bus}_va_degree_std']) * (np.pi / 180))

        # Convert measurements dataframe to tensor for efficiency
        bus_measurements_tensor = torch.tensor(self.measurements_bus_ts_df.values, dtype=torch.float32)
        edge_measurements_tensor = torch.tensor(self.measurements_line_ts_df.values, dtype=torch.float32)

        if len(results_df) > 0:
            # Select only columns ending with '_va_degree' or '_vm_pu'
            results_df = results_df.filter(regex='(_va_degree|_vm_pu)$')
            results_df = results_df.apply(lambda x: x.replace('', pd.NA))  # Replace empty strings with NaN (if applicable)
            results_df.fillna(method='ffill', inplace=True)
                        
            for bus in self.net.bus.index:
                results_df[f'bus_{bus}_va_degree'] = ((results_df[f'bus_{bus}_va_degree']) * (np.pi / 180)) #% (2 * np.pi)
            results_tensor = torch.tensor(results_df.values, dtype=torch.float32)
            
        datetime_index = self._adjust_df_start_time(pd.DataFrame(data = {'time_id' : pd.date_range(start='2016-01-01 00:00:00', end='2016-12-31 23:45:00', freq='15T')}))

        # Pre-allocate tensors for all timestamps
        num_timestamps = bus_measurements_tensor.shape[0]

        x_tensor = torch.zeros((num_timestamps, num_buses, num_bus_features + 3), dtype=torch.float32)  # addition of time features
        edge_attr_tensor = torch.zeros((num_timestamps, num_edges, num_edge_features), dtype=torch.float32)
        if len(results_df) > 0:
            y_tensor = torch.zeros((num_timestamps, num_buses, 2), dtype=torch.float32)

        for i in range(num_timestamps):
            # Extract time-related features
            current_time = datetime_index.loc[i, 'time_id']
            time_info = torch.tensor(
                [
                    i+1,  # Timestamp index
                    current_time.hour + current_time.minute / 60,  # Fractional hour
                    current_time.dayofweek  # Day of the week (0=Monday, 6=Sunday)
                ],
                dtype=torch.float32
            )
            
            # Repeat time_info for all buses at this timestamp
            time_info_repeated = time_info.repeat(num_buses, 1)  # Shape: (num_buses, 3)

            # Reshape and combine bus features with time info
            x = bus_measurements_tensor[i].reshape(num_buses, num_bus_features)  # Shape: (num_buses, num_bus_features)
            x = torch.cat([x, time_info_repeated], dim=1)  # Concatenate along feature dimension (axis=1)

            # Store in pre-allocated tensor
            x_tensor[i, :, :] = x

            edge_attr = edge_measurements_tensor[i].reshape(num_edges, num_edge_features)
            edge_attr_tensor[i, :, :] = edge_attr

            if len(results_df) > 0:
                y = results_tensor[i].reshape(num_buses, 2)
                y_tensor[i, :, :] = y

        # Partition indices
        num_timestamps = x_tensor.shape[0]
        train_idx = int(num_timestamps * 0.8)
        val_idx = int(num_timestamps * 0.9)
        train_indices = range(0, train_idx)

       # Compute scaling based on training data
        x_train = x_tensor[train_indices]
        edge_attr_train = edge_attr_tensor[train_indices]

        x_mask = x_train != 0.
        x_set_mean = torch.nan_to_num((x_train * x_mask).sum(dim=(0, 1)) / x_mask.sum(dim=(0, 1)))
        x_set_std = torch.nan_to_num(torch.sqrt(((x_train - x_set_mean) ** 2 * x_mask).sum(dim=(0, 1)) / x_mask.sum(dim=(0, 1))))

        edge_attr_mask = edge_attr_train != 0.
        edge_attr_set_mean = torch.nan_to_num((edge_attr_train * edge_attr_mask).sum(dim=(0, 1)) / edge_attr_mask.sum(dim=(0, 1)))
        edge_attr_set_std = torch.nan_to_num(torch.sqrt(((edge_attr_train - edge_attr_set_mean) ** 2 * edge_attr_mask).sum(dim=(0, 1)) / edge_attr_mask.sum(dim=(0, 1))))


        # Normalize node and edge features
        x_set = torch.nan_to_num((x_tensor - x_set_mean) * (x_tensor != 0) / x_set_std)
        x_set[:, :, 8:11] = x_tensor[:, :, 8:11]

        edge_attr_set = torch.nan_to_num((edge_attr_tensor - edge_attr_set_mean) * (edge_attr_tensor != 0) / edge_attr_set_std)
        edge_attr_set[:, :, 6:] = edge_attr_tensor[:, :, 6:]

        # Determine active edges based on the 'closed_line' status for all timestamps
        active_edges_masks = self.measurements_line_ts_df[
                                                            [f'line_{idx}_closed_line' for idx in self.net.line.index] + 
                                                            [f'trafo_{idx}_closed_line' for idx in self.net.trafo.index]
                                                        ] == 1

        # Create a unique identifier for each edge
        self.edge_param['unique_id'] = self.edge_param.index.astype(str) + '_' + self.edge_param['from_bus'].astype(str) + '_' + self.edge_param['to_bus'].astype(str)

        # Create a mapping from the unique identifiers to the new indices
        index_mapping = {unique_id: new_idx for new_idx, unique_id in enumerate(self.edge_param['unique_id'])}

        # Reconstruct data_list from normalized tensors
        data_list = []
        
        for i in range(num_timestamps):
            x = x_set[i, :, :]

            # Get active edges for the current timestamp
            active_edges = self.edge_param[active_edges_masks.values[i]]
            edge_index = torch.tensor(active_edges[['from_bus', 'to_bus']].values.T, dtype=torch.long)

            # Map the active edge indices to the new indices using unique identifiers
            mapped_active_edge_indices = [index_mapping[unique_id] for unique_id in active_edges.unique_id]

            # Extract edge attributes for active edges
            active_edge_attr = edge_attr_set[i, mapped_active_edge_indices, :]
            
            if len(results_df) > 0:
                y = y_tensor[i, :, :]
            else:
                y = torch.zeros(x.size(0), dtype=torch.float32)

            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=active_edge_attr, y=y)
            data.validate(raise_on_error=True)
            data_list.append(data)


        train_data = data_list[:train_idx]
        val_data = data_list[train_idx:val_idx]
        test_data = data_list[val_idx:]

        self.train_idx = 0
        self.train_len = len(train_data)

        self.val_idx = train_idx
        self.val_len = len(val_data)

        self.test_idx = val_idx
        self.test_len = len(test_data)
            
        for bus in self.net.bus.index:
            self.measurements_bus_ts_df[f'bus_{bus}_va_degree'] = ((self.measurements_bus_ts_df[f'bus_{bus}_va_degree']) * (180 /np.pi)) #% (2 * np.pi)
            self.measurements_bus_ts_df[f'bus_{bus}_va_degree_std'] = ((self.measurements_bus_ts_df[f'bus_{bus}_va_degree_std']) * (180 /np.pi))

        return train_data, val_data, test_data, x_set_mean[:8], x_set_std[:8], edge_attr_set_mean[:6], edge_attr_set_std[:6]

    def get_bus_param(self):
        df_bus_param = pd.DataFrame(index=self.net.bus.index)
        df_bus_param["vn_kv"] = self.net.bus["vn_kv"]
        df_bus_param["bool_slack"] = (df_bus_param["vn_kv"]==df_bus_param["vn_kv"].max()).astype(float)
        df_bus_param["bool_zero_inj"] = 0.
        for i in self.net.bus.index:
            if (not i in self.net.load["bus"].values):
                if df_bus_param["bool_slack"][i]==0.:
                    df_bus_param["bool_zero_inj"][i]=1.
                    
        self.node_param = df_bus_param

    def get_edge_param(self):
    
        self.net.bus["name"] = np.arange(self.net.bus.index.size)
        
        edge_length = self.net.line["length_km"]
        edge_r = self.net.line["r_ohm_per_km"] * edge_length
        edge_x = self.net.line["x_ohm_per_km"] * edge_length
        
        edge_c = self.net.line["c_nf_per_km"] * edge_length
        edge_b =  -2 * np.pi * self.net.f_hz * edge_c * 1e-9
        edge_g = self.net.line["g_us_per_km"] * edge_length * 1e-6
        
        edge_imax = self.net.line["max_i_ka"]
        t_sn = self.net.trafo["sn_mva"]
        
        t_r = (self.net.trafo["vkr_percent"]/100) * (self.net.sn_mva/t_sn)
        t_z = (self.net.trafo["vk_percent"]/100) * (self.net.sn_mva/t_sn)
        t_x_square =  t_z.pow(2) - t_r.pow(2) 
        t_x = t_x_square.pow(0.5)
        
        t_g = (self.net.trafo["pfe_kw"]/1000) * (self.net.sn_mva/t_sn**2)
        t_y = (self.net.trafo["i0_percent"]/100)
        t_b_square = t_y**2 - t_g**2
        t_b = t_b_square.pow(0.5)
        
        Z_trafo = (self.net.trafo["vn_lv_kv"]**2/self.net.sn_mva)
        
        t_R = t_r * Z_trafo
        t_X = t_x * Z_trafo
        t_G = t_g/Z_trafo
        t_B = t_b/Z_trafo
        
        edge_r = pd.concat([edge_r,t_R])
        edge_x = pd.concat([edge_x,t_X])
        edge_b = pd.concat([edge_b,t_B])
        edge_g = pd.concat([edge_g,t_G])
        
        t_phase_shift = self.net.trafo["shift_degree"]*np.pi/180
        
        edge_phase_shift = np.concatenate((np.zeros(self.net.line.index.size), t_phase_shift.values))
        
        edge_switch =  pd.DataFrame(data ={"closed": True}, index = np.concatenate([self.net.line.index,self.net.trafo.index]))
        edge_ind = -1
        for i in self.net.switch.index:
            old_ind = edge_ind
            edge_ind = self.net.switch["element"][i]

            if edge_ind == old_ind:
                edge_switch.loc[edge_ind] = (self.net.switch["closed"][i] and self.net.switch["closed"][switch_old])
            else:
                edge_switch.loc[edge_ind] = self.net.switch["closed"][i]
            
            switch_old = i
                
        bool_closed_line = edge_switch["closed"].values.astype("float64")

        edge_source = pd.concat([self.net.line["from_bus"],self.net.trafo["hv_bus"]])
        edge_target = pd.concat([self.net.line["to_bus"],self.net.trafo["lv_bus"]])
        edge_source_index =  edge_source.values
        edge_target_index =  edge_target.values
        new_edge_source = self.net.bus['name'][edge_source_index].values.astype(float)
        new_edge_target = self.net.bus['name'][edge_target_index].values.astype(float)
        
        
        edge_Z = edge_r.values + 1j * edge_x.values
        edge_Y = np.reciprocal(edge_Z)
        edge_Ys = edge_g - 1j* edge_b
        
        edge_imax_or_sn = pd.concat([edge_imax, t_sn])
        
        
        self.edge_param = pd.DataFrame({'from_bus': new_edge_source, 'to_bus': new_edge_target, 'G': np.real(edge_Y), 'B': np.imag(edge_Y), 'Gs': np.nan_to_num(np.real(edge_Ys)), 'Bs': np.nan_to_num(np.imag(edge_Ys)), 'closed line': bool_closed_line, 'phase shift': edge_phase_shift, 'imax or sn': edge_imax_or_sn})

    def create_switch_element_matrix(self):
        """
        Create a sparse matrix representing the relationship between switches and elements.
        Handles non-continuous indices for both switches and elements by mapping them to continuous indices.
        """
        # Map non-continuous switch indices to continuous indices
        switch_indices = self.net.switch.index.tolist()
        switch_idx_map = {original_idx: continuous_idx for continuous_idx, original_idx in enumerate(switch_indices)}

        # Map non-continuous element indices to continuous indices
        element_indices = self.edge_param.index.tolist()
        element_idx_map = {original_idx: continuous_idx for continuous_idx, original_idx in enumerate(element_indices)}

        # Initialize lists for sparse matrix construction
        row_indices = []
        col_indices = []
        data = []

        # Iterate over switches to populate the matrix
        for switch_idx, row in self.net.switch.iterrows():
            element_idx = row['element']
            continuous_row_idx = switch_idx_map[switch_idx]  # Map switch index to continuous
            continuous_col_idx = element_idx_map[element_idx]  # Map element index to continuous
            row_indices.append(continuous_row_idx)
            col_indices.append(continuous_col_idx)
            data.append(1)  # Indicating a relationship

        # Create the sparse matrix
        n_switches_continuous = len(switch_indices)  # Number of continuous switch indices
        n_elements_continuous = len(element_indices)  # Number of continuous element indices
        switch_element_matrix = sp.coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_switches_continuous, n_elements_continuous),
        )
        return switch_element_matrix
    
    def compute_operational_states(self):
        """
        Compute the operational states of all elements over the entire time series using switch states.

        Returns:
            pandas.DataFrame: A DataFrame containing the operational states of all elements over the entire time series.
                The DataFrame has time steps as the index and element names as the columns.
        """
        # Precompute the switch-element matrix
        switch_element_matrix = self.create_switch_element_matrix()

        # Convert the switch states DataFrame to a numpy array
        switch_states = self.res_switch_closed_df.to_numpy()

        # Compute operational states using matrix multiplication
        # Multiply switch states (time steps x switches) with the sparse matrix (switches x elements)
        element_states = switch_states @ switch_element_matrix.toarray()

        # Elements are operational if all their associated switches are closed
        operational_states = (element_states == switch_element_matrix.sum(axis=0)).astype(float)

        # Convert the result to a DataFrame with time steps as the index
        operational_states_df = pd.DataFrame(
            operational_states,
            index=self.res_switch_closed_df.index,
            columns=self.edge_param.index,
        )

        return operational_states_df
    
    def update_edge_params_with_operational_states(self):
        """
        Update edge parameters dynamically based on the computed operational states.

        This method computes the operational states for all elements and updates the edge parameters
        by repeating them for all time steps and updating the 'closed line' status based on the computed
        operational states.

        Returns:
            pd.DataFrame: The updated edge parameters with dynamic values.
        """
        if ('switch', 'closed') in self.profiles.keys():
            # Compute the operational states for all elements
            operational_states_df = self.compute_operational_states()
        else:
            self.res_switch_closed_df = pd.DataFrame(np.ones((len(self.values_bus_ts_df.index), len(self.net.switch.index))), index=self.values_bus_ts_df.index, columns=[f"switch_{i}_closed" for i in self.net.switch.index])
            operational_states_df = pd.DataFrame(np.tile(self.edge_param['closed line'].values.T, (len(self.values_bus_ts_df.index), 1)), index=self.values_bus_ts_df.index, columns=self.edge_param.index)

        # Repeat the edge parameters for all time steps and update the 'closed line' status
        edge_params_dynamic = pd.concat(
            [
                self.edge_param.assign(time_step=time_idx, closed_line=operational_states_df.loc[time_idx])
                for time_idx in operational_states_df.index
            ]
        )
        edge_params_dynamic = self._transform_edge_dataframe(edge_params_dynamic)

        return edge_params_dynamic
    
    def compute_slp_for_bus(self, bus_idx):
        # Apply the appropriate load and generation SLPs
        # Extract the load and sgen profiles from self.slps
        # Get all loads and sgens for this bus
        loads = self.net.load[self.net.load.bus == bus_idx]
        sgens = self.net.sgen[self.net.sgen.bus == bus_idx]

        # Initialize arrays for load and sgen profiles
        slp_load_total = np.zeros(len(self.values_bus_ts_df))
        slp_sgen_total = np.zeros(len(self.values_bus_ts_df))

        # Process loads
        for _, load in loads.iterrows():
            profile_type = load['profile'][:2].lower()
            if profile_type == 'lv':
                profile_type = 'h0'
            if profile_type in self.slps:
                slp_load = self.slps[profile_type]
                if slp_load.sum() != 0:
                    slp_load = slp_load * (self.profiles[('load', 'p_mw')][load.name].sum() / slp_load.sum())
                slp_load_total += slp_load

        # Process sgens
        for _, sgen in sgens.iterrows():
            sgen_type = sgen['type'][:2].lower()
            if sgen_type == 'lv':
                sgen_type = 'pv'
            if sgen_type in self.slps:
                slp_sgen = self.slps[sgen_type]
                if slp_sgen.sum() != 0:
                    slp_sgen = slp_sgen * (self.profiles[('sgen', 'p_mw')][sgen.name].sum() / slp_sgen.sum())
                slp_sgen_total += slp_sgen

        return slp_load_total, slp_sgen_total

    
    def get_SLP_profiles(self, year=2016):
        # Standortdaten für eine beispielhafte PV-Anlage in Deutschland
        latitude, longitude = 52.379189, 9.768954  # Hanover, Germany
        tz = 'utc'

        # Erstellen eines Standortobjekts
        site = pvlib.location.Location(latitude, longitude, tz=tz)

        # Generieren von Wetterdaten für ein Jahr
        times = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:45', freq='15min')
        weather = site.get_clearsky(times)

        # Definieren der PV-Systemparameter
        system = pvlib.pvsystem.PVSystem(
            surface_tilt=30,
            surface_azimuth=180,
            module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
            inverter_parameters={'pdc0': 240},
            racking_model='freestanding',  # Specify racking model
            module_type='glass_polymer'    # Specify module type
        )

        # Berechnen der PV-Leistung
        mc = pvlib.modelchain.ModelChain(system, site, clearsky_model= weather, aoi_model='physical', spectral_model='no_loss', losses_model='pvwatts')

        mc.run_model(weather)

        # Extrahieren des AC-Leistungsprofils
        ac_power = mc.results.ac # in W

        slp = bdew.ElecSlp(year)
        time_id = slp.get_profile(1).index
        slps = slp.all_load_profiles(time_id)
        dyn_ho = slp.create_dynamic_h0_profile()
        slps['h0'] = dyn_ho
        slps['pv'] = ac_power.values

        # Normalize all profiles
        slps = slps.div(slps.sum(axis=0), axis=1)

        self.slps = self._adjust_df_start_time(slps)
    
    def _adjust_df_start_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts a single profile to start at a specific offset and appends the left-out portion to the end.
        
        Parameters:
        - profile_data (pd.DataFrame): The DataFrame containing the profile data.
        
        Returns:
        - pd.DataFrame: The adjusted profile with the specified offset.
        """
        if not isinstance(df, (pd.Series, pd.DataFrame)):
            raise ValueError("The profile_data must be a pandas Series or DataFrame.")

        # Shift the profile
        adjusted_df = pd.concat([df.iloc[self.start_offset:], df.iloc[:self.start_offset]]).reset_index(drop=True)

        adjusted_df = adjusted_df[:self.length]
        
        logger.info(f"Dataframe adjusted to start at offset {self.start_offset} and cut to the length of {self.length}.")
        
        return adjusted_df

        
    @staticmethod
    def _assign_datetime_index(df: pd.DataFrame, datetime_index: pd.DatetimeIndex) -> pd.DataFrame:
        # df = df.iloc[:len(datetime_index)].copy()  # Truncate or pad as necessary
        df.index = datetime_index['time_id']
        return df

    @staticmethod
    def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove any 'Unnamed' columns from a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to remove 'Unnamed' columns from.

        Returns:
        pd.DataFrame: The DataFrame with 'Unnamed' columns removed.
        """
        return df.drop([col for col in df.columns if 'Unnamed:' in col], axis=1)
    
    @staticmethod
    def _sort_bus_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally sort bus-related columns. 
        (You can customize sorting logic if needed.)
        
        Args:
            df (pd.DataFrame): The input DataFrame containing bus-related columns.
        
        Returns:
            pd.DataFrame: The DataFrame with bus-related columns sorted according to the specified logic.
        """
        # Extract the bus columns
        bus_columns = [col for col in df.columns if 'bus' in col]
        
        # Define the desired order of suffixes
        order = {'_vm_pu': 0, '_vm_pu_std': 1, '_va_degree': 2, '_va_degree_std': 3, '_p_mw': 4, '_p_mw_std': 5, '_q_mvar': 6, '_q_mvar_std': 7,'_vn_kv': 8, '_bool_slack': 9, '_bool_zero_inj': 10}
        
        # Sort the bus columns
        sorted_bus_columns = sorted(
            bus_columns,
            key=lambda x: (int(x.split('_')[1]), order.get('_' + '_'.join(x.split('_')[2:]), float('inf')))
        )
        
        # Reorder the DataFrame columns
        sorted_columns = [col for col in df.columns if col not in bus_columns] + sorted_bus_columns
        return df[sorted_columns]
    
    @staticmethod
    def _sort_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
            """
            Optionally sort bus-related columns. 
            (You can customize sorting logic if needed.)
            
            Args:
                df (pd.DataFrame): The input DataFrame containing bus-related columns.
            
            Returns:
                pd.DataFrame: The DataFrame with bus-related columns sorted according to the specified logic.
            """
            # Extract the bus columns
            line_columns = [col for col in df.columns if ('line' in col.split('_')[0])]
            trafo_columns = [col for col in df.columns if ('trafo' in col.split('_')[0])]
            
            # Define the desired order of suffixes
            order_line = {'_p_from_mw': 0, '_p_from_mw_std': 1, '_q_from_mvar': 2, '_q_from_mvar_std': 3, '_G': 4, '_B': 5,'_G_1': 6, '_B_1': 7, '_Gs': 8, '_Bs': 9, '_closed_line': 10, '_phase_shift': 11,  '_imax_or_sn': 12,}
            order_trafo = {'_p_hv_mw': 0, '_p_hv_mw_std': 1, '_q_hv_mvar': 2, '_q_hv_mvar_std': 3, '_G': 4, '_B': 5,'_G_1': 6, '_B_1': 7, '_Gs': 8, '_Bs': 9, '_closed_line': 10, '_phase_shift': 11,  '_imax_or_sn': 12,}
            
            # Sort the bus columns
            sorted_line_columns = sorted(
                line_columns,
                key=lambda x: (int(x.split('_')[1]), order_line.get('_' + '_'.join(x.split('_')[2:]), float('inf')))
            )
            sorted_trafo_columns = sorted(
                trafo_columns,
                key=lambda x: (int(x.split('_')[1]), order_trafo.get('_' + '_'.join(x.split('_')[2:]), float('inf')))
            )
            # Reorder the DataFrame columns
            sorted_columns = sorted_line_columns + sorted_trafo_columns
            return df[sorted_columns]
    
    @staticmethod
    def _apply_error(value, rel_error_bound, abs_error_bound):
        """
        Applies error to a given value.

        Args:
            value (float): The original value.
            rel_error_bound (float): The relative error bound.
            abs_error_bound (float): The absolute error bound.

        Returns:
            tuple: A tuple containing the modified value and the standard deviation.
        """
        std = np.abs(rel_error_bound * value) + abs_error_bound
        error = np.random.normal(loc=0., scale = std)
        value_error = value + error
        return  value_error, std
    
    @staticmethod
    def _transform_node_dataframe(df):
        """
        Transforms a node dataframe into a new dataframe with specific column names.

        Args:
            df (pd.DataFrame): The input node dataframe.

        Returns:
            pd.DataFrame: The transformed dataframe with new column names.
        """
        new_df = pd.DataFrame()
        
        for bus_id in df.index:
            new_df[f'bus_{bus_id}_vn_kv'] = [df.loc[bus_id, 'vn_kv']]
            new_df[f'bus_{bus_id}_bool_slack'] = [df.loc[bus_id, 'bool_slack']]
            new_df[f'bus_{bus_id}_bool_zero_inj'] = [df.loc[bus_id, 'bool_zero_inj']]
        
        return new_df
        
    @staticmethod
    def _transform_edge_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Efficiently transform the edge DataFrame to a time-step indexed format with dynamically
        generated column names for line/transformer parameters.

        Args:
            df (pd.DataFrame): The input DataFrame containing edge data.

        Returns:
            pd.DataFrame: The transformed DataFrame with time-step indexed format.
        """
        # Initialize a dictionary to hold transformed data
        transformed_data = {}

        # Iterate over the rows of the DataFrame
        for _, row in df.iterrows():
            # Determine the element type ('line' or 'trafo')
            element_type = 'line' if row['phase shift'] == 0 else 'trafo'
            time_step = int(row['time_step'])

            # Generate column names dynamically
            row_data = {
                f"{element_type}_{row.name}_G": row['G'],
                f"{element_type}_{row.name}_B": row['B'],
                f"{element_type}_{row.name}_G_1": row['G'],
                f"{element_type}_{row.name}_B_1": row['B'],
                f"{element_type}_{row.name}_Gs": row['Gs'],
                f"{element_type}_{row.name}_Bs": row['Bs'],
                f"{element_type}_{row.name}_closed_line": row['closed_line'],
                f"{element_type}_{row.name}_phase_shift": row['phase shift'],
                f"{element_type}_{row.name}_imax_or_sn": row['imax or sn']
            }

            # Store these in the transformed dictionary by time step
            if time_step not in transformed_data:
                transformed_data[time_step] = {}

            transformed_data[time_step].update(row_data)

        # Convert the dictionary to a DataFrame
        transformed_df = pd.DataFrame.from_dict(transformed_data, orient='index')
        transformed_df.index.name = 'time_step'

        return transformed_df.reset_index()
