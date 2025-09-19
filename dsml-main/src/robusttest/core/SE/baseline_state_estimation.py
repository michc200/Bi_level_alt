import copy
import logging
import pandas as pd
import pandapower as pp
import multiprocessing as mp
from robusttest.core.grid_time_series import GridTimeSeries
import pickle 

logger = logging.getLogger("BaselineStateEstimation")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')


def update_measurements_for_net(net, measurements_bus, measurements_line, measurement_mapping):
    """
    Update the measurements for the network copy at a given time step.

    Args:
        net: A copy of the Pandapower network.
        t: The current time step index.
    """

    # Cache columns for buses and lines
    bus_cols = measurements_bus.index
    line_cols = measurements_line.index

    for element, mapping in measurement_mapping.items():
        if element == "bus":
            df = measurements_bus
            side = None
            cols = bus_cols
        else:
            df = measurements_line
            side = 'from' if element == 'line' else 'hv'
            cols = line_cols

        # Process measurements for each type
        for param, meas_type in mapping.items():
            # Filter relevant columns for the current measurement type
            relevant_cols = [col for col in cols if param in col and "_std" not in col]
            std_cols = [f"{col}_std" for col in relevant_cols]

            # Add measurements to the network
            for col, std_col in zip(relevant_cols, std_cols):
                idx = int(col.split('_')[1])  # Extract element index from the column name
                meas_value = df.loc[col]
                std_dev = df.loc[std_col]
                if meas_value == 0:
                    continue
                # Create the measurement in Pandapower
                pp.create_measurement(
                    net, meas_type, element, value=meas_value,
                    std_dev=std_dev, element=idx, side=side
                )

    return net


def collect_results_from_net(net, t):
    """
    Collect the state estimation results from a network copy for the current time step.

    Args:
        net (pandapower.networks.Network): A copy of the Pandapower network.
        t (int): The current time step index.

    Returns:
        dict: A dictionary of state estimation results for buses at the given time step.
            The dictionary contains the following keys:
            - "time" (int): The current time step index.
            - "bus_{index}_vm_pu" (float): The voltage magnitude at bus {index} in per unit.
            - "bus_{index}_va_degree" (float): The voltage angle at bus {index} in degrees.
            - "error" (str, optional): If there is a missing estimation result, this key will be present with an error message.

    Raises:
        None

    Examples:
        >>> net = pandapower.networks.create_cigre_network_mv()
        >>> t = 0
        >>> result = GridTimeseries.collect_results_from_net(net, t)
        >>> print(result)
        {'time': 0, 'bus_0_vm_pu': 1.0, 'bus_0_va_degree': 0.0, 'bus_1_vm_pu': 1.0, 'bus_1_va_degree': 0.0, ...}
    """
    try:
        # Extract bus results (voltage magnitude and angle)
        bus_results = net.res_bus_est[['vm_pu', 'va_degree']]

        # Flatten results into a dictionary
        result_dict = {"time": t}
        for index, row in bus_results.iterrows():
            for col, value in row.items():
                result_dict[f"bus_{index}_{col}"] = value

        return result_dict
    except KeyError as e:
        # Handle missing estimation results gracefully
        return {"time": t, "error": f"Missing result: {e}"}
    

# Module-level global initializers
def init_globals(shared_data):
    """
    Initialize globals for multiprocessing. Shared objects are passed and set as global variables.
    """
    global shared_net_dict, shared_mapping, shared_measurements_bus_ts_df, shared_measurements_line_ts_df, shared_res_switch_closed_df
    shared_net_dict = shared_data["net_dict"]
    shared_mapping = shared_data["measurement_mapping"]
    shared_measurements_bus_ts_df = shared_data["measurements_bus_ts_df"]
    shared_measurements_line_ts_df = shared_data["measurements_line_ts_df"]
    shared_res_switch_closed_df = shared_data["res_switch_closed_df"]

def process_time_step(t):
    """
    Worker function to process a single time step.
    """
    try:
        measurements_bus_t = shared_measurements_bus_ts_df.iloc[t]
        measurements_line_t = shared_measurements_line_ts_df.iloc[t]
        res_switch_closed_t = shared_res_switch_closed_df.iloc[t]

        net = pp.from_json_dict(shared_net_dict)

        # Update measurements and switch states
        net = update_measurements_for_net(net, measurements_bus_t, measurements_line_t, shared_mapping)
        net.switch["closed"] = res_switch_closed_t.values == 1

        # Perform state estimation
        pp.estimation.estimate(net, init="results", maximum_iterations=100, tolerance=1e-4)

        # Collect results
        return collect_results_from_net(net, t)
    except Exception as e:
        return {"time": t, "error": str(e)}

class BaselineStateEstimation:
    """
    Class for performing baseline state estimation.

    Args:
        net: A copy of the Pandapower network.
        grid_time_series: GridTimeSeries object containing time series measurements and switch status.
    """

    def __init__(self, grid_time_series : GridTimeSeries):
        self.grid_time_series = grid_time_series
        self.measurement_mapping = {
            "bus": {"vm_pu": "v", "va_degree": "va", "p_mw": "p", "q_mvar": "q"},
            "line": {"p_from_mw": "p", "q_from_mvar": "q"},
            "trafo": {"p_hv_mw": "p", "q_hv_mvar": "q"},
        }

    
    def run_parallel_state_estimation(self, n_jobs=-1) -> pd.DataFrame:
        """
        Perform state estimation in parallel for a series of time steps with periodic progress logging.

        Returns:
            pd.DataFrame: DataFrame containing state estimation results for all time steps.
        """
        logger.info("Starting parallel state estimation.")
        pp.runpp(self.grid_time_series.net)

        # Create a list of time step indices
        time_steps = self.grid_time_series.measurements_bus_ts_df.index.tolist()

        # Use a Manager for shared global state
        with mp.Manager() as manager:
            shared_data = manager.dict({
                "net_dict": self.grid_time_series.net.copy(),
                "measurement_mapping": self.measurement_mapping,
                "measurements_bus_ts_df": self.grid_time_series.measurements_bus_ts_df,
                "measurements_line_ts_df": self.grid_time_series.measurements_line_ts_df,
                "res_switch_closed_df": self.grid_time_series.res_switch_closed_df
            })

            results = manager.list()  # Shared results list

            # Initialize multiprocessing pool
            with mp.Pool(processes=n_jobs, initializer=init_globals, initargs=(shared_data,)) as pool:
                async_results = []
                for t in time_steps:
                    async_results.append(pool.apply_async(process_time_step, args=(t,)))

                # Monitor progress
                for i, async_result in enumerate(async_results):
                    try:
                        result = async_result.get()  # Retrieve the result
                        results.append(result)

                        # Log completion every 100 time steps
                        if (i + 1) % 100 == 0 or i == len(async_results) - 1:
                            logger.info(f"Completed {i + 1}/{len(async_results)} time steps.")
                        if len(result.keys()) < 10:
                            logger.error(f"Error retrieving result for time step {result}")
                    except Exception as e:
                        logger.error(f"Error retrieving result for time step {i}: {str(e)}")

            # Combine results into a DataFrame
            results_df = pd.DataFrame(list(results))


            # Cleanup shared data explicitly
            shared_data.clear()  # Clear shared data dictionary

        logger.info("State estimation completed in parallel.")
        self.baseline_se_results_df = results_df

        return results_df
    
    def save(self, path: str):
        pickle.dump(self, open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))