import pandas as pd
import numpy as np
import logging
import simbench as sb
import pandapower as pp
from pandapower.topology import unsupplied_buses
from robusttest.core.grid_time_series import GridTimeSeries 
from robusttest.core.grid_utils.grid_uncertainty import GridUncertainty
import random
from sklearn.model_selection import train_test_split
from typing import Tuple, List

# Configure logging
logger = logging.getLogger("SwitchingProfiles")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')

class SwitchingProfiles:
    def __init__(self, grid_ts: GridTimeSeries):
        self.net = grid_ts.net
        self.profiles = grid_ts.profiles
        self.random_seed = grid_ts.random_seed

    def find_redundend_lines(self, bus: int) -> List[int]:
        """
        Find redundant lines connected to a bus.
        """
        switch_state = self.net.switch['closed']
        self.net.switch['closed'] = True

        redundant_lines = []
        for line in self.net.line[(self.net.line['from_bus'] == bus) | (self.net.line['to_bus'] == bus)].index:
            self.net.switch.loc[self.net.switch['element'] == line, 'closed'] = False
            unsupplied = unsupplied_buses(self.net)
            if len(unsupplied) == 0:
                redundant_lines.append(line)
            self.net.switch.loc[self.net.switch['element'] == line, 'closed'] = True

        # self.net.switch['closed'] = switch_state

        logger.info(f"Found redundant lines for bus {bus}: {redundant_lines}")

        return redundant_lines
    

    def generate_random_switching_profile(self, line_pairs: List[Tuple[int, int]], switch_states=None) -> None:
        """
        Create a DataFrame with switch states over time for the given line pairs.

        Args:
            line_pairs: List of tuples representing line pairs to toggle.
        """
        time_steps = len(self.profiles[('load', 'p_mw')].index)

        # Map non-continuous switch indices to continuous indices
        switch_indices = self.net.switch.index.tolist()
        switch_idx_map = {original_idx: continuous_idx for continuous_idx, original_idx in enumerate(switch_indices)}
        np.random.seed(self.random_seed)

        # Initialize switch states to True (closed)
        if switch_states is None:
            switch_states = np.full((time_steps, len(switch_indices)), True)

        try:
            for line1, line2 in line_pairs:
                # Retrieve and map the indices of switches associated with the line pairs
                switches = [
                    switch_idx_map[idx] 
                    for idx in self.get_line_switch_indices(line1) + self.get_line_switch_indices(line2)
                ]

                # Generate random time indices for state changes
                n_changes = random.randint(1, 10)
                time_indices = random.choices(range(time_steps), k=n_changes)

                # Sort the time indices in ascending order
                time_indices.sort()

                # Set the switch states for the selected time indices
                for i in range(len(time_indices)):
                    time_idx = time_indices[i]
                    next_time_idx = time_indices[i + 1] if i + 1 < len(time_indices) else None  # Handle the last index safely

                    if i == 0:
                        # For the first time index, set switches[:2] to False and switches[2:] to True
                        for idx in switches[:2]:
                            switch_states[:time_idx, idx] = False
                        for idx in switches[2:]:
                            switch_states[:time_idx, idx] = True

                    # Toggle switches[:2] between time_idx and next_time_idx
                    for idx in switches[:2]:
                        if next_time_idx:
                            switch_states[time_idx:next_time_idx, idx] = not switch_states[time_idx - 1, idx]
                        else:
                            switch_states[time_idx:, idx] = not switch_states[time_idx - 1, idx]

                    # Optionally toggle switches[2:] between time_idx and next_time_idx
                    for idx in switches[2:]:
                        if next_time_idx:
                            switch_states[time_idx:next_time_idx, idx] = not switch_states[time_idx - 1, idx]
                        else:
                            switch_states[time_idx:, idx] = not switch_states[time_idx - 1, idx]

            switch_profile = pd.DataFrame(
                switch_states,
                index=self.profiles[('load', 'p_mw')].index,
                columns=switch_indices,
            )

            # Add the switch profile to the profiles dictionary
            self.profiles[('switch', 'closed')] = switch_profile
        except Exception as e:
            logger.error(f"Error creating switch profile: {e}")

    def get_line_switch_indices(self, line: int) -> List[int]:
            """Retrieve switch indices associated with a line.

            Args:
                line (int): The line number.

            Returns:
                list[int]: A list of switch indices associated with the line.
            """
            return list(self.net.switch[self.net.switch['element'] == line].index)
