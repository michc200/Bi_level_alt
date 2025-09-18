import logging
from scipy.spatial.distance import euclidean
from pandapower.topology import unsupplied_buses, create_nxgraph, connected_component
import pandapower as pp
import numpy as np
import random


logger = logging.getLogger("GridUncertainty")


class GridUncertainty:
    def __init__(self, net):
        self.net = net

    def remove_lines_between_buses(self, bus_ids):
        try:
            # Find lines where both buses are in the bus_ids list
            lines_to_remove = self.net.line[
                self.net.line.apply(
                    lambda row: row['from_bus'] in bus_ids and row['to_bus'] in bus_ids, axis=1
                )
            ]

            # Log lines to be removed
            if not lines_to_remove.empty:
                logger.info(f"Removing lines: {lines_to_remove.index.tolist()}")
            else:
                logger.info("No lines to remove for the specified buses.")

            # Drop the lines from the network
            self.net.line.drop(index=lines_to_remove.index, inplace=True)

            # Remove associated switches
            switches_to_remove = self.net.switch[
                self.net.switch['element'].isin(lines_to_remove.index)
            ]
            self.net.switch.drop(index=switches_to_remove.index, inplace=True)

            logger.info(f"Removed {len(lines_to_remove)} lines and associated switches.")
        except Exception as e:
            logger.error(f"Error removing lines between buses: {e}")

    def recover_unsupplied_buses(self, mode='location'):
        try:
            # Find unsupplied buses
            unsupplied = set(unsupplied_buses(self.net))
            if not unsupplied:
                logger.info("No unsupplied buses found.")
                return

            # Find all initially supplied buses, excluding high-voltage transformer buses
            hv_buses = set(self.net.trafo['hv_bus'])
            supplied = (set(self.net.bus.index) - unsupplied) - hv_buses
            logger.info(f"Initial unsupplied buses: {unsupplied}")
            logger.info(f"Initial supplied buses (excluding HV transformer buses): {supplied}")

            # Recover unsupplied buses
            while unsupplied:
                for bus in list(unsupplied):  # Iterate over a snapshot since we'll modify the set
                    if mode == 'location':
                        # Connect to the geographically closest supplied bus
                        unsupplied_coords = self.net.bus_geodata.loc[bus, ['x', 'y']]
                        closest_bus = None
                        min_distance = float('inf')

                        # Find the closest supplied bus
                        for supplied_bus in supplied:
                            supplied_coords = self.net.bus_geodata.loc[supplied_bus, ['x', 'y']]
                            distance = euclidean(unsupplied_coords, supplied_coords)
                            if distance < min_distance:
                                min_distance = distance
                                closest_bus = supplied_bus

                        if closest_bus is not None:
                            # Add a line between the unsupplied bus and the closest supplied bus
                            logger.info(f"Connecting unsupplied bus {bus} to supplied bus {closest_bus}.")
                            self.add_line_between_buses(bus, closest_bus)

                            # Update the sets
                            supplied.add(bus)  # Newly connected bus becomes supplied
                            unsupplied.remove(bus)  # Remove it from the unsupplied set
                        else:
                            logger.warning(f"Could not find a supplied bus to connect for bus {bus}.")
                            break  # Exit if no connection is possible

                    elif mode == 'random':
                        # Connect to a random supplied bus
                        random_bus = random.choice(list(supplied))
                        logger.info(f"Connecting unsupplied bus {bus} to random supplied bus {random_bus}.")
                        self.add_line_between_buses(bus, random_bus)

                        # Update the sets
                        supplied.add(bus)  # Newly connected bus becomes supplied
                        unsupplied.remove(bus)  # Remove it from the unsupplied set

                    else:
                        logger.warning(f"Invalid mode: {mode}. Please choose 'location' or 'random'.")
                        break  # Exit if mode is invalid

            # Final connectivity check
            logger.info("Performing final connectivity check.")
            slack_buses = set(self.net.ext_grid['bus'])  # Buses connected to the external grid (slack buses)
            connected_buses = set()
            netx = create_nxgraph(self.net)

            for slack_bus in slack_buses:
                # Collect all buses in the connected component of each slack bus
                connected_buses.update(set(connected_component(netx, slack_bus)))

            all_buses = set(self.net.bus.index)
            if all_buses == connected_buses:
                logger.info("All buses are successfully connected to the external grid.")
            else:
                unconnected_buses = all_buses - connected_buses
                logger.warning(f"Some buses are still unconnected to the external grid: {unconnected_buses}")

        except Exception as e:
            logger.error(f"Error recovering unsupplied buses: {e}")
    

    
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

        
    def add_line_parameter_random_errors(self, line_ids, errors = {
                "length_km": 0.25,  # 25% error for line length
                "r_ohm_per_km": 0.02,  # 2% error for line resistance
                "x_ohm_per_km": 0.02  # 2% error for line reactance
            }):
        try:
            # Define errors for different parameters
            

            for line_id in line_ids:
                for param_name, max_error in errors.items():
                    if param_name in self.net.line.columns:
                        std = np.abs(max_error * self.net.line.at[line_id, param_name])
                        error = np.random.normal(loc=0., scale = std)
                        self.net.line.at[line_id, param_name] += error
                    else:
                        logger.warning(f"Unsupported line parameter: {param_name}")

            logger.info(f"Applied random percentage errors to line parameters for lines: {line_ids}")

        except Exception as e:
            logger.error(f"Error applying random percentage errors to line parameters: {e}")