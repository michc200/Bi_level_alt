import copy
import numpy as np
import pandapower as pp
from scipy.spatial.distance import euclidean
from robusttest.core.grid_utils.grid_uncertainty import GridUncertainty
import logging

logger = logging.getLogger("PertrubTopology")

def perturb_topology(net, num_lines_to_remove=0, num_lines_to_add=0):
    """
    Steps:
        1. load topology
        2. randomly remove lines (<- control: e.g. how many?)
        3. check connectivity
        4. if yes, return; else revert step 2 and retry. 
    """
    if num_lines_to_remove == 0 and num_lines_to_add == 0:
        return 0, net
    
    max_attempts = 20
    # 1. load topology
    lines_indices = np.array(net.line.index)
    lines_from_bus = net.line['from_bus'].values # from 0, shape (num_lines,)
    lines_to_bus = net.line['to_bus'].values # shape (num_lines,)
    line_numbers = np.arange(start=0, stop=len(lines_from_bus))
    bus_numbers = net.bus[~net.bus.index.isin(net.trafo.hv_bus.values)].index # shape (num_buses,)
    
    rng = np.random.default_rng()
    # 2. remove lines
    net_perturbed = copy.deepcopy(net)
    to_be_removed = rng.choice(line_numbers, size=num_lines_to_remove, replace=False)
    pp.drop_lines(net_perturbed, lines_indices[to_be_removed])

    logger.info(f"Removed lines {to_be_removed}.")
    
    # 3. add lines
    for _ in range(num_lines_to_add):
        from_bus, to_bus = rng.choice(bus_numbers, size=2, replace=False)
        from_coords = net_perturbed.bus_geodata.loc[from_bus, ['x', 'y']]
        to_coords = net_perturbed.bus_geodata.loc[to_bus, ['x', 'y']]
        length_km = euclidean(from_coords, to_coords) * abs(np.random.normal(40,20))
        copied_line = net.line.iloc[rng.choice(line_numbers, size=1, replace=False)]
        new_line = pp.create_line_from_parameters(
            net_perturbed, 
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

        pp.create_switch(net_perturbed, bus=from_bus, element=new_line, et='l', closed=True)
        pp.create_switch(net_perturbed, bus=to_bus, element=new_line, et='l', closed=True)

        logger.info(f"Added line and switches between buses {from_bus} and {to_bus}.")
        
    net_uncertain = GridUncertainty(net_perturbed)
    net_uncertain.recover_unsupplied_buses(mode='random')
    net_perturbed = net_uncertain.net
        
    return 0, net_perturbed