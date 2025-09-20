from torch_geometric.utils import scatter, get_laplacian
import torch
import os

# Set CUDA device
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pflow(y, edge_index, node_param, edge_param, phase_shift=True):
    """
    
    Power flow equations to compute other estimated variables from the outputs of the model

    """
    V_n = node_param[:,0]
    V_hv = V_n.max()
    V_lv = V_n.min()  # Considering only two V level
    
    ratio = torch.tensor(V_hv/V_lv)

    # Store output values separately
    V = y[:,0] # in pu
    Theta = y[:,1] # in rad

    indices_from = edge_index[0]
    indices_to = edge_index[1]

    # Extact edge characteristics from A matrix
    Y1_ij = edge_param[:,0]
    Y2_ij = edge_param[:,1]

    Ys1_ij = edge_param[:,2]
    Ys2_ij = edge_param[:,3]

    # Gather V and theta on both sides of each edge
    V_i = torch.gather(V,0, indices_from)  # torch.float32, [n_samples*n_edges, 1], in p.u.
    Th_i = torch.gather(Theta,0, indices_from)  # torch.float32, [n_samples*n_edges, 1], in p.u.
    V_j = torch.gather(V,0, indices_to)  # torch.float32, [n_samples*n_edges, 1], in rad
    Th_j = torch.gather(Theta,0, indices_to)  # torch.float32, [n_samples*n_edges, 1], in ra

    # Compute h(U) = V_i, theta_i, P_i, Q_i, P_ij, Q_ij, I_ij
    
    if phase_shift:
        shift = 0
    else:  
        shift = torch.tensor(edge_param[:,5])
        
    trafo_pos = (torch.tensor(edge_param[:, 5]) != 0).int()
    imax_or_sn = torch.tensor(edge_param[:,6])

    P_ij_from = (- V_i * V_j * (Y1_ij * torch.cos(Th_i - Th_j - shift) + Y2_ij * torch.sin(Th_i - Th_j - shift)) + (Y1_ij + Ys1_ij / 2) * V_i ** 2) * V_lv**2  

    Q_ij_from = (V_i * V_j * (- Y1_ij * torch.sin(Th_i - Th_j - shift) + Y2_ij * torch.cos(Th_i - Th_j - shift)) - (Y2_ij + Ys2_ij / 2) * V_i ** 2) * V_lv**2  

    P_ij_to = (- V_i * V_j * ( Y1_ij * torch.cos(Th_i - Th_j - shift) - Y2_ij * torch.sin(Th_i - Th_j - shift)) + (Y1_ij + Ys1_ij / 2) * V_j ** 2) * V_lv**2  

    Q_ij_to = (V_i * V_j * (Y1_ij * torch.sin(Th_i - Th_j - shift) + Y2_ij * torch.cos(Th_i - Th_j - shift)) - (Y2_ij + Ys2_ij / 2) * V_j ** 2) * V_lv**2 

    I_ij_from = (torch.complex(P_ij_from, -Q_ij_from).abs() / (V_i * V_lv * torch.sqrt(torch.tensor(3))))

    I_ij_from = I_ij_from/(1.- (trafo_pos*(1.-ratio)))


    I_ij_to = torch.complex(P_ij_to, -Q_ij_to).abs() / (V_j * V_lv * torch.sqrt(torch.tensor(3)))

    # Calculating line and trafo loading

    loading_lines = ((1.- trafo_pos) * torch.maximum(I_ij_from, I_ij_to)) / imax_or_sn
    loading_trafo = (trafo_pos * torch.maximum(I_ij_from * V_hv, I_ij_to * V_lv))/ imax_or_sn

    return loading_lines, loading_trafo, P_ij_from, Q_ij_from, P_ij_to, Q_ij_to, I_ij_from, I_ij_to  # loading in %, P in MW, Q in MVAr, I in kA

def gsp_wls_edge(input, edge_input, output, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, num_samples, node_param, edge_param):
    
    total_nodes =input.shape[0]
    
    z = input[:,::2] # V, theta, P, Q (buses) [batch*num_nodes ,4]
    edge_z = edge_input[:,:4:2] # Pflow, Qflow (lines) [batch*num_lines, 2]
    z_mask = z != 0.
    edge_z_mask = edge_z != 0.
    edge_Z = (edge_z*edge_std[:4:2] + edge_mean[:4:2]) * edge_z_mask
    Z = (z*x_std[::2] + x_mean[::2]) * z_mask
    r_inv = input[:,1::2]  # Cov(V)^-1, Cov(theta)^-1, Cov(P)^-1, Cov(Q)^-1 (buses) [batch*num_nodes,4]
    r_mask = r_inv!=0.
    r_edge_inv = edge_input[:,1:4:2]
    r_edge_mask = r_edge_inv !=0.
    
    R_inv = (r_inv*x_std[1::2] + x_mean[1::2]) * r_mask
    R_edge_inv = (r_edge_inv*edge_std[1:4:2] + edge_mean[1:4:2]) * r_edge_mask
    
    v_i = output[:,0:1]*x_std[:1]  + x_mean[:1] #  # [batch*num_nodes, 1]  
    theta_i = output[:,1:]*x_std[2:3] + x_mean[2:3] #  # [batch*num_nodes, 1]  
    theta_i *= (1.- node_param[:,1:2]) # Enforce theta_slack = 0.
    
    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param, phase_shift = False)
    
    loading = loading_lines + loading_trafos
    
    indices_from = edge_index[0] # [batch*num_edges,1]
    indices_to = edge_index[1] # [batch*num_edges,1]
    
    L = get_laplacian(edge_index=edge_index) # [batch*num_nodes, batch*num_nodes]
    Ld = torch.sparse_coo_tensor(L[0],L[1]).to_dense()
    
    
    #Summing flow to balance in buses, negative signs in sum to follow conventions from PandaPower

    p_i = -scatter(p_to,indices_to,dim_size=total_nodes) - scatter(p_from, indices_from, dim_size=total_nodes) # [batch*num_nodes, 1]
    q_i = -scatter(q_to,indices_to, dim_size=total_nodes) - scatter(q_from, indices_from, dim_size=total_nodes) # [batch*num_nodes, 1]
    
    theta_ij = torch.abs(torch.gather(theta_i[:,0],0, indices_from) - torch.gather(theta_i[:,0],0,indices_to))

    h = torch.concatenate([v_i, theta_i, torch.unsqueeze(p_i,1), torch.unsqueeze(q_i,1)], dim = 1) # [batch*num_nodes, 4]
    
    h_edge = torch.concatenate([ torch.unsqueeze(p_from,1),  torch.unsqueeze(q_from,1)], dim =1)

    delta = Z - h  # [batch*num_nodes, 4]
    
    delta_edge = edge_Z - h_edge
    
    meas_node_weights = torch.tensor([reg_coefs['lam_v'], reg_coefs['lam_v'], reg_coefs['lam_p'], reg_coefs['lam_p']])
    meas_edge_weights = torch.tensor([reg_coefs['lam_pf'], reg_coefs['lam_pf']])
    
    meas_node_weights, meas_edge_weights = (meas_node_weights.to(device), meas_edge_weights.to(device))

    J_sample = torch.sum(torch.mul(delta**2 * R_inv, meas_node_weights), axis=1)
    J_sample_edge = torch.sum(torch.mul(delta_edge**2 * R_edge_inv, meas_edge_weights), axis=1)
    
    J = torch.mean(J_sample) + torch.mean(J_sample_edge[J_sample_edge != 0]) # [1,1]
    trafo_pos = (torch.tensor(edge_param[:, 5]) == 0).int()

    J_v = reg_coefs['lam_reg']*torch.mean(torch.relu(v_i - 1.1) + torch.relu(0.9 - v_i))**2
    J_theta = reg_coefs['lam_reg'] * torch.mean(torch.relu(torch.mul(theta_ij,trafo_pos) - 0.5))**2
    J_loading = reg_coefs['lam_reg'] *torch.mean(torch.relu(loading - 1.5))**2
    
    J_reg = J +  J_v +  J_theta +  J_loading # [1,1]
    
    return J_reg

def compute_wls_loss(input, edge_input, output, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, node_param, edge_param):
    total_nodes = input.shape[0]

    # Extract normalized states
    z = input[:, ::2]  # V, theta, P, Q (buses)
    edge_z = edge_input[:, :4:2]  # Pflow, Qflow (lines)
    z_mask = z != 0.
    edge_z_mask = edge_z != 0.

    # Denormalize
    Z = (z * x_std[::2] + x_mean[::2]) * z_mask
    edge_Z = (edge_z * edge_std[:4:2] + edge_mean[:4:2]) * edge_z_mask

    # Inverse variances
    r_inv = input[:, 1::2]
    r_edge_inv = edge_input[:, 1:4:2]
    r_mask = r_inv != 0.
    r_edge_mask = r_edge_inv != 0.
    R_inv = (r_inv * x_std[1::2] + x_mean[1::2]) * r_mask
    R_edge_inv = (r_edge_inv * edge_std[1:4:2] + edge_mean[1:4:2]) * r_edge_mask

    # Predicted states
    v_i = output[:, 0:1] * x_std[:1] + x_mean[:1]
    theta_i = output[:, 1:] * x_std[2:3] + x_mean[2:3]
    theta_i *= (1. - node_param[:, 1:2])  # enforce slack bus

    # Flows
    _, _, p_from, q_from, p_to, q_to, _, _ = get_pflow(
        torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param, phase_shift=False
    )

    indices_from, indices_to = edge_index
    p_i = -scatter(p_to, indices_to, dim_size=total_nodes) - scatter(p_from, indices_from, dim_size=total_nodes)
    q_i = -scatter(q_to, indices_to, dim_size=total_nodes) - scatter(q_from, indices_from, dim_size=total_nodes)

    # Build measurement residuals
    h = torch.concatenate([v_i, theta_i, p_i.unsqueeze(1), q_i.unsqueeze(1)], dim=1)
    h_edge = torch.concatenate([p_from.unsqueeze(1), q_from.unsqueeze(1)], dim=1)
    delta = Z - h
    delta_edge = edge_Z - h_edge

    # Weighted Least Squares loss
    meas_node_weights = torch.tensor([reg_coefs['lam_v'], reg_coefs['lam_v'], reg_coefs['lam_p'], reg_coefs['lam_p']], device=device)
    meas_edge_weights = torch.tensor([reg_coefs['lam_pf'], reg_coefs['lam_pf']], device=device)

    J_sample = torch.sum(delta**2 * R_inv * meas_node_weights, axis=1)
    J_sample_edge = torch.sum(delta_edge**2 * R_edge_inv * meas_edge_weights, axis=1)

    J_wls = torch.mean(J_sample) + torch.mean(J_sample_edge[J_sample_edge != 0])
    return J_wls, v_i, theta_i

def compute_physical_loss(v_i, theta_i, edge_index, edge_param, node_param, reg_coefs):
    # Compute flows again for physical constraints
    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(
        torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param, phase_shift=False
    )
    loading = loading_lines + loading_trafos

    indices_from, indices_to = edge_index
    theta_ij = torch.abs(torch.gather(theta_i[:, 0], 0, indices_from) -
                         torch.gather(theta_i[:, 0], 0, indices_to))

    trafo_pos = (torch.tensor(edge_param[:, 5]) == 0).int()

    # Regularization terms
    J_v = reg_coefs['lam_reg'] * torch.mean(torch.relu(v_i - 1.1) + torch.relu(0.9 - v_i))**2
    J_theta = reg_coefs['lam_reg'] * torch.mean(torch.relu(theta_ij * trafo_pos - 0.5))**2
    J_loading = reg_coefs['lam_reg'] * torch.mean(torch.relu(loading - 1.5))**2

    J_phys = J_v + J_theta + J_loading
    return J_phys

def compute_v_theta(output, x_mean, x_std, node_param):
    """
    Compute predicted voltage magnitudes (v_i) and angles (theta_i).

    Args:
        output (torch.Tensor): Model output of shape [num_nodes, 2] (or more).
        x_mean (torch.Tensor): Mean used for denormalization (size at least 3).
        x_std (torch.Tensor): Std used for denormalization (size at least 3).
        node_param (torch.Tensor): Node parameters (for slack bus enforcement).

    Returns:
        v_i (torch.Tensor): Denormalized voltage magnitudes [num_nodes, 1].
        theta_i (torch.Tensor): Denormalized voltage angles [num_nodes, 1].
    """
    # Voltage magnitudes
    v_i = output[:, 0:1] * x_std[:1] + x_mean[:1]

    # Voltage angles
    theta_i = output[:, 1:2] * x_std[2:3] + x_mean[2:3]

    # Enforce slack bus (assuming node_param[:,1] indicates slack bus)
    theta_i *= (1.0 - node_param[:, 1:2])

    return v_i, theta_i


if __name__ == "__main__":
    pass
    print("Test")