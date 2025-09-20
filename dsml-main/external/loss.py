#!/usr/bin/env python3
"""
Loss Functions for DSML State Estimation

This module contains the loss functions used for state estimation in the DSML pipeline.
Includes both WLS (Weighted Least Squares) loss and physical constraint losses.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch_geometric.utils import scatter, get_laplacian
from src.robusttest.core.SE.pf_funcs import get_pflow, gsp_wls_edge

# Force CPU usage
device = torch.device('cpu')

def wls_loss(output, input, edge_input, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, node_param, edge_param):
    """
    Compute Weighted Least Squares (WLS) loss for state estimation.

    PyTorch-style loss function that extracts the WLS component from the original gsp_wls_edge function.

    Args:
        output: Model output (voltage magnitude and angle) - PyTorch convention: predictions first
        input: Input node features
        edge_input: Input edge features
        x_mean, x_std: Node normalization parameters
        edge_mean, edge_std: Edge normalization parameters
        edge_index: Edge connectivity
        reg_coefs: Regularization coefficients
        node_param: Node parameters
        edge_param: Edge parameters

    Returns:
        torch.Tensor: WLS loss scalar
    """
    total_nodes = input.shape[0]

    z = input[:, ::2]  # V, theta, P, Q (buses)
    edge_z = edge_input[:, :4:2]  # Pflow, Qflow (lines)
    z_mask = z != 0.
    edge_z_mask = edge_z != 0.
    edge_Z = (edge_z * edge_std[:4:2] + edge_mean[:4:2]) * edge_z_mask
    Z = (z * x_std[::2] + x_mean[::2]) * z_mask
    r_inv = input[:, 1::2]  # Cov(V)^-1, Cov(theta)^-1, Cov(P)^-1, Cov(Q)^-1 (buses)
    r_mask = r_inv != 0.
    r_edge_inv = edge_input[:, 1:4:2]
    r_edge_mask = r_edge_inv != 0.

    R_inv = (r_inv * x_std[1::2] + x_mean[1::2]) * r_mask
    R_edge_inv = (r_edge_inv * edge_std[1:4:2] + edge_mean[1:4:2]) * r_edge_mask

    v_i = output[:, 0:1] * x_std[:1] + x_mean[:1]  # [batch*num_nodes, 1]
    theta_i = output[:, 1:] * x_std[2:3] + x_mean[2:3]  # [batch*num_nodes, 1]
    theta_i *= (1. - node_param[:, 1:2])  # Enforce theta_slack = 0.

    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(
        torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param, phase_shift=False
    )

    indices_from = edge_index[0]
    indices_to = edge_index[1]

    # Summing flow to balance in buses, negative signs in sum to follow conventions from PandaPower
    p_i = -scatter(p_to, indices_to, dim_size=total_nodes) - scatter(p_from, indices_from, dim_size=total_nodes)
    q_i = -scatter(q_to, indices_to, dim_size=total_nodes) - scatter(q_from, indices_from, dim_size=total_nodes)

    h = torch.concatenate([v_i, theta_i, torch.unsqueeze(p_i, 1), torch.unsqueeze(q_i, 1)], dim=1)
    h_edge = torch.concatenate([torch.unsqueeze(p_from, 1), torch.unsqueeze(q_from, 1)], dim=1)

    delta = Z - h
    delta_edge = edge_Z - h_edge

    meas_node_weights = torch.tensor([reg_coefs['lam_v'], reg_coefs['lam_v'], reg_coefs['lam_p'], reg_coefs['lam_p']])
    meas_edge_weights = torch.tensor([reg_coefs['lam_pf'], reg_coefs['lam_pf']])

    meas_node_weights, meas_edge_weights = (meas_node_weights.to(device), meas_edge_weights.to(device))

    J_sample = torch.sum(torch.mul(delta**2 * R_inv, meas_node_weights), axis=1)
    J_sample_edge = torch.sum(torch.mul(delta_edge**2 * R_edge_inv, meas_edge_weights), axis=1)

    J_wls = torch.mean(J_sample) + torch.mean(J_sample_edge[J_sample_edge != 0])

    return J_wls


def physical_loss(output, x_mean, x_std, edge_index, edge_param, node_param, reg_coefs):
    """
    Compute physical constraint losses for state estimation.

    PyTorch-style loss function that extracts the physical constraint components from the original gsp_wls_edge function.

    Args:
        output: Model output (voltage magnitude and angle) - PyTorch convention: predictions first
        x_mean, x_std: Node normalization parameters
        edge_index: Edge connectivity
        edge_param: Edge parameters
        node_param: Node parameters
        reg_coefs: Regularization coefficients

    Returns:
        torch.Tensor: Physical constraint loss scalar
    """
    # Denormalize predictions
    v_i = output[:, 0:1] * x_std[:1] + x_mean[:1]
    theta_i = output[:, 1:] * x_std[2:3] + x_mean[2:3]
    theta_i *= (1. - node_param[:, 1:2])  # Enforce theta_slack = 0.
    loading_lines, loading_trafos, p_from, q_from, p_to, q_to, i_from, i_to = get_pflow(
        torch.concat([v_i, theta_i], axis=1), edge_index, node_param, edge_param, phase_shift=False
    )

    loading = loading_lines + loading_trafos

    indices_from = edge_index[0]
    indices_to = edge_index[1]
    theta_ij = torch.abs(torch.gather(theta_i[:, 0], 0, indices_from) - torch.gather(theta_i[:, 0], 0, indices_to))

    trafo_pos = (torch.tensor(edge_param[:, 5]) == 0).int()

    J_v = reg_coefs['lam_reg'] * torch.mean(torch.relu(v_i - 1.1) + torch.relu(0.9 - v_i))**2
    J_theta = reg_coefs['lam_reg'] * torch.mean(torch.relu(torch.mul(theta_ij, trafo_pos) - 0.5))**2
    J_loading = reg_coefs['lam_reg'] * torch.mean(torch.relu(loading - 1.5))**2

    J_phys = J_v + J_theta + J_loading
    return J_phys


def wls_and_physical_loss(output, input, edge_input, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, node_param, edge_param, lambda_wls=1.0, lambda_physical=1.0):
    """
    Combined WLS and physical constraint loss for state estimation.

    PyTorch-style loss function that combines both WLS and physical constraint losses with configurable weights.

    Args:
        output: Model output (voltage magnitude and angle) - PyTorch convention: predictions first
        input: Input node features
        edge_input: Input edge features
        x_mean, x_std: Node normalization parameters
        edge_mean, edge_std: Edge normalization parameters
        edge_index: Edge connectivity
        reg_coefs: Regularization coefficients
        node_param: Node parameters
        edge_param: Edge parameters
        lambda_wls: Weight for WLS loss component (default: 1.0)
        lambda_physical: Weight for physical constraint loss component (default: 1.0)

    Returns:
        torch.Tensor: Combined loss scalar
    """
    # Compute WLS loss
    wls_loss_val = wls_loss(output, input, edge_input, x_mean, x_std, edge_mean, edge_std, edge_index, reg_coefs, node_param, edge_param)

    # Compute physical loss
    physical_loss_val = physical_loss(output, x_mean, x_std, edge_index, edge_param, node_param, reg_coefs)

    # Combine with weights
    combined_loss = lambda_wls * wls_loss_val + lambda_physical * physical_loss_val

    return combined_loss


if __name__ == "__main__":
    print("DSML Loss Functions - CPU Optimized")
    print("PyTorch-style loss functions:")
    print("- wls_loss: Weighted Least Squares loss")
    print("- physical_loss: Physical constraint loss")
    print("- wls_and_physical_loss: Combined loss with configurable lambda weights")
    print("Uses original get_pflow and gsp_wls_edge from src.robusttest.core.SE.pf_funcs")