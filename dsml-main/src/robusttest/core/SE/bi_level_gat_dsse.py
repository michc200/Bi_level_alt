import torch
import torch.nn as nn
from typing import Optional

import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import GATv2Conv, Sequential
from robusttest.core.SE.pf_funcs import compute_wls_loss, compute_physical_loss, compute_v_theta
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.pool import avg_pool_neighbor_x


class FAIR_GAT_BILEVEL_Lightning(pl.LightningModule):
    """
    Bi-level GAT for DSSE in Lightning.
    Leader: odd layers + projection -> minimize WLS loss
    Follower: even layers -> minimize Physical loss
    """
    def __init__(self, hyperparameters,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', fairness_alpha=100.0,
                 lr_g=1e-3, lr_f=1e-2, weight_decay=1e-5,
                 x_mean=None, x_std=None, edge_mean=None, edge_std=None,
                 reg_coefs=None, time_info=True, time_feat_dim=0):

        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # Model hyperparameters
        dim_feat = hyperparameters['dim_nodes']
        dim_hidden = hyperparameters['dim_hid']
        dim_dense = hyperparameters['dim_dense']
        dim_out = hyperparameters['dim_out']
        heads = hyperparameters['heads']
        num_layers = hyperparameters['gnn_layers']
        edge_dim = hyperparameters['dim_lines']
        dropout = hyperparameters['dropout_rate']
        self.num_nfeat = hyperparameters['num_nfeat']
        self.num_efeat = hyperparameters['dim_lines']

        # GAT backbone (bi-level compatible)
        self.model = GAT_DSSE_BiLevel(dim_feat=dim_feat,
                                      dim_hidden=dim_hidden,
                                      dim_dense=dim_dense,
                                      dim_out=dim_out,
                                      num_layers=num_layers,
                                      edge_dim=edge_dim,
                                      heads=heads,
                                      concat=concat,
                                      slope=slope,
                                      self_loops=self_loops,
                                      dropout=dropout,
                                      nonlin=nonlin,
                                      time_info=time_info,
                                      time_feat_dim=time_feat_dim)

        # Data normalization / reg
        self.x_mean = x_mean
        self.x_std = x_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.reg_coefs = reg_coefs
        self.use_time_info = time_info

        # Split parameters for leader/follower
        layers_params = list(self.model.model.children())
        self.leader_params = []
        self.follower_params = []
        for idx, layer in enumerate(layers_params[:-3]):  # exclude projection
            if isinstance(layer, GATv2Conv):
                if idx % 2 == 0:
                    self.leader_params += list(layer.parameters())
                else:
                    self.follower_params += list(layer.parameters())

        # Optimizers
        self.optimizer_G = Adam(self.leader_params, lr=lr_g, weight_decay=weight_decay*10)
        self.optimizer_F = Adam(self.follower_params, lr=lr_f, weight_decay=weight_decay)
        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

        self.fairness_alpha = fairness_alpha

    def forward(self, x_nodes, edge_index, edge_input, time_info=None):
        return self.model(x_nodes, edge_index, edge_input, time_info)
    
    def process_input(self, batch): # This is currently unused
        combined_tensor = batch.x[:, :self.num_nfeat]

        non_zero_mask = combined_tensor != 0
        for _ in range(int(10)):
            batch = avg_pool_neighbor_x(batch)
            x_nodes_gnn = batch.x[:,:self.num_nfeat]

            # Place zero values (processed) from batch_2.x into the combined tensor
            combined_tensor[~non_zero_mask] = x_nodes_gnn[~non_zero_mask]
            batch.x = combined_tensor

        return combined_tensor

    def optimize_step(self, x_nodes, edge_index, edge_input,
                      node_param, edge_param, num_samples, time_info=None, k_follower=1):
        """One bi-level optimization step."""
        # Follower: Physical loss
        for _ in range(k_follower):
            self.optimizer_F.zero_grad()
            y_pred = self.forward(x_nodes, edge_index, edge_input, time_info)
            v_i, theta_i = compute_v_theta(output=y_pred, x_mean=self.x_mean, x_std=self.x_std, node_param=node_param)
            follower_loss = compute_physical_loss(node_param=node_param,
                                                  edge_index=edge_index, 
                                                  edge_param=edge_param, 
                                                  v_i=v_i,
                                                  theta_i=theta_i,
                                                  reg_coefs=self.reg_coefs)
            follower_loss.backward()
            self.optimizer_F.step()

        # Leader: WLS loss
        self.optimizer_G.zero_grad()
        y_pred = self.forward(x_nodes, edge_index, edge_input, time_info)
        wls_loss, _, _ = compute_wls_loss(input=x_nodes,
                                          edge_input=edge_input,
                                          output=y_pred,
                                          x_mean=self.x_mean,
                                          x_std=self.x_std,
                                          edge_mean=self.edge_mean,
                                          edge_std=self.edge_std,
                                          edge_index=edge_index,
                                          reg_coefs=self.reg_coefs,
                                          node_param=node_param,
                                          edge_param=edge_param,
                                          num_samples=num_samples)
        total_loss = wls_loss + follower_loss
        wls_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_params, max_norm=1.0)
        self.optimizer_G.step()

        return wls_loss, follower_loss, total_loss

    def training_step(self, batch, batch_idx):
        x  = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        x_nodes_gnn = x_nodes.clone() # self.process_input(batch)
        if self.use_time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]
        num_samples = batch.batch[-1] + 1
        time_info = batch.x[:, self.hparams.hyperparameters['dim_nodes']+3:] \
                    if self.use_time_info else None

        wls_loss, follower_loss, total_loss = self.optimize_step(
            x_nodes, edge_index, edge_input, node_param, edge_param, num_samples, time_info
        )

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wls_loss", wls_loss, on_step=True, on_epoch=True)
        self.log("train_phys_loss", follower_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x  = batch.x.clone()
        edge_index  = batch.edge_index
        edge_attr  = batch.edge_attr
        y = batch.y  # Target values for MSE loss
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        x_nodes_gnn = x_nodes.clone() # self.process_input(batch)
        num_samples = batch.batch[-1] + 1
        if self.use_time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]

        y_pred = self.forward(x_nodes, edge_index, edge_input, time_info)
        wls_loss, v_i, theta_i = compute_wls_loss(input=x_nodes, 
                                                  edge_input=edge_input, 
                                                  output=y_pred,
                                                  x_mean=self.x_mean,
                                                  x_std=self.x_std,
                                                  edge_mean=self.edge_mean,
                                                  edge_std=self.edge_std,
                                                  edge_index=edge_index,
                                                  reg_coefs=self.reg_coefs,
                                                  node_param=node_param, 
                                                  edge_param=edge_param)
        phys_loss = compute_physical_loss(node_param=node_param,
                                         edge_index=edge_index, 
                                         edge_param=edge_param, 
                                         v_i=v_i,
                                         theta_i=theta_i,
                                         reg_coefs=self.reg_coefs)
        total_loss = wls_loss + phys_loss

        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_wls_loss", wls_loss, on_step=False, on_epoch=True)
        self.log("val_phys_loss", phys_loss, on_step=False, on_epoch=True)

        return total_loss

    def predict_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        x_nodes = x[:,:self.num_nfeat]
        x_nodes_gnn = x_nodes.clone() # self.process_input(batch)
        if self.use_time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)
        
        edge_input = edge_attr[:,:self.num_efeat]
        y_pred = self.forward(x_nodes, edge_index, edge_input, time_info)

        v_i = y_pred[:, 0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = y_pred[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1.- node_param[:,1:2])
        return v_i, theta_i

    def configure_optimizers(self):
        return [self.optimizer_G, self.optimizer_F], [self.scheduler_G, self.scheduler_F]
# -------------------------
# Bi-level compatible GAT model
# -------------------------
class GAT_DSSE_BiLevel(nn.Module):
    """
    GAT model compatible with bi-level optimization.
    Supports optional concatenation of time_info features.
    """
    def __init__(self, dim_feat, dim_hidden, dim_dense, dim_out,
                 num_layers, edge_dim, heads=1, concat=True,
                 slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', model='gat', time_info=False,
                 time_feat_dim=0):
        super().__init__()
        self.dim_feat = dim_feat
        self.dim_hidden = dim_hidden
        self.dim_dense = dim_dense
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops
        self.time_info = time_info
        self.time_feat_dim = time_feat_dim

        # Activation
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = nn.LeakyReLU()
        else:
            raise ValueError("Invalid nonlin type")

        nn_layers = []
        in_ch = self.dim_feat + (self.time_feat_dim if self.time_info else 0)

        if model == 'gat':
            nn_layers.append((GATv2ConvNorm(in_channels=in_ch, out_channels=self.dim_hidden,
                                         heads=self.heads, concat=self.concat,
                                         negative_slope=self.slope, dropout=self.dropout,
                                         add_self_loops=self.loop, edge_dim=self.edge_dim),
                              'x, edge_index, edge_attr -> x'))
            nn_layers.append(self.nonlin)
            for _ in range(self.num_layers - 1):
                nn_layers.append((GATv2ConvNorm(in_channels=self.dim_hidden, out_channels=self.dim_hidden,
                                             heads=self.heads, concat=self.concat,
                                             negative_slope=self.slope, dropout=self.dropout,
                                             add_self_loops=self.loop, edge_dim=self.edge_dim),
                                  'x, edge_index, edge_attr -> x'))
                nn_layers.append(self.nonlin)
        else:
            raise ValueError("Invalid model type")

        # Projection head
        nn_layers.append(nn.Linear(self.dim_hidden, self.dim_dense))
        nn_layers.append(self.nonlin)
        nn_layers.append(nn.Linear(self.dim_dense, self.dim_out))

        self.model = Sequential('x, edge_index, edge_attr', nn_layers)

    def forward(self, x, edge_index, edge_attr, time_info=None):
        if self.time_info and time_info is not None:
            x = torch.cat([x, time_info], dim=1)
        return self.model(x, edge_index, edge_attr)
    

class LipschitzNorm(nn.Module):
    """
    Scales pre-softmax logits e_ij per head to control sensitivity.
    Shapes:
      e_ij: [E, H]
      x_i, x_j: [E, H, C]
      index: [E]  – destination node index for softmax groups
    """
    def __init__(self, att_norm: float = 4.0, eps: float = 1e-12):
        super().__init__()
        self.att_norm = float(att_norm)
        self.eps = float(eps)

    def forward(self, e_ij: Tensor, x_i: Tensor, x_j: Tensor, index: Tensor) -> Tensor:
        ni = torch.norm(x_i, dim=-1)         # [E, H]
        nj = torch.norm(x_j, dim=-1)         # [E, H]
        max_nj_per_node = torch.scatter_reduce(
            torch.zeros(index.max().item() + 1, nj.size(-1), device=nj.device, dtype=nj.dtype),
            dim=0,
            index=index.unsqueeze(-1).expand_as(nj),
            src=nj,
            reduce="amax",
            include_self=False)
        denom = self.att_norm * (ni + max_nj_per_node[index]) + self.eps
        return e_ij / denom


class GATv2ConvNorm(GATv2Conv):
    """
    GATv2 with optional Lipschitz normalization on attention logits.
    Set `enable_lip=True` to activate.
    """
    def __init__(self,
                 *args,
                 enable_lip: bool = True,
                 lipschitz_norm: Optional[nn.Module] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lip = enable_lip
        self.lipschitz_norm = lipschitz_norm if lipschitz_norm is not None else LipschitzNorm()

    # Keep the same signature to stay TorchScript-friendly
    def edge_update(self,
                    x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)

        # Pre-softmax logits per edge per head
        e_ij = (x * self.att).sum(dim=-1)  # [E, H]

        # Lipschitz scaling before softmax
        if self.enable_lip and self.lipschitz_norm is not None:
            # x_i, x_j are [E, H, C] already from MessagePassing expansion
            e_ij = self.lipschitz_norm(e_ij, x_i, x_j, index)

        alpha = softmax(e_ij, index, ptr, dim_size)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
