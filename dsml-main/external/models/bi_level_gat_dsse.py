import torch
import torch.nn as nn
from typing import Optional

import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.pool import avg_pool_neighbor_x

from external.loss import wls_loss, physical_loss, wls_and_physical_loss


class FAIR_GAT_BILEVEL_Lightning(pl.LightningModule):
    """
    Bi-level GAT for DSSE in Lightning.
    Leader: odd layers + projection -> minimize WLS loss
    Follower: even layers -> minimize Physical loss
    """
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, time_info = True, loss_type='gsp_wls', loss_kwargs=None,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', fairness_alpha=100.0,
                 lr_g=1e-4, lr_f=1e-3, weight_decay=1e-5,
                 time_feat_dim=0):

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

        # Collect the actual GAT conv modules from the Sequential model
        # Leader (odd layers + projection), Follower (even layers)
        self.leader_params = [p for i, l in enumerate(self.model.layers) if i % 2 == 0 for p in l.parameters()]
        self.leader_params += list(self.model.projection.parameters())
        self.follower_params = [p for i, l in enumerate(self.model.layers) if i % 2 == 1 for p in l.parameters()]

        if len(self.leader_params) == 0 or len(self.follower_params) == 0:
            raise RuntimeError("Leader/Follower param lists are empty — check model.layers collection")

        # Optimizers
        self.optimizer_G = Adam(self.leader_params, lr=lr_g, weight_decay=weight_decay)
        self.optimizer_F = Adam(self.follower_params, lr=lr_f, weight_decay=weight_decay)
        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

        self.fairness_alpha = fairness_alpha

    def forward(self, x, edge_index, edge_input):
        """
        Full x is passed in. We slice node features + time info here
        before calling the GAT backbone.
        """
        # Base node features
        x_nodes_gnn = x[:, :self.num_nfeat]

        # Time features if used
        if self.use_time_info:
            time_info = x[:, self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

        return self.model(x_nodes_gnn, edge_index, edge_input)
    
    def optimize_step(self, x, edge_index, edge_input,
                      node_param, edge_param, num_samples, k_follower=1):
        """One bi-level optimization step using full x."""
        # Follower: Physical loss
        for _ in range(k_follower):
            self.optimizer_F.zero_grad()
            y_pred = self.forward(x, edge_index, edge_input)
            follower_loss = physical_loss(output=y_pred,
                                          x_mean=self.x_mean, x_std=self.x_std,
                                          edge_index=edge_index, edge_param=edge_param,
                                          node_param=node_param, reg_coefs=self.reg_coefs)
            follower_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.follower_params, max_norm=5.0)
            # sanitize grads (replace NaN/Inf with zero)
            for p in self.follower_params:
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6)
            self.optimizer_F.step()

        # Leader: WLS loss
        self.optimizer_G.zero_grad()
        y_pred = self.forward(x, edge_index, edge_input)
        x_gnn = x[:, :self.num_nfeat]
        leader_loss = wls_loss(output=y_pred, input=x_gnn, edge_input=edge_input,
                               x_mean=self.x_mean, x_std=self.x_std,
                               edge_mean=self.edge_mean, edge_std=self.edge_std,
                               edge_index=edge_index, reg_coefs=self.reg_coefs,
                               node_param=node_param, edge_param=edge_param)
        
        total_loss = leader_loss + follower_loss
        leader_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_params, max_norm=5.0)
        for p in self.leader_params:
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6)
        self.optimizer_G.step()

        return leader_loss, follower_loss, total_loss

    def training_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        node_param = x[:, :self.num_nfeat]
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]
        num_samples = batch.batch[-1] + 1

        leader_loss, follower_loss, total_loss = self.optimize_step(
            x, edge_index, edge_input, node_param, edge_param, num_samples
        )

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wls_loss", leader_loss, on_step=True, on_epoch=True)
        self.log("train_phys_loss", follower_loss, on_step=True, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        node_param = x[:, self.num_nfeat:self.num_nfeat+3]
        x_gnn = x[:, :self.num_nfeat]
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]
        num_samples = batch.batch[-1] + 1

        y_pred = self.forward(x, edge_index, edge_input)

        follower_loss = physical_loss(output=y_pred, x_mean=self.x_mean, x_std=self.x_std,
                                      edge_index=edge_index, edge_param=edge_param,
                                      node_param=node_param, reg_coefs=self.reg_coefs)
        leader_loss = wls_loss(output=y_pred, input=x_gnn, edge_input=edge_input,
                               x_mean=self.x_mean, x_std=self.x_std,
                               edge_mean=self.edge_mean, edge_std=self.edge_std,
                               edge_index=edge_index, reg_coefs=self.reg_coefs,
                               node_param=node_param, edge_param=edge_param)
        total_loss = leader_loss + follower_loss

        self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_wls_loss", leader_loss, on_step=False, on_epoch=True)
        self.log("val_phys_loss", follower_loss, on_step=False, on_epoch=True)

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
        y_pred = self.forward(x, edge_index, edge_input)

        v_i = y_pred[:, 0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = y_pred[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1.- node_param[:,1:2])
        return v_i, theta_i

    def configure_optimizers(self):
        return (
            [self.optimizer_G, self.optimizer_F],
            [
                {"scheduler": self.scheduler_G, "interval": "epoch"},
                {"scheduler": self.scheduler_F, "interval": "epoch"},
            ],
        )
    
# -------------------------
# Bi-level compatible GAT model
# -------------------------
class GAT_DSSE_BiLevel(nn.Module):
    """
    GAT model compatible with bi-level optimization.
    Uses explicit layers + projection, like old GAT_NORM_DSSE.
    """
    def __init__(self, dim_feat, dim_hidden, dim_dense, dim_out,
                 num_layers, edge_dim, heads=1, concat=True,
                 slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', time_info=False,
                 time_feat_dim=0, lipschitz_norm=None):

        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops
        self.time_info = time_info
        self.time_feat_dim = time_feat_dim

        # activation
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = nn.LeakyReLU()
        else:
            raise ValueError("Invalid nonlin type")

        if lipschitz_norm is None:
            lipschitz_norm = LipschitzNorm(att_norm=4.0, eps=1e-12)

        # -----------------------
        # GAT layers (explicit)
        # -----------------------
        self.layers = nn.ModuleList()
        in_ch = dim_feat
        for _ in range(num_layers):
            layer = GATv2ConvNorm(
                in_channels=in_ch,
                out_channels=dim_hidden,
                heads=heads,
                concat=concat,
                negative_slope=slope,
                dropout=dropout,
                add_self_loops=self.loop,
                edge_dim=edge_dim,
                enable_lip=True,
                lipschitz_norm=lipschitz_norm,
            )
            self.layers.append(layer)
            in_ch = dim_hidden * heads if concat else dim_hidden

        # -----------------------
        # Projection head
        # -----------------------
        self.projection = nn.Sequential(
            nn.Linear(in_ch, dim_dense),
            self.nonlin,
            nn.Linear(dim_dense, dim_out),
        )

    def forward(self, x, edge_index, edge_attr, time_info=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = self.nonlin(x)
        return self.projection(x)    

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
        denom = self.att_norm * (ni + max_nj_per_node[index])
        # clamp denom to avoid tiny values
        denom = torch.clamp(denom, min=self.eps)
        # also clamp numerators to avoid huge values
        e_ij = torch.clamp(e_ij, -1e4, 1e4)
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
        e_ij = torch.clamp(e_ij, -20, 20) # TODO: check if this helps grad

        alpha = softmax(e_ij, index, ptr, dim_size)  # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha