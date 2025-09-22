import torch
import torch.nn as nn
from typing import Optional

import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.pool import avg_pool_neighbor_x

from external.loss import wls_loss, physical_loss, wls_and_physical_loss


class FAIR_GAT_BILEVEL_Lightning_Stable(pl.LightningModule):
    """
    Stabilized Bi-level GAT for DSSE in Lightning.
    Leader: odd layers + projection -> minimize WLS loss
    Follower: even layers -> minimize Physical loss
    """
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, 
                 time_info=True, loss_type='gsp_wls', loss_kwargs=None,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', fairness_alpha=100.0,
                 lr_g=1e-5, lr_f=1e-4, weight_decay=1e-6,  # Reduced learning rates
                 time_feat_dim=0, 
                 # New stability parameters
                 grad_clip_val=1.0, loss_clip_val=100.0, 
                 warmup_epochs=10, balance_ratio=0.1):

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

        # Stability parameters
        self.grad_clip_val = grad_clip_val
        self.loss_clip_val = loss_clip_val
        self.warmup_epochs = warmup_epochs
        self.balance_ratio = balance_ratio
        self.current_epoch_count = 0

        # GAT backbone (bi-level compatible)
        self.model = GAT_DSSE_BiLevel_Stable(
            dim_feat=dim_feat,
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

        # Collect parameters more carefully
        self.leader_params = []
        self.follower_params = []
        
        # Alternate layers between leader and follower
        for i, layer in enumerate(self.model.layers):
            if i % 2 == 0:  # Even indices -> Leader
                self.leader_params.extend(layer.parameters())
            else:  # Odd indices -> Follower
                self.follower_params.extend(layer.parameters())
        
        # Projection always belongs to leader
        self.leader_params.extend(self.model.projection.parameters())

        if len(self.leader_params) == 0 or len(self.follower_params) == 0:
            raise RuntimeError("Leader/Follower param lists are empty")

        # More conservative optimizers
        self.optimizer_G = Adam(self.leader_params, lr=lr_g, weight_decay=weight_decay, 
                               betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        self.optimizer_F = Adam(self.follower_params, lr=lr_f, weight_decay=weight_decay,
                               betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
        
        # Use plateau scheduler for more stability
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.8, 
                                           patience=5, min_lr=1e-7)
        self.scheduler_F = ReduceLROnPlateau(self.optimizer_F, mode='min', factor=0.8,
                                           patience=5, min_lr=1e-7)

        self.fairness_alpha = fairness_alpha
        
        # Track loss history for stability monitoring
        self.loss_history = {'leader': [], 'follower': [], 'total': []}

    def forward(self, x, edge_index, edge_input):
        """Forward pass with gradient clipping"""
        x_nodes_gnn = x[:, :self.num_nfeat]

        if self.use_time_info:
            time_info = x[:, self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)

        return self.model(x_nodes_gnn, edge_index, edge_input)
    
    def clip_loss(self, loss):
        """Clip loss values to prevent explosion"""
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(self.loss_clip_val, device=loss.device, requires_grad=True)
        return torch.clamp(loss, 0, self.loss_clip_val)
    
    def sanitize_gradients(self, params):
        """Clean gradients of NaN/Inf values"""
        for p in params:
            if p.grad is not None:
                # Replace NaN/Inf with zeros
                p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1e6, neginf=-1e6)
                # Additional clipping
                p.grad.data = torch.clamp(p.grad.data, -10.0, 10.0)
    
    def optimize_step(self, x, edge_index, edge_input,
                      node_param, edge_param, num_samples, k_follower=1):
        """Stabilized bi-level optimization step"""
        
        # Check if we're in warmup phase
        warmup_factor = min(1.0, self.current_epoch_count / self.warmup_epochs) if self.warmup_epochs > 0 else 1.0
        
        # Follower optimization with stability checks
        follower_loss_accumulated = 0.0
        for step in range(k_follower):
            self.optimizer_F.zero_grad()
            
            try:
                y_pred = self.forward(x, edge_index, edge_input)
                follower_loss = physical_loss(
                    output=y_pred,
                    x_mean=self.x_mean, x_std=self.x_std,
                    edge_index=edge_index, edge_param=edge_param,
                    node_param=node_param, reg_coefs=self.reg_coefs)
                
                follower_loss = self.clip_loss(follower_loss)
                follower_loss_accumulated += follower_loss.item()
                
                # Scale loss during warmup
                scaled_loss = follower_loss * warmup_factor
                scaled_loss.backward()
                
                # Gradient sanitization and clipping
                self.sanitize_gradients(self.follower_params)
                torch.nn.utils.clip_grad_norm_(self.follower_params, max_norm=self.grad_clip_val)
                
                # Check gradient health before stepping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.follower_params, max_norm=float('inf'))
                if grad_norm < 1e6 and not torch.isnan(grad_norm):  # Only step if gradients are reasonable
                    self.optimizer_F.step()
                else:
                    print(f"Skipping follower step due to large gradient norm: {grad_norm}")
                    
            except Exception as e:
                print(f"Error in follower step {step}: {e}")
                follower_loss = torch.tensor(self.loss_clip_val, device=x.device, requires_grad=True)
                break

        # Leader optimization
        self.optimizer_G.zero_grad()
        
        try:
            y_pred = self.forward(x, edge_index, edge_input)
            x_gnn = x[:, :self.num_nfeat]
            
            leader_loss = wls_loss(
                output=y_pred, input=x_gnn, edge_input=edge_input,
                x_mean=self.x_mean, x_std=self.x_std,
                edge_mean=self.edge_mean, edge_std=self.edge_std,
                edge_index=edge_index, reg_coefs=self.reg_coefs,
                node_param=node_param, edge_param=edge_param)
            
            leader_loss = self.clip_loss(leader_loss)
            
            # Balance the losses to prevent one from dominating
            if len(self.loss_history['leader']) > 0 and len(self.loss_history['follower']) > 0:
                leader_avg = sum(self.loss_history['leader'][-10:]) / len(self.loss_history['leader'][-10:])
                follower_avg = sum(self.loss_history['follower'][-10:]) / len(self.loss_history['follower'][-10:])
                
                if follower_avg > 0:
                    balance_weight = min(10.0, leader_avg / follower_avg) * self.balance_ratio
                    total_loss = leader_loss + balance_weight * follower_loss
                else:
                    total_loss = leader_loss + follower_loss
            else:
                total_loss = leader_loss + follower_loss
            
            # Scale during warmup
            scaled_loss = leader_loss * warmup_factor
            scaled_loss.backward()
            
            # Gradient sanitization and clipping
            self.sanitize_gradients(self.leader_params)
            torch.nn.utils.clip_grad_norm_(self.leader_params, max_norm=self.grad_clip_val)
            
            # Check gradient health
            grad_norm = torch.nn.utils.clip_grad_norm_(self.leader_params, max_norm=float('inf'))
            if grad_norm < 1e6 and not torch.isnan(grad_norm):
                self.optimizer_G.step()
            else:
                print(f"Skipping leader step due to large gradient norm: {grad_norm}")
                
        except Exception as e:
            print(f"Error in leader step: {e}")
            leader_loss = torch.tensor(self.loss_clip_val, device=x.device, requires_grad=True)
            total_loss = leader_loss + follower_loss

        # Update loss history
        self.loss_history['leader'].append(leader_loss.item())
        self.loss_history['follower'].append(follower_loss.item())
        self.loss_history['total'].append(total_loss.item())
        
        # Keep only recent history
        for key in self.loss_history:
            if len(self.loss_history[key]) > 100:
                self.loss_history[key] = self.loss_history[key][-50:]

        return leader_loss, follower_loss, total_loss

    def training_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        node_param = x[:, :self.num_nfeat]
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]
        num_samples = batch.batch[-1] + 1

        # Adaptive k_follower based on loss balance
        if len(self.loss_history['leader']) > 10 and len(self.loss_history['follower']) > 10:
            leader_recent = sum(self.loss_history['leader'][-5:]) / 5
            follower_recent = sum(self.loss_history['follower'][-5:]) / 5
            k_follower = 2 if follower_recent > leader_recent * 2 else 1
        else:
            k_follower = 1

        leader_loss, follower_loss, total_loss = self.optimize_step(
            x, edge_index, edge_input, node_param, edge_param, num_samples, k_follower=k_follower
        )

        # Log with finite check
        if torch.isfinite(total_loss):
            self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        if torch.isfinite(leader_loss):
            self.log("train_wls_loss", leader_loss, on_step=True, on_epoch=True)
        if torch.isfinite(follower_loss):
            self.log("train_phys_loss", follower_loss, on_step=True, on_epoch=True)
        
        # Log learning rates
        self.log("lr_leader", self.optimizer_G.param_groups[0]['lr'], on_step=True)
        self.log("lr_follower", self.optimizer_F.param_groups[0]['lr'], on_step=True)
        
        return total_loss

    def on_train_epoch_end(self):
        """Update epoch counter and schedulers"""
        self.current_epoch_count += 1
        
        # Update schedulers with average loss from this epoch
        if len(self.loss_history['total']) > 0:
            avg_loss = sum(self.loss_history['total'][-20:]) / len(self.loss_history['total'][-20:])
            self.scheduler_G.step(avg_loss)
            self.scheduler_F.step(avg_loss)

    def validation_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        node_param = x[:, self.num_nfeat:self.num_nfeat+3]
        x_gnn = x[:, :self.num_nfeat]
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]

        try:
            y_pred = self.forward(x, edge_index, edge_input)

            follower_loss = physical_loss(output=y_pred, x_mean=self.x_mean, x_std=self.x_std,
                                        edge_index=edge_index, edge_param=edge_param,
                                        node_param=node_param, reg_coefs=self.reg_coefs)
            leader_loss = wls_loss(output=y_pred, input=x_gnn, edge_input=edge_input,
                                 x_mean=self.x_mean, x_std=self.x_std,
                                 edge_mean=self.edge_mean, edge_std=self.edge_std,
                                 edge_index=edge_index, reg_coefs=self.reg_coefs,
                                 node_param=node_param, edge_param=edge_param)
            
            # Clip validation losses too
            follower_loss = self.clip_loss(follower_loss)
            leader_loss = self.clip_loss(leader_loss)
            total_loss = leader_loss + follower_loss

            if torch.isfinite(total_loss):
                self.log("val_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            if torch.isfinite(leader_loss):
                self.log("val_wls_loss", leader_loss, on_step=False, on_epoch=True)
            if torch.isfinite(follower_loss):
                self.log("val_phys_loss", follower_loss, on_step=False, on_epoch=True)

        except Exception as e:
            print(f"Error in validation step: {e}")
            total_loss = torch.tensor(self.loss_clip_val, device=x.device)

        return total_loss

    def predict_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        x_nodes = x[:,:self.num_nfeat]
        x_nodes_gnn = x_nodes.clone()
        
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
                {
                    "scheduler": self.scheduler_G,
                    "monitor": "val_total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
                {
                    "scheduler": self.scheduler_F,
                    "monitor": "val_total_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]
        )


class GAT_DSSE_BiLevel_Stable(nn.Module):
    """Stabilized GAT model with improved normalization"""
    def __init__(self, dim_feat, dim_hidden, dim_dense, dim_out,
                 num_layers, edge_dim, heads=1, concat=True,
                 slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', time_info=False,
                 time_feat_dim=0):

        super().__init__()
        self.num_layers = num_layers
        self.heads = heads
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops
        self.time_info = time_info
        self.time_feat_dim = time_feat_dim

        # More conservative activation
        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01)  # Less aggressive
        else:
            self.nonlin = nn.ReLU()  # Default to ReLU

        # Improved Lipschitz norm
        lipschitz_norm = StableLipschitzNorm(att_norm=2.0, eps=1e-8)

        # GAT layers with batch normalization
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        in_ch = dim_feat
        for i in range(num_layers):
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
            
            # Add layer normalization
            out_dim = dim_hidden * heads if concat else dim_hidden
            self.layer_norms.append(nn.LayerNorm(out_dim))
            
            in_ch = out_dim

        # More stable projection with batch norm and residual
        self.projection = nn.Sequential(
            nn.LayerNorm(in_ch),
            nn.Linear(in_ch, dim_dense),
            nn.BatchNorm1d(dim_dense),
            self.nonlin,
            nn.Dropout(0.1),
            nn.Linear(dim_dense, dim_dense // 2),
            nn.BatchNorm1d(dim_dense // 2),
            self.nonlin,
            nn.Linear(dim_dense // 2, dim_out),
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Conservative weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x, edge_index, edge_attr, time_info=None):
        for i, (layer, norm) in enumerate(zip(self.layers, self.layer_norms)):
            residual = x if i > 0 and x.size(-1) == layer.out_channels * (layer.heads if layer.concat else 1) else None
            
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            
            # Residual connection where possible
            if residual is not None:
                x = x + 0.1 * residual  # Scaled residual
                
            x = self.nonlin(x)
            
        return self.projection(x)


class StableLipschitzNorm(nn.Module):
    """More stable version of Lipschitz normalization"""
    def __init__(self, att_norm: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.att_norm = float(att_norm)
        self.eps = float(eps)

    def forward(self, e_ij: Tensor, x_i: Tensor, x_j: Tensor, index: Tensor) -> Tensor:
        # Compute norms with numerical stability
        ni = torch.norm(x_i, dim=-1, p=2) + self.eps  # [E, H]
        nj = torch.norm(x_j, dim=-1, p=2) + self.eps  # [E, H]
        
        # More stable max reduction
        max_nj_per_node = torch.zeros(index.max().item() + 1, nj.size(-1), 
                                    device=nj.device, dtype=nj.dtype)
        max_nj_per_node = max_nj_per_node.scatter_reduce(
            dim=0,
            index=index.unsqueeze(-1).expand_as(nj),
            src=nj,
            reduce="amax",
            include_self=False
        ) + self.eps
        
        denom = self.att_norm * (ni + max_nj_per_node[index]) + self.eps
        
        # Stable division with clipping
        ratio = e_ij / denom
        return torch.clamp(ratio, -10.0, 10.0)


class GATv2ConvNorm(GATv2Conv):
    """Stabilized GATv2 with improved normalization"""
    def __init__(self, *args, enable_lip: bool = True,
                 lipschitz_norm: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lip = enable_lip
        self.lipschitz_norm = lipschitz_norm if lipschitz_norm is not None else StableLipschitzNorm()

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor, dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, negative_slope=0.01)  # Less aggressive slope

        # Pre-softmax logits
        e_ij = (x * self.att).sum(dim=-1)

        # Lipschitz scaling with stability check
        if self.enable_lip and self.lipschitz_norm is not None:
            e_ij = self.lipschitz_norm(e_ij, x_i, x_j, index)
        
        # Additional stability clipping
        e_ij = torch.clamp(e_ij, -8, 8)

        alpha = softmax(e_ij, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha