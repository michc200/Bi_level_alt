import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import GATv2Conv, Sequential
from robusttest.core.SE.pf_funcs import compute_wls_loss, compute_physical_loss, compute_v_theta

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
        x_nodes = batch.x[:, :self.hparams.hyperparameters['dim_feat']]
        node_param = batch.x[:, self.hparams.hyperparameters['dim_feat']:self.hparams.hyperparameters['dim_feat']+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.hyperparameters['edge_dim']]
        edge_param = batch.edge_attr[:, self.hparams.hyperparameters['edge_dim']:]
        num_samples = batch.batch[-1] + 1
        time_info = batch.x[:, self.hparams.hyperparameters['dim_feat']+3:] if self.use_time_info else None

        wls_loss, follower_loss, total_loss = self.optimize_step(
            x_nodes, edge_index, edge_input, node_param, edge_param, num_samples, time_info
        )

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wls_loss", wls_loss, on_step=True, on_epoch=True)
        self.log("train_phys_loss", follower_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_nodes = batch.x[:, :self.hparams.hyperparameters['dim_feat']]
        node_param = batch.x[:, self.hparams.hyperparameters['dim_feat']:self.hparams.hyperparameters['dim_feat']+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.hyperparameters['edge_dim']]
        edge_param = batch.edge_attr[:, self.hparams.hyperparameters['edge_dim']:]
        num_samples = batch.batch[-1] + 1
        time_info = batch.x[:, self.hparams.hyperparameters['dim_feat']+3:] if self.use_time_info else None

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
        x_nodes = batch.x[:, :self.hparams.hyperparameters['dim_feat']]
        node_param = batch.x[:, self.hparams.hyperparameters['dim_feat']:self.hparams.hyperparameters['dim_feat']+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.hyperparameters['edge_dim']]
        time_info = batch.x[:, self.hparams.hyperparameters['dim_feat']+3:] if self.use_time_info else None

        y_pred = self.forward(x_nodes, edge_index, edge_input, time_info)

        v_i = y_pred[:, 0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = y_pred[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1. - node_param[:, 1:2])

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
            nn_layers.append((GATv2Conv(in_channels=in_ch, out_channels=self.dim_hidden,
                                         heads=self.heads, concat=self.concat,
                                         negative_slope=self.slope, dropout=self.dropout,
                                         add_self_loops=self.loop, edge_dim=self.edge_dim),
                              'x, edge_index, edge_attr -> x'))
            nn_layers.append(self.nonlin)
            for _ in range(self.num_layers - 1):
                nn_layers.append((GATv2Conv(in_channels=self.dim_hidden, out_channels=self.dim_hidden,
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
