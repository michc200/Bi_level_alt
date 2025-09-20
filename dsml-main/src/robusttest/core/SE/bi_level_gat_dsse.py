import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.nn import GATv2Conv, Sequential
from robusttest.core.SE.pf_funcs import compute_wls_loss, compute_physical_loss, compute_v_theta

class FAIR_GAT_BILEVEL_Lightning(pl.LightningModule):
    """
    Bi-level GAT for DSSE in Lightning style.
    Leader: odd layers + projection -> minimize WLS loss
    Follower: even layers -> minimize Physical loss
    """
    def __init__(self, dim_feat, dim_dense, dim_out, num_layers, edge_dim,
                 heads=1, concat=True, slope=0.2, self_loops=True, dropout=0.0,
                 nonlin='leaky_relu', fairness_alpha=100.0,
                 lr_g=1e-3, lr_f=1e-2, weight_decay=1e-5,
                 x_mean=None, x_std=None, edge_mean=None, edge_std=None,
                 reg_coefs=None):

        super().__init__()
        self.save_hyperparameters()

        # GAT backbone as Sequential
        nn_layers = []
        for i in range(num_layers):
            in_ch = dim_feat if i == 0 else dim_feat
            nn_layers.append((GATv2Conv(in_channels=in_ch, out_channels=dim_feat,
                                         heads=heads, concat=concat, negative_slope=slope,
                                         dropout=dropout, add_self_loops=self_loops,
                                         edge_dim=edge_dim), 'x, edge_index, edge_attr -> x'))
            nn_layers.append(nn.LeakyReLU() if nonlin == 'leaky_relu' else nn.ReLU())

        # Projection head
        nn_layers.extend([nn.Linear(dim_feat, dim_dense), nn.LeakyReLU(), nn.Linear(dim_dense, dim_out)])
        self.model = Sequential('x, edge_index, edge_attr', nn_layers)

        # Save data normalization / reg
        self.x_mean = x_mean
        self.x_std = x_std
        self.edge_mean = edge_mean
        self.edge_std = edge_std
        self.reg_coefs = reg_coefs

        # Split parameters for leader/follower
        layers_params = list(self.model.children())
        self.leader_params = []
        self.follower_params = []
        for idx, layer in enumerate(layers_params[:-3]):  # exclude projection
            if isinstance(layer, GATv2Conv):
                if idx % 2 == 0:
                    self.leader_params += list(layer.parameters())
                else:
                    self.follower_params += list(layer.parameters())
        self.leader_params += list(layers_params[-3:].parameters())  # projection

        # Optimizers
        self.optimizer_G = Adam(self.leader_params, lr=lr_g, weight_decay=weight_decay*10)
        self.optimizer_F = Adam(self.follower_params, lr=lr_f, weight_decay=weight_decay)
        self.scheduler_G = ExponentialLR(self.optimizer_G, gamma=0.99)
        self.scheduler_F = ExponentialLR(self.optimizer_F, gamma=0.99)

        self.fairness_alpha = fairness_alpha

    def forward(self, x_nodes, edge_index, edge_input):
        return self.model(x_nodes, edge_index, edge_input)

    def optimize_step(self, x_nodes, edge_index, edge_input,
                      node_param, edge_param, num_samples, k_follower=1):
        """Perform one bi-level optimization step."""
        # Follower (Physical loss)
        for _ in range(k_follower):
            self.optimizer_F.zero_grad()
            y_pred = self.forward(x_nodes, edge_index, edge_input)
            v_i, theta_i = compute_v_theta(output=y_pred, x_mean=self.x_mean, x_std=self.x_std, node_param=node_param)
            follower_loss = compute_physical_loss(node_param=node_param,
                                                edge_index=edge_index, 
                                                edge_param=edge_param, 
                                                v_i=v_i,
                                                theta_i=theta_i,
                                                reg_coefs=self.reg_coefs)
            follower_loss.backward()
            self.optimizer_F.step()

        # Leader (WLS loss)
        self.optimizer_G.zero_grad()
        y_pred = self.forward(x_nodes, edge_index, edge_input)
        wls_loss, _, _ = compute_wls_loss(input=x_nodes, edge_input=edge_input, output=y_pred,
                                x_mean=self.x_mean, x_std=self.x_std,
                                edge_mean=self.edge_mean, edge_std=self.edge_std,
                                edge_index=edge_index, reg_coefs=self.reg_coefs,
                                node_param=node_param, edge_param=edge_param,
                                num_samples=num_samples)
        total_loss = wls_loss + follower_loss
        wls_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leader_params, max_norm=1.0)
        self.optimizer_G.step()

        return wls_loss, follower_loss, total_loss

    def training_step(self, batch, batch_idx):
        x_nodes = batch.x[:, :self.hparams.dim_feat]
        node_param = batch.x[:, self.hparams.dim_feat:self.hparams.dim_feat+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.edge_dim]
        edge_param = batch.edge_attr[:, self.hparams.edge_dim:]
        num_samples = batch.batch[-1] + 1

        wls_loss, follower_loss, total_loss = self.optimize_step(
            x_nodes, edge_index, edge_input, node_param, edge_param, num_samples
        )

        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_wls_loss", wls_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_phys_loss", follower_loss, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x_nodes = batch.x[:, :self.hparams.dim_feat]
        node_param = batch.x[:, self.hparams.dim_feat:self.hparams.dim_feat+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.edge_dim]
        edge_param = batch.edge_attr[:, self.hparams.edge_dim:]
        num_samples = batch.batch[-1] + 1

        y_pred = self.forward(x_nodes, edge_index, edge_input)
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
        self.log("val_wls_loss", wls_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_phys_loss", phys_loss, on_step=False, on_epoch=True, prog_bar=False)

        return total_loss

    def predict_step(self, batch, batch_idx):
        x_nodes = batch.x[:, :self.hparams.dim_feat]
        node_param = batch.x[:, self.hparams.dim_feat:self.hparams.dim_feat+3]
        edge_index = batch.edge_index
        edge_input = batch.edge_attr[:, :self.hparams.edge_dim]

        y_pred = self.forward(x_nodes, edge_index, edge_input)

        v_i = y_pred[:, 0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = y_pred[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1. - node_param[:, 1:2])

        return v_i, theta_i

    def configure_optimizers(self):
        return [self.optimizer_G, self.optimizer_F], [self.scheduler_G, self.scheduler_F]

if __name__ == '__main__':
    print('test')