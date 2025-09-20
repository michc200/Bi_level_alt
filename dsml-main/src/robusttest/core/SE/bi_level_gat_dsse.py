import torch
from torch import nn
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch_geometric.nn import GATv2Conv, Sequential
from torch_geometric.nn.models import GAT, MLP
from torch_geometric.nn.pool import avg_pool_neighbor_x
from torch.nn import LeakyReLU, Linear
import torch.nn.functional as F
from torch.nn import Module
from robusttest.core.SE.pf_funcs import gsp_wls_edge, compute_wls_loss, compute_physical_loss

# Add path for external loss functions
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from external.loss import wls_loss, physical_loss, wls_and_physical_loss
import logging
from torch.nn.functional import mse_loss

logger = logging.getLogger("BI_LEVEL_GAT_DSSE")

class BiLevelGAT_DSSE_Lightning(pl.LightningModule):
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, time_info=True, loss_type='gsp_wls', loss_kwargs=None):
        super().__init__()

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

        # Initialize bi-level GNN model
        self.model = BiLevelGAT_DSSE(dim_feat=dim_feat, dim_hidden=dim_hidden, dim_dense=dim_dense, dim_out=dim_out,
                                     num_layers=num_layers, edge_dim=edge_dim, heads=heads, dropout=dropout)

        # Save other required parameters
        self.x_mean = torch.tensor(x_mean)
        self.x_std = torch.tensor(x_std)
        self.edge_mean = torch.tensor(edge_mean)
        self.edge_std = torch.tensor(edge_std)

        # Extract reg_coefs from the unified loss_kwargs structure
        self.reg_coefs = {k: v for k, v in reg_coefs.items() if k not in ['lambda_wls', 'lambda_physical']}

        self.num_nfeat = hyperparameters['num_nfeat']
        self.num_efeat = hyperparameters['dim_lines']
        self.lr = hyperparameters['lr']
        self.time_info = time_info

        # Loss configuration
        self.loss_type = loss_type  # Options: 'gsp_wls', 'wls', 'physical', 'wls_and_physical', 'mse'
        self.loss_kwargs = loss_kwargs if loss_kwargs is not None else {}

        # Loss tracker
        self.train_loss = []
        self.val_loss = []

    def forward(self, x, edge_index, edge_attr):
        """Forward pass through the bi-level GNN."""
        return self.model(x, edge_index, edge_attr)

    def process_input(self, batch):
        combined_tensor = batch.x[:, :self.num_nfeat]

        non_zero_mask = combined_tensor != 0
        for _ in range(int(10)):
            batch = avg_pool_neighbor_x(batch)
            x_nodes_gnn = batch.x[:,:self.num_nfeat]

            # Place zero values (processed) from batch_2.x into the combined tensor
            combined_tensor[~non_zero_mask] = x_nodes_gnn[~non_zero_mask]
            batch.x = combined_tensor

        return combined_tensor

    def training_step(self, batch, batch_idx):
        """One training step."""
        x = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        y = batch.y  # Target values for MSE loss
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]

        x_nodes_gnn = x_nodes.clone()
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]

        output = self(x_nodes_gnn,
                      edge_index,
                      edge_input)

        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """One validation step."""
        x = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        y = batch.y  # Target values for MSE loss
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]

        x_nodes_gnn = x_nodes.clone()

        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]

        output = self(x_nodes_gnn,
                      edge_index,
                      edge_input)

        # Calculate loss based on the selected method
        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch.x.clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]

        x_nodes = x[:,:self.num_nfeat]
        x_nodes_gnn = x_nodes.clone()

        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes_gnn = torch.cat([x_nodes_gnn, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]

        output = self(x_nodes_gnn,
                      edge_index,
                      edge_input)

        v_i = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = output[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1.- node_param[:,1:2])

        return v_i, theta_i

    def calculate_loss(self, x, edge_input, output, edge_index, node_param, edge_param, num_samples, y=None):
        """Custom loss function with configurable loss types."""

        if self.loss_type == 'mse':
            # MSE loss requires target values and denormalization
            if y is None:
                raise ValueError("Target values (y) are required for MSE loss")

            # Denormalize output
            output_denorm = output.clone()
            output_denorm[:, 0:1] = output_denorm[:, 0:1] * self.x_std[:1] + self.x_mean[:1]
            output_denorm[:, 1:] = output_denorm[:, 1:] * self.x_std[2:3] + self.x_mean[2:3]
            output_denorm[:, 1:] *= (1. - node_param[:, 1:2])  # Enforce theta_slack = 0

            return mse_loss(output_denorm, y)

        elif self.loss_type == 'gsp_wls':
            # Original combined loss function
            return gsp_wls_edge(input=x, edge_input=edge_input,
                                output=output, x_mean=self.x_mean,
                                x_std=self.x_std, edge_mean=self.edge_mean,
                                edge_std=self.edge_std, edge_index=edge_index,
                                reg_coefs=self.reg_coefs, num_samples=num_samples,
                                node_param=node_param, edge_param=edge_param)

        elif self.loss_type == 'wls':
            # WLS loss only
            return wls_loss(output=output, input=x, edge_input=edge_input,
                           x_mean=self.x_mean, x_std=self.x_std,
                           edge_mean=self.edge_mean, edge_std=self.edge_std,
                           edge_index=edge_index, reg_coefs=self.reg_coefs,
                           node_param=node_param, edge_param=edge_param)

        elif self.loss_type == 'physical':
            # Physical loss only
            return physical_loss(output=output, x_mean=self.x_mean, x_std=self.x_std,
                                edge_index=edge_index, edge_param=edge_param,
                                node_param=node_param, reg_coefs=self.reg_coefs)

        elif self.loss_type == 'wls_and_physical':
            # Combined loss with configurable weights
            lambda_wls = self.loss_kwargs.get('lambda_wls', 1.0)
            lambda_physical = self.loss_kwargs.get('lambda_physical', 1.0)

            return wls_and_physical_loss(output=output, input=x, edge_input=edge_input,
                                        x_mean=self.x_mean, x_std=self.x_std,
                                        edge_mean=self.edge_mean, edge_std=self.edge_std,
                                        edge_index=edge_index, reg_coefs=self.reg_coefs,
                                        node_param=node_param, edge_param=edge_param,
                                        lambda_wls=lambda_wls, lambda_physical=lambda_physical)

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Options: 'gsp_wls', 'wls', 'physical', 'wls_and_physical', 'mse'")

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = Adamax(self.parameters(), lr=self.lr)

        # Set up a learning rate scheduler that reduces the LR if no progress is made
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'train_loss_epoch',
        }

        return [optimizer], [scheduler]


class BiLevelGAT_DSSE(nn.Module):
    def __init__(self, dim_feat, dim_hidden, dim_dense, dim_out, num_layers, edge_dim, heads=1, concat=True, slope=0.2, self_loops=True, dropout=0., nonlin='leaky_relu', model='gat'):
        super().__init__()
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.dim_feat = dim_feat
        self.dim_dense = dim_dense
        self.edge_dim = edge_dim
        self.dim_hidden = dim_hidden

        self.channels = dim_feat
        self.heads = heads
        self.dim_out = dim_out
        self.concat = concat
        self.slope = slope
        self.dropout = dropout
        self.loop = self_loops
        self.num_layers = num_layers

        if nonlin == 'relu':
            self.nonlin = nn.ReLU()
        elif nonlin == 'tanh':
            self.nonlin = nn.Tanh()
        elif nonlin == 'leaky_relu':
            self.nonlin = LeakyReLU()
        else:
            raise Exception('invalid activation type')

        # Bi-level architecture: Local and Global processing
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()

        # Local level: Node-centric processing
        local_layer = []
        if model == 'gat':
            hyperparameters = {"in_channels": self.dim_feat, "out_channels": self.dim_hidden,
                              "heads": self.heads, "concat": self.concat, "negative_slope": self.slope,
                              "dropout": self.dropout, "add_self_loops": self.loop, "edge_dim": self.edge_dim}
            local_layer.extend([(GATv2Conv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])

            for l in range(self.num_layers - 1):
                hyperparameters = {"in_channels": self.dim_hidden, "out_channels": self.dim_hidden,
                                  "heads": self.heads, "concat": self.concat, "negative_slope": self.slope,
                                  "dropout": self.dropout, "add_self_loops": self.loop, "edge_dim": self.edge_dim}
                local_layer.extend([(GATv2Conv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])
        else:
            raise Exception('invalid model type')

        self.local_model = Sequential('x, edge_index, edge_attr', local_layer)

        # Global level: System-wide processing
        global_layer = []
        if model == 'gat':
            # Global processing with different attention patterns
            hyperparameters = {"in_channels": self.dim_hidden, "out_channels": self.dim_hidden,
                              "heads": self.heads, "concat": self.concat, "negative_slope": self.slope,
                              "dropout": self.dropout, "add_self_loops": self.loop, "edge_dim": self.edge_dim}
            global_layer.extend([(GATv2Conv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])

            for l in range(self.num_layers - 1):
                hyperparameters = {"in_channels": self.dim_hidden, "out_channels": self.dim_hidden,
                                  "heads": self.heads, "concat": self.concat, "negative_slope": self.slope,
                                  "dropout": self.dropout, "add_self_loops": self.loop, "edge_dim": self.edge_dim}
                global_layer.extend([(GATv2Conv(**hyperparameters), 'x, edge_index, edge_attr -> x'), self.nonlin])

        self.global_model = Sequential('x, edge_index, edge_attr', global_layer)

        # Fusion mechanism
        self.fusion_layer = Linear(in_features=2 * self.dim_hidden, out_features=self.dim_hidden)

        # Final prediction layers
        self.prediction_layers = nn.Sequential(
            Linear(in_features=self.dim_hidden, out_features=self.dim_dense),
            self.nonlin,
            Linear(in_features=self.dim_dense, out_features=self.dim_out)
        )

    def forward(self, x, edge_index, edge_attr):
        # Local level processing
        local_features = self.local_model(x, edge_index, edge_attr)

        # Global level processing
        global_features = self.global_model(x, edge_index, edge_attr)

        # Fusion of local and global features
        fused_features = torch.cat([local_features, global_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        fused_features = self.nonlin(fused_features)

        # Final prediction
        output = self.prediction_layers(fused_features)

        return output