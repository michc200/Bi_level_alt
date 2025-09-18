import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch_geometric.nn import MLP
from robusttest.core.SE.pf_funcs import gsp_wls_edge
from torch_geometric.utils import to_dense_batch
from torch.nn.functional import mse_loss  # Import MSE loss function

class MLP_DSSE_Lightning(pl.LightningModule):
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, time_info=True, use_mse_loss=False):
        super().__init__()
        self.save_hyperparameters()

        # Model hyperparameters
        dim_feat = hyperparameters['dim_nodes']
        dim_hidden = hyperparameters['dim_hid']
        self.dim_out = hyperparameters['dim_out']
        num_layers = hyperparameters['mlp_layers']
        num_nodes = hyperparameters['num_nodes']
        dropout = hyperparameters['dropout_rate']

        # Initialize MLP model
        self.model = MLP(in_channels=dim_feat*num_nodes, 
                         hidden_channels=dim_hidden,
                         out_channels=self.dim_out*num_nodes,
                         num_layers=num_layers,
                         dropout=dropout,
                         act='leaky_relu')

        # Save other required parameters
        self.x_mean = torch.tensor(x_mean)
        self.x_std = torch.tensor(x_std)
        self.edge_mean = torch.tensor(edge_mean)
        self.edge_std = torch.tensor(edge_std)
        self.reg_coefs = reg_coefs
        self.num_nfeat = hyperparameters['num_nfeat']
        self.num_efeat = hyperparameters['dim_lines']
        self.num_nodes = hyperparameters['num_nodes']
        self.lr = hyperparameters['lr']
        self.time_info = time_info
        self.use_mse_loss = use_mse_loss  # Flag to toggle MSE loss

        # Loss tracker
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """One training step."""
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        y = batch.y  # Target values for MSE loss

        x_nodes = x[:, :self.num_nfeat]
        node_param = x[:, self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:, self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]

        x_nodes_padded, mask = to_dense_batch(x_nodes, batch.batch)  # Shape: [num_samples, max_num_nodes, num_features]

        # Flatten node features for MLP
        x_mlp_input = x_nodes_padded.view(x_nodes_padded.size(0), -1)  # Shape: [num_samples, max_num_nodes * num_features]

        output = self(x_mlp_input).reshape(self.num_nodes*num_samples, self.dim_out)

        x_nodes = x[:, :self.num_nfeat]

        # Calculate loss based on the selected method
        if self.use_mse_loss:
            output[:,0:1] = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
            output[:, 1:] = output[:, 1:] * self.x_std[2:3] + self.x_mean[2:3] #* self.x_std[2:3] + self.x_mean[2:3]
            output[:, 1:] *= (1.- node_param[:,1:2])
            loss = mse_loss(output, y)  # MSE loss with batch.y as target
        else:
            loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """One validation step."""
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        y = batch.y  # Target values for MSE loss

        x_nodes = x[:, :self.num_nfeat]
        node_param = x[:, self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:, self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]

        x_nodes_padded, mask = to_dense_batch(x_nodes, batch.batch)  # Shape: [num_samples, max_num_nodes, num_features]

        # Flatten node features for MLP
        x_mlp_input = x_nodes_padded.view(x_nodes_padded.size(0), -1)  # Shape: [num_samples, max_num_nodes * num_features]

        output = self(x_mlp_input).reshape(self.num_nodes*num_samples, self.dim_out)
        
        x_nodes = x[:, :self.num_nfeat]

        # Calculate loss based on the selected method
        if self.use_mse_loss:
            output[:,0:1] = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
            output[:, 1:] = output[:, 1:] * self.x_std[2:3] + self.x_mean[2:3] #* self.x_std[2:3] + self.x_mean[2:3]
            output[:, 1:] *= (1.- node_param[:,1:2])
            loss = mse_loss(output, y)  # MSE loss with batch.y as target
        else:
            loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    
    def predict_step(self, batch, batch_idx):
        x = batch.x

        x_nodes = x[:, :self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)

        # Reshape x_nodes to fit MLP input
        x_nodes_padded, mask = to_dense_batch(x_nodes, batch.batch)  # Shape: [num_samples, max_num_nodes, num_features]

        # Flatten node features for MLP
        x_mlp_input = x_nodes_padded.view(x_nodes_padded.size(0), -1)  # Shape: [num_samples, max_num_nodes * num_features]

        # Forward pass
        output = self(x_mlp_input).reshape(self.num_nodes*num_samples, self.dim_out)
        
        v_i = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = output[:, 1:] * self.x_std[2:3] + self.x_mean[2:3] #* self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1.- node_param[:,1:2])

        return v_i, theta_i

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = Adamax(self.parameters(), lr=self.lr)

        # Set up a learning rate scheduler that reduces the LR if no progress is made
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'train_loss',  # Replace with your actual validation metric
        }

        return [optimizer], [scheduler]

    def calculate_loss(self, x, edge_input, output, edge_index, node_param, edge_param, num_samples):
        """Custom loss function."""
        return gsp_wls_edge(
            input=x,
            edge_input=edge_input,
            output=output,
            x_mean=self.x_mean,
            x_std=self.x_std,
            edge_mean=self.edge_mean,
            edge_std=self.edge_std,
            edge_index=edge_index,
            reg_coefs=self.reg_coefs,
            num_samples=num_samples,
            node_param=node_param,
            edge_param=edge_param
        )
