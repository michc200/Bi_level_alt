import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torch_geometric.nn import GCN, GAT
from robusttest.core.SE.pf_funcs import gsp_wls_edge

class GCN_DSSE_Lightning(pl.LightningModule):
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, model = 'GCN', time_info = True):
        super().__init__()
        self.save_hyperparameters()

        # Model hyperparameters
        dim_feat = hyperparameters['dim_nodes']
        dim_hidden = hyperparameters['dim_hid']
        dim_out = hyperparameters['dim_out']
        num_layers = hyperparameters['gcn_layers']
        dropout = hyperparameters['dropout_rate']

        # Initialize GCN model
        if model == 'GCN':
            self.model = GCN(in_channels=dim_feat, 
                             hidden_channels=dim_feat,
                             num_layers=num_layers,
                             out_channels=dim_out,
                             dropout=dropout)
        elif model == 'GAT':
            self.model = GAT(in_channels=dim_feat, 
                             hidden_channels=dim_feat,
                             num_layers=num_layers,
                             out_channels=dim_out,
                             dropout=dropout)
        else:
            raise ValueError("Model not recognized. Please choose 'GCN' or 'GAT'.")

        # Save other required parameters
        self.x_mean = torch.tensor(x_mean)
        self.x_std = torch.tensor(x_std)
        self.edge_mean = torch.tensor(edge_mean)
        self.edge_std = torch.tensor(edge_std)
        self.reg_coefs = reg_coefs
        self.num_nfeat = hyperparameters['num_nfeat']
        self.num_efeat = hyperparameters['dim_lines']
        self.lr = hyperparameters['lr']
        self.time_info = time_info

        # Loss tracker
        self.train_loss = []
        self.val_loss = []

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = batch.edge_attr
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]

        output = self(x_nodes, edge_index)
        x_nodes = x[:,:self.num_nfeat]
        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = batch.edge_attr
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:,:self.num_efeat]
        edge_param = edge_attr[:,self.num_efeat:]

        output = self(x_nodes, edge_index)
        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index

        x_nodes = x[:, :self.num_nfeat]
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        node_param = x[:,self.num_nfeat+3:]

        # Forward pass
        output = self(x_nodes, edge_index)
        
        v_i = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = output[:, 1:] * self.x_std[2:3] + self.x_mean[2:3] #
        theta_i *= (1.- node_param[:,1:2])

        return v_i, theta_i

    def calculate_loss(self, x, edge_input, output, edge_index, node_param, edge_param, num_samples):
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

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = Adamax(self.parameters(), lr=self.lr)

        # Set up a learning rate scheduler that reduces the LR if no progress is made
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'train_loss',  # Replace with your actual validation metric
        }

        return [optimizer], [scheduler]