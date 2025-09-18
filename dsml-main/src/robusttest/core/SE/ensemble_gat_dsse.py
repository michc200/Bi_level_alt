import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
import pytorch_lightning as pl
from robusttest.core.SE.gat_dsse import GAT_DSSE_Lightning
from robusttest.core.SE.pf_funcs import gsp_wls_edge
import torch_geometric

class EnsembleGAT_DSSE(pl.LightningModule):
    def __init__(self, hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs, train_dataset, time_info = True):
        super().__init__()
        self.save_hyperparameters(ignore=['train_dataset'])
        
        self.unique_setups = self._get_unique_setups(train_dataset)
        self.num_models = len(self.unique_setups)
        
        self.models = nn.ModuleList([
            GAT_DSSE_Lightning(hyperparameters, x_mean, x_std, edge_mean, edge_std, reg_coefs)
            for _ in range(self.num_models)
        ])
        
        # Other parameters and initializations
        self.x_mean = torch.tensor(x_mean)
        self.x_std = torch.tensor(x_std)
        self.edge_mean = torch.tensor(edge_mean)
        self.edge_std = torch.tensor(edge_std)
        self.reg_coefs = reg_coefs
        self.num_nfeat = hyperparameters['num_nfeat']
        self.num_efeat = hyperparameters['dim_lines']
        self.lr = hyperparameters['lr']
        self.time_info = time_info

        self.train_dataset = train_dataset

    def _get_unique_setups(self, dataset):
        unique_setups = {}
        for data in dataset:
            setup_key = frozenset(self._get_edge_set(data.edge_index))
            if setup_key not in unique_setups:
                unique_setups[setup_key] = len(unique_setups)
        return unique_setups
    
    @staticmethod
    def _get_edge_set(edge_index):
        return set(frozenset(edge) for edge in edge_index.t().tolist())

    def _calculate_similarity(self, setup_key1, setup_key2):
        edge_set1 = self._get_edge_set(setup_key1)
        edge_set2 = self._get_edge_set(setup_key2)
        # Calculate Jaccard similarity
        intersection = len(edge_set1.intersection(edge_set2))
        union = len(edge_set1.union(edge_set2))
        return intersection / union
    
    def setup_key_collate(self, batch):
        setup_keys = {}
        for data in batch:
            setup_key = frozenset(self._get_edge_set(data.edge_index))
            if setup_key not in setup_keys:
                setup_keys[setup_key] = []
            setup_keys[setup_key].append(data)
        
        sorted_batches = []
        for key, data_list in setup_keys.items():
            sorted_batches.append(torch_geometric.data.Batch.from_data_list(data_list))
        
        return sorted_batches

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=self.setup_key_collate
        )

    def training_step(self, batch, batch_idx):
        # all elements in a batch should have the same setup_key
        edge_index = batch[0].edge_index[:, batch[0].batch[batch[0].edge_index[0]] == 0]
        setup_key = frozenset(self._get_edge_set(edge_index))
        model_index = self.unique_setups[setup_key]
        
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]

        output = self.models[model_index](x_nodes, edge_index, edge_input)
        x_nodes = x[:,:self.num_nfeat]
        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        num_samples = batch.batch[-1] + 1
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:, :self.num_efeat]
        edge_param = edge_attr[:, self.num_efeat:]

        # Forward pass
        output = self.forward(x_nodes, edge_index, edge_input)
        x_nodes = x[:,:self.num_nfeat]
        
        loss = self.calculate_loss(x_nodes, edge_input, output, edge_index, node_param, edge_param, num_samples)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        x_nodes = x[:,:self.num_nfeat]
        node_param = x[:,self.num_nfeat:self.num_nfeat+3]
        if self.time_info:
            time_info = x[:,self.num_nfeat+3:]
            x_nodes = torch.cat([x_nodes, time_info], dim=1)
        edge_input = edge_attr[:, :self.num_efeat]

        # Forward pass
        output = self.forward(x_nodes, edge_index, edge_input)

        v_i = output[:,0:1] * self.x_std[:1] + self.x_mean[:1]
        theta_i = output[:, 1:] * self.x_std[2:3]  + self.x_mean[2:3] #* self.x_std[2:3] + self.x_mean[2:3]
        theta_i *= (1.- node_param[:,1:2])

        return v_i, theta_i

    def forward(self, x, edge_index, edge_attr):
        setup_key = frozenset(self._get_edge_set(edge_index))
        similarities = [self._calculate_similarity(setup_key, key) for key in self.unique_setups.keys()]
        
        similarities = similarities - torch.min(similarities)
        total_similarity = sum(similarities)
        weights = [sim / total_similarity for sim in similarities]
        
        weights = torch.tensor(weights)
        
        outputs = [model(x, edge_index, edge_attr) for model in self.models]
        return sum(w * out for w, out in zip(weights, outputs))

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
    