import torch
import scipy
import numpy as np
from torch import optim
from anndata import AnnData
import pytorch_lightning as pl
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset

class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations, optional Dropout and optional BatchNorm.
    """

    def __init__(self, sizes: List[int], dropout: int = 0.0, batch_norm: bool = True, layer_norm: bool = False, last_layer_act: str = "linear", device: torch.device = 'cpu') -> None:
        """
        Args:
            sizes (list): size of layers
            dropout (int, optional): Dropout probability. Defaults to 0.0.
            batch_norm (bool, optional): batch norm. Defaults to True.
            layer_norm (bool, optional): layern norm, common in Transformers
            last_layer_act (str, optional): activation function of last layer. Defaults to "linear".
            device (torch.device, optional)
        Raises:
            ValueError: _description_
        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                torch.nn.LayerNorm(sizes[s + 1])
                if layer_norm and s < len(sizes) - 2 and not batch_norm
                else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
                if s < len(sizes) - 2
                else None
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)
        
        self.network.apply(init_weights)
        self.device = device
        self.to(device)
        
        self.sizes = sizes
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.last_layer_act = last_layer_act
        print(self.network, flush=True)

    def forward(self, x) -> torch.Tensor:
        if self.activation == 'ReLU':
            return self.relu(self.network(x)) 
        return self.network(x)
    
    def embedding(self, x) -> torch.Tensor:
        for layer in self.network[:-1]:
            x = layer(x)
        return x
    
    
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        
class PerturbationDataset(Dataset):
    """
    Dataset for perturbation classification. Needed for training a model that classifies the perturbe cells and takes as pert. embedding the second to last layer.
    """
    def __init__(self, adata: np.array, target_col: str = "perturbations", label_col: str = "perturbations", layer_key: str = None,):
        """
        Args:
            adata (anndata): anndata with observations and labels
            target_col (str, Optional): key with the perturbation labels numerically encoded
            label_col (str, Optional): key with the perturbation labels
            layer_key (str, Optional): key of the layer to be used as data, otherwise .X
        """
        
        if layer_key:
            self.data = adata.layers[layer_key]
        else:
            self.data = adata.X
            
        self.labels = adata.obs[target_col]
        self.pert_labels = adata.obs[label_col]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a sample and corresponding perturbations applied (labels)
        """

        sample = self.data[idx].A if scipy.sparse.issparse(self.data) else self.data[idx]
        num_label = self.labels[idx]
        str_label = self.pert_labels[idx]

        return sample, num_label, str_label
        
        
class PerturbationClassifier(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, layers: list = [512], dropout: int = 0.0, batch_norm: bool = True, layer_norm: bool = False, last_layer_act: str = "linear", lr=1e-4, seed=42):
        """
        Inputs:
            layers - list: layers of the MLP
        """
        super().__init__()
        self.save_hyperparameters()
        if model:
            self.net = model
        else:
            self._create_model()

    def _create_model(self):
        self.net = MLP(sizes=self.hparams.layers, dropout=self.hparams.dropout,
                       batch_norm=self.hparams.batch_norm, layer_norm=self.hparams.layer_norm,
                       last_layer_act=self.hparams.last_layer_act)
        
        print(self.net, flush=True)

    def forward(self, x):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, 1] 
        """
        x = self.net(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)

        return optimizer

    def training_step(self, batch, batch_idx):

        x, y, _ = batch
        x = x.to(torch.float32)
        y = y.to(torch.long)
       
        y_hat = self.forward(x)
        
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        
        x, y, _ = batch
        x = x.to(torch.float32)
        y = y.to(torch.long)
       
        y_hat = self.forward(x)
        
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        
        x, y, _ = batch
        x = x.to(torch.float32)
        y = y.to(torch.long)
       
        y_hat = self.forward(x)
        
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        
        return loss
    
    def embedding(self, x):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, 1] 
        """
        x = self.net.embedding(x)
        return x
    
    def get_embeddings(self, batch):

        x, _, y = batch
        x = x.to(torch.float32)
       
        embedding = self.embedding(x)
        return embedding, y