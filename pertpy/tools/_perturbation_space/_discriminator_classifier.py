from __future__ import annotations

from typing import TYPE_CHECKING

import anndata
import pytorch_lightning as pl
import scipy
import torch
from anndata import AnnData
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace

if TYPE_CHECKING:
    import numpy as np


class DiscriminatorClassifierSpace(PerturbationSpace):
    """Leveraging discriminator classifier. Fit a regressor model to the data and take the feature space.

    See here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19.
    We use either the coefficients of the model for each perturbation as a feature or train a classifier example
    (simple MLP or logistic regression) and take the penultimate layer as feature space and apply pseudobulking approach.
    """

    def load(  # type: ignore
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        layer_key: str = None,
        hidden_dim: list[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        batch_size: int = 256,
        test_split_size: float = 0.2,
        validation_split_size: float = 0.25,
    ):
        """Creates a neural network model using the specified parameters (hidden_dim, dropout, batch_norm). Further
         parameters such as the number of classes to predict (number of perturbations) are obtained from the provided
         AnnData object directly.

        It further creates dataloaders and fixes class imbalance due to control.
        Sets the device to a GPU if available.

        Args:
            adata: AnnData object of size cells x genes
            target_col: .obs column that stores the perturbations. Defaults to "perturbations".
            layer_key: Layer in adata to use. Defaults to None.
            hidden_dim: list of hidden layers of the neural network. For instance: [512, 256].
            dropout: amount of dropout applied, constant for all layers. Defaults to 0.
            batch_norm: Whether to apply batch normalization. Defaults to True.
            batch_size: The batch size, i.e. the number of datapoints to use in one forward/backward pass. Defaults to 256.
            test_split_size: Fraction of data to put in the test set. Default to 0.2.
            validation_split_size: Fraction of data to put in the validation set of the resultant train set.
                E.g. a test_split_size of 0.2 and a validation_split_size of 0.25 means that 25% of 80% of the data
                will be used for validation. Defaults to 0.25.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.papalexi_2021()['rna']
            >>> dcs = pt.tl.DiscriminatorClassifierSpace()
            >>> dcs.load(adata, target_col="gene_target")
        """
        if layer_key is not None and layer_key not in adata.obs.columns:
            raise ValueError(f"Layer key {layer_key} not found in adata. {layer_key}")

        if target_col not in adata.obs:
            raise ValueError(f"Column {target_col!r} does not exist in the .obs attribute.")

        if hidden_dim is None:
            hidden_dim = [512]

        # Labels are strings, one hot encoding for classification
        n_classes = len(adata.obs[target_col].unique())
        labels = adata.obs[target_col]
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        adata.obs["encoded_perturbations"] = encoded_labels

        # Split the data in train, test and validation
        X = list(range(0, adata.n_obs))
        y = adata.obs[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split_size, stratify=y_train
        )

        train_dataset = PLDataset(
            adata=adata[X_train], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        val_dataset = PLDataset(
            adata=adata[X_val], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        test_dataset = PLDataset(
            adata=adata[X_test], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )  # we don't need to pass y_test since the label selection is done inside

        # Fix class unbalance (likely to happen in perturbation datasets)
        # Usually control cells are overrepresented such that predicting control all time would give good results
        # Cells with rare perturbations are sampled more
        class_weights = 1.0 / torch.bincount(torch.tensor(train_dataset.labels.values))
        train_weights = class_weights[train_dataset.labels]
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        # Define the network
        sizes = [adata.n_vars] + hidden_dim + [n_classes]
        self.net = MLP(sizes=sizes, dropout=dropout, batch_norm=batch_norm)

        # Define a dataset that gathers all the data and dataloader for getting embeddings
        total_dataset = PLDataset(
            adata=adata, target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        self.entire_dataset = DataLoader(total_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4)

        return self

    def train(self, max_epochs: int = 40, val_epochs_check: int = 5, patience: int = 2):
        """Trains and tests the neural network model defined in the load step.

        Args:
            max_epochs: max epochs for training. Default to 40.
            val_epochs_check: test performance on validation dataset after every val_epochs_check training epochs.
            patience: number of validation performance checks without improvement, after which the early stopping flag
                is activated and training is therefore stopped.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.papalexi_2021()['rna']
            >>> dcs = pt.tl.DiscriminatorClassifierSpace()
            >>> dcs.load(adata, target_col="gene_target")
            >>> dcs.train(max_epochs=5)
        """
        self.trainer = pl.Trainer(
            min_epochs=1,
            max_epochs=max_epochs,
            check_val_every_n_epoch=val_epochs_check,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience)],
            devices="auto",
            accelerator="auto",
        )

        self.model = PerturbationClassifier(model=self.net)

        self.trainer.fit(
            model=self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader
        )
        self.trainer.test(model=self.model, dataloaders=self.test_dataloader)

    def get_embeddings(self) -> AnnData:
        """Obtain the embeddings of the data, i.e., the values in the last layer of the MLP.

        Returns:
            AnnData whose `X` attribute is the perturbation embedding and whose .obs['perturbations'] are the names of the perturbations.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.papalexi_2021()['rna']
            >>> dcs = pt.tl.DiscriminatorClassifierSpace()
            >>> dcs.load(adata, target_col="gene_target")
            >>> dcs.train()
            >>> embeddings = dcs.get_embeddings()
        """
        with torch.no_grad():
            self.model.eval()
            for dataset_count, batch in enumerate(self.entire_dataset):
                emb, y = self.model.get_embeddings(batch)
                batch_adata = AnnData(X=emb.cpu().numpy())
                batch_adata.obs["perturbations"] = y
                if dataset_count == 0:
                    pert_adata = batch_adata
                else:
                    pert_adata = anndata.concat([pert_adata, batch_adata])

        return pert_adata


class MLP(torch.nn.Module):
    """
    A multilayer perceptron with ReLU activations, optional Dropout and optional BatchNorm.
    """

    def __init__(
        self,
        sizes: list[int],
        dropout: float = 0.0,
        batch_norm: bool = True,
        layer_norm: bool = False,
        last_layer_act: str = "linear",
    ) -> None:
        """
        Args:
            sizes: size of layers.
            dropout: Dropout probability. Defaults to 0.0.
            batch_norm: specifies if batch norm should be applied. Defaults to True.
            layer_norm:  specifies if layer norm should be applied, as commonly used in Transformers. Defaults to False.
            last_layer_act: activation function of last layer. Defaults to "linear".
        """
        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1]) if batch_norm and s < len(sizes) - 2 else None,
                torch.nn.LayerNorm(sizes[s + 1]) if layer_norm and s < len(sizes) - 2 and not batch_norm else None,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout) if s < len(sizes) - 2 else None,
            ]

        layers = [layer for layer in layers if layer is not None][:-1]
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = torch.nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        self.network = torch.nn.Sequential(*layers)

        self.network.apply(init_weights)

        self.sizes = sizes
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.last_layer_act = last_layer_act

    def forward(self, x) -> torch.Tensor:
        if self.activation == "ReLU":
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


class PLDataset(Dataset):
    """
    Dataset for perturbation classification.
    Needed for training a model that classifies the perturbed cells and takes as perturbation embedding the second to last layer.
    """

    def __init__(
        self,
        adata: np.array,
        target_col: str = "perturbations",
        label_col: str = "perturbations",
        layer_key: str = None,
    ):
        """
        Args:
            adata: AnnData object with observations and labels.
            target_col: key with the perturbation labels numerically encoded. Defaults to 'perturbations'.
            label_col: key with the perturbation labels. Defaults to 'perturbations'.
            layer_key: key of the layer to be used as data, otherwise .X
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
        """Returns a sample and corresponding perturbations applied (labels)"""

        sample = self.data[idx].A if scipy.sparse.issparse(self.data) else self.data[idx]
        num_label = self.labels[idx]
        str_label = self.pert_labels[idx]

        return sample, num_label, str_label


class PerturbationClassifier(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        layers: list = [512],  # noqa
        dropout: float = 0.0,
        batch_norm: bool = True,
        layer_norm: bool = False,
        last_layer_act: str = "linear",
        lr=1e-4,
        seed=42,
    ):
        """
        Args:
            layers: list of layers of the MLP
            dropout: dropout probability
            batch_norm: whether to apply batch norm
            layer_norm: whether to apply layer norm
            last_layer_act: activation function of last layer
            lr: learning rate
            seed: random seed
        """
        super().__init__()
        self.save_hyperparameters()
        if model:
            self.net = model
        else:
            self._create_model()

    def _create_model(self):
        self.net = MLP(
            sizes=self.hparams.layers,
            dropout=self.hparams.dropout,
            batch_norm=self.hparams.batch_norm,
            layer_norm=self.hparams.layer_norm,
            last_layer_act=self.hparams.last_layer_act,
        )

    def forward(self, x):
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
            x: Input features of shape [Batch, SeqLen, 1]
        """
        x = self.net.embedding(x)
        return x

    def get_embeddings(self, batch):
        x, _, y = batch
        x = x.to(torch.float32)

        embedding = self.embedding(x)
        return embedding, y
