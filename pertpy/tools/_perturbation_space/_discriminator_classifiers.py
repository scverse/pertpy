from __future__ import annotations

import warnings

import anndata
import numpy as np
import pandas as pd
import scipy
import torch
from anndata import AnnData
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class LRClassifierSpace(PerturbationSpace):
    """Fits a logistic regression model to the data and takes the feature space as embedding.

    We fit one logistic regression model per perturbation. After training, the coefficients of the logistic regression
    model are used as the feature space. This results in one embedding per perturbation.
    """

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        layer_key: str = None,
        embedding_key: str = None,
        test_split_size: float = 0.2,
        max_iter: int = 1000,
    ):
        """Fits a logistic regression model to the data and takes the coefficients of the logistic regression model as perturbation embedding.

        Args:
            adata: AnnData object of size cells x genes
            target_col: .obs column that stores the perturbations.
            layer_key: Layer in adata to use.
            embedding_key: Key of the embedding in obsm to be used as data for the logistic regression classifier.
                Can only be specified if layer_key is None.
            test_split_size: Fraction of data to put in the test set.
            max_iter: Maximum number of iterations taken for the solvers to converge.

        Returns:
            AnnData object with the logistic regression coefficients as the embedding in X and the perturbations as .obs['perturbations'].

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> rcs = pt.tl.LRClassifierSpace()
            >>> pert_embeddings = rcs.compute(adata, embedding_key="X_pca", target_col="perturbation_name")
        """
        if layer_key is not None and layer_key not in adata.obs.columns:
            raise ValueError(f"Layer key {layer_key} not found in adata.")

        if embedding_key is not None and embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key {embedding_key} not found in adata.obsm.")

        if layer_key is not None and embedding_key is not None:
            raise ValueError("Cannot specify both layer_key and embedding_key.")

        if target_col not in adata.obs:
            raise ValueError(f"Column {target_col!r} does not exist in the .obs attribute.")

        if layer_key is not None:
            regression_data = adata.layers[layer_key]
        elif embedding_key is not None:
            regression_data = adata.obsm[embedding_key]
        else:
            regression_data = adata.X

        regression_labels = adata.obs[target_col]

        # Save adata observations for embedding annotations in get_embeddings
        adata_obs = adata.obs.reset_index(drop=True)
        adata_obs = adata_obs.groupby(target_col).agg(
            lambda pert_group: np.nan if len(set(pert_group)) != 1 else list(set(pert_group))[0]
        )

        # Fit a logistic regression model for each perturbation
        regression_model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        regression_embeddings = {}
        regression_scores = {}

        for perturbation in regression_labels.unique():
            labels = np.where(regression_labels == perturbation, 1, 0)
            X_train, X_test, y_train, y_test = train_test_split(
                regression_data, labels, test_size=test_split_size, stratify=labels
            )

            regression_model.fit(X_train, y_train)
            regression_embeddings[perturbation] = regression_model.coef_
            regression_scores[perturbation] = regression_model.score(X_test, y_test)

        # Save the regression embeddings and scores in an AnnData object
        pert_adata = AnnData(X=np.array(list(regression_embeddings.values())).squeeze())
        pert_adata.obs["perturbations"] = list(regression_embeddings.keys())
        pert_adata.obs["classifier_score"] = list(regression_scores.values())

        # Save adata observations for embedding annotations
        for obs_name in adata_obs.columns:
            if not adata_obs[obs_name].isnull().values.any():
                pert_adata.obs[obs_name] = pert_adata.obs["perturbations"].map(
                    {pert: adata_obs.loc[pert][obs_name] for pert in adata_obs.index}
                )

        return pert_adata


# Ensure backward compatibility with DiscriminatorClassifierSpace
def DiscriminatorClassifierSpace():
    warnings.warn(
        "The DiscriminatorClassifierSpace class is deprecated and will be removed in the future."
        "Please use the MLPClassifierSpace or the LRClassifierSpace class instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return MLPClassifierSpace()


class MLPClassifierSpace(PerturbationSpace):
    """Fits an ANN classifier to the data and takes the feature space (weights in the last layer) as embedding.

    We train the ANN to classify the different perturbations. After training, the penultimate layer is used as the
    feature space, resulting in one embedding per cell. Consider employing the PseudoBulk or another PerturbationSpace
    to obtain one embedding per perturbation.

    See here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19.
    """

    def compute(  # type: ignore
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
        max_epochs: int = 20,
        val_epochs_check: int = 2,
        patience: int = 2,
    ) -> AnnData:
        """Creates cell embeddings by training a MLP classifier model to distinguish between perturbations.

        A model is created using the specified parameters (hidden_dim, dropout, batch_norm). Further parameters such as
        the number of classes to predict (number of perturbations) are obtained from the provided AnnData object directly.
        Dataloaders that take into account class imbalances are created. Next, the model is trained and tested, using the
        GPU if available. The embeddings are obtained by passing the data through the model and extracting the values in
        the last layer of the MLP. You will get one embedding per cell, so be aware that you might need to apply another
        perturbation space to aggregate the embeddings per perturbation.

        Args:
            adata: AnnData object of size cells x genes
            target_col: .obs column that stores the perturbations.
            layer_key: Layer in adata to use.
            hidden_dim: List of number of neurons in each hidden layers of the neural network. For instance, [512, 256]
                will create a neural network with two hidden layers, the first with 512 neurons and the second with 256 neurons.
            dropout: Amount of dropout applied, constant for all layers.
            batch_norm: Whether to apply batch normalization.
            batch_size: The batch size, i.e. the number of datapoints to use in one forward/backward pass.
            test_split_size: Fraction of data to put in the test set. Default to 0.2.
            validation_split_size: Fraction of data to put in the validation set of the resultant train set.
                E.g. a test_split_size of 0.2 and a validation_split_size of 0.25 means that 25% of 80% of the data
                will be used for validation.
            max_epochs: Maximum number of epochs for training.
            val_epochs_check: Test performance on validation dataset after every val_epochs_check training epochs.
                Note that this affects early stopping, as the model will be stopped if the validation performance does not
                improve for patience epochs.
            patience: Number of validation performance checks without improvement, after which the early stopping flag
                is activated and training is therefore stopped.

        Returns:
            AnnData whose `X` attribute is the perturbation embedding and whose .obs['perturbations'] are the names of the perturbations.
            The AnnData will have shape (n_cells, n_features) where n_features is the number of features in the last layer of the MLP.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.norman_2019()
            >>> dcs = pt.tl.MLPClassifierSpace()
            >>> cell_embeddings = dcs.compute(adata, target_col="perturbation_name")
        """
        if layer_key is not None and layer_key not in adata.obs.columns:
            raise ValueError(f"Layer key {layer_key} not found in adata.")

        if target_col not in adata.obs:
            raise ValueError(f"Column {target_col!r} does not exist in the .obs attribute.")

        if hidden_dim is None:
            hidden_dim = [512]

        # Labels are strings, one hot encoding for classification
        n_classes = len(adata.obs[target_col].unique())
        labels = adata.obs[target_col].values.reshape(-1, 1)
        encoder = OneHotEncoder()
        encoded_labels = encoder.fit_transform(labels).toarray()
        adata.obs["encoded_perturbations"] = [np.float32(label) for label in encoded_labels]

        # Split the data in train, test and validation
        X = list(range(adata.n_obs))
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
        train_weights = 1 / (1 + torch.sum(torch.tensor(train_dataset.labels), dim=1))
        train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Define the network
        sizes = [adata.n_vars] + hidden_dim + [n_classes]
        self.net = MLP(sizes=sizes, dropout=dropout, batch_norm=batch_norm)

        # Define a dataset that gathers all the data and dataloader for getting embeddings
        total_dataset = PLDataset(
            adata=adata, target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        self.entire_dataset = DataLoader(total_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=0)

        # Save adata observations for embedding annotations in get_embeddings
        self.adata_obs = adata.obs.reset_index(drop=True)

        self.trainer = Trainer(
            min_epochs=1,
            max_epochs=max_epochs,
            check_val_every_n_epoch=val_epochs_check,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=patience)],
            devices="auto",
            accelerator="auto",
        )

        self.mlp = PerturbationClassifier(model=self.net, batch_size=self.train_dataloader.batch_size)

        self.trainer.fit(model=self.mlp, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        self.trainer.test(model=self.mlp, dataloaders=self.test_dataloader)

        # Obtain cell embeddings
        with torch.no_grad():
            self.mlp.eval()
            for dataset_count, batch in enumerate(self.entire_dataset):
                emb, y = self.mlp.get_embeddings(batch)
                emb = torch.squeeze(emb)
                batch_adata = AnnData(X=emb.cpu().numpy())
                batch_adata.obs["perturbations"] = y
                if dataset_count == 0:
                    pert_adata = batch_adata
                else:
                    pert_adata = batch_adata if dataset_count == 0 else anndata.concat([pert_adata, batch_adata])

        # Add .obs annotations to the pert_adata. Because shuffle=False and num_workers=0, the order of the data is stable
        # and we can just add the annotations from the original AnnData object
        pert_adata.obs = pert_adata.obs.reset_index(drop=True)
        if "perturbations" in self.adata_obs.columns:
            self.adata_obs = self.adata_obs.drop("perturbations", axis=1)
        pert_adata.obs = pd.concat([pert_adata.obs, self.adata_obs], axis=1)

        # Drop the 'encoded_perturbations' colums, since this stores the one-hot encoded labels as numpy arrays,
        # which would cause errors in the downstream processing of the AnnData object (e.g. when plotting)
        pert_adata.obs = pert_adata.obs.drop("encoded_perturbations", axis=1)

        return pert_adata

    def load(self, adata, **kwargs):
        """This method is deprecated and will be removed in the future. Please use the compute method instead."""
        raise DeprecationWarning(
            "The load method is deprecated and will be removed in the future. Please use the compute method instead."
        )

    def train(self, **kwargs):
        """This method is deprecated and will be removed in the future. Please use the compute method instead."""
        raise DeprecationWarning(
            "The train method is deprecated and will be removed in the future. Please use the compute method instead."
        )

    def get_embeddings(self, **kwargs):
        """This method is deprecated and will be removed in the future. Please use the compute method instead."""
        raise DeprecationWarning(
            "The get_embeddings method is deprecated and will be removed in the future. Please use the compute method instead."
        )


class MLP(torch.nn.Module):
    """A multilayer perceptron with ReLU activations, optional Dropout and optional BatchNorm."""

    def __init__(
        self,
        sizes: list[int],
        dropout: float = 0.0,
        batch_norm: bool = True,
        layer_norm: bool = False,
        last_layer_act: str = "linear",
    ) -> None:
        """Multilayer perceptron with ReLU activations, optional Dropout and optional BatchNorm.

        Args:
            sizes: size of layers.
            dropout: Dropout probability.
            batch_norm: specifies if batch norm should be applied.
            layer_norm:  specifies if layer norm should be applied, as commonly used in Transformers.
            last_layer_act: activation function of last layer.
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
    """Dataset for perturbation classification.

    Needed for training a model that classifies the perturbed cells and takes as perturbation embedding the second to last layer.
    """

    def __init__(
        self,
        adata: np.array,
        target_col: str = "perturbations",
        label_col: str = "perturbations",
        layer_key: str = None,
    ):
        """PyTorch lightning Dataset for perturbation classification.

        Args:
            adata: AnnData object with observations and labels.
            target_col: key with the perturbation labels numerically encoded.
            label_col: key with the perturbation labels.
            layer_key: key of the layer to be used as data, otherwise .X.
        """
        if layer_key:
            self.data = adata.layers[layer_key]
        else:
            self.data = adata.X

        self.labels = adata.obs[target_col]
        self.pert_labels = adata.obs[label_col]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Returns a sample and corresponding perturbations applied (labels)."""
        sample = self.data[idx].toarray().squeeze() if scipy.sparse.issparse(self.data) else self.data[idx]
        num_label = self.labels.iloc[idx]
        str_label = self.pert_labels.iloc[idx]

        return sample, num_label, str_label


class PerturbationClassifier(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        layers: list = [512],  # noqa
        dropout: float = 0.0,
        batch_norm: bool = True,
        layer_norm: bool = False,
        last_layer_act: str = "linear",
        lr=1e-4,
        seed=42,
    ):
        """Perturbation Classifier.

        Args:
            model: model to be trained
            batch_size: batch size
            layers: list of layers of the MLP
            dropout: dropout probability
            batch_norm: whether to apply batch norm
            layer_norm: whether to apply layer norm
            last_layer_act: activation function of last layer
            lr: learning rate
            seed: random seed.
        """
        super().__init__()
        self.batch_size = batch_size
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            Network output tensor
        """
        x = self.net(x)
        return x

    def configure_optimizers(self) -> optim.Adam:
        """Configure optimizer for the model.

        Returns:
            Adam optimizer with weight decay
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        return optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a training step.

        Args:
            batch: Tuple of (input, target, metadata)
            batch_idx: Index of the current batch

        Returns:
            Loss value
        """
        x, y, _ = batch
        x = x.to(torch.float32)

        y_hat = self.forward(x)

        y = torch.argmax(y, dim=1)
        y_hat = y_hat.squeeze()

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a validation step.

        Args:
            batch: Tuple of (input, target, metadata)
            batch_idx: Index of the current batch

        Returns:
            Loss value
        """
        x, y, _ = batch
        x = x.to(torch.float32)

        y_hat = self.forward(x)

        y = torch.argmax(y, dim=1)
        y_hat = y_hat.squeeze()

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a test step.

        Args:
            batch: Tuple of (input, target, metadata)
            batch_idx: Index of the current batch

        Returns:
            Loss value
        """
        x, y, _ = batch
        x = x.to(torch.float32)

        y_hat = self.forward(x)

        y = torch.argmax(y, dim=1)
        y_hat = y_hat.squeeze()

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True, batch_size=self.batch_size)

        return loss

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input features.

        Args:
            x: Input tensor of shape [Batch, SeqLen, 1]

        Returns:
            Embedded representation of the input
        """
        x = self.net.embedding(x)
        return x

    def get_embeddings(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract embeddings from a batch.

        Args:
            batch: Tuple of (input, target, metadata)

        Returns:
            Tuple of (embeddings, metadata)
        """
        x, _, y = batch
        x = x.to(torch.float32)

        embedding = self.embedding(x)
        return embedding, y
