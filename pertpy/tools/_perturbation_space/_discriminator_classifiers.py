from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import scipy
from anndata import AnnData
from fast_array_utils.conv import to_dense
from flax.training import train_state
from jax import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

        adata_obs = adata.obs.reset_index(drop=True)
        adata_obs = adata_obs.groupby(target_col).agg(
            lambda pert_group: np.nan if len(set(pert_group)) != 1 else list(set(pert_group))[0]
        )

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

        pert_adata = AnnData(X=np.array(list(regression_embeddings.values())).squeeze())
        pert_adata.obs["perturbations"] = list(regression_embeddings.keys())
        pert_adata.obs["classifier_score"] = list(regression_scores.values())

        for obs_name in adata_obs.columns:
            if not adata_obs[obs_name].isnull().values.any():
                pert_adata.obs[obs_name] = pert_adata.obs["perturbations"].map(
                    {pert: adata_obs.loc[pert][obs_name] for pert in adata_obs.index}
                )

        return pert_adata


class MLP(nn.Module):
    """A multilayer perceptron with ReLU activations, optional Dropout and optional BatchNorm."""

    sizes: list[int]
    dropout: float = 0.0
    batch_norm: bool = True
    layer_norm: bool = False
    last_layer_act: str = "linear"

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        for i in range(len(self.sizes) - 1):
            x = nn.Dense(self.sizes[i + 1])(x)

            if i < len(self.sizes) - 2:
                if self.batch_norm:
                    x = nn.BatchNorm(use_running_average=not training)(x)
                elif self.layer_norm:
                    x = nn.LayerNorm()(x)

                x = nn.relu(x)

                if self.dropout > 0 and training:
                    x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)

        if self.last_layer_act == "ReLU":
            x = nn.relu(x)

        return x

    @nn.compact
    def embedding(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i in range(len(self.sizes) - 2):
            x = nn.Dense(self.sizes[i + 1])(x)

            if self.batch_norm:
                x = nn.BatchNorm(use_running_average=True)(x)
            elif self.layer_norm:
                x = nn.LayerNorm()(x)

            x = nn.relu(x)

            if self.dropout > 0 and training:
                x = nn.Dropout(rate=self.dropout, deterministic=True)(x)

        return x


class TrainState(train_state.TrainState):
    batch_stats: Any


def create_train_state(rng: jnp.ndarray, model: nn.Module, input_shape: tuple[int, ...], lr: float) -> TrainState:
    dummy_input = jnp.ones((1,) + input_shape)
    rng, init_rng, dropout_rng = random.split(rng, 3)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input, training=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    tx = optax.adamw(learning_rate=lr, weight_decay=0.1)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)


@jax.jit
def train_step(state: TrainState, batch: tuple[jnp.ndarray, jnp.ndarray], rng: jnp.ndarray) -> tuple[TrainState, float]:
    def loss_fn(params):
        x, y = batch
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_batch_stats = state.apply_fn(
            variables, x, training=True, mutable=["batch_stats"], rngs={"dropout": rng}
        )

        y_indices = jnp.argmax(y, axis=1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_indices).mean()
        return loss, new_batch_stats

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats["batch_stats"])

    return state, loss


@jax.jit
def val_step(state: TrainState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> float:
    x, y = batch
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, x, training=False)

    y_indices = jnp.argmax(y, axis=1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_indices).mean()
    return loss


@jax.jit
def get_embeddings(state: TrainState, x: jnp.ndarray) -> jnp.ndarray:
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    return state.apply_fn(variables, x, training=False, method="embedding")


class JAXDataset:
    """Dataset for perturbation classification.

    Needed for training a model that classifies the perturbed cells and takes as perturbation embedding the second to last layer.
    """

    def __init__(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        label_col: str = "perturbations",
        layer_key: str = None,
    ):
        """JAX Dataset for perturbation classification.

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

        if target_col in adata.obs.columns:
            self.labels = adata.obs[target_col].values
        elif target_col in adata.obsm:
            self.labels = adata.obsm[target_col]
        else:
            raise ValueError(f"Target column {target_col} not found in obs or obsm")

        self.pert_labels = adata.obs[label_col].values

        if scipy.sparse.issparse(self.data):
            self.data = to_dense(self.data)

        self.data = jnp.array(self.data, dtype=jnp.float32)
        self.labels = jnp.array(self.labels, dtype=jnp.float32)

    def __len__(self):
        return self.data.shape[0]

    def get_batch(self, indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, list]:
        """Returns a batch of samples and corresponding perturbations applied (labels)."""
        batch_data = self.data[indices]
        batch_labels = self.labels[indices]
        batch_pert_labels = [self.pert_labels[i] for i in indices]
        return batch_data, batch_labels, batch_pert_labels


def create_batched_indices(
    dataset_size: int, rng: jnp.ndarray, batch_size: int, n_batches: int, weights: jnp.ndarray | None = None
) -> list:
    """Create batched indices for training, optionally with weighted sampling."""
    batches = []
    for _ in range(n_batches):
        rng, batch_rng = random.split(rng)
        if weights is not None:
            batch_indices = random.choice(batch_rng, dataset_size, shape=(batch_size,), p=weights)
        else:
            batch_indices = random.choice(batch_rng, dataset_size, shape=(batch_size,), replace=False)
        batches.append(batch_indices)
    return batches


class MLPClassifierSpace(PerturbationSpace):
    """Fits an ANN classifier to the data and takes the feature space (weights in the last layer) as embedding.

    We train the ANN to classify the different perturbations. After training, the penultimate layer is used as the
    feature space, resulting in one embedding per cell. Consider employing the PseudoBulk or another PerturbationSpace
    to obtain one embedding per perturbation.

    See here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19.
    """

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        layer_key: str = None,
        hidden_dim: list[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = True,
        batch_size: int = 128,
        test_split_size: float = 0.2,
        validation_split_size: float = 0.25,
        max_epochs: int = 20,
        val_epochs_check: int = 2,
        patience: int = 2,
        lr: float = 1e-4,
        seed: int = 42,
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
            hidden_dim: List of number of neurons in each hidden layers of the neural network.
                For instance, [512, 256] will create a neural network with two hidden layers, the first with 512 neurons and the second with 256 neurons.
            dropout: Amount of dropout applied, constant for all layers.
            batch_norm: Whether to apply batch normalization.
            batch_size: The batch size, i.e. the number of datapoints to use in one forward/backward pass.
            test_split_size: Fraction of data to put in the test set. Default to 0.2.
            validation_split_size: Fraction of data to put in the validation set of the resultant train set.
                E.g. a test_split_size of 0.2 and a validation_split_size of 0.25 means that 25% of 80% of the data will be used for validation.
            max_epochs: Maximum number of epochs for training.
            val_epochs_check: Test performance on validation dataset after every val_epochs_check training epochs.
                Note that this affects early stopping, as the model will be stopped if the validation performance does not improve for patience epochs.
            patience: Number of validation performance checks without improvement, after which the early stopping flag
                is activated and training is therefore stopped.
            lr: Learning rate for training.
            seed: Random seed for reproducibility.

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
        adata = adata.copy()
        adata.obsm["encoded_perturbations"] = encoded_labels.astype(np.float32)

        X = list(range(adata.n_obs))
        y = adata.obs[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split_size, stratify=y_train
        )

        train_dataset = JAXDataset(
            adata=adata[X_train], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        val_dataset = JAXDataset(
            adata=adata[X_val], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        test_dataset = JAXDataset(
            adata=adata[X_test], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        total_dataset = JAXDataset(
            adata=adata, target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )

        rng = random.PRNGKey(seed)
        rng, init_rng, train_rng = random.split(rng, 3)

        sizes = [adata.n_vars] + hidden_dim + [n_classes]
        model = MLP(sizes=sizes, dropout=dropout, batch_norm=batch_norm)

        state = create_train_state(init_rng, model, (adata.n_vars,), lr)

        # Create weighted sampling for class imbalance
        weights = 1.0 / (1.0 + jnp.sum(train_dataset.labels, axis=1))
        weights = weights / jnp.sum(weights)

        n_batches_per_epoch = len(train_dataset) // batch_size
        train_batches = create_batched_indices(
            len(train_dataset), train_rng, batch_size, max_epochs * n_batches_per_epoch, weights
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            epoch_train_loss = 0

            epoch_start = epoch * n_batches_per_epoch
            epoch_end = (epoch + 1) * n_batches_per_epoch
            epoch_batches = train_batches[epoch_start:epoch_end]

            for _n_train_batches, batch_indices in enumerate(epoch_batches, 1):
                rng, step_rng = random.split(rng)
                batch_data, batch_labels, *_ = train_dataset.get_batch(batch_indices)
                state, loss = train_step(state, (batch_data, batch_labels), step_rng)
                epoch_train_loss += loss

            if (epoch + 1) % val_epochs_check == 0:
                val_losses = []
                for i in range(0, len(val_dataset), batch_size):
                    val_indices = jnp.arange(i, min(i + batch_size, len(val_dataset)))
                    val_batch_data, val_batch_labels, _ = val_dataset.get_batch(val_indices)
                    val_loss = val_step(state, (val_batch_data, val_batch_labels))
                    val_losses.append(val_loss)

                avg_val_loss = jnp.mean(jnp.array(val_losses))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

        # Test evaluation
        test_losses = []
        for i in range(0, len(test_dataset), batch_size):
            test_indices = jnp.arange(i, min(i + batch_size, len(test_dataset)))
            test_batch_data, test_batch_labels, _ = test_dataset.get_batch(test_indices)
            test_loss = val_step(state, (test_batch_data, test_batch_labels))
            test_losses.append(test_loss)

        # Extract embeddings
        embeddings_list = []
        labels_list = []

        for i in range(0, len(total_dataset), batch_size * 2):
            indices = jnp.arange(i, min(i + batch_size * 2, len(total_dataset)))
            batch_data, _, batch_pert_labels = total_dataset.get_batch(indices)
            batch_embeddings = get_embeddings(state, batch_data)

            embeddings_list.append(batch_embeddings)
            labels_list.extend(batch_pert_labels)

        all_embeddings = jnp.concatenate(embeddings_list, axis=0)

        pert_adata = AnnData(X=np.array(all_embeddings))
        pert_adata.obs["perturbations"] = labels_list

        adata_obs = adata.obs.reset_index(drop=True)
        if "perturbations" in adata_obs.columns:
            adata_obs = adata_obs.drop("perturbations", axis=1)

        obs_subset = adata_obs.iloc[: len(pert_adata.obs)].copy()
        cols_to_add = [col for col in obs_subset.columns if col not in ["perturbations", "encoded_perturbations"]]
        new_cols_data = {col: obs_subset[col].values for col in cols_to_add}

        if new_cols_data:
            pert_adata.obs = pd.concat(
                [pert_adata.obs, pd.DataFrame(new_cols_data, index=pert_adata.obs.index)], axis=1
            )

        return pert_adata
