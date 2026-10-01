from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from anndata import AnnData
from fast_array_utils.conv import to_dense
from flax.training import train_state
from jax import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from pertpy._parallel import _MAX_BLOCK_ELEMENTS
from pertpy._types import CSBase, cast_dense, cast_frame
from pertpy.tools._perturbation_space._perturbation_space import (
    PerturbationSpace,
    _carry_constant_obs,
)


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
def train_steps(state: TrainState, batches: tuple[jnp.ndarray, jnp.ndarray], rngs: jnp.ndarray) -> TrainState:
    """Apply :func:`train_step` to each of the stacked batches in turn."""

    def step(state: TrainState, batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[TrainState, None]:
        x, y, rng = batch
        return train_step(state, (x, y), rng)[0], None

    return jax.lax.scan(step, state, (*batches, rngs))[0]


@jax.jit
def val_step(state: TrainState, batch: tuple[jnp.ndarray, jnp.ndarray]) -> float:
    x, y = batch
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, x, training=False)

    y_indices = jnp.argmax(y, axis=1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_indices).mean()
    return loss  # type: ignore[return-value]


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
        target_col: str = "perturbation",
        label_col: str = "perturbation",
        layer_key: str | None = None,
    ):
        """JAX Dataset for perturbation classification.

        Args:
            adata: AnnData object with observations and labels.
            target_col: key with the perturbation labels numerically encoded.
            label_col: key with the perturbation labels.
            layer_key: key of the layer to be used as data, otherwise .X.
        """
        data = adata.layers[layer_key] if layer_key else adata.X

        labels: np.ndarray
        if target_col in adata.obs.columns:
            labels = np.asarray(adata.obs[target_col].values)
        elif target_col in adata.obsm:
            labels = cast_dense(adata.obsm[target_col])
        else:
            raise ValueError(f"Target column {target_col} not found in obs or obsm")

        self.pert_labels = np.asarray(adata.obs[label_col].values)

        # Keep sparse data sparse and densify only the requested batch to avoid materializing the full dense matrix.
        self.is_sparse = isinstance(data, CSBase)
        self.data: CSBase | np.ndarray = (
            data.tocsr() if isinstance(data, CSBase) else np.asarray(data, dtype=np.float32)
        )
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return self.data.shape[0]

    def get_batch(self, indices: np.ndarray | jnp.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns a batch of samples and corresponding perturbations applied (labels).

        Indices of several batches stacked along the first axis return stacked batches.
        """
        idx = np.asarray(indices)
        data = self.data
        if isinstance(data, CSBase):
            batch_data = data[idx.ravel()].toarray().astype(np.float32, copy=False).reshape(*idx.shape, -1)
        else:
            batch_data = data[idx]
        return batch_data, self.labels[idx], self.pert_labels[idx]


def _split_keys(rng: jnp.ndarray, n: int) -> jnp.ndarray:
    """The ``n`` subkeys produced by repeatedly calling ``rng, subkey = random.split(rng)``."""

    def step(carry: jnp.ndarray, _) -> tuple[jnp.ndarray, jnp.ndarray]:
        carry, subkey = random.split(carry)
        return carry, subkey

    return jax.lax.scan(step, rng, length=n)[1]


def create_batched_indices(
    dataset_size: int, rng: jnp.ndarray, batch_size: int, n_batches: int, weights: jnp.ndarray | None = None
) -> np.ndarray:
    """Create batched indices for training, optionally with weighted sampling."""
    if n_batches == 0:
        return np.empty((0, batch_size), dtype=np.int32)

    def draw(batch_rng: jnp.ndarray) -> jnp.ndarray:
        if weights is not None:
            return random.choice(batch_rng, dataset_size, shape=(batch_size,), p=weights)
        return random.choice(batch_rng, dataset_size, shape=(batch_size,), replace=False)

    return np.asarray(jax.vmap(draw)(_split_keys(rng, n_batches)))


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
        *,
        target_col: str = "perturbation",
        layer_key: str | None = None,
        embedding_key: str | None = None,
        hidden_dim: list[int] | None = None,
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
        """Creates a perturbation embedding by training a MLP classifier model to distinguish between perturbations.

        A model is created using the specified parameters (hidden_dim, dropout, batch_norm). Further parameters such as
        the number of classes to predict (number of perturbations) are obtained from the provided AnnData object directly.
        Dataloaders that take into account class imbalances are created. Next, the model is trained and tested, using the
        GPU if available. The penultimate-layer activations are extracted for every cell and averaged per perturbation,
        yielding one embedding per perturbation.

        Args:
            adata: AnnData object of size cells x genes
            target_col: `.obs` column that stores the label of the perturbation applied to each cell.
            layer_key: Layer in adata to use.
            embedding_key: `.obsm` embedding to train the classifier on. Mutually exclusive with layer_key.
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
            AnnData with one observation per perturbation, the averaged penultimate-layer embedding in `.X` and the perturbation labels in `.obs[target_col]`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.norman_2019()
            >>> dcs = pt.tl.MLPClassifierSpace()
            >>> pert_embeddings = dcs.compute(adata, target_col="perturbation_name")
        """
        if layer_key is not None and layer_key not in adata.layers:
            raise ValueError(f"Layer key {layer_key} not found in adata.")

        if embedding_key is not None and embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key {embedding_key} not found in adata.obsm.")

        if layer_key is not None and embedding_key is not None:
            raise ValueError("Cannot specify both layer_key and embedding_key.")

        if target_col not in adata.obs:
            raise ValueError(f"Column {target_col!r} does not exist in the .obs attribute.")

        if hidden_dim is None:
            hidden_dim = [512]

        if embedding_key is not None:
            work = AnnData(X=adata.obsm[embedding_key])
        else:
            work = AnnData(X=adata.X if layer_key is None else adata.layers[layer_key])
        work.obs_names = adata.obs_names.tolist()
        work.obs = cast_frame(adata.obs).copy()
        adata = work
        layer_key = None

        # Labels are strings, one hot encoding for classification
        n_classes = len(adata.obs[target_col].unique())
        labels = to_dense(adata.obs[target_col]).reshape(-1, 1)
        encoder = OneHotEncoder()
        encoded_labels = encoder.fit_transform(labels).toarray()
        adata.obsm["encoded_perturbations"] = encoded_labels.astype(np.float32)

        X = list(range(adata.n_obs))
        y = adata.obs[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split_size, stratify=y, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split_size, stratify=y_train, random_state=seed
        )

        train_dataset = JAXDataset(
            adata=adata[X_train], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
        )
        val_dataset = JAXDataset(
            adata=adata[X_val], target_col="encoded_perturbations", label_col=target_col, layer_key=layer_key
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
        labels = jnp.asarray(train_dataset.labels)
        weights = 1.0 / (labels @ jnp.sum(labels, axis=0))
        weights = weights / jnp.sum(weights)

        n_batches_per_epoch = len(train_dataset) // batch_size
        n_steps = max_epochs * n_batches_per_epoch
        train_batches = create_batched_indices(len(train_dataset), train_rng, batch_size, n_steps, weights)
        step_rngs = np.asarray(_split_keys(rng, n_steps))

        best_val_loss: float | jax.Array = float("inf")
        patience_counter = 0

        step_elements = batch_size * (adata.n_vars + n_classes)
        steps_per_chunk = max(1, min(n_batches_per_epoch, _MAX_BLOCK_ELEMENTS // step_elements))

        for epoch in range(max_epochs):
            epoch_end = (epoch + 1) * n_batches_per_epoch
            for start in range(epoch * n_batches_per_epoch, epoch_end, steps_per_chunk):
                steps = slice(start, min(start + steps_per_chunk, epoch_end))
                batch_data, batch_labels, _ = train_dataset.get_batch(train_batches[steps])
                state = train_steps(state, (batch_data, batch_labels), step_rngs[steps])

            if (epoch + 1) % val_epochs_check == 0:
                val_losses = []
                for i in range(0, len(val_dataset), batch_size):
                    val_indices = np.arange(i, min(i + batch_size, len(val_dataset)))
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

        # Extract embeddings
        embeddings_list = []
        for i in range(0, len(total_dataset), batch_size * 2):
            indices = np.arange(i, min(i + batch_size * 2, len(total_dataset)))
            batch_data, _, _ = total_dataset.get_batch(indices)
            embeddings_list.append(get_embeddings(state, batch_data))

        all_embeddings = np.asarray(jnp.concatenate(embeddings_list, axis=0))

        # Average the per-cell embeddings within each perturbation to obtain one embedding per perturbation.
        cell_embeddings = pd.DataFrame(all_embeddings)
        cell_embeddings[target_col] = [str(label) for label in total_dataset.pert_labels]
        aggregated = cell_embeddings.groupby(target_col, observed=True).mean()

        pert_adata = AnnData(X=aggregated.to_numpy(dtype=np.float32))
        pert_adata.obs_names = aggregated.index.astype(str).tolist()
        pert_adata.obs[target_col] = pd.Categorical(aggregated.index.astype(str))

        _carry_constant_obs(pert_adata, cast_frame(adata.obs), target_col)
        pert_adata.obs[target_col] = cast_frame(pert_adata.obs)[target_col].astype("category")

        return pert_adata
