from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from anndata import AnnData
from fast_array_utils.conv import to_dense
from flax import serialization
from scipy import stats

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._logger import logger
from pertpy._types import cast_dense, cast_frame, cast_matrix

from ._scgenvae import JaxSCGENVAE
from ._train import DEFAULT_EPS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, train_module
from ._utils import balancer, extractor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.pyplot import Figure

font = {"family": "Arial", "size": 14}

SETUP_KEY = "_scgen_setup"


class Scgen:
    """JAX implementation of scGen for batch removal and perturbation prediction.

    The latent space supports vector arithmetic: the difference between the latent means of a
    stimulated and a control population, added to unperturbed cells of another cell type, predicts
    that cell type's response.
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        **model_kwargs,
    ):
        """Initialise the model.

        Args:
            adata: AnnData that :meth:`~pertpy.tools.Scgen.setup_anndata` has been run on.
            n_hidden: Width of every hidden layer.
            n_latent: Dimensionality of the latent space.
            n_layers: Number of hidden layers in the encoder and in the decoder.
            dropout_rate: Dropout applied to every hidden layer.
            **model_kwargs: Passed to :class:`~pertpy.tools._scgen._scgenvae.JaxSCGENVAE`.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
        """
        if SETUP_KEY not in adata.uns:
            raise ValueError("Please run `Scgen.setup_anndata` on this AnnData before initializing the model.")

        self.adata = adata
        self._setup: dict[str, Any] = dict(adata.uns[SETUP_KEY])
        self.init_params_ = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            **model_kwargs,
        }
        self.module = JaxSCGENVAE(n_input=adata.n_vars, **self.init_params_)
        self._eval_module = self.module.clone(training=False)
        self.params: Any = None
        self.model_state: Any = None
        self.history: dict[str, list[float]] = {}
        self.is_trained_ = False

    def __repr__(self) -> str:
        params = self.init_params_
        return (
            f"Scgen model with n_hidden: {params['n_hidden']}, n_latent: {params['n_latent']}, "
            f"n_layers: {params['n_layers']}, dropout_rate: {params['dropout_rate']}"
        )

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        layer: str | None = None,
    ) -> None:
        """Register the fields that scGen reads from ``adata``.

        scGen expects log-normalized expression in ``adata.X`` or in ``layer``.

        Args:
            adata: AnnData to register. Modified in place.
            batch_key: ``adata.obs`` column holding the condition or batch.
                If `None`, a constant column is added, which leaves nothing for
                :meth:`~pertpy.tools.Scgen.batch_removal` to correct.
            labels_key: ``adata.obs`` column holding the cell type.
                If `None`, a constant column is added.
            layer: ``adata.layers`` key holding the expression to model.
                If `None`, ``adata.X`` is used.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
        """
        for name, key in (("batch_key", batch_key), ("labels_key", labels_key)):
            if key is not None and key not in adata.obs:
                raise KeyError(f"{name} {key!r} was not found in adata.obs.")
        if layer is not None and layer not in adata.layers:
            raise KeyError(f"layer {layer!r} was not found in adata.layers.")

        if batch_key is None:
            batch_key = "_scgen_batch"
            adata.obs[batch_key] = pd.Categorical(np.zeros(adata.n_obs, dtype=np.int64))
        if labels_key is None:
            labels_key = "_scgen_labels"
            adata.obs[labels_key] = pd.Categorical(np.zeros(adata.n_obs, dtype=np.int64))

        adata.uns[SETUP_KEY] = {"batch_key": batch_key, "labels_key": labels_key, "layer": layer}

    @property
    def batch_key(self) -> str:
        """``adata.obs`` column holding the condition or batch."""
        return self._setup["batch_key"]

    @property
    def labels_key(self) -> str:
        """``adata.obs`` column holding the cell type."""
        return self._setup["labels_key"]

    @property
    def is_trained(self) -> bool:
        """Whether :meth:`~pertpy.tools.Scgen.train` has been run."""
        return self.is_trained_

    def _check_if_trained(self) -> None:
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")

    def _validate_anndata(self, adata: AnnData | None = None) -> AnnData:
        if adata is None:
            return self.adata
        if list(adata.var_names) != list(self.adata.var_names):
            raise ValueError("adata has different var_names than the AnnData the model was initialized with.")
        for key in (self.batch_key, self.labels_key):
            if key not in adata.obs:
                raise KeyError(f"{key!r} was not found in adata.obs.")
        return adata

    def _get_x(self, adata: AnnData) -> np.ndarray:
        layer = self._setup["layer"]
        x = adata.layers[layer] if layer is not None else adata.X
        return np.asarray(to_dense(x), dtype=np.float32)

    @property
    def _variables(self) -> dict[str, Any]:
        self._check_if_trained()
        return {"params": self.params, **self.model_state}

    def train(
        self,
        *,
        max_epochs: int | None = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        early_stopping: bool = False,
        early_stopping_patience: int = 45,
        early_stopping_min_delta: float = 0.0,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        eps: float = DEFAULT_EPS,
        max_norm: float | None = None,
        seed: int = 0,
    ) -> None:
        """Train the model.

        Args:
            max_epochs: Passes over the training set.
                Defaults to ``min(round((20000 / n_cells) * 400), 400)``.
            batch_size: Minibatch size.
            train_size: Fraction of cells used for training.
            validation_size: Fraction of cells used for validation.
                Defaults to everything not used for training.
            shuffle_set_split: Whether to shuffle before splitting rather than splitting sequentially.
            early_stopping: Whether to stop once the validation loss stops improving.
            early_stopping_patience: Epochs without improvement before stopping.
            early_stopping_min_delta: Minimum improvement that counts as progress.
            lr: Adam learning rate.
            weight_decay: Decoupled weight decay.
            eps: Adam epsilon.
            max_norm: Global gradient norm to clip to.
            seed: Seed for the split, the minibatch shuffling and the model initialization.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
        """
        state, history = train_module(
            self.module,
            self._get_x(self.adata),
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            max_norm=max_norm,
            seed=seed,
        )
        self.params = state.params
        self.model_state = state.state
        self.history = history
        self.is_trained_ = True

    def _encode(self, x: np.ndarray, *, give_mean: bool, n_samples: int, key: jnp.ndarray) -> jnp.ndarray:
        out = cast(
            "dict[str, Any]",
            self._eval_module.apply(
                self._variables, jnp.asarray(x), n_samples=n_samples, method="inference", rngs={"z": key}
            ),
        )
        return out["qz"].mean if give_mean else out["z"]

    def _decode(self, z: object) -> np.ndarray:
        out = cast(
            "dict[str, Any]", self._eval_module.apply(self._variables, jnp.asarray(cast_dense(z)), method="generative")
        )
        return np.asarray(out["px"])

    def get_latent_representation(
        self,
        adata: AnnData | np.ndarray | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        n_samples: int = 1,
        batch_size: int = 1024,
        *,
        seed: int = 0,
    ) -> np.ndarray:
        """Return the latent representation for each cell.

        Args:
            adata: AnnData with the same variables as the AnnData the model was initialized with.
                If `None`, that AnnData is used.
                A dense expression matrix is also accepted.
            indices: Indices of cells to use. If `None`, all cells are used.
            give_mean: Whether to return the mean of the latent distribution rather than a sample.
            n_samples: Number of latent samples to draw when ``give_mean`` is `False`.
            batch_size: Minibatch size used while encoding.
            seed: Seed for the latent sampling.

        Returns:
            Low-dimensional representation for each cell.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> latent_X = model.get_latent_representation()
        """
        self._check_if_trained()

        if adata is None or isinstance(adata, AnnData):
            x = self._get_x(self._validate_anndata(adata))
        else:
            x = np.asarray(to_dense(adata), dtype=np.float32)
        if indices is not None:
            x = x[np.asarray(indices)]

        key = jax.random.PRNGKey(seed)
        latent = []
        for start in range(0, x.shape[0], batch_size):
            key, z_key = jax.random.split(key)
            latent.append(
                self._encode(x[start : start + batch_size], give_mean=give_mean, n_samples=n_samples, key=z_key)
            )
        concat_axis = 0 if ((n_samples == 1) or give_mean) else 1

        return np.asarray(jnp.concatenate(latent, axis=concat_axis))

    def get_decoded_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int = 1024,
        *,
        seed: int = 0,
    ) -> np.ndarray:
        """Get decoded expression.

        Args:
            adata: AnnData with the same variables as the AnnData the model was initialized with.
                If `None`, that AnnData is used.
            indices: Indices of cells to use. If `None`, all cells are used.
            batch_size: Minibatch size used while decoding.
            seed: Seed for the latent sampling.

        Returns:
            Decoded expression for each cell.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> decoded_X = model.get_decoded_expression()
        """
        self._check_if_trained()

        latent = self.get_latent_representation(adata, indices=indices, batch_size=batch_size, seed=seed)
        return self._decode(latent)

    def predict(
        self,
        ctrl_key=None,
        stim_key=None,
        adata_to_predict=None,
        celltype_to_predict=None,
        restrict_arithmetic_to="all",
    ) -> tuple[AnnData, Any]:
        """Predicts the cell type provided by the user in stimulated condition.

        Args:
            ctrl_key: Key for `control` part of the `data` found in `condition_key`.
            stim_key: Key for `stimulated` part of the `data` found in `condition_key`.
            adata_to_predict: Adata for unperturbed cells you want to be predicted.
            celltype_to_predict: The cell type you want to be predicted.
            restrict_arithmetic_to: Dictionary of celltypes you want to be observed for prediction.

        Returns:
            :class:`numpy.ndarray` of predicted cells in primary space.
        delta: float
            Difference between stimulated and control cells in latent space

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> pred, delta = model.predict(ctrl_key="ctrl", stim_key="stim", celltype_to_predict="CD4 T cells")
        """
        # use keys registered from `setup_anndata()`
        cell_type_key = self.labels_key
        condition_key = self.batch_key

        if restrict_arithmetic_to == "all":
            ctrl_x = self.adata[self.adata.obs[condition_key] == ctrl_key, :]
            stim_x = self.adata[self.adata.obs[condition_key] == stim_key, :]
            ctrl_x = balancer(ctrl_x, cell_type_key)
            stim_x = balancer(stim_x, cell_type_key)
        else:
            key = list(restrict_arithmetic_to.keys())[0]
            values = restrict_arithmetic_to[key]
            subset = self.adata[self.adata.obs[key].isin(values)]
            ctrl_x = subset[subset.obs[condition_key] == ctrl_key, :]
            stim_x = subset[subset.obs[condition_key] == stim_key, :]
            if len(values) > 1:
                ctrl_x = balancer(ctrl_x, cell_type_key)
                stim_x = balancer(stim_x, cell_type_key)
        if celltype_to_predict is not None and adata_to_predict is not None:
            raise Exception("Please provide either a cell type or adata not both!")
        if celltype_to_predict is None and adata_to_predict is None:
            raise Exception("Please provide a cell type name or adata for your unperturbed cells")
        if celltype_to_predict is not None:
            ctrl_pred = extractor(
                self.adata,
                celltype_to_predict,
                condition_key=condition_key,
                cell_type_key=cell_type_key,
                ctrl_key=ctrl_key,
                stim_key=stim_key,
            )[1]
        else:
            ctrl_pred = adata_to_predict

        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        rng = np.random.default_rng()
        cd_ind = rng.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = rng.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[cd_ind, :]
        stim_adata = stim_x[stim_ind, :]

        latent_ctrl = self._avg_vector(ctrl_adata)
        latent_stim = self._avg_vector(stim_adata)

        delta = latent_stim - latent_ctrl

        latent_cd = self.get_latent_representation(ctrl_pred)

        stim_pred = delta + latent_cd
        predicted_cells = self._decode(stim_pred)

        predicted_adata = AnnData(
            X=np.array(predicted_cells),
            obs=ctrl_pred.obs.copy(),
            var=ctrl_pred.var.copy(),
            obsm=ctrl_pred.obsm.copy(),
        )
        return predicted_adata, delta

    def _avg_vector(self, adata):
        return np.mean(self.get_latent_representation(adata), axis=0)

    def batch_removal(self, adata: AnnData | None = None) -> AnnData:
        """Removes batch effects.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.

        Returns:
            A corrected `~anndata.AnnData` object.
            AnnData of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent"].
            A reference to the original AnnData is in `corrected.raw` if the input adata had no `raw` attribute.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> corrected_adata = model.batch_removal()
        """
        adata = self._validate_anndata(adata)
        latent_all = self.get_latent_representation(adata)
        # use keys registered from `setup_anndata()`
        cell_label_key = self.labels_key
        batch_key = self.batch_key

        adata_latent = AnnData(latent_all)
        adata_latent.obs = cast_frame(adata.obs).copy(deep=True)
        unique_cell_types = np.unique(adata_latent.obs[cell_label_key])
        shared_ct = []
        not_shared_ct = []
        for cell_type in unique_cell_types:
            temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type].copy()
            if len(np.unique(temp_cell.obs[batch_key])) < 2:
                cell_type_ann = adata_latent[adata_latent.obs[cell_label_key] == cell_type]
                not_shared_ct.append(cell_type_ann)
                continue
            temp_cell = adata_latent[adata_latent.obs[cell_label_key] == cell_type].copy()
            batch_list = {}
            batch_ind = {}
            max_batch = 0
            max_batch_ind = ""
            batches = np.unique(temp_cell.obs[batch_key])
            for i in batches:
                temp = temp_cell[temp_cell.obs[batch_key] == i]
                temp_ind = temp_cell.obs[batch_key] == i
                if max_batch < len(temp):
                    max_batch = len(temp)
                    max_batch_ind = i
                batch_list[i] = temp
                batch_ind[i] = temp_ind
            max_batch_ann = batch_list[max_batch_ind]
            max_batch_mean = np.average(max_batch_ann.X, axis=0)
            for study in batch_list:
                batch_list[study] = batch_list[study].copy()
                delta = max_batch_mean - np.average(batch_list[study].X, axis=0)
                batch_list[study].X = delta + batch_list[study].X
            shared_ct.append(temp_cell)

        all_shared_ann = ad.concat(shared_ct, label="concat_batch", index_unique=None)
        if "concat_batch" in all_shared_ann.obs.columns:
            del cast_frame(all_shared_ann.obs)["concat_batch"]
        if len(not_shared_ct) < 1:
            corrected = AnnData(
                self._decode(all_shared_ann.X),
                obs=cast_frame(all_shared_ann.obs),
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names].copy()
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names.tolist()
                corrected.raw = adata_raw
            corrected.obsm["latent"] = cast_matrix(all_shared_ann.X)
            corrected.obsm["corrected_latent"] = self.get_latent_representation(corrected)
            return corrected
        else:
            all_not_shared_ann = ad.concat(not_shared_ct, label="concat_batch", index_unique=None)
            all_corrected_data = ad.concat(
                [all_shared_ann, all_not_shared_ann], label="concat_batch", index_unique=None
            )
            if "concat_batch" in all_corrected_data.obs.columns:
                del cast_frame(all_corrected_data.obs)["concat_batch"]
            corrected = AnnData(
                self._decode(all_corrected_data.X),
                obs=cast_frame(all_corrected_data.obs),
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names].copy()
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names.tolist()
                corrected.raw = adata_raw
            corrected.obsm["latent"] = cast_matrix(all_corrected_data.X)
            corrected.obsm["corrected_latent"] = self.get_latent_representation(corrected)

            return corrected

    def save(self, dir_path: str | Path, *, overwrite: bool = False, save_anndata: bool = False) -> None:
        """Save the trained model to a directory.

        Args:
            dir_path: Directory to write to.
            overwrite: Whether to overwrite an existing directory.
            save_anndata: Whether to also write the AnnData the model was trained on.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
            >>> model.train(max_epochs=10)
            >>> model.save("scgen_model")
        """
        self._check_if_trained()

        path = Path(dir_path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")
        path.mkdir(parents=True, exist_ok=True)

        weights = {"params": self.params, "model_state": self.model_state}
        (path / "model.msgpack").write_bytes(serialization.msgpack_serialize(jax.device_get(weights)))
        (path / "attr.json").write_text(
            json.dumps(
                {
                    "init_params": self.init_params_,
                    "setup": self._setup,
                    "var_names": list(self.adata.var_names),
                    "history": self.history,
                }
            )
        )
        if save_anndata:
            self.adata.write(path / "adata.h5ad")

    @classmethod
    def load(cls, dir_path: str | Path, adata: AnnData | None = None) -> Scgen:
        """Load a model saved with :meth:`~pertpy.tools.Scgen.save`.

        Args:
            dir_path: Directory written by :meth:`~pertpy.tools.Scgen.save`.
            adata: AnnData to attach to the model.
                Required unless the model was saved with ``save_anndata=True``.

        Returns:
            The loaded model.

        Examples:
            >>> import pertpy as pt
            >>> model = pt.tl.Scgen.load("scgen_model")
        """
        path = Path(dir_path)
        attrs = json.loads((path / "attr.json").read_text())

        if adata is None:
            adata_path = path / "adata.h5ad"
            if not adata_path.exists():
                raise ValueError("This model was saved without its AnnData. Please pass `adata`.")
            adata = ad.read_h5ad(adata_path)
        if list(adata.var_names) != attrs["var_names"]:
            raise ValueError("adata has different var_names than the AnnData the model was trained on.")

        adata.uns[SETUP_KEY] = attrs["setup"]
        model = cls(adata, **attrs["init_params"])

        weights = serialization.msgpack_restore((path / "model.msgpack").read_bytes())
        model.params = weights["params"]
        model.model_state = weights["model_state"]
        model.history = attrs["history"]
        model.is_trained_ = True
        return model

    def plot_reg_mean_plot(  # pragma: no cover # noqa: D417
        self,
        adata,
        condition_key: str,
        axis_keys: dict[str, str],
        labels: dict[str, str],
        *,
        gene_list: list[str] | None = None,
        top_100_genes: list[str] | None = None,
        verbose: bool = False,
        legend: bool = True,
        title: str | None = None,
        x_coeff: float = 0.30,
        y_coeff: float = 0.8,
        fontsize: float = 14,
        show: bool = False,
        save: str | bool | None = None,
        **kwargs,
    ) -> tuple[float, float] | float:
        """Plots mean matching for a set of specified genes.

        Args:
            adata:  AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                    AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                    corresponding to batch and cell type metadata, respectively.
            condition_key: The key for the condition
            axis_keys: Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
                       {`x`: `Key for x-axis`, `y`: `Key for y-axis`}.
            labels: Dictionary of axes labels of the form {`x`: `x-axis-name`, `y`: `y-axis name`}.
            gene_list: list of gene names to be plotted.
            top_100_genes: List of the top 100 differentially expressed genes. Specify if you want the top 100 DEGs to be assessed extra.
            verbose: Specify if you want information to be printed while creating the plot.
            legend: Whether to plot a legend.
            title: Set if you want the plot to display a title.
            x_coeff: Offset to print the R^2 value in x-direction.
            y_coeff: Offset to print the R^2 value in y-direction.
            fontsize: Fontsize used for text in the plot.
            show: if `True`, will show to the plot after saving it.
            save: Specify if the plot should be saved or not.
            **kwargs:

        Returns:
            Returns R^2 value for all genes and R^2 value for top 100 DEGs if `top_100_genes` is not `None`.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> scg = pt.tl.Scgen(data)
            >>> scg.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> pred, delta = scg.predict(ctrl_key='ctrl', stim_key='stim', celltype_to_predict='CD4 T cells')
            >>> pred.obs['label'] = 'pred'
            >>> eval_adata = data[data.obs['cell_type'] == 'CD4 T cells'].copy().concatenate(pred)
            >>> r2_value = scg.plot_reg_mean_plot(eval_adata, condition_key='label', axis_keys={"x": "pred", "y": "stim"}, \
                labels={"x": "predicted", "y": "ground truth"}, save=False, show=True)

        Preview:
            .. image:: /_static/docstring_previews/scgen_reg_mean.png
        """
        import seaborn as sns

        sns.set_theme()
        sns.set_theme(color_codes=True)

        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                logger.info(f"top_100 DEGs mean: {r_value_diff**2}")
        x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.mean(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            logger.info(
                f"All genes mean: {r_value**2}",
            )
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs["range"]
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if gene_list is not None:
            texts = []
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
                plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
                # if "y1" in axis_keys.keys():
                # y1_bar = y1[j]
                # plt.text(x_bar, y1_bar, i, fontsize=11, color="black")
        if gene_list is not None:
            adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops={"arrowstyle": "->", "color": "grey", "lw": 0.5},
                force_static=(0.0, 0.0),
            )
        if legend:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            plt.title("", fontsize=fontsize)
        else:
            plt.title(title, fontsize=fontsize)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value**2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= " + f"{r_value_diff**2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if isinstance(save, str):
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    def plot_reg_var_plot(  # pragma: no cover # noqa: D417
        self,
        adata,
        condition_key: str,
        axis_keys: dict[str, str],
        labels: dict[str, str],
        *,
        gene_list: list[str] | None = None,
        top_100_genes: list[str] | None = None,
        legend: bool = True,
        title: str | None = None,
        verbose: bool = False,
        x_coeff: float = 0.3,
        y_coeff: float = 0.8,
        fontsize: float = 14,
        show: bool = True,
        save: str | bool | None = None,
        **kwargs,
    ) -> tuple[float, float] | float:
        """Plots variance matching for a set of specified genes.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.
            condition_key: Key of the condition.
            axis_keys: Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
                       {"x": "Key for x-axis", "y": "Key for y-axis"}.
            labels: Dictionary of axes labels of the form {"x": "x-axis-name", "y": "y-axis name"}.
            gene_list: list of gene names to be plotted.
            top_100_genes: List of the top 100 differentially expressed genes. Specify if you want the top 100 DEGs to be assessed extra.
            legend: Whether to plot a legend.
            title: Set if you want the plot to display a title.
            verbose: Specify if you want information to be printed while creating the plot.
            x_coeff: Offset to print the R^2 value in x-direction.
            y_coeff: Offset to print the R^2 value in y-direction.
            fontsize: Fontsize used for text in the plot.
            show: if `True`, will show to the plot after saving it.
            save: Specify if the plot should be saved or not.
        """
        import seaborn as sns

        sns.set_theme()
        sns.set_theme(color_codes=True)

        sc.tl.rank_genes_groups(adata, groupby=condition_key, n_genes=100, method="wilcoxon")
        diff_genes = top_100_genes
        stim = adata[adata.obs[condition_key] == axis_keys["y"]]
        ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
        if diff_genes is not None:
            if hasattr(diff_genes, "tolist"):
                diff_genes = diff_genes.tolist()
            adata_diff = adata[:, diff_genes]
            stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
            ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
            x_diff = np.asarray(np.var(ctrl_diff.X, axis=0)).ravel()
            y_diff = np.asarray(np.var(stim_diff.X, axis=0)).ravel()
            m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(x_diff, y_diff)
            if verbose:
                logger.info(f"Top 100 DEGs var: {r_value_diff**2}")
        if "y1" in axis_keys:
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = np.asarray(np.var(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.var(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            logger.info(f"All genes var: {r_value**2}")
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs["range"]
            ax.set_xticks(np.arange(start, stop, step))
            ax.set_yticks(np.arange(start, stop, step))
        # _p1 = plt.scatter(x, y, marker=".", label=f"{axis_keys['x']}-{axis_keys['y']}")
        # plt.plot(x, m * x + b, "-", color="green")
        ax.set_xlabel(labels["x"], fontsize=fontsize)
        ax.set_ylabel(labels["y"], fontsize=fontsize)
        if "y1" in axis_keys:
            y1 = np.asarray(np.var(real_stim.X, axis=0)).ravel()
            _ = plt.scatter(
                x,
                y1,
                marker="*",
                c="grey",
                alpha=0.5,
                label=f"{axis_keys['x']}-{axis_keys['y1']}",
            )
        if gene_list is not None:
            for i in gene_list:
                j = adata.var_names.tolist().index(i)
                x_bar = x[j]
                y_bar = y[j]
                plt.text(x_bar, y_bar, i, fontsize=11, color="black")
                plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
                if "y1" in axis_keys:
                    y1_bar = y1[j]
                    plt.text(x_bar, y1_bar, "*", color="black", alpha=0.5)
        if legend:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if title is None:
            plt.title("", fontsize=12)
        else:
            plt.title(title, fontsize=12)
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - y_coeff * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value**2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
        if diff_genes is not None:
            ax.text(
                max(x) - max(x) * x_coeff,
                max(y) - (y_coeff + 0.15) * max(y),
                r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= " + f"{r_value_diff**2:.2f}",
                fontsize=kwargs.get("textsize", fontsize),
            )

        if isinstance(save, str):
            plt.savefig(save, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        if diff_genes is not None:
            return r_value**2, r_value_diff**2
        else:
            return r_value**2

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_binary_classifier(  # pragma: no cover # noqa: D417
        self,
        scgen: Scgen,
        adata: AnnData | None,
        delta: np.ndarray,
        ctrl_key: str,
        stim_key: str,
        *,
        fontsize: float = 14,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plots the dot product between delta and latent representation of a linear classifier.

        Builds a linear classifier based on the dot product between
        the difference vector and the latent representation of each
        cell and plots the dot product results between delta and latent representation.

        Args:
            scgen: ScGen object that was trained.
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been set up with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.
            delta: Difference between stimulated and control cells in latent space
            ctrl_key: Key for `control` part of the `data` found in `condition_key`.
            stim_key: Key for `stimulated` part of the `data` found in `condition_key`.
            fontsize: Set the font size of the plot.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
        """
        plt.close("all")
        adata = scgen._validate_anndata(adata)
        condition_key = scgen.batch_key
        cd = adata[adata.obs[condition_key] == ctrl_key, :]
        stim = adata[adata.obs[condition_key] == stim_key, :]
        all_latent_cd = scgen.get_latent_representation(cd.X)  # type: ignore[arg-type]
        all_latent_stim = scgen.get_latent_representation(stim.X)  # type: ignore[arg-type]
        dot_cd = np.zeros(len(all_latent_cd))
        dot_sal = np.zeros(len(all_latent_stim))
        for ind, vec in enumerate(all_latent_cd):
            dot_cd[ind] = np.dot(delta, vec)
        for ind, vec in enumerate(all_latent_stim):
            dot_sal[ind] = np.dot(delta, vec)
        plt.hist(
            dot_cd,
            label=ctrl_key,
            bins=50,
        )
        plt.hist(dot_sal, label=stim_key, bins=50)
        plt.axvline(0, color="k", linestyle="dashed", linewidth=1)
        plt.title("  ", fontsize=fontsize)
        plt.xlabel("  ", fontsize=fontsize)
        plt.ylabel("  ", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        ax = plt.gca()
        ax.grid(False)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None


# compatibility
SCGEN = Scgen
