from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from jax import Array
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from scvi.utils import setup_anndata_dsp

from ._jax_scgenvae import JaxSCGENVAE
from ._utils import balancer, extractor

if TYPE_CHECKING:
    from collections.abc import Sequence

font = {"family": "Arial", "size": 14}


class SCGEN(JaxTrainingMixin, BaseModelClass):
    """Jax Implementation of scGen model for batch removal and perturbation prediction."""

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 800,
        n_latent: int = 100,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.module = JaxSCGENVAE(
            n_input=self.summary_stats.n_vars,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCGEN Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: " "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

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
            `np nd-array` of predicted cells in primary space.
        delta: float
            Difference between stimulated and control cells in latent space

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.SCGEN.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.SCGEN(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> pred, delta = model.predict(ctrl_key='ctrl', stim_key='stim', celltype_to_predict='CD4 T cells')
        """
        # use keys registered from `setup_anndata()`
        cell_type_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        condition_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key

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
                condition_key,
                cell_type_key,
                ctrl_key,
                stim_key,
            )[1]
        else:
            ctrl_pred = adata_to_predict

        eq = min(ctrl_x.X.shape[0], stim_x.X.shape[0])
        cd_ind = np.random.choice(range(ctrl_x.shape[0]), size=eq, replace=False)
        stim_ind = np.random.choice(range(stim_x.shape[0]), size=eq, replace=False)
        ctrl_adata = ctrl_x[cd_ind, :]
        stim_adata = stim_x[stim_ind, :]

        latent_ctrl = self._avg_vector(ctrl_adata)
        latent_stim = self._avg_vector(stim_adata)

        delta = latent_stim - latent_ctrl

        latent_cd = self.get_latent_representation(ctrl_pred)

        stim_pred = delta + latent_cd
        predicted_cells = self.module.as_bound().generative(stim_pred)["px"]

        predicted_adata = AnnData(
            X=np.array(predicted_cells),
            obs=ctrl_pred.obs.copy(),
            var=ctrl_pred.var.copy(),
            obsm=ctrl_pred.obsm.copy(),
        )
        return predicted_adata, delta

    def _avg_vector(self, adata):
        return np.mean(self.get_latent_representation(adata), axis=0)

    def get_decoded_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> Array:
        """Get decoded expression.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model.
            indices: Indices of cells in adata to use. If `None`, all cells are used.
            batch_size: Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns:
            Decoded expression for each cell

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.SCGEN.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.SCGEN(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> decoded_X = model.get_decoded_expression()
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        decoded = []
        for tensors in scdl:
            _, generative_outputs = self.module.as_bound()(tensors, compute_loss=False)
            px = generative_outputs["px"]
            decoded.append(px)

        return jnp.concatenate(decoded)

    def batch_removal(self, adata: AnnData | None = None) -> AnnData:
        """Removes batch effects.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
                   corresponding to batch and cell type metadata, respectively.

        Returns:
            corrected: `~anndata.AnnData`
            AnnData of corrected gene expression in adata.X and corrected latent space in adata.obsm["latent"].
            A reference to the original AnnData is in `corrected.raw` if the input adata had no `raw` attribute.

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.SCGEN.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.SCGEN(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> corrected_adata = model.batch_removal()
        """
        adata = self._validate_anndata(adata)
        latent_all = self.get_latent_representation(adata)
        # use keys registered from `setup_anndata()`
        cell_label_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).original_key
        batch_key = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key

        adata_latent = AnnData(latent_all)
        adata_latent.obs = adata.obs.copy(deep=True)
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
            for study in batch_list:
                delta = np.average(max_batch_ann.X, axis=0) - np.average(batch_list[study].X, axis=0)
                batch_list[study].X = delta + batch_list[study].X
                temp_cell[batch_ind[study]].X = batch_list[study].X
            shared_ct.append(temp_cell)

        all_shared_ann = AnnData.concatenate(*shared_ct, batch_key="concat_batch", index_unique=None)
        if "concat_batch" in all_shared_ann.obs.columns:
            del all_shared_ann.obs["concat_batch"]
        if len(not_shared_ct) < 1:
            corrected = AnnData(
                np.array(self.module.as_bound().generative(all_shared_ann.X)["px"]),
                obs=all_shared_ann.obs,
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_shared_ann.X
            corrected.obsm["corrected_latent"] = self.get_latent_representation(corrected)
            return corrected
        else:
            all_not_shared_ann = AnnData.concatenate(*not_shared_ct, batch_key="concat_batch", index_unique=None)
            all_corrected_data = AnnData.concatenate(
                all_shared_ann,
                all_not_shared_ann,
                batch_key="concat_batch",
                index_unique=None,
            )
            if "concat_batch" in all_shared_ann.obs.columns:
                del all_corrected_data.obs["concat_batch"]
            corrected = AnnData(
                np.array(self.module.as_bound().generative(all_corrected_data.X)["px"]),
                obs=all_corrected_data.obs,
            )
            corrected.var_names = adata.var_names.tolist()
            corrected = corrected[adata.obs_names]
            if adata.raw is not None:
                adata_raw = AnnData(X=adata.raw.X, var=adata.raw.var)
                adata_raw.obs_names = adata.obs_names
                corrected.raw = adata_raw
            corrected.obsm["latent"] = all_corrected_data.X
            corrected.obsm["corrected_latent"] = self.get_latent_representation(corrected)

            return corrected

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        labels_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        scGen expects the expression data to come from `adata.X`

        %(param_batch_key)s
        %(param_labels_key)s

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.SCGEN.setup_anndata(data, batch_key="label", labels_key="cell_type")
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, None, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def to_device(self, device):
        pass

    @property
    def device(self):
        return self.module.device

    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Return the latent representation for each cell.

        Args:
            adata: AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
                   AnnData object used to initialize the model.
            indices: Indices of cells in adata to use. If `None`, all cells are used.
            batch_size: Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.

        Returns:
            Low-dimensional representation for each cell

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.SCGEN.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.SCGEN(data)
            >>> model.train(max_epochs=10, batch_size=64, early_stopping=True, early_stopping_patience=5)
            >>> latent_X = model.get_latent_representation()
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)

        jit_inference_fn = self.module.get_jit_inference_fn(inference_kwargs={"n_samples": n_samples})

        latent = []
        for array_dict in scdl:
            out = jit_inference_fn(self.module.rngs, array_dict)
            if give_mean:
                z = out["qz"].mean
            else:
                z = out["z"]
            latent.append(z)
        concat_axis = 0 if ((n_samples == 1) or give_mean) else 1
        latent = jnp.concatenate(latent, axis=concat_axis)  # type: ignore

        return self.module.as_numpy_array(latent)
