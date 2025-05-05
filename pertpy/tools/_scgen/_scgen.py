from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from adjustText import adjust_text
from anndata import AnnData
from jax import Array
from lamin_utils import logger
from scipy import stats
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from scvi.utils import setup_anndata_dsp

from pertpy._doc import _doc_params, doc_common_plot_args

from ._scgenvae import JaxSCGENVAE
from ._utils import balancer, extractor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.pyplot import Figure

font = {"family": "Arial", "size": 14}


class Scgen(JaxTrainingMixin, BaseModelClass):
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
            f"Scgen Model with the following params: \nn_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, dropout_rate: "
            f"{dropout_rate}"
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
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
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
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
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
            give_mean: Whether to return the mean
            n_samples: The number of samples to use.

        Returns:
            Low-dimensional representation for each cell

        Examples:
            >>> import pertpy as pt
            >>> data = pt.dt.kang_2018()
            >>> pt.tl.Scgen.setup_anndata(data, batch_key="label", labels_key="cell_type")
            >>> model = pt.tl.Scgen(data)
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
            z = out["qz"].mean if give_mean else out["z"]
            latent.append(z)
        concat_axis = 0 if ((n_samples == 1) or give_mean) else 1
        latent = jnp.concatenate(latent, axis=concat_axis)  # type: ignore

        return self.module.as_numpy_array(latent)

    def plot_reg_mean_plot(  # pragma: no cover # noqa: D417
        self,
        adata,
        condition_key: str,
        axis_keys: dict[str, str],
        labels: dict[str, str],
        *,
        gene_list: list[str] = None,
        top_100_genes: list[str] = None,
        verbose: bool = False,
        legend: bool = True,
        title: str = None,
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
                logger.info("top_100 DEGs mean: ", r_value_diff**2)
        x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.mean(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            logger.info("All genes mean: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
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

        if save:
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
        gene_list: list[str] = None,
        top_100_genes: list[str] = None,
        legend: bool = True,
        title: str = None,
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
                logger.info("Top 100 DEGs var: ", r_value_diff**2)
        if "y1" in axis_keys:
            real_stim = adata[adata.obs[condition_key] == axis_keys["y1"]]
        x = np.asarray(np.var(ctrl.X, axis=0)).ravel()
        y = np.asarray(np.var(stim.X, axis=0)).ravel()
        m, b, r_value, p_value, std_err = stats.linregress(x, y)
        if verbose:
            logger.info("All genes var: ", r_value**2)
        df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
        ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
        ax.tick_params(labelsize=fontsize)
        if "range" in kwargs:
            start, stop, step = kwargs.get("range")
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

        if save:
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
        condition_key = scgen.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).original_key
        cd = adata[adata.obs[condition_key] == ctrl_key, :]
        stim = adata[adata.obs[condition_key] == stim_key, :]
        all_latent_cd = scgen.get_latent_representation(cd.X)
        all_latent_stim = scgen.get_latent_representation(stim.X)
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
