from __future__ import annotations

import uuid
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numba import njit, prange
from rich.progress import track
from scanpy.get import _get_obs_rep, _set_obs_rep
from scipy.sparse import csr_matrix, issparse

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._types import CSRBase
from pertpy.preprocessing._guide_rna_mixture import PoissonGaussMixture

if TYPE_CHECKING:
    from matplotlib.pyplot import Figure


class GuideAssignment:
    """Assign cells to guide RNAs."""

    @singledispatchmethod
    def assign_by_threshold(
        self,
        data: AnnData | np.ndarray | CSRBase,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ):
        """Simple threshold based gRNA assignment function.

        Each cell is assigned to gRNA with at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            data: The (annotated) data matrix of shape `n_obs` × `n_vars`.
                  Rows correspond to cells and columns to genes.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_layer: Assigned guide will be saved on adata.layers[output_key].

        Examples:
            Each cell is assigned to gRNA that occurs at least 5 times in the respective cell.

            >>> import pertpy as pt
            >>> mdata = pt.data.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_by_threshold(gdo, assignment_threshold=5)
        """
        raise NotImplementedError(
            f"No implementation found for {type(data)}. Must be numpy array, sparse matrix, or AnnData object."
        )

    @assign_by_threshold.register(AnnData)
    def _assign_by_threshold_anndata(
        self,
        adata: AnnData,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ) -> None:
        X = _get_obs_rep(adata, layer=layer)
        guide_assignments = self.assign_by_threshold(X, assignment_threshold=assignment_threshold)
        _set_obs_rep(adata, guide_assignments, layer=output_layer)

    @assign_by_threshold.register(np.ndarray)
    def _assign_by_threshold_numpy(self, X: np.ndarray, /, *, assignment_threshold: float) -> np.ndarray:
        return np.where(assignment_threshold <= X, 1, 0)

    @staticmethod
    @njit(parallel=True)
    def _threshold_sparse_numba(data: np.ndarray, threshold: float) -> np.ndarray:
        out = np.zeros_like(data, dtype=np.int8)
        for i in prange(data.shape[0]):
            if data[i] >= threshold:
                out[i] = 1
        return out

    @assign_by_threshold.register(CSRBase)
    def _assign_by_threshold_sparse(self, X: CSRBase, /, *, assignment_threshold: float) -> CSRBase:
        new_data = self._threshold_sparse_numba(X.data, assignment_threshold)
        return csr_matrix((new_data, X.indices, X.indptr), shape=X.shape)

    @singledispatchmethod
    def assign_to_max_guide(
        self,
        data: AnnData | np.ndarray | CSRBase,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> np.ndarray | None:
        """Simple threshold based max gRNA assignment function.

        Each cell is assigned to the most expressed gRNA if it has at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            data: The (annotated) data matrix of shape `n_obs` × `n_vars`.
                  Rows correspond to cells and columns to genes.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            obs_key: Assigned guide will be saved on adata.obs[output_key].
            no_grna_assigned_key: The key to return if no gRNA is expressed enough.

        Examples:
            Each cell is assigned to the most expressed gRNA if it has at least 5 counts.

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_to_max_guide(gdo, assignment_threshold=5)
        """
        raise NotImplementedError(
            f"No implementation found for {type(data)}. Must be numpy array, sparse matrix, or AnnData object."
        )

    @assign_to_max_guide.register(AnnData)
    def assign_to_max_guide_anndata(
        self,
        adata: AnnData,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> None:
        X = _get_obs_rep(adata, layer=layer)
        guide_assignments = self.assign_to_max_guide(
            X, var=adata.var, assignment_threshold=assignment_threshold, no_grna_assigned_key=no_grna_assigned_key
        )
        adata.obs[obs_key] = guide_assignments

    @assign_to_max_guide.register(np.ndarray)
    def assign_to_max_guide_numpy(
        self,
        X: np.ndarray,
        /,
        *,
        var: pd.DataFrame,
        assignment_threshold: float,
        no_grna_assigned_key: str = "Negative",
    ) -> np.ndarray:
        assigned_grna = np.where(
            X.max(axis=1).squeeze() >= assignment_threshold,
            var.index[X.argmax(axis=1).squeeze()],
            no_grna_assigned_key,
        )

        return assigned_grna

    @staticmethod
    @njit(parallel=True)
    def _assign_max_guide_sparse(indptr, data, indices, assignment_threshold, assigned_grna):
        n_rows = len(indptr) - 1
        for i in range(n_rows):
            row_start = indptr[i]
            row_end = indptr[i + 1]

            if row_end > row_start:
                data_row = data[row_start:row_end]
                indices_row = indices[row_start:row_end]
                max_pos = np.argmax(data_row)
                if data_row[max_pos] >= assignment_threshold:
                    assigned_grna[i] = indices_row[max_pos]
        return assigned_grna

    @assign_to_max_guide.register(CSRBase)
    def assign_to_max_guide_sparse(
        self, X: CSRBase, /, *, var: pd.DataFrame, assignment_threshold: float, no_grna_assigned_key: str = "Negative"
    ) -> np.ndarray:
        n_rows = X.shape[0]

        assigned_positions = np.zeros(n_rows, dtype=np.int32) - 1  # -1 means not assigned
        assigned_positions = self._assign_max_guide_sparse(
            X.indptr, X.data, X.indices, assignment_threshold, assigned_positions
        )

        assigned_grna = np.full(n_rows, no_grna_assigned_key, dtype=object)
        mask = assigned_positions >= 0
        var_index_array = np.array(var.index)
        if np.any(mask):
            assigned_grna[mask] = var_index_array[assigned_positions[mask]]

        return assigned_grna

    def assign_mixture_model(
        self,
        adata: AnnData,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        assigned_guides_key: str = "assigned_guide",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        show_progress: bool = False,
        **mixture_model_kwargs,
    ) -> np.ndarray | None:
        """Assigns gRNAs to cells using a mixture model.

        Args:
            adata: AnnData object containing gRNA values.
            model: The model to use for the mixture model. Currently only `Poisson_Gauss_Mixture` is supported.
            assigned_guides_key: Assigned guide will be saved on adata.obs[output_key].
            no_grna_assigned_key: The key to return if a cell is negative for all gRNAs.
            max_assignments_per_cell: The maximum number of gRNAs that can be assigned to a cell.
            multiple_grna_assigned_key: The key to return if multiple gRNAs are assigned to a cell.
            multiple_grna_assignment_string: The string to use to join multiple gRNAs assigned to a cell.
            only_return_results: Whether input AnnData is not modified and the result is returned as an np.ndarray.
            show_progress: Whether to shows progress bar.
            mixture_model_kwargs: Are passed to the mixture model.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_mixture_model(gdo)
        """
        if model == "poisson_gauss_mixture":
            mixture_model = PoissonGaussMixture(**mixture_model_kwargs)
        else:
            raise ValueError("Model not implemented. Please use 'poisson_gauss_mixture'.")

        res = pd.DataFrame(0, index=adata.obs_names, columns=adata.var_names)
        fct = track if show_progress else lambda iterable: iterable
        for gene in fct(adata.var_names):
            is_nonzero = (
                np.ravel((adata[:, gene].X != 0).todense()) if issparse(adata.X) else np.ravel(adata[:, gene].X != 0)
            )
            if sum(is_nonzero) < 2:
                warn(f"Skipping {gene} as there are less than 2 cells expressing the guide at all.", stacklevel=2)
                continue
            # We are only fitting the model to the non-zero values, the rest is
            # automatically assigned to the negative class
            data = adata[is_nonzero, gene].X.todense().A1 if issparse(adata.X) else adata[is_nonzero, gene].X
            data = np.ravel(data)

            if np.any(data < 0):
                raise ValueError(
                    "Data contains negative values. Please use non-negative data for guide assignment with the Mixture Model."
                )

            # Log2 transform the data so positive population is approximately normal
            data = np.log2(data)
            assignments = mixture_model.run_model(data)
            res.loc[adata.obs_names[is_nonzero][assignments == "Positive"], gene] = 1

            # Add the parameters to the adata.var DataFrame
            for params_name, param in mixture_model.params.items():
                if param.ndim == 0:
                    if params_name not in adata.var.columns:
                        adata.var[params_name] = np.nan
                    adata.var.loc[gene, params_name] = param.item()
                else:
                    for i, p in enumerate(param):
                        if f"{params_name}_{i}" not in adata.var.columns:
                            adata.var[f"{params_name}_{i}"] = np.nan
                        adata.var.loc[gene, f"{params_name}_{i}"] = p

        # Assign guides to cells
        # Some cells might have multiple guides assigned
        series = pd.Series(no_grna_assigned_key, index=adata.obs_names)
        num_guides_assigned = res.sum(1)
        series.loc[(num_guides_assigned <= max_assignments_per_cell) & (num_guides_assigned != 0)] = res.apply(
            lambda row: row.index[row == 1].tolist(), axis=1
        ).str.join(multiple_grna_assignment_string)
        series.loc[num_guides_assigned > max_assignments_per_cell] = multiple_grna_assigned_key

        if only_return_results:
            return series.values

        adata.obs[assigned_guides_key] = series.values

        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_heatmap(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        order_by: np.ndarray | str | None = None,
        key_to_save_order: str = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Heatmap plotting of guide RNA expression matrix.

        Assuming guides have sparse expression, this function reorders cells
        and plots guide RNA expression so that a nice sparse representation is achieved.
        The cell ordering can be stored and reused in future plots to obtain consistent
        plots before and after analysis of the guide RNA expression.
        Note: This function expects a log-normalized or binary data.

        Args:
            adata: Annotated data matrix containing gRNA values
            layer: Key to the layer containing log normalized count values of the gRNAs.
                   adata.X is used if layer is None.
            order_by: The order of cells in y axis.
                      If None, cells will be reordered to have a nice sparse representation.
                      If a string is provided, adata.obs[order_by] will be used as the order.
                      If a numpy array is provided, the array will be used for ordering.
            key_to_save_order: The obs key to save cell orders in the current plot. Only saves if not None.
            {common_plot_args}
            kwargs: Are passed to sc.pl.heatmap.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
            Order of cells in the y-axis will be saved on `adata.obs[key_to_save_order]` if provided.

        Examples:
            Each cell is assigned to gRNA that occurs at least 5 times in the respective cell, which is then
            visualized using a heatmap.

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_by_threshold(gdo, assignment_threshold=5)
            >>> ga.plot_heatmap(gdo)
        """
        data = adata.X if layer is None else adata.layers[layer]

        if order_by is None:
            if issparse(data):
                max_values = data.max(axis=1).toarray().squeeze()
                data_argmax = data.argmax(axis=1).A.squeeze()
                max_guide_index = np.where(max_values != data.min(axis=1).toarray().squeeze(), data_argmax, -1)
            else:
                max_guide_index = np.where(
                    data.max(axis=1).squeeze() != data.min(axis=1).squeeze(), data.argmax(axis=1).squeeze(), -1
                )
            order = np.argsort(max_guide_index)
        elif isinstance(order_by, str):
            order = np.argsort(adata.obs[order_by])
        else:
            order = order_by

        temp_col_name = f"_tmp_pertpy_grna_plot_{uuid.uuid4()}"
        adata.obs[temp_col_name] = pd.Categorical(["" for _ in range(adata.shape[0])])

        if key_to_save_order is not None:
            adata.obs[key_to_save_order] = pd.Categorical(order)

        try:
            fig = sc.pl.heatmap(
                adata[order, :],
                var_names=adata.var.index.tolist(),
                groupby=temp_col_name,
                cmap="viridis",
                use_raw=False,
                dendrogram=False,
                layer=layer,
                show=False,
                **kwargs,
            )
        finally:
            del adata.obs[temp_col_name]

        if return_fig:
            return fig
        plt.show()
        return None
