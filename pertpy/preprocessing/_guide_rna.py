from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from rich.progress import track
from scipy.sparse import issparse

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy.preprocessing._guide_rna_mixture import PoissonGaussMixture

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.pyplot import Figure


class GuideAssignment:
    """Assign cells to guide RNAs."""

    def assign_by_threshold(
        self,
        adata: AnnData,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
        only_return_results: bool = False,
    ) -> np.ndarray | None:
        """Simple threshold based gRNA assignment function.

        Each cell is assigned to gRNA with at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            adata: AnnData object containing gRNA values.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_layer: Assigned guide will be saved on adata.layers[output_key].
            only_return_results: Whether to input AnnData is not modified and the result is returned as an :class:`np.ndarray`.

        Examples:
            Each cell is assigned to gRNA that occurs at least 5 times in the respective cell.

            >>> import pertpy as pt
            >>> mdata = pt.data.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_by_threshold(gdo, assignment_threshold=5)
        """
        counts = adata.X if layer is None else adata.layers[layer]
        if scipy.sparse.issparse(counts):
            counts = counts.toarray()

        assigned_grnas = np.where(counts >= assignment_threshold, 1, 0)
        assigned_grnas = scipy.sparse.csr_matrix(assigned_grnas)
        if only_return_results:
            return assigned_grnas
        adata.layers[output_layer] = assigned_grnas

        return None

    def assign_to_max_guide(
        self,
        adata: AnnData,
        assignment_threshold: float,
        layer: str | None = None,
        output_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
        only_return_results: bool = False,
    ) -> np.ndarray | None:
        """Simple threshold based max gRNA assignment function.

        Each cell is assigned to the most expressed gRNA if it has at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            adata: AnnData object containing gRNA values.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_key: Assigned guide will be saved on adata.obs[output_key]. default value is `assigned_guide`.
            no_grna_assigned_key: The key to return if no gRNA is expressed enough.
            only_return_results: Whether to input AnnData is not modified and the result is returned as an np.ndarray.

        Examples:
            Each cell is assigned to the most expressed gRNA if it has at least 5 counts.

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_to_max_guide(gdo, assignment_threshold=5)
        """
        counts = adata.X if layer is None else adata.layers[layer]
        if scipy.sparse.issparse(counts):
            counts = counts.toarray()

        assigned_grna = np.where(
            counts.max(axis=1).squeeze() >= assignment_threshold,
            adata.var.index[counts.argmax(axis=1).squeeze()],
            no_grna_assigned_key,
        )

        if only_return_results:
            return assigned_grna
        adata.obs[output_key] = assigned_grna

        return None

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
        uns_key: str = "guide_assignment_params",
        show_progress: bool = False,
        **mixture_model_kwargs,
    ) -> np.ndarray | None:
        """Assigns gRNAs to cells using a mixture model.

        Args:
            adata: AnnData object containing gRNA values.
            model: The model to use for the mixture model. Currently only `Poisson_Gauss_Mixture` is supported.
            output_key: Assigned guide will be saved on adata.obs[output_key].
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

        if uns_key not in adata.uns:
            adata.uns[uns_key] = {}
        elif type(adata.uns[uns_key]) is not dict:
            raise ValueError(f"adata.uns['{uns_key}'] should be a dictionary. Please remove it or change the key.")

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
            adata.uns[uns_key][gene] = mixture_model.params

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
    def plot_heatmap(
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
            if scipy.sparse.issparse(data):
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
