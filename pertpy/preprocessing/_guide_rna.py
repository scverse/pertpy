from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
import scipy

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


class GuideAssignment:
    """Offers simple guide assigment based on count thresholds."""

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
            adata: Annotated data matrix containing gRNA values
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_layer: Assigned guide will be saved on adata.layers[output_key].
            only_return_results: If True, input AnnData is not modified and the result is returned as an np.ndarray.

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
        no_grna_assigned_key: str = "NT",
        only_return_results: bool = False,
    ) -> np.ndarray | None:
        """Simple threshold based max gRNA assignment function.

        Each cell is assigned to the most expressed gRNA if it has at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            adata: Annotated data matrix containing gRNA values
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_key: Assigned guide will be saved on adata.obs[output_key]. default value is `assigned_guide`.
            no_grna_assigned_key: The key to return if no gRNA is expressed enough.
            only_return_results: If True, input AnnData is not modified and the result is returned as an np.ndarray.

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

    def plot_heatmap(
        self,
        adata: AnnData,
        layer: str | None = None,
        order_by: np.ndarray | str | None = None,
        key_to_save_order: str = None,
        **kwargs,
    ) -> list[Axes]:
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
            kwargs: Are passed to sc.pl.heatmap.

        Returns:
            List of Axes. Alternatively you can pass save or show parameters as they will be passed to sc.pl.heatmap.
            Order of cells in the y-axis will be saved on adata.obs[key_to_save_order] if provided.

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
            axis_group = sc.pl.heatmap(
                adata[order, :],
                var_names=adata.var.index.tolist(),
                groupby=temp_col_name,
                cmap="viridis",
                use_raw=False,
                dendrogram=False,
                layer=layer,
                **kwargs,
            )
        finally:
            del adata.obs[temp_col_name]

        return axis_group
