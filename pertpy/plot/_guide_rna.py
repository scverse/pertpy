from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes


class GuideRnaPlot:
    @staticmethod
    def heatmap(
        adata: AnnData,
        layer: str | None = None,
        order_by: np.ndarray | str | None = None,
        key_to_save_order: str = None,
        **kwds,
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
            order_by: The order of cells in y axis. Defaults to None.
                      If None, cells will be reordered to have a nice sparse representation.
                      If a string is provided, adata.obs[order_by] will be used as the order.
                      If a numpy array is provided, the array will be used for ordering.
            key_to_save_order: The obs key to save cell orders in the current plot. Only saves if not None.
            kwds: Are passed to sc.pl.heatmap.

        Returns:
            List of Axes. Alternatively you can pass save or show parameters as they will be passed to sc.pl.heatmap.
            Order of cells in the y axis will be saved on adata.obs[key_to_save_order] if provided.
        """
        grna = AnnData(adata.X if layer is None else adata.layers[layer], var=adata.var, obs=adata.obs[[]])

        if order_by is None:
            grna.obs["max_guide_index"] = np.where(
                np.array(grna.X.max(axis=1)).squeeze() != grna.X.min(), np.array(grna.X.argmax(axis=1)).squeeze(), -1
            )
            order = np.argsort(grna.obs["max_guide_index"])
        elif isinstance(order_by, str):
            order = adata.obs[order_by]
        else:
            order = order_by

        grna = grna[order].copy()
        grna.obs["dummy_group"] = ""
        if key_to_save_order is not None:
            adata.obs[key_to_save_order] = order
        return sc.pl.heatmap(
            grna, grna.var.index.tolist(), groupby="dummy_group", cmap="viridis", dendrogram=False, **kwds
        )
