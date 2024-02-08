from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


class GuideRnaPlot:
    @staticmethod
    def heatmap(
        adata: AnnData,
        layer: str | None = None,
        order_by: np.ndarray | str | None = None,
        key_to_save_order: str = None,
        **kwargs,
    ):
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
             kwargs: Are passed to sc.pl.heatmap.

         Returns:
             List of Axes. Alternatively you can pass save or show parameters as they will be passed to sc.pl.heatmap.
             Order of cells in the y axis will be saved on adata.obs[key_to_save_order] if provided.

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
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.preprocessing import GuideAssignment

        ga = GuideAssignment()
        ga.plot_heatmap(adata=adata, layer=layer, order_by=order_by, key_to_save_order=key_to_save_order, kwargs=kwargs)
