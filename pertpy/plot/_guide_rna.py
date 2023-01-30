import logging
from typing import Optional, Union

import numpy as np
import scanpy as sc
from anndata import AnnData


class GuideRnaPlot:
    @staticmethod
    def heatmap(
        adata: AnnData,
        layer: Optional[str] = None,
        order_by: Union[np.ndarray, str, None] = None,
        key_to_save_order: str = None,
        **kwds,
    ) -> None:
        """\
        Simple gRNA plotting
        Parameters
        ----------
        adata
            Annotated data matrix containing gRNA values
        layer
            Key to the layer containing log normalized count values of the gRNAs.
            adata.X is used if layer is None.
        order_by
            The order of cells in y axis. Defaults to None.
            If None, cells will be reordered to have a nice sparse representation.
            If a string is provided, adata.obs[order_by] will be used as the order.
            If a numpy array is provided, the array will be used for ordering.
        key_to_save_order
            The obs key to save cell orders in the current plot. Only saves if not None.
        kwds,
            Are passed to sc.pl.heatmap
        Returns
        -------
        The heatmap plot of the cells versus guide RNAs will be shown.
        Order of cells in the y axis will be saved on adata.obs[key_to_save_order] if `key_to_save_order` is provided.
        """
        grna = AnnData(adata.X if layer is None else adata.layers[layer], var=adata.var, obs=adata.obs[[]])

        # TODO: move to utils
        max_entry = grna.X.max()
        if abs(max_entry - int(max_entry)) < 1e-8 and max_entry > 1:
            logging.warning("The data seems unnormalized. Please log normalize to get a better plot.")
        # UO to here

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
        sc.pl.heatmap(grna, grna.var.index.tolist(), groupby="dummy_group", cmap="viridis", dendrogram=False, **kwds)
        if key_to_save_order is not None:
            adata.obs[key_to_save_order] = order
