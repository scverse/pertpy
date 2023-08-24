from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    from anndata import AnnData


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
            output_layer: Assigned guide will be saved on adata.layers[output_key]. Defaults to `assigned_guides`.
            only_return_results: If True, input AnnData is not modified and the result is returned as an np.ndarray.
                                 Defaults to False.
        """
        counts = adata.X if layer is None else adata.layers[layer]
        if scipy.sparse.issparse(counts):
            counts = counts.A

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
                   assignment_threshold: If a gRNA is available for at least `assignment_threshold`, it will be recognized as assigned.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_key: Assigned guide will be saved on adata.obs[output_key]. default value is `assigned_guide`.
            no_grna_assigned_key: The key to return if no gRNA is expressed enough.
            only_return_results: If True, input AnnData is not modified and the result is returned as an np.ndarray.
        """
        counts = adata.X if layer is None else adata.layers[layer]
        if scipy.sparse.issparse(counts):
            counts = counts.A

        assigned_grna = np.where(
            counts.max(axis=1).squeeze() >= assignment_threshold,
            adata.var.index[counts.argmax(axis=1).squeeze()],
            no_grna_assigned_key,
        )

        if only_return_results:
            return assigned_grna
        adata.obs[output_key] = assigned_grna

        return None
