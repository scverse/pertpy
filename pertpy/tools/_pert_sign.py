from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from pynndescent import NNDescent
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, issparse


def pert_sign(
    adata: AnnData,
    pert_key: str,
    control: str,
    split_by: Optional[str] = None,
    n_neighbors: int = 20,
    use_rep: Optional[str] = None,
    n_pcs: Optional[int] = None,
    batch_size: Optional[int] = None,
    copy: bool = False,
    **kwargs
):
    """Calculate perturbation signature.

    For each cell, we identify `n_neighbors` cells from the control pool with
    the most similar mRNA expression profiles. The perturbation signature is calculated by subtracting
    the averaged mRNA expression profile of the control neighbors from the mRNA expression profile
    of each cell.

    Args:
        adata: The annotated data object.
        pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
        control: Control category from the `pert_key` column.
        split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
            the perturbation signature for every replicate separately.
        n_neighbors: Number of neighbors from the control to use for the perturbation signature.
        use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
            If `None`, the representation is chosen automatically:
            For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
            If 'X_pca' is not present, it’s computed with default parameters.
        n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        batch_size: Size of batch to calculate the perturbation signature.
            If 'None', the perturbation signature is calcuated in the full mode, requiring more memory.
            The batched mode is very inefficient for sparse data.
        copy: Determines whether a copy of the `adata` is returned.
        **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

    Returns:
        If `copy=True`, returns the copy of `adata` with the perturbation signature in `.layers["X_pert"]`.
        Otherwise writes the perturbation signature directly to `.layers["X_pert"]` of the provided `adata`.
    """
    if copy:
        adata = adata.copy()

    adata.layers["X_pert"] = adata.X.copy()

    control_mask = adata.obs[pert_key] == control

    if split_by is None:
        split_masks = [np.full(adata.n_obs, True, dtype=bool)]
    else:
        split_obs = adata.obs[split_by]
        cats = split_obs.unique()
        split_masks = [split_obs == cat for cat in cats]

    R = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    for split_mask in split_masks:
        control_mask_split = control_mask & split_mask

        R_split = R[split_mask]
        R_control = R[control_mask_split]

        eps = kwargs.pop("epsilon", 0.1)
        nn_index = NNDescent(R_control, **kwargs)
        indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)

        X_control = np.expm1(adata.X[control_mask_split])

        n_split = split_mask.sum()
        n_control = X_control.shape[0]

        if batch_size is None:
            col_indices = np.ravel(indices)
            row_indices = np.repeat(np.arange(n_split), n_neighbors)

            neigh_matrix = csr_matrix(
                (np.ones_like(col_indices, dtype=np.float64), (row_indices, col_indices)), shape=(n_split, n_control)
            )
            neigh_matrix /= n_neighbors
            adata.layers["X_pert"][split_mask] -= np.log1p(neigh_matrix @ X_control)
        else:
            is_sparse = issparse(X_control)
            split_indices = np.where(split_mask)[0]
            for i in range(0, n_split, batch_size):
                size = min(i + batch_size, n_split)
                select = slice(i, size)

                batch = np.ravel(indices[select])
                split_batch = split_indices[select]

                size = size - i

                # sparse is very slow
                means_batch = X_control[batch]
                means_batch = means_batch.toarray() if is_sparse else means_batch
                means_batch = means_batch.reshape(size, n_neighbors, -1).mean(1)

                adata.layers["X_pert"][split_batch] -= np.log1p(means_batch)

    if copy:
        return adata
