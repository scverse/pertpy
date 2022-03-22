import numpy as np
import scanpy as sc
from pynndescent import NNDescent
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, issparse


def pert_sign(
    adata,
    pert_key,
    control,
    split_by=None,
    n_neighbors=20,
    use_rep=None,
    n_pcs=None,
    batch_size=None,
    copy=False,
    **kwargs
):
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
