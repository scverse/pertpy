import numpy as np
import scanpy as sc
from pynndescent import NNDescent
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, issparse


def pert_sign(adata, pert_key, control, n_neighbors=20, use_rep=None, n_pcs=None, batch_size=None, **kwargs):
    control_mask = adata.obs[pert_key] == control

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    X_control = X[control_mask]
    X_pert = X[~control_mask]

    eps = kwargs.pop("epsilon", 0.1)
    nn_index = NNDescent(X_control, **kwargs)
    indices, _ = nn_index.query(X_pert, k=n_neighbors, epsilon=eps)

    X_control = adata.X[control_mask]
    X_pert = adata.X[~control_mask]

    n_pert = X_pert.shape[0]
    n_control = X_control.shape[0]

    if batch_size is None:
        col_indices = np.ravel(indices)
        row_indices = np.repeat(np.arange(n_pert), n_neighbors)

        neigh_matrix = csr_matrix(
            (np.ones_like(col_indices, dtype=np.float64), (row_indices, col_indices)), shape=(n_pert, n_control)
        )
        neigh_matrix /= n_neighbors

        X_pert -= neigh_matrix @ X_control
    else:
        is_sparse = issparse(X_pert)
        for i in range(0, n_pert, batch_size):
            size = min(i + batch_size, n_pert)
            select = slice(i, size)
            batch = np.ravel(indices[select])

            size = size - i

            # sparse is very slow
            means_batch = X_control[batch]
            means_batch = means_batch.toarray() if is_sparse else means_batch
            means_batch = means_batch.reshape(size, n_neighbors, -1).mean(1)

            X_pert[select] -= means_batch

    adata_pert = sc.AnnData(X=X_pert, obs=adata.obs[~control_mask])
    return adata_pert
