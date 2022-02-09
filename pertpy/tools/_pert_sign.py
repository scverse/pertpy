import numpy as np
import scanpy as sc
from pynndescent import NNDescent
from scanpy.tools._utils import _choose_representation
from scipy.sparse import block_diag, csr_matrix, issparse


def pert_sign(adata, pert_key, control, n_neighbors=20, use_rep=None, n_pcs=None, **kwargs):
    control_mask = adata.obs[pert_key] == control

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    X_control = X[control_mask]
    X_pert = X[~control_mask]

    eps = kwargs.pop("epsilon", 0.1)
    nn_index = NNDescent(X_control, **kwargs)
    indices, _ = nn_index.query(X_pert, k=n_neighbors, epsilon=eps)

    X_control = adata.X[control_mask]
    X_pert = adata.X[~control_mask]

    is_sparse = issparse(X_pert)

    batch_size = 200
    n_pert = indices.shape[0]

    for i in range(0, n_pert, batch_size):
        size = min(i + batch_size, n_pert)
        select = slice(i, size)
        batch = np.ravel(indices[select])

        size = size - i

        # sparse is very slow
        means_batch = X_control[batch].toarray() if is_sparse else X_control[batch]
        means_batch = means_batch.reshape(size, n_neighbors, -1).mean(1)
        X_pert[select] -= means_batch

    #            for sparse
    #            means_block = block_diag([np.ones((1, n_neighbors))] * size, format='csr') / n_neighbors
    #            means_block = means_block @ X_control[batch]
    #            X_pert[select] -= means_block

    adata_pert = sc.AnnData(X=X_pert, obs=adata.obs[~control_mask])
    return adata_pert
