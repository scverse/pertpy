import numpy as np
from anndata import AnnData


class DifferentialSpace:
    """Subtract mean of the control from the perturbation."""

    def __call__(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        reference_key: str = "control",
        *,
        layer_key: str = None,
        new_layer_key: str = "differential_response",
        embedding_key: str = None,
        new_embedding_key: str = "differential_response",
        copy: bool = False,
        **kwargs,
    ):
        if reference_key not in adata.obs[target_col].unique():
            raise ValueError(
                f"Reference key {reference_key} not found in {target_col}. {reference_key} must be in obs column {target_col}."
            )

        if copy:
            adata = adata.copy()

        control_mask = adata.obs[target_col] == reference_key

        if layer_key:
            diff_matrix = adata.layers[layer_key] - np.mean(adata.layers[layer_key][~control_mask, :], axis=0)
            adata[new_layer_key] = diff_matrix

        elif embedding_key:
            diff_matrix = adata.obsm[embedding_key] - np.mean(adata.obsm[embedding_key][~control_mask, :], axis=0)
            adata.obsm[new_embedding_key] = diff_matrix
        else:
            diff_matrix = adata.X - np.mean(adata.X[~control_mask, :], axis=0)
            adata.X = diff_matrix

        return adata


class CentroidSpace:
    """Determines the centroids of a pre-computed embedding (e.g. UMAP)."""

    def __call__(self, adata: AnnData, embedding_key: str = "X_umap", *args, **kwargs) -> AnnData:
        # TODO test this
        if embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")

        embedding = adata.obsm[embedding_key]
        centroids = np.mean(embedding, axis=0)

        ps_adata = adata.copy()
        ps_adata.X = centroids.reshape(1, -1)

        return ps_adata


class PseudobulkSpace:
    """Determines pseudobulks of an AnnData object."""

    def __call__(self, *args, **kwargs):
        # TODO implement
        pass
