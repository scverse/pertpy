import numpy as np
from anndata import AnnData
from sklearn.cluster import KMeans, DBSCAN


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

    def __call__(self, adata: AnnData, target_col: str = "perturbations", layer_key: str = None, *args, **kwargs):
        
        # Create a DataFrame from the .obs attribute of the AnnData object
        obs_df = adata.obs
        # Group the observations based on the 'condition' column
        grouped = obs_df.groupby(target_col)

        if layer_key:
            X = np.empty((len(adata.obs[target_col].unique()), adata.layers[layer_key].shape[1]))
            index = []
            i = 0
            for group_name, group_data in grouped:
                indices = group_data.index
                index.append(group_name)
                X[i, :] = np.mean(adata[indices].layers[layer_key], axis=0)
                i += 1

        else:
            X = np.empty((len(adata.obs[target_col].unique()), adata.X.shape[1]))
            index = []
            i = 0
            for group_name, group_data in grouped:
                indices = group_data.index
                index.append(group_name)
                X[i, :] = np.mean(adata[indices].X, axis=0)
                i += 1

        ps_data = AnnData(X=X)
        ps_data.obs_names = index
            
        return ps_data


class KMeansSpace:
    """Cluster the given data using K-Means"""
    
    def __call__(self, adata: AnnData, layer_key: str = None, embedding_key: str = "X_umap", n_clusters = 3, *args, **kwargs) -> AnnData:
        # TODO pass kwargs
        
        if embedding_key is not None:
            if embedding_key not in adata.obsm_keys():
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            else:
                self.X = adata.obsm[embedding_key]
        
        elif layer_key is not None:
            if layer_key not in adata.obsm_keys():
                raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")
            else:
                self.X = adata.layers[layer_key]
                
        else:
            self.X = adata.X
        
        clustering = KMeans(n_clusters=n_clusters, n_init='auto').fit(self.X)
        adata.obs['K-Means'] = clustering.labels_

        return adata
    
    
class DBScanSpace:
    """Cluster the given data using K-Means"""
    
    def __call__(self, adata: AnnData, layer_key: str = None, embedding_key: str = "X_umap", *args, **kwargs) -> AnnData:
        # TODO pass kwargs
        
        if embedding_key is not None:
            if embedding_key not in adata.obsm_keys():
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            else:
                self.X = adata.obsm[embedding_key]
        
        elif layer_key is not None:
            if layer_key not in adata.obsm_keys():
                raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")
            else:
                self.X = adata.layers[layer_key]
                
        else:
            self.X = adata.X
        
        clustering = DBSCAN().fit(self.X)
        adata.obs['DBSCAN'] = clustering.labels_

        return adata