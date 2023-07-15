import numpy as np
from anndata import AnnData
from sklearn.cluster import DBSCAN, KMeans

from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace
from pertpy.tools._perturbation_space._clustering import ClusteringSpace


class DifferentialSpace(PerturbationSpace):
    """Subtract mean of the control from the perturbation."""

    def __call__(  # type: ignore
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
        """Subtract mean of the control from the perturbation.
        
        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            reference_key: indicates the control perturbation
            layer_key: if specified and exists in the adata, the pseudobulk computation is done by using it. Otherwise, computation is done with .X
            new_layer_key: the results are stored in the given layer 
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, computation is done with .X
            new_embedding_key: the results are stored in a new embedding named as 'new_embedding_key'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
        """
        
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


class CentroidSpace(PerturbationSpace):
    """Determines the centroids of a pre-computed embedding (e.g. UMAP)."""

    def __call__(
        self,
        adata: AnnData, 
        embedding_key: str = "X_umap", 
        **kwargs) -> AnnData:  # type: ignore
        """
            Takes as input an Anndata object of size cells x genes. 
            If 'embedding_key' is specified and exists in the adata, the clustering is done with that embedding. Otherwise, it raises error
            Returns an new Anndata object in which each observation is a perturbation and its X the centroid
        """
        # TODO test this
        
        if embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")

        embedding = adata.obsm[embedding_key]
        centroids = np.mean(embedding, axis=0)

        ps_adata = adata.copy()
        ps_adata.X = centroids.reshape(1, -1)

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Determines pseudobulks of an AnnData object."""

    def __call__(
        self, 
        adata: AnnData, 
        target_col: str = "perturbations", 
        layer_key: str = None, 
        embedding_key: str = None,
        **kwargs):  # type: ignore
        """Determines pseudobulks of an AnnData object.
        
        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: if specified and exists in the adata, the pseudobulk computation is done by using it. Otherwise, computation is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, computation is done with .X
        """
    
        grouped = adata.obs.groupby(target_col)

        if layer_key:
            if layer_key in adata.layers.keys():
                X = np.empty((len(adata.obs[target_col].unique()), adata.layers[layer_key].shape[1]))
                index = []
                i = 0
                for group_name, group_data in grouped:
                    indices = group_data.index
                    index.append(group_name)
                    X[i, :] = np.mean(adata[indices].layers[layer_key], axis=0)
                    i += 1
            else:
                raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
            
        if embedding_key:
            if embedding_key in adata.obsm.keys():
                X = np.empty((len(adata.obs[target_col].unique()), adata.obsm[embedding_key].shape[1]))
                index = []
                pert_index = 0
                for group_name, group_data in grouped:
                    indices = group_data.index
                    index.append(group_name)
                    X[i, :] = np.mean(adata[indices].obsm[embedding_key], axis=0)
                    pert_index += 1
            else:
                raise ValueError(f"Layer {embedding_key!r} does not exist in the .layers attribute.")
            
        else:
            X = np.empty((len(adata.obs[target_col].unique()), adata.X.shape[1]))
            index = []
            pert_index = 0
            for group_name, group_data in grouped:
                indices = group_data.index
                index.append(group_name)
                X[i, :] = np.mean(adata[indices].X, axis=0)
                pert_index += 1

        ps_data = AnnData(X=X)
        ps_data.obs_names = index

        return ps_data


class KMeansSpace(ClusteringSpace):
    """Cluster the given data using K-Means"""

    def __call__(  # type: ignore
        self,
        adata: AnnData,
        layer_key: str = None,
        embedding_key: str = None,
        cluster_key: str = None,
        copy: bool = False,
        return_object: bool = False,
        **kwargs,
    ) -> AnnData:
        """
        Args:
            adata: Anndata object of size cells x genes
            layer_key: if specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels. Default 'k-means'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
        """

        if copy:
            adata = adata.copy()
            
        if cluster_key is None:
            cluster_key = 'k-means'

        if embedding_key is not None:
            if embedding_key not in adata.obsm_keys():
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            else:
                self.X = adata.obsm[embedding_key]

        elif layer_key is not None:
            if layer_key not in adata.layers.keys():
                raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")
            else:
                self.X = adata.layers[layer_key]

        else:
            self.X = adata.X

        clustering = KMeans(**kwargs).fit(self.X)
        adata.obs[cluster_key] = clustering.labels_

        if return_object:
            return adata, clustering
        
        return adata


class DBSCANSpace(ClusteringSpace):
    """Cluster the given data using DBSCAN"""

    def __call__(  # type: ignore
        self, 
        adata: AnnData, 
        layer_key: str = None, 
        embedding_key: str = None, 
        cluster_key: str = None,
        copy: bool = True, 
        return_object: bool = False,
        **kwargs
    ) -> AnnData:
        """
        Args:
            adata: Anndata object of size cells x genes
            layer_key: if specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels. Default 'k-means'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
        """

        if copy:
            adata = adata.copy()
            
        if cluster_key is None:
            cluster_key = 'dbscan'

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

        clustering = DBSCAN(**kwargs).fit(self.X)
        adata.obs[cluster_key] = clustering.labels_

        if return_object:
            return adata, clustering
        
        return adata
