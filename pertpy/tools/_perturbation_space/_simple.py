import numpy as np
from anndata import AnnData
from sklearn.cluster import DBSCAN, KMeans
import decoupler as dc

from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


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

    def compute(
        self,
        adata: AnnData, 
        target_col: str = "perturbations", 
        embedding_key: str = "X_umap", 
        ) -> AnnData:  # type: ignore
        """Takes as input an Anndata object of size cells x genes with a precomputed embedding and compute the centroids of the embedding.
            
        Args:
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            embedding_key: if specified and exists in the adata, the pseudobulk computation is done by using it. Otherwise, raises error.
            Returns an new Anndata object in which each observation is a perturbation and its X the centroid
        """

        if embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
        
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        grouped = adata.obs.groupby(target_col)
        
        X = np.empty((len(adata.obs[target_col].unique()), adata.obsm[embedding_key].shape[1]))
        index = []
        pert_index = 0
        for group_name, group_data in grouped:
            indices = group_data.index
            index.append(group_name)
            X[pert_index, :] = np.mean(adata[indices].obsm[embedding_key], axis=0)
            pert_index += 1
            
        ps_adata = AnnData(X=X)
        ps_adata.obs_names = index

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Determines pseudobulks of an AnnData object."""

    def compute(
        self, 
        adata: AnnData, 
        target_col: str = "perturbations", 
        layer_key: str = None, 
        **kwargs
        )  -> AnnData:  # type: ignore
        """Determines pseudobulks of an AnnData object. It uses Decoupler implementation.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: if specified and exists in the adata, the pseudobulk computation is done by using it. Otherwise, computation is done with .X
            mode: How to perform the pseudobulk. Available options are sum, mean or median. 
        """
    
        if 'groups_col' not in kwargs:
            kwargs['groups_col'] = "perturbations"
            
        if layer_key is not None and layer_key not in adata.layers.keys():
            raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
        
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")
            
        ps_adata = dc.get_pseudobulk(
            adata,
            sample_col=target_col,
            layer=layer_key,
            **kwargs
        )

        return ps_adata


class KMeansSpace(ClusteringSpace):
    """Cluster the given data using K-Means"""

    def compute(  # type: ignore
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
            cluster_key = "k-means"

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

    def compute(  # type: ignore
        self, 
        adata: AnnData, 
        layer_key: str = None, 
        embedding_key: str = None, 
        cluster_key: str = None,
        copy: bool = True,
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
            cluster_key = "dbscan"

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
