from __future__ import annotations

import decoupler as dc
import numpy as np
from anndata import AnnData
from sklearn.cluster import DBSCAN, KMeans

from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class CentroidSpace(PerturbationSpace):
    def compute(
        self,
        adata: AnnData,
        embedding_key: str = "X_umap",
    ) -> AnnData:  # type: ignore
        """Computes the centroids of a pre-computed embedding such as UMAP.

        Args:
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
        Returns:
            A new Anndata object in which each observation is a perturbation and its X the centroid.
        """
        if embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")

        embedding = adata.obsm[embedding_key]
        centroids = np.mean(embedding, axis=0)

        ps_adata = adata.copy()
        ps_adata.X = centroids.reshape(1, -1)

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    def compute(
        self, 
        adata: AnnData, 
        target_col: str = "perturbations", 
        layer_key: str = None, 
        embedding_key: str = None, 
        **kwargs
    ) -> AnnData:  # type: ignore
        """Determines pseudobulks of an AnnData object. It uses Decoupler implementation.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with .X
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            mode: How to perform the pseudobulk. Available options are sum, mean or median.
        """
        if "groups_col" not in kwargs:
            kwargs["groups_col"] = "perturbations"

        if layer_key is not None and embedding_key is not None:
            raise ValueError(f"Please, select just either layer or embedding for computation.")

        if layer_key is not None and layer_key not in adata.layers.keys():
            raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
        
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")
        
        if embedding_key is not None:
            if embedding_key not in adata.obsm_keys():
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            else:
                adata_emb = AnnData(X=adata.obsm[embedding_key])
                adata_emb.obs_names = adata.obs_names
                adata_emb.obs = adata.obs
                adata = adata_emb

        ps_adata = dc.get_pseudobulk(adata, sample_col=target_col, layer=layer_key, **kwargs)  # type: ignore

        return ps_adata


class KMeansSpace(ClusteringSpace):
    def compute(  # type: ignore
        self,
        adata: AnnData,
        layer_key: str = None,
        embedding_key: str = None,
        cluster_key: str = None,
        copy: bool = False,
        return_object: bool = False,
        **kwargs,
    ) -> tuple[AnnData, object] | AnnData:
        """Computes K-Means clustering of the expression values.

        Args:
            adata: Anndata object of size cells x genes
            layer_key: if specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels. Default 'k-means'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
            **kwargs: Are passed to sklearn's KMeans.

        Returns:

        """
        if copy:
            adata = adata.copy()

        if cluster_key is None:
            cluster_key = "k-means"
            
        if layer_key is not None and embedding_key is not None:
            raise ValueError(f"Please, select just either layer or embedding for computation.")

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
    ) -> tuple[AnnData, object | AnnData]:
        """Computes a clustering using Density-based spatial clustering of applications (DBSCAN).

        Args:
            adata: Anndata object of size cells x genes
            layer_key: If specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels. Defaults to 'k-means'
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
