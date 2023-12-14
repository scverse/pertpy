from __future__ import annotations

import decoupler as dc
import numpy as np
from anndata import AnnData
from sklearn.cluster import DBSCAN, KMeans

from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace


class CentroidSpace(PerturbationSpace):
    """Computes the centroids per perturbation of a pre-computed embedding."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        layer_key: str = None,
        embedding_key: str = "X_umap",
    ) -> AnnData:  # type: ignore
        """Computes the centroids of a pre-computed embedding such as UMAP.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with .X
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.

        Returns:
            AnnData object with one observation per perturbation, storing the embedding data of the
            centroid of the respective perturbation.

        Examples:
            Compute the centroids of a UMAP embedding of the papalexi_2021 dataset:

            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> mdata = pt.dt.papalexi_2021()
            >>> sc.pp.pca(mdata["rna"])
            >>> sc.pp.neighbors(mdata['rna'])
            >>> sc.tl.umap(mdata["rna"])
            >>> cs = pt.tl.CentroidSpace()
            >>> cs_adata = cs.compute(mdata["rna"], target_col="gene_target")
        """

        X = None
        if layer_key is not None and embedding_key is not None:
            raise ValueError("Please, select just either layer or embedding for computation.")

        if embedding_key is not None:
            if embedding_key not in adata.obsm_keys():
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            else:
                X = np.empty((len(adata.obs[target_col].unique()), adata.obsm[embedding_key].shape[1]))

        if layer_key is not None:
            if layer_key not in adata.layers.keys():
                raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
            else:
                X = np.empty((len(adata.obs[target_col].unique()), adata.layers[layer_key].shape[1]))

        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        grouped = adata.obs.groupby(target_col)

        if X is None:
            X = np.empty((len(adata.obs[target_col].unique()), adata.obsm[embedding_key].shape[1]))

        index = []
        pert_index = 0
        for group_name, group_data in grouped:
            indices = group_data.index
            if layer_key is not None:
                points = adata[indices].layers[layer_key]
            elif embedding_key is not None:
                points = adata[indices].obsm[embedding_key]
            else:
                points = adata[indices].X
            index.append(group_name)
            centroid = np.mean(points, axis=0)  # find centroid of cloud of points
            closest_point = min(
                points, key=lambda point: np.linalg.norm(point - centroid)
            )  # Find the point in the array closest to the centroid
            X[pert_index, :] = closest_point
            pert_index += 1

        ps_adata = AnnData(X=X)
        ps_adata.obs_names = index

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Determines pseudobulks using decoupler."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        layer_key: str = None,
        embedding_key: str = None,
        **kwargs,
    ) -> AnnData:  # type: ignore
        """Determines pseudobulks of an AnnData object. It uses Decoupler implementation.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with .X
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            **kwargs: Are passed to decoupler's get_pseuobulk.

        Returns:
             AnnData object with one observation per perturbation.

        Examples:
            >>> import pertpy as pp
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
        """
        if "groups_col" not in kwargs:
            kwargs["groups_col"] = "perturbations"

        if layer_key is not None and embedding_key is not None:
            raise ValueError("Please, select just either layer or embedding for computation.")

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
    """Computes K-Means clustering of the expression values."""

    def compute(  # type: ignore
        self,
        adata: AnnData,
        layer_key: str = None,
        embedding_key: str = None,
        cluster_key: str = "k-means",
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
            If return_object is True, the adata and the clustering object is returned.
            Otherwise, only the adata is returned. The adata is updated with a new .obs column as specified in cluster_key,
             that stores the cluster labels.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> kmeans = pt.tl.KMeansSpace()
            >>> kmeans_adata = kmeans.compute(mdata["rna"], n_clusters=26)
        """
        if copy:
            adata = adata.copy()

        if layer_key is not None and embedding_key is not None:
            raise ValueError("Please, select just either layer or embedding for computation.")

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
        cluster_key: str = "dbscan",
        copy: bool = True,
        return_object: bool = False,
        **kwargs,
    ) -> tuple[AnnData, object | AnnData]:
        """Computes a clustering using Density-based spatial clustering of applications (DBSCAN).

        Args:
            adata: Anndata object of size cells x genes
            layer_key: If specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels. Defaults to 'dbscan'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
            **kwargs: Are passed to sklearn's DBSCAN.

        Returns:
            If return_object is True, the adata and the clustering object is returned.
            Otherwise, only the adata is returned. The adata is updated with a new .obs column as specified in cluster_key,
             that stores the cluster labels.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> dbscan = pt.tl.DBSCANSpace()
            >>> dbscan_adata = dbscan.compute(mdata["rna"])
        """
        if copy:
            adata = adata.copy()

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

        clustering = DBSCAN(**kwargs).fit(self.X)
        adata.obs[cluster_key] = clustering.labels_

        if return_object:
            return adata, clustering

        return adata
