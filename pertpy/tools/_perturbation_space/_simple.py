from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from decoupler import get_pseudobulk as dc_get_pseudobulk
from decoupler import plot_psbulk_samples as dc_plot_psbulk_samples
from sklearn.cluster import DBSCAN, KMeans

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace

if TYPE_CHECKING:
    from matplotlib.pyplot import Figure


class CentroidSpace(PerturbationSpace):
    """Computes the centroids per perturbation of a pre-computed embedding."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        layer_key: str = None,
        embedding_key: str = "X_umap",
        keep_obs: bool = True,
    ) -> AnnData:  # type: ignore
        """Computes the centroids of a pre-computed embedding such as UMAP.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with .X
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            keep_obs: Whether .obs columns in the input AnnData should be kept in the output pseudobulk AnnData. Only .obs columns with the same value for
                each cell of one perturbation are kept.

        Returns:
            AnnData object with one observation per perturbation, storing the embedding data of the
            centroid of the respective perturbation.

        Examples:
            Compute the centroids of a UMAP embedding of the papalexi_2021 dataset:

            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> mdata = pt.dt.papalexi_2021()
            >>> sc.pp.pca(mdata["rna"])
            >>> sc.pp.neighbors(mdata["rna"])
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
            if layer_key not in adata.layers:
                raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
            else:
                X = np.empty((len(adata.obs[target_col].unique()), adata.layers[layer_key].shape[1]))

        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        grouped = adata.obs.groupby(target_col)

        if X is None:
            X = np.empty((len(adata.obs[target_col].unique()), adata.obsm[embedding_key].shape[1]))

        index = []
        for pert_index, (group_name, group_data) in enumerate(grouped):
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

        ps_adata = AnnData(X=X)
        ps_adata.obs_names = index
        ps_adata.obs[target_col] = index

        if embedding_key is not None:
            ps_adata.obsm[embedding_key] = X

        if keep_obs:  # Save the values of the obs columns of interest in the ps_adata object
            obs_df = adata.obs
            obs_df = obs_df.groupby(target_col).agg(
                lambda pert_group: np.nan if len(set(pert_group)) != 1 else list(set(pert_group))[0]
            )
            for obs_name in obs_df.columns:
                if not obs_df[obs_name].isnull().values.any():
                    mapping = {pert: obs_df.loc[pert][obs_name] for pert in index}
                    ps_adata.obs[obs_name] = ps_adata.obs[target_col].map(mapping)

        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Determines pseudobulks using decoupler."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        groups_col: str = None,
        layer_key: str = None,
        embedding_key: str = None,
        **kwargs,
    ) -> AnnData:  # type: ignore
        """Determines pseudobulks of an AnnData object. It uses Decoupler implementation.

        Args:
            adata: Anndata object of size cells x genes
            target_col: .obs column that stores the label of the perturbation applied to each cell.
            groups_col: Optional .obs column that stores a grouping label to consider for pseudobulk computation.
                The summarized expression per perturbation (target_col) and group (groups_col) is computed.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with .X
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            **kwargs: Are passed to decoupler's get_pseuobulk.

        Returns:
             AnnData object with one observation per perturbation.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target")
        """
        if layer_key is not None and embedding_key is not None:
            raise ValueError("Please, select just either layer or embedding for computation.")

        if layer_key is not None and layer_key not in adata.layers:
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

        adata.obs[target_col] = adata.obs[target_col].astype("category")
        ps_adata = dc_get_pseudobulk(adata, sample_col=target_col, layer=layer_key, groups_col=groups_col, **kwargs)  # type: ignore

        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_psbulk_samples(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        groupby: str,
        *,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Plot the pseudobulk samples of an AnnData object.

        Plot the count number vs. the number of cells per pseudobulk sample.

        Args:
            adata: Anndata containing pseudobulk samples.
            groupby: `.obs` column to color the samples by.
            {common_plot_args}
            **kwargs: Are passed to decoupler's plot_psbulk_samples.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata, target_col="Patient", groups_col="Cluster", mode="sum", min_cells=10, min_counts=1000
            ... )
            >>> ps.plot_psbulk_samples(pdata, groupby=["Patient", "Major celltype"], figsize=(12, 4))

        Preview:
            .. image:: /_static/docstring_previews/pseudobulk_samples.png
        """
        fig = dc_plot_psbulk_samples(adata, groupby, return_fig=True, **kwargs)

        if return_fig:
            return fig
        plt.show()
        return None


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
            if layer_key not in adata.layers:
                raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")
            else:
                self.X = adata.layers[layer_key]

        else:
            self.X = adata.X

        clustering = KMeans(**kwargs).fit(self.X)
        adata.obs[cluster_key] = clustering.labels_
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

        if return_object:
            return adata, clustering

        return adata


class DBSCANSpace(ClusteringSpace):
    """Cluster the given data using DBSCAN."""

    def compute(  # type: ignore
        self,
        adata: AnnData,
        layer_key: str = None,
        embedding_key: str = None,
        cluster_key: str = "dbscan",
        copy: bool = True,
        return_object: bool = False,
        **kwargs,
    ) -> tuple[AnnData, object] | AnnData:
        """Computes a clustering using Density-based spatial clustering of applications (DBSCAN).

        Args:
            adata: Anndata object of size cells x genes
            layer_key: If specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with .X
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with .X
            cluster_key: name of the .obs column to store the cluster labels.
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
            if layer_key not in adata.layers:
                raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")
            else:
                self.X = adata.layers[layer_key]

        else:
            self.X = adata.X

        clustering = DBSCAN(**kwargs).fit(self.X)
        adata.obs[cluster_key] = clustering.labels_
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")

        if return_object:
            return adata, clustering

        return adata
