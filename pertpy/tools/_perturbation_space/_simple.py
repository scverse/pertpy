from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import DBSCAN, KMeans

from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import PerturbationSpace, _resolve_matrix


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
            keep_obs: Whether .obs columns in the input AnnData should be kept in the output pseudobulk AnnData.
                Only .obs columns with the same value for each cell of one perturbation are kept.

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
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        coords = _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key)

        groups = adata.obs.groupby(target_col, observed=True)
        index = list(groups.groups.keys())
        X = np.empty((len(index), coords.shape[1]), dtype=coords.dtype)
        for pert_index, (_, group_data) in enumerate(groups):
            row_idx = adata.obs_names.get_indexer(group_data.index)
            points = coords[row_idx]
            centroid = points.mean(axis=0)
            closest = np.argmin(np.linalg.norm(points - centroid, axis=1))
            X[pert_index, :] = points[closest]

        ps_adata = AnnData(X=X)
        ps_adata.obs_names = index
        ps_adata.obs[target_col] = index

        if embedding_key is not None:
            ps_adata.obsm[embedding_key] = X

        if keep_obs:  # Save the values of the obs columns of interest in the ps_adata object
            obs_df = adata.obs.groupby(target_col, observed=True).agg(
                lambda pert_group: np.nan if len(set(pert_group)) != 1 else next(iter(set(pert_group)))
            )
            for obs_name in obs_df.columns:
                if not obs_df[obs_name].isnull().values.any():
                    mapping = {pert: obs_df.loc[pert][obs_name] for pert in index}
                    ps_adata.obs[obs_name] = ps_adata.obs[target_col].map(mapping)

        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Calculates pseudobulks."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        groups_col: str = None,
        layer_key: str = None,
        embedding_key: str = None,
        mode: Literal["count_nonzero", "mean", "sum", "var", "median"] = "sum",
    ) -> AnnData:  # type: ignore
        """Determines pseudobulks of an AnnData object.

        Args:
            adata: Anndata object of size cells x genes
            target_col: `.obs` column that stores the label of the perturbation applied to each cell.
            groups_col: Optional `.obs` column that stores a grouping label to consider for pseudobulk computation.
                The summarized expression per perturbation (target_col) and group (groups_col) is computed.
            layer_key: If specified pseudobulk computation is done by using the specified layer. Otherwise, computation is done with `.X`.
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            mode: Pseudobulk aggregation function

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
            if embedding_key not in adata.obsm:
                raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
            adata_emb = AnnData(X=adata.obsm[embedding_key])
            adata_emb.obs_names = adata.obs_names
            adata_emb.obs = adata.obs.copy()
            adata = adata_emb
        else:
            adata = adata.copy()
        adata.obs[target_col] = adata.obs[target_col].astype("category")
        grouping_cols = [target_col] if groups_col is None else [target_col, groups_col]
        original_obs = adata.obs.copy()
        ps_adata = sc.get.aggregate(adata, by=grouping_cols, func=mode, layer=layer_key)

        if None in ps_adata.layers:
            del ps_adata.layers[None]

        if mode in ps_adata.layers:
            ps_adata.X = ps_adata.layers[mode]

        missing_cols = [col for col in original_obs.columns if col not in ps_adata.obs.columns]

        if missing_cols:
            grouped = original_obs.groupby(grouping_cols, observed=False)[missing_cols].first()
            if len(grouping_cols) == 1:
                index = pd.Index(ps_adata.obs[grouping_cols[0]])
            else:
                index = pd.MultiIndex.from_frame(ps_adata.obs[grouping_cols])
            grouped = grouped.reindex(index)
            grouped.index = ps_adata.obs.index
            ps_adata.obs = pd.concat([ps_adata.obs, grouped], axis=1)

        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata


def _run_clustering(
    estimator,
    adata: AnnData,
    *,
    layer_key: str | None,
    embedding_key: str | None,
    cluster_key: str,
    copy: bool,
    return_object: bool,
) -> tuple[AnnData, object] | AnnData:
    """Shared body for KMeansSpace/DBSCANSpace — resolve coords, fit, write labels."""
    if copy:
        adata = adata.copy()
    coords = _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key)
    fitted = estimator.fit(coords)
    adata.obs[cluster_key] = pd.Categorical(fitted.labels_)
    return (adata, fitted) if return_object else adata


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
            layer_key: if specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with `.X`.
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with `.X`.
            cluster_key: name of the .obs column to store the cluster labels. Default 'k-means'
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
            **kwargs: Are passed to sklearn's KMeans.

        Returns:
            If return_object is True, the adata and the clustering object is returned.
            Otherwise, only the adata is returned. The adata is updated with a new .obs column as specified in cluster_key, that stores the cluster labels.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> kmeans = pt.tl.KMeansSpace()
            >>> kmeans_adata = kmeans.compute(mdata["rna"], n_clusters=26)
        """
        return _run_clustering(
            KMeans(**kwargs),
            adata,
            layer_key=layer_key,
            embedding_key=embedding_key,
            cluster_key=cluster_key,
            copy=copy,
            return_object=return_object,
        )


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
            layer_key: If specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with `.X`.
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with `.X`.
            cluster_key: name of the .obs column to store the cluster labels.
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
            **kwargs: Are passed to sklearn's DBSCAN.

        Returns:
            If return_object is True, the adata and the clustering object is returned.
            Otherwise, only the adata is returned. The adata is updated with a new .obs column as specified in cluster_key, that stores the cluster labels.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> dbscan = pt.tl.DBSCANSpace()
            >>> dbscan_adata = dbscan.compute(mdata["rna"])
        """
        return _run_clustering(
            DBSCAN(**kwargs),
            adata,
            layer_key=layer_key,
            embedding_key=embedding_key,
            cluster_key=cluster_key,
            copy=copy,
            return_object=return_object,
        )
