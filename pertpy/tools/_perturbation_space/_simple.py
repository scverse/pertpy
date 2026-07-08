from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import HDBSCAN, KMeans

from pertpy._logger import logger
from pertpy.tools._perturbation_space._clustering import ClusteringSpace
from pertpy.tools._perturbation_space._perturbation_space import (
    PerturbationSpace,
    _carry_constant_obs,
    _resolve_matrix,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pertpy.tools._distances._distances import Metric


class CentroidSpace(PerturbationSpace):
    """Computes the centroids per perturbation of a pre-computed embedding."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        layer_key: str | None = None,
        embedding_key: str | None = "X_umap",
        keep_obs: bool = True,
    ) -> AnnData:
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

        if keep_obs:
            _carry_constant_obs(ps_adata, adata.obs, target_col)

        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata


class PseudobulkSpace(PerturbationSpace):
    """Calculates pseudobulks."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        groups_col: str | None = None,
        layer_key: str | None = None,
        embedding_key: str | None = None,
        mode: Literal["count_nonzero", "mean", "sum", "var", "median"] = "sum",
    ) -> AnnData:
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


class DistanceSpace(PerturbationSpace):
    """Represents each perturbation by its statistical distance to every other perturbation."""

    def compute(
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        metric: Metric = "edistance",
        layer_key: str | None = None,
        embedding_key: str | None = None,
        groups: Sequence[str] | None = None,
        **kwargs,
    ) -> AnnData:
        """Computes a perturbation space from pairwise distances between perturbations.

        Wraps :meth:`~pertpy.tools.Distance.pairwise` so that any distance metric available in :class:`~pertpy.tools.Distance` defines a perturbation space.
        Each perturbation is represented by its vector of distances to all perturbations, and the full distance matrix is additionally stored in ``.obsp["distances"]`` so it can feed clustering (``metric="precomputed"``), :meth:`nearest_perturbations` and :meth:`plot_similarity` directly.

        Args:
            adata: Anndata object of size cells x genes.
            target_col: `.obs` column that stores the label of the perturbation applied to each cell.
            metric: Distance metric, passed to :class:`~pertpy.tools.Distance`.
            layer_key: If specified, the distances are computed on this layer. Otherwise `.X` or the embedding is used.
            embedding_key: `.obsm` embedding to compute distances from. Mutually exclusive with `layer_key`; defaults to `X_pca` internally when neither is given.
            groups: Subset of perturbations to compute distances for. If None, all perturbations are used.
            **kwargs: Passed to :meth:`~pertpy.tools.Distance.pairwise`.

        Returns:
            AnnData with one observation per perturbation whose `.X` and `.obsp["distances"]` store the pairwise distance matrix.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ds = pt.tl.DistanceSpace()
            >>> ds_adata = ds.compute(mdata["rna"], target_col="gene_target", metric="edistance", embedding_key="X_pca")
        """
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        from pertpy.tools._distances._distances import Distance

        distance = Distance(metric=metric, layer_key=layer_key, obsm_key=embedding_key)
        df = distance.pairwise(
            adata, groupby=target_col, groups=None if groups is None else list(groups), show_progressbar=False, **kwargs
        )
        if isinstance(df, tuple):
            df = df[0]

        index = df.index.astype(str)
        matrix = df.to_numpy(dtype=float)
        ps_adata = AnnData(X=matrix)
        ps_adata.obs_names = index
        ps_adata.var_names = index
        ps_adata.obsp["distances"] = matrix
        ps_adata.obs[target_col] = pd.Categorical(df.index)

        _carry_constant_obs(ps_adata, adata.obs, target_col)
        ps_adata.obs[target_col] = ps_adata.obs[target_col].astype("category")

        return ps_adata


class EmbeddingSpace(PerturbationSpace):
    """Builds a perturbation space from a precomputed per-perturbation embedding."""

    def compute(
        self,
        adata: AnnData,
        embedding: Mapping[str, Sequence[float]] | pd.DataFrame,
        target_col: str = "perturbation",
    ) -> AnnData:
        """Aligns an external per-perturbation embedding to the perturbations present in the data.

        Useful for bringing in perturbation representations that are defined outside the expression matrix, such as gene or drug embeddings from foundation models (scGPT, Geneformer, UCE), knowledge graphs or chemical fingerprints.
        Per-cell embeddings stored in `.obsm` do not need this and can be aggregated with :class:`PseudobulkSpace` or :class:`CentroidSpace` via `embedding_key`.

        Args:
            adata: AnnData whose `.obs[target_col]` holds the perturbation labels to align against.
            embedding: Mapping from perturbation name to embedding vector, or a DataFrame indexed by perturbation name.
            target_col: `.obs` column that stores the label of the perturbation applied to each cell.

        Returns:
            AnnData with one observation per perturbation present in both the data and the embedding.

        Examples:
            >>> import pertpy as pt
            >>> import pandas as pd
            >>> adata = pt.dt.norman_2019()
            >>> gene_embedding = pd.DataFrame(...)  # index: perturbation names, values: embedding
            >>> es = pt.tl.EmbeddingSpace()
            >>> es_adata = es.compute(adata, gene_embedding, target_col="perturbation_name")
        """
        if target_col not in adata.obs:
            raise ValueError(f"Obs {target_col!r} does not exist in the .obs attribute.")

        emb_df = (
            embedding
            if isinstance(embedding, pd.DataFrame)
            else pd.DataFrame.from_dict(dict(embedding), orient="index")
        )
        emb_df = emb_df.copy()
        emb_df.index = emb_df.index.astype(str)

        present = pd.Index(adata.obs[target_col].astype(str).unique())
        keep = present.intersection(emb_df.index)
        if keep.empty:
            raise ValueError(f"No overlap between perturbations in .obs[{target_col!r}] and the embedding index.")
        missing = present.difference(emb_df.index)
        if len(missing):
            logger.warning(
                f"{len(missing)} perturbations are missing from the embedding and were dropped, e.g. {list(missing[:5])}."
            )

        emb_df = emb_df.loc[keep]
        ps_adata = AnnData(X=emb_df.to_numpy(dtype=float))
        ps_adata.obs_names = keep
        ps_adata.obs[target_col] = pd.Categorical(keep)

        _carry_constant_obs(ps_adata, adata.obs, target_col)
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
    """Shared body for KMeansSpace/HDBSCANSpace — resolve coords, fit, write labels."""
    if copy:
        adata = adata.copy()
    coords = _resolve_matrix(adata, layer_key=layer_key, embedding_key=embedding_key)
    fitted = estimator.fit(coords)
    adata.obs[cluster_key] = pd.Categorical(fitted.labels_)
    return (adata, fitted) if return_object else adata


class KMeansSpace(ClusteringSpace):
    """Computes K-Means clustering of the expression values."""

    def compute(
        self,
        adata: AnnData,
        layer_key: str | None = None,
        embedding_key: str | None = None,
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


class HDBSCANSpace(ClusteringSpace):
    """Cluster the given data using HDBSCAN."""

    def compute(
        self,
        adata: AnnData,
        layer_key: str | None = None,
        embedding_key: str | None = None,
        cluster_key: str = "hdbscan",
        copy: bool = True,
        return_object: bool = False,
        **kwargs,
    ) -> tuple[AnnData, object] | AnnData:
        """Computes a clustering using hierarchical density-based spatial clustering of applications (HDBSCAN).

        HDBSCAN extends DBSCAN by converting it into a hierarchical clustering algorithm, removing the need to pick a single density threshold (`eps`) and handling clusters of varying density.

        Args:
            adata: Anndata object of size cells x genes
            layer_key: If specified and exists in the adata, the clustering is done by using it. Otherwise, clustering is done with `.X`.
            embedding_key: if specified and exists in the adata, the clustering is done with that embedding. Otherwise, clustering is done with `.X`.
            cluster_key: name of the .obs column to store the cluster labels.
            copy: if True returns a new Anndata of same size with the new column; otherwise it updates the initial adata
            return_object: if True returns the clustering object
            **kwargs: Are passed to sklearn's HDBSCAN.

        Returns:
            If return_object is True, the adata and the clustering object is returned.
            Otherwise, only the adata is returned. The adata is updated with a new .obs column as specified in cluster_key, that stores the cluster labels.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> hdbscan = pt.tl.HDBSCANSpace()
            >>> hdbscan_adata = hdbscan.compute(mdata["rna"])
        """
        return _run_clustering(
            HDBSCAN(**kwargs),
            adata,
            layer_key=layer_key,
            embedding_key=embedding_key,
            cluster_key=cluster_key,
            copy=copy,
            return_object=return_object,
        )
