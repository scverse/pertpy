from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from anndata import AnnData
from rich.progress import track
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests


class Distance:
    """Distance class, used to compute distances between groups of cells. 
    
    The distance metric can be specified by the user. The class also provides a 
    method to compute the pairwise distances between all groups of cells.
    Currently available metrics:
    - "edistance": Energy distance
    - "pseudobulk": Pseudobulk distance
    - "mean_pairwise": Mean pairwise distance

    Attributes:
        metric: Name of distance metric.
        obsm_key: Name of embedding in adata.obsm to use.
        metric_fct: Distance metric function.
    """

    def __init__(
        self,
        metric: str = "edistance",
        obsm_key: str = "X_pca",
    ) -> None:
        """Initialize Distance class.

        Args:
            metric: Distance metric to use. (default: "edistance")
            obsm_key: Name of embedding in adata.obsm to use. (default: "X_pca")
        """
        metric_fct: AbstractDistance = None
        if metric == "edistance":
            metric_fct = Edistance()
        elif metric == "pseudobulk":
            metric_fct = PseudobulkDistance()
        elif metric == "mean_pairwise":
            metric_fct = MeanPairwiseDistance()
        else:
            raise ValueError(f"Metric {metric} not recognized.")
        self.metric_fct = metric_fct
        self.obsm_key = obsm_key
        self.metric = metric

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).

        Returns:
            float: Distance between X and Y.
        """
        return self.metric_fct(X, Y, **kwargs)

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        groups: list[str] | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Get pairwise distances between groups of cells.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            groups: List of groups to compute pairwise distances for. 
                If None, uses all groups. (default: None)
            verbose: Whether to show progress bar. (default: True)

        Returns:
            pd.DataFrame: Dataframe with pairwise distances.
        """
        # TODO: This could make use of the precomputed distances if we store
        # those in the adata object
        groups = adata.obs[groupby].unique() if groups is None else groups
        df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        embedding = adata.obsm[self.obsm_key].copy()
        grouping = adata.obs[groupby].copy()
        fct = track if verbose else lambda iterable: iterable
        for index_x, group_x in enumerate(fct(groups)):
            cells_x = embedding[grouping == group_x].copy()
            for group_y in groups[index_x:]:
                cells_y = embedding[grouping == group_y].copy()
                dist = self.metric_fct(cells_x, cells_y, **kwargs)
                df.loc[group_x, group_y] = dist
                df.loc[group_y, group_x] = dist
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric}"
        return df


class AbstractDistance(ABC):
    """Abstract class of distance metrics between two sets of vectors."""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed: bool = None

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    @abstractmethod
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        """Compute a distance between vectors X and Y with precomputed distances.

        Args:
            P: Pairwise distance matrix of shape (n_samples, n_samples).
            idx: Boolean array of shape (n_samples,) indicating which
            samples belong to X (or Y, since each metric is symmetric).

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")


# Specific distance metrics
class Edistance(AbstractDistance):
    """Edistance metric."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        # TODO: inherit docstring from parent class
        sigma_X = pairwise_distances(X, X, metric="euclidean").mean()
        sigma_Y = pairwise_distances(Y, Y, metric="euclidean").mean()
        delta = pairwise_distances(X, Y, metric="euclidean").mean()
        return 2 * delta - sigma_X - sigma_Y

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        # TODO: inherit docstring from parent class
        sigma_X = P[idx, :][:, idx].mean()
        sigma_Y = P[~idx, :][:, ~idx].mean()
        delta = P[idx, :][:, ~idx].mean()
        return 2 * delta - sigma_X - sigma_Y


# class MMD(AbstractDistance):
#     # TODO: implement MMD metric
#     # e.g.
#     # https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
#     # https://github.com/calico/scmmd
#     pass

# class WassersteinDistance(AbstractDistance):
#     """Wasserstein distance metric.
#     NOTE: I bet the Theislab OT people have a better solution for this.
#     """
#     # TODO: implement Wasserstein distance metric


class PseudobulkDistance(AbstractDistance):
    """Euclidean distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0), ord=2, **kwargs)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("PseudobulkDistance cannot be called on a pairwise distance matrix.")


class MeanPairwiseDistance(AbstractDistance):
    """Mean of the pairwise euclidean distance between two groups of cells."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distances(X, Y, **kwargs).mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()
