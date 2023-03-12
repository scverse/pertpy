from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from anndata import AnnData
from rich.progress import track
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests

"""
DEV NOTES:

User interface:
- A function to get distances between any two or more groups of cells with a
distance metric of choice
- get p-values for permutation test with a distance metric of choice for all or
    some of their groups

Implementation:
- An abstract base class "AbstractDistance" for distance metrics.
- A class for each distance metric that inherits from the abstract base class.
- An interface / API / wrapper class "Distance" which is accessible to the user
and selects the appropriate distance metric class based on the user input.
TODO:
- Inherit docstrings from parent classes
- Store precomputed distances in adata object
- Implement MMD
- Implement Wasserstein distance
"""


class Distance:
    """Distance class.

    Used to compute distances between groups of cells. The distance metric can
    be specified by the user. The class also provides a method to compute
    the pairwise distances between all groups of cells.

    Attributes:
        metric_fct (AbstractDistance): Distance metric function.
        obsm_key (str): Name of embedding in adata.obsm to use.
        metric (str): Name of distance metric.
    """

    def __init__(
        self,
        metric: str = "edistance",
        obsm_key: str = "X_pca",
    ) -> None:
        """Initialize Distance class.

        Args:
            metric (str, optional): Distance metric to use. Defaults to edistance.
            obsm_key (str, optional): Name of embedding in adata.obsm to use.
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
            X (np.ndarray): First vector of shape (n_samples, n_features).
            Y (np.ndarray): Second vector of shape (n_samples, n_features).

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
            adata (AnnData): Annotated data matrix.
            groupby (str): Column name in adata.obs.
            groups (Optional[List[str]], optional): List of groups to compute
            pairwise distances for. Defaults to None (all groups).
            verbose (bool, optional): Whether to show progress bar. Defaults to
            True.

        Returns:
            pd.DataFrame: Dataframe with pairwise distances.
        """
        # TODO: This could make use of the precomputed distances if we store
        # those in the adata object
        groups = adata.obs[groupby].unique() if groups is None else groups
        df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        X = adata.obsm[self.obsm_key].copy()
        y = adata.obs[groupby].copy()
        fct = track if verbose else lambda x: x
        for i, p1 in enumerate(fct(groups)):
            x1 = X[y == p1].copy()
            for p2 in groups[i:]:
                x2 = X[y == p2].copy()
                dist = self.metric_fct(x1, x2, **kwargs)
                df.loc[p1, p2] = dist
                df.loc[p2, p1] = dist
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
            X (np.ndarray): First vector of shape (n_samples, n_features).
            Y (np.ndarray): Second vector of shape (n_samples, n_features).

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    @abstractmethod
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        """Compute a distance between vectors X and Y with precomputed distances.

        Args:
            P (np.ndarray): Pairwise distance matrix of shape (n_samples, n_samples).
            idx (np.ndarray): Boolean array of shape (n_samples,) indicating which
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

    # NOTE: I think this might be basically the same as PseudobulkDistance

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distances(X, Y, **kwargs).mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()
