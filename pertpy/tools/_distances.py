from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from ot import unif
from ot.gromov import gromov_wasserstein
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

"""
DEV NOTES:

User interface:
- A function to get distances between any two or more groups of cells with a
distance metric of choice
- get p-values for permutation test with a distance metric of choice for all or
    some of their groups

Internally:
- There will be two types of PermutationTest classes:
    - One that recomputes the pairwise distance matrix every time and works for
      any metric.
    - One that uses the precomputed pairwise distance matrix and works only for
    metrics that can be called on the pairwise distance matrix. Will be faster.
NOTE: This is a bit redundant, maybe merge? Make it Fast automatically
if Metric is callable on P?
NOTE: Maybe move the tests to a separate file? (e.g. _tests.py) Might lead to
confusion with DE testing though...

TODO:
- Make PermutationTest allow for multiple controls (accept list of controls)
- Inherit docstrings from parent classes

- Options how to handle Metrics for PermutationTest:
    1. (x) Have a Metric class with two function: one of them on X and Y, the other on P.
    The latter is only used by PermutationTest internally. The first one is accessable
    as __call__ to the user.
    2. Have separate, second Metric class for pairwise distances, e.g. P_Edistance.
    3. (x) Have a wrapper function that takes a metric and returns a new metric that
    is callable on X and Y, and uses the pairwise distance matrix P internally.
    This function will also be accessable to the user.
I think 1+3 is the best solution.
"""

# wrapper fcts maybe useful to user (see above)
def distance(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    if metric == "edistance":
        return Edistance()(x, y)
    elif metric == "wasserstein":
        return Wasserstein()(x, y)


def get_distances(
    adata: AnnData,
    groups: list[str],
    metric: str | Metric = "edistance",
    **kwargs,
) -> pd.DataFrame:
    """Get pairwise distances between groups of cells.

    Args:
        adata (AnnData): Annotated data matrix.
        groups (List[str]): List of group names. # or None, then use all groups
        metric (Union[str, Metric], optional): Distance metric. Defaults to 'edistance'.

    Returns:
        pd.DataFrame: Dataframe with pairwise distances.
    """
    raise NotImplementedError("get_distances is not implemented yet.")


class Metric:
    """Metric class of distance metrics between groups of cells. (OUTDATED)"""

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X (np.ndarray): First vector of shape (n_samples, n_features).
            Y (np.ndarray): Second vector of shape (n_samples, n_features).

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    def from_Pidx(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        """Compute a distance between vectors X and Y with precomputed distances.

        Args:
            P (np.ndarray): Pairwise distance matrix of shape (n_samples, n_samples).
            idx (np.ndarray): Boolean array of shape (n_samples,) indicating which
            samples belong to X (or Y, since each metric is symmetric).

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")


class Edistance(Metric):
    """Edistance metric."""

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        # TODO: inherit docstring from parent class
        sigma_X = pairwise_distances(X, X, metric="euclidean").mean()
        sigma_Y = pairwise_distances(Y, Y, metric="euclidean").mean()
        delta = pairwise_distances(X, Y, metric="euclidean").mean()
        return 2 * delta - sigma_X - sigma_Y

    def from_Pidx(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        # TODO: inherit docstring from parent class
        sigma_X = P[idx, :][:, idx].mean()
        sigma_Y = P[~idx, :][:, ~idx].mean()
        delta = P[idx, :][:, ~idx].mean()
        return 2 * delta - sigma_X - sigma_Y


class MMD(Metric):
    # TODO: implement MMD metric
    # e.g.
    # https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    # https://github.com/calico/scmmd
    pass


class Wasserstein(Metric):
    """Wasserstein distance metric.
    NOTE: I bet the Theislab OT people have a better solution for this.
    """

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        C1 = cdist(X, X)
        C2 = cdist(Y, Y)

        C1 /= C1.max()
        C2 /= C2.max()

        p = unif(len(C1))
        q = unif(len(C2))

        gw0, log0 = gromov_wasserstein(C1, C2, p, q, "square_loss", verbose=False, log=True)
        dist = log0["gw_dist"]
        return dist

    def from_Pidx(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        C1 = P[idx, :][:, idx]
        C2 = P[~idx, :][:, ~idx]
        C1 /= C1.max()
        C2 /= C2.max()
        p = unif(len(C1))
        q = unif(len(C2))

        _, log0 = gromov_wasserstein(C1, C2, p, q, "square_loss", verbose=False, log=True)
        dist = log0["gw_dist"]
        return dist


class PseudobulkDistance(Metric):
    """Euclidean distance between pseudobulk vectors."""

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0), ord=2, **kwargs)

    def from_Pidx(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("PseudobulkDistance cannot be called on a pairwise distance matrix.")


class MeanPairwiseDistance(Metric):
    """Mean of the pairwise euclidean distance between two groups of cells."""

    # NOTE: I think this might be basically the same as PseudobulkDistance
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distances(X, Y, **kwargs).mean()

    def from_Pidx(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()


class PermutationTest:
    """Runs permutation test for all groups of cells against a contrast group ("control").

    Args:
    metric (Metric): Distance metric to use.
    n_perms (int): Number of permutations to run.
    embedding (str): Name of embedding to use for distance computation.
    alpha (float): Significance level.
    correction (str): Multiple testing correction method.
    verbose (bool): Whether to print progress.

    Returns:
    pandas.DataFrame: Results of the permutation test, with columns:
        - distance: distance between the contrast group and the group
        - pvalue: p-value of the permutation test
        - significant: whether the group is significantly different from the contrast group
        - pvalue_adj: p-value after multiple testing correction
        - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction
    """

    def __init__(
        self,
        metric: Metric,
        n_perms: int = 1000,
        embedding: str = "X_pca",
        alpha: float = 0.05,
        correction: str = "holm-sidak",
        verbose: bool = True,
    ):
        self.metric = metric
        self.n_perms = n_perms
        self.embedding = embedding
        # self.groups = groups
        self.alpha = alpha
        self.correction = correction
        # self.contrast = contrast
        self.verbose = verbose

    def __call__(self, adata: AnnData, groupby: str, contrast: str) -> pd.DataFrame:
        """Run permutation test for metrics that take x and y."""
        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group '{contrast}' not found in '{groupby}' of adata.obs.")
        fct = tqdm if self.verbose else lambda x: x
        emb = adata.obsm[self.embedding]
        res = []

        # Generate the null distribution
        for i in fct(range(self.n_perms)):
            # per perturbation, shuffle with control and compute e-distance
            df = pd.DataFrame(index=groups, columns=["distance"], dtype=float)
            for group in groups:
                if group == contrast:
                    continue
                # Shuffle the labels of the groups
                mask = adata.obs[groupby].isin([group, contrast])
                labels = adata.obs[groupby].values[mask]
                shuffled_labels = np.random.permutation(labels)
                idx = shuffled_labels == group

                X = emb[mask][idx]  # shuffled group
                Y = emb[mask][~idx]  # shuffled contrast
                dist = self.metric(X, Y)

                df.loc[group, "distance"] = dist
            res.append(df.sort_index())

        # Generate the empirical distribution
        for group in groups:
            if group == contrast:
                continue
            X = emb[adata.obs[groupby] == group]
            Y = emb[adata.obs[groupby] == contrast]
            df.loc[group, "distance"] = self.metric(X, Y)

        # Evaluate the test
        # count times shuffling resulted in larger distance
        results = np.array(pd.concat([r["distance"] - df["distance"] for r in res], axis=1) > 0, dtype=int)
        n_failures = pd.Series(np.clip(np.sum(results, axis=1), 1, np.inf), index=df.index)
        pvalues = n_failures / self.n_perms

        # Apply multiple testing correction
        significant_adj, pvalue_adj, _, _ = multipletests(pvalues.values, alpha=self.alpha, method=self.correction)

        # Aggregate results
        tab = pd.DataFrame(
            {
                "distance": df["distance"],
                "pvalue": pvalues,
                "significant": pvalues < self.alpha,
                "pvalue_adj": pvalue_adj,
                "significant_adj": significant_adj,
            },
            index=df.index,
        )

        # Set the contrast group
        tab.loc[contrast, "distance"] = 0
        tab.loc[contrast, "pvalue"] = 1
        tab.loc[contrast, "significant"] = False
        tab.loc[contrast, "pvalue_adj"] = 1
        tab.loc[contrast, "significant_adj"] = False

        return tab


class FastPermutationTest(PermutationTest):
    """Like PermutationTest, but Metric needs to have a from_Pidx method."""

    # TODO: copy docstring from PermutationTest
    def __init__(
        self,
        metric: Metric,
        n_perms: int = 1000,
        embedding: str = "X_pca",
        alpha: float = 0.05,
        correction: str = "holm-sidak",
        verbose: bool = True,
    ):
        self.metric = metric
        self.n_perms = n_perms
        self.embedding = embedding
        # self.groups = groups
        self.alpha = alpha
        self.correction = correction
        # self.contrast = contrast
        self.verbose = verbose

    def __call__(self, adata: AnnData, groupby: str, contrast: str) -> pd.DataFrame:
        """Run permutation test for metrics that take x and y."""
        dist = "euclidean"

        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group '{contrast}' not found in '{groupby}' of adata.obs.")
        fct = tqdm if self.verbose else lambda x: x
        emb = adata.obsm[self.embedding]
        res = []

        # Precompute the pairwise distances
        pwds = {}
        for group in groups:
            x = adata[adata.obs[groupby].isin([group, contrast])].obsm[self.embedding].copy()
            pwd = pairwise_distances(x, x, metric=dist)
            pwds[group] = pwd

        # Generate the null distribution
        for i in fct(range(self.n_perms)):
            # per perturbation, shuffle with control and compute e-distance
            df = pd.DataFrame(index=groups, columns=["distance"], dtype=float)
            for group in groups:
                if group == contrast:
                    continue
                # Shuffle the labels of the groups
                mask = adata.obs[groupby].isin([group, contrast])
                labels = adata.obs[groupby].values[mask]
                shuffled_labels = np.random.permutation(labels)
                idx = shuffled_labels == group

                # quicker
                P = pwds[group]
                dist = self.metric.from_Pidx(P, idx)

                df.loc[group, "distance"] = dist
            res.append(df.sort_index())

        # Generate the empirical distribution
        for group in groups:
            if group == contrast:
                continue
            # quicker
            mask = adata.obs[groupby].isin([group, contrast])
            labels = adata.obs[groupby].values[mask]
            idx = labels == group
            P = pwds[group]
            dist = self.metric.from_Pidx(P, idx)
            df.loc[group, "distance"] = dist

        # Evaluate the test
        # count times shuffling resulted in larger distance
        results = np.array(pd.concat([r["distance"] - df["distance"] for r in res], axis=1) > 0, dtype=int)
        n_failures = pd.Series(np.clip(np.sum(results, axis=1), 1, np.inf), index=df.index)
        pvalues = n_failures / self.n_perms

        # Apply multiple testing correction
        significant_adj, pvalue_adj, _, _ = multipletests(pvalues.values, alpha=self.alpha, method=self.correction)

        # Aggregate results
        tab = pd.DataFrame(
            {
                "distance": df["distance"],
                "pvalue": pvalues,
                "significant": pvalues < self.alpha,
                "pvalue_adj": pvalue_adj,
                "significant_adj": significant_adj,
            },
            index=df.index,
        )

        # Set the contrast group
        tab.loc[contrast, "distance"] = 0
        tab.loc[contrast, "pvalue"] = 1
        tab.loc[contrast, "significant"] = False
        tab.loc[contrast, "pvalue_adj"] = 1
        tab.loc[contrast, "significant_adj"] = False
        return tab


class Etest(FastPermutationTest):
    """Permutation test for e-distance."""

    # NOTE: I think would be nice for users to have a class for each metric maybe?
    def __init__(
        self,
        n_perms: int = 1000,
        embedding: str = "X_pca",
        groups: list[str] | None = None,
        alpha: float = 0.05,
        correction: str = "holm-sidak",
        contrast: str = "control",
        verbose: bool = True,
    ):
        super().__init__(Edistance(), n_perms, embedding, groups, alpha, correction, contrast, verbose)
