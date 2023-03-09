from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from _distances import Distance
from anndata import AnnData
from rich.progress import track
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests

"""
DEV NOTES:
- There will be a single PermutationTest class that internally does one of the following strategies:
    - recomputes the pairwise distance matrix every time and works for
      any metric that has .
    - uses the precomputed pairwise distance matrix and works only for
    metrics that can be called on the pairwise distance matrix. Will be faster.

TODO:
- Make PermutationTest allow for multiple controls (accept list of controls)
"""


class PermutationTest:
    """Runs permutation tests for all groups of cells against a contrast group ("control").

    Args:
    metric (str): Distance metric to use.
    n_perms (int): Number of permutations to run.
    obsm_key (str): Name of embedding to use for distance computation.
    alpha (float): Significance level.
    correction (str): Multiple testing correction method.
    """

    def __init__(
        self,
        metric: str,
        n_perms: int = 1000,
        obsm_key: str = "X_pca",
        alpha: float = 0.05,
        correction: str = "holm-sidak",
    ):
        self.metric = metric
        self.n_perms = n_perms
        self.obsm_key = obsm_key
        self.alpha = alpha
        self.correction = correction

    def __call__(self, adata: AnnData, groupby: str, contrast: str, verbose: bool = True) -> pd.DataFrame:
        """Run permutation test.

        Args:
        adata (anndata.AnnData): Annotated data matrix.
        groupby (str): Key in adata.obs for grouping cells.
        contrast (str): Name of the contrast group.
        verbose (bool): Whether to print progress.

        Returns:
        pandas.DataFrame: Results of the permutation test, with columns:
            - distance: distance between the contrast group and the group
            - pvalue: p-value of the permutation test
            - significant: whether the group is significantly different from the contrast group
            - pvalue_adj: p-value after multiple testing correction
            - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction
        """

        if Distance(self.metric, self.obsm_key).metric.accepts_precomputed:
            # Much faster if the metric can be called on the precomputed
            # distance matrix, but not all metrics can do that.
            return self.test_precomputed(adata, groupby, contrast, verbose)
        else:
            return self.test_xy(adata, groupby, contrast, verbose)

    def test_xy(self, adata: AnnData, groupby: str, contrast: str, verbose: bool = True) -> pd.DataFrame:
        """Run permutation test for metrics that take x and y."""
        distance = Distance(self.metric, self.obsm_key)
        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group {contrast} not found in {groupby} of adata.obs.")
        fct = track if verbose else lambda x: x
        emb = adata.obsm[self.obsm_key]
        res = []

        # Generate the null distribution
        for _i in fct(range(self.n_perms)):
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
                dist = distance(X, Y)

                df.loc[group, "distance"] = dist
            res.append(df.sort_index())

        # Generate the empirical distribution
        for group in groups:
            if group == contrast:
                continue
            X = emb[adata.obs[groupby] == group]
            Y = emb[adata.obs[groupby] == contrast]
            df.loc[group, "distance"] = distance(X, Y)

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

    def test_precomputed(self, adata: AnnData, groupby: str, contrast: str, verbose: bool = True) -> pd.DataFrame:
        """Run permutation test for metrics that take precomputed distances."""
        dist = "euclidean"  # TODO: make this an argument? This is the metric for the precomputed distances

        distance = Distance(self.metric, self.obsm_key)
        if not distance.metric.accepts_precomputed:
            raise ValueError(f"Metric {self.metric} does not accept precomputed distances.")

        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group {contrast} not found in {groupby} of adata.obs.")
        fct = track if verbose else lambda x: x

        # Precompute the pairwise distances
        pwds = {}
        for group in groups:
            x = adata[adata.obs[groupby].isin([group, contrast])].obsm[self.obsm_key].copy()
            pwd = pairwise_distances(x, x, metric=dist)
            pwds[group] = pwd

        # Generate the null distribution
        res = []
        for _i in fct(range(self.n_perms)):
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

                # much quicker
                P = pwds[group]
                dist = distance.metric_fct.from_precomputed(P, idx)

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
            dist = distance.metric_fct.from_precomputed(P, idx)
            df.loc[group, "distance"] = dist

        # Evaluate the test
        # count times shuffling resulted in larger distance
        results = np.array(pd.concat([r["distance"] - df["distance"] for r in res], axis=1) > 0, dtype=int)
        n_failures = pd.Series(np.clip(np.sum(results, axis=1), 1, np.inf), index=df.index)
        pvalues = n_failures / self.n_perms
        # return results, n_failures, res, df  # DEBUGGING

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
