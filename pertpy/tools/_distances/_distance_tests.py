from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.metrics import pairwise_distances
from statsmodels.stats.multitest import multipletests

from ._distances import Distance

if TYPE_CHECKING:
    from anndata import AnnData


class DistanceTest:
    """Run permutation tests using a distance of choice between groups of cells.

    Performs Monte-Carlo permutation testing using a distance metric of choice
    as test statistic. Tests all groups of cells against a specified contrast
    group (which normally would be your "control" cells).

    Args:
        metric: Distance metric to use between groups of cells.
        n_perms: Number of permutations to run.
        layer_key: Name of the counts layer containing raw counts to calculate distances for.
                   Mutually exclusive with 'obsm_key'.
                   If equal to `None` the parameter is ignored.
        obsm_key: Name of embedding in adata.obsm to use.
                  Mutually exclusive with 'counts_layer_key'.
                  Defaults to None, but is set to "X_pca" if not set explicitly internally.
        alpha: Significance level.
        correction: Multiple testing correction method.
        cell_wise_metric: Metric to use between single cells.

    Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.distance_example()
        >>> distance_test = pt.tl.DistanceTest("edistance", n_perms=1000)
        >>> tab = distance_test(adata, groupby="perturbation", contrast="control")
    """

    def __init__(
        self,
        metric: str,
        n_perms: int = 1000,
        layer_key: str = None,
        obsm_key: str = None,
        alpha: float = 0.05,
        correction: str = "holm-sidak",
        cell_wise_metric=None,
    ):
        self.metric = metric
        self.n_perms = n_perms

        if layer_key and obsm_key:
            raise ValueError(
                "Cannot use 'counts_layer_key' and 'obsm_key' at the same time.\n"
                "Please provide only one of the two keys."
            )
        if not layer_key and not obsm_key:
            obsm_key = "X_pca"
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self.alpha = alpha
        self.correction = correction
        self.cell_wise_metric = (
            cell_wise_metric if cell_wise_metric else Distance(self.metric, obsm_key=self.obsm_key).cell_wise_metric
        )

        self.distance = Distance(
            self.metric,
            layer_key=self.layer_key,
            obsm_key=self.obsm_key,
            cell_wise_metric=self.cell_wise_metric,
        )

    def __call__(
        self,
        adata: AnnData,
        groupby: str,
        contrast: str,
        show_progressbar: bool = True,
    ) -> pd.DataFrame:
        """Run a permutation test using the specified distance metric, testing all groups of cells against a specified contrast group ("control").

        Args:
            adata: Annotated data matrix.
            groupby: Key in adata.obs for grouping cells.
            contrast: Name of the contrast group.
            show_progressbar: Whether to print progress.

        Returns:
            pandas.DataFrame: Results of the permutation test, with columns:
                - distance: distance between the contrast group and the group
                - pvalue: p-value of the permutation test
                - significant: whether the group is significantly different from the contrast group
                - pvalue_adj: p-value after multiple testing correction
                - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> distance_test = pt.tl.DistanceTest("edistance", n_perms=1000)
            >>> tab = distance_test(adata, groupby="perturbation", contrast="control")
        """
        if self.distance.metric_fct.accepts_precomputed:
            # Much faster if the metric can be called on the precomputed
            # distance matrix, but not all metrics can do that.
            return self.test_precomputed(adata, groupby, contrast, show_progressbar)
        else:
            return self.test_xy(adata, groupby, contrast, show_progressbar)

    def test_xy(self, adata: AnnData, groupby: str, contrast: str, show_progressbar: bool = True) -> pd.DataFrame:
        """Run permutation test for metric not supporting precomputed distances.

        Runs a permutation test for a metric that can not be computed using
        precomputed pairwise distances, but need the actual data points. This is
        generally slower than test_precomputed.

        Args:
            adata: Annotated data matrix.
            groupby: Key in adata.obs for grouping cells.
            contrast: Name of the contrast group.
            show_progressbar: Whether to print progress.

        Returns:
            pandas.DataFrame: Results of the permutation test, with columns:
                - distance: distance between the contrast group and the group
                - pvalue: p-value of the permutation test
                - significant: whether the group is significantly different from the contrast group
                - pvalue_adj: p-value after multiple testing correction
                - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> distance_test = pt.tl.DistanceTest("edistance", n_perms=1000)
            >>> test_results = distance_test.test_xy(adata, groupby="perturbation", contrast="control")
        """
        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group {contrast} not found in {groupby} of adata.obs.")
        fct = track if show_progressbar else lambda iterable: iterable
        embedding = adata.obsm[self.obsm_key]

        # Generate the null distribution
        results = []
        for _permutation in fct(range(self.n_perms)):
            # per perturbation, shuffle with control and compute e-distance
            df = pd.DataFrame(index=groups, columns=["distance"], dtype=float)
            for group in groups:
                if group == contrast:
                    continue
                # Shuffle the labels of the groups
                mask = adata.obs[groupby].isin([group, contrast])
                labels = adata.obs[groupby].values[mask]
                rng = np.random.default_rng()
                shuffled_labels = rng.permutation(labels)
                idx = shuffled_labels == group

                X = embedding[mask][idx]  # shuffled group
                Y = embedding[mask][~idx]  # shuffled contrast
                dist = self.distance(X, Y)

                df.loc[group, "distance"] = dist
            results.append(df.sort_index())

        # Generate the empirical distribution
        for group in groups:
            if group == contrast:
                continue
            X = embedding[adata.obs[groupby] == group]
            Y = embedding[adata.obs[groupby] == contrast]
            df.loc[group, "distance"] = self.distance(X, Y)

        # Evaluate the test
        # count times shuffling resulted in larger distance
        comparison_results = np.array(
            pd.concat([r["distance"] - df["distance"] for r in results], axis=1) > 0,
            dtype=int,
        )
        n_failures = pd.Series(np.clip(np.sum(comparison_results, axis=1), 1, np.inf), index=df.index)
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
        """Run permutation test for metrics that take precomputed distances.

        Args:
            adata: Annotated data matrix.
            groupby: Key in adata.obs for grouping cells.
            contrast: Name of the contrast group.
            cell_wise_metric: Metric to use for pairwise distances.
            verbose: Whether to print progress.

        Returns:
            pandas.DataFrame: Results of the permutation test, with columns:
                - distance: distance between the contrast group and the group
                - pvalue: p-value of the permutation test
                - significant: whether the group is significantly different from the contrast group
                - pvalue_adj: p-value after multiple testing correction
                - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> distance_test = pt.tl.DistanceTest("edistance", n_perms=1000)
            >>> test_results = distance_test.test_precomputed(adata, groupby="perturbation", contrast="control")
        """
        if not self.distance.metric_fct.accepts_precomputed:
            raise ValueError(f"Metric {self.metric} does not accept precomputed distances.")

        groups = adata.obs[groupby].unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group {contrast} not found in {groupby} of adata.obs.")
        fct = track if verbose else lambda iterable: iterable

        # Precompute the pairwise distances
        precomputed_distances = {}
        for group in groups:
            if self.layer_key:
                cells = adata[adata.obs[groupby].isin([group, contrast])].layers[self.layer_key].copy()
            else:
                cells = adata[adata.obs[groupby].isin([group, contrast])].obsm[self.obsm_key].copy()
            pwd = pairwise_distances(cells, cells, metric=self.distance.cell_wise_metric)
            precomputed_distances[group] = pwd

        # Generate the null distribution
        results = []
        for _permutation in fct(range(self.n_perms)):
            # per perturbation, shuffle with control and compute e-distance
            df = pd.DataFrame(index=groups, columns=["distance"], dtype=float)
            for group in groups:
                if group == contrast:
                    continue
                # Shuffle the labels of the groups
                mask = adata.obs[groupby].isin([group, contrast])
                labels = adata.obs[groupby].values[mask]
                rng = np.random.default_rng()
                shuffled_labels = rng.permutation(labels)
                idx = shuffled_labels == group

                precomputed_distance = precomputed_distances[group]
                distance_result = self.distance.metric_fct.from_precomputed(precomputed_distance, idx)

                df.loc[group, "distance"] = distance_result
            results.append(df.sort_index())

        # Generate the empirical distribution
        for group in groups:
            if group == contrast:
                continue
            mask = adata.obs[groupby].isin([group, contrast])
            labels = adata.obs[groupby].values[mask]
            idx = labels == group

            precomputed_distance = precomputed_distances[group]
            distance_result = self.distance.metric_fct.from_precomputed(precomputed_distance, idx)

            df.loc[group, "distance"] = distance_result

        # Evaluate the test
        # count times shuffling resulted in larger distance
        comparison_results = np.array(
            pd.concat([r["distance"] - df["distance"] for r in results], axis=1) > 0,
            dtype=int,
        )
        n_failures = pd.Series(np.clip(np.sum(comparison_results, axis=1), 1, np.inf), index=df.index)
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
