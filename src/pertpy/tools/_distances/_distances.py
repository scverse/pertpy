from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
from fast_array_utils.conv import to_dense
from numba import jit, prange
from pandas import Series
from rich.progress import track
from scipy.spatial.distance import cosine, mahalanobis
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, r2_score

from pertpy._jax import jax_import
from pertpy._types import CSBase, cast_dense, cast_frame, cast_matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Iterator

    import jax
    from anndata import AnnData
    from ott.geometry.geometry import Geometry


_EUCLIDEAN, _RBF, _POLYNOMIAL = range(3)


@jit(nopython=True, cache=True)
def _pair_kernel(x: np.ndarray, y: np.ndarray, kernel: int, gamma: float, degree: float) -> float:
    """Euclidean distance, rbf kernel or polynomial kernel without offset between two vectors."""
    total = 0.0
    if kernel == _POLYNOMIAL:
        for k in range(x.shape[0]):
            total += x[k] * y[k]
        return (gamma * total) ** degree
    for k in range(x.shape[0]):
        diff = x[k] - y[k]
        total += diff * diff
    return np.sqrt(total) if kernel == _EUCLIDEAN else np.exp(-gamma * total)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _pairwise_kernel_sum_within(
    X: np.ndarray, kernel: int = _EUCLIDEAN, gamma: float = 0.0, degree: float = 0.0
) -> float:
    """Sum of :func:`_pair_kernel` over all ordered pairs of rows of `X`, including every row with itself."""
    n_samples = X.shape[0]
    total = 0.0
    for i in prange((n_samples + 1) // 2):
        row_total = 0.0
        for j in range(i + 1, n_samples):
            row_total += _pair_kernel(X[i], X[j], kernel, gamma, degree)
        mirror = n_samples - 1 - i
        if mirror != i:
            for j in range(mirror + 1, n_samples):
                row_total += _pair_kernel(X[mirror], X[j], kernel, gamma, degree)
        total += row_total
    diagonal = 0.0
    for i in prange(n_samples):
        diagonal += _pair_kernel(X[i], X[i], kernel, gamma, degree)
    return 2 * total + diagonal


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _pairwise_kernel_sum_between(
    X: np.ndarray, Y: np.ndarray, kernel: int = _EUCLIDEAN, gamma: float = 0.0, degree: float = 0.0
) -> float:
    """Sum of :func:`_pair_kernel` over all pairs of a row of `X` and a row of `Y`."""
    total = 0.0
    for i in prange(X.shape[0]):
        for j in range(Y.shape[0]):
            total += _pair_kernel(X[i], Y[j], kernel, gamma, degree)
    return total


@jit(nopython=True, parallel=True, cache=True)
def _ks_statistics(X: np.ndarray, Y: np.ndarray, cdf_x: np.ndarray, cdf_y: np.ndarray) -> np.ndarray:
    """Two-sided two-sample Kolmogorov-Smirnov statistic of every column, as computed by :func:`scipy.stats.ks_2samp`.

    `cdf_x` and `cdf_y` hold the values the empirical CDFs of `X` and `Y` can take.
    """
    n_x, n_y = X.shape[0], Y.shape[0]
    stats = np.empty(X.shape[1], dtype=cdf_x.dtype)
    for k in prange(X.shape[1]):
        x = np.sort(X[:, k])
        y = np.sort(Y[:, k])
        if np.isnan(x[-1]) or np.isnan(y[-1]):
            stats[k] = np.nan
            continue
        i = j = 0
        d_min = d_max = cdf_x[0] - cdf_y[0]
        while i < n_x or j < n_y:
            value = x[i] if j == n_y or (i < n_x and x[i] <= y[j]) else y[j]
            while i < n_x and x[i] <= value:
                i += 1
            while j < n_y and y[j] <= value:
                j += 1
            diff = cdf_x[i] - cdf_y[j]
            d_min = min(d_min, diff)
            d_max = max(d_max, diff)
        d_min = min(max(-d_min, 0), 1)
        stats[k] = d_min if d_min > d_max else d_max
    return stats


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _masked_row_sums(P: np.ndarray, mask: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum of each of the `rows` of `P` over the columns inside and outside of `mask`."""
    inside = np.empty(rows.size)
    outside = np.empty(rows.size)
    for i in prange(rows.size):
        sum_inside = 0.0
        sum_outside = 0.0
        for j in range(P.shape[1]):
            if mask[j]:
                sum_inside += P[rows[i], j]
            else:
                sum_outside += P[rows[i], j]
        inside[i] = sum_inside
        outside[i] = sum_outside
    return inside, outside


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _exponential_kernel_sums(points: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Sum of the exponential kernel between every row of `grid` and all rows of `points`."""
    sums = np.empty(grid.shape[0])
    for g in prange(grid.shape[0]):
        total = 0.0
        for i in range(points.shape[0]):
            total += np.exp(-_pair_kernel(grid[g], points[i], _EUCLIDEAN, 0.0, 0.0) / bandwidth)
        sums[g] = total
    return sums


@jit(nopython=True, cache=True)
def _nb_size_mle(counts: np.ndarray) -> float:
    """Maximum likelihood size of a negative binomial fitted to `counts`, NaN if they are not overdispersed."""
    n = counts.size
    mean = counts.mean()
    var = ((counts - mean) ** 2).mean()
    if var <= mean:
        return np.nan
    n_above = n - np.cumsum(np.bincount(counts))
    lower, upper = -np.inf, np.inf
    log_size = np.log(mean**2 / (var - mean))
    for _ in range(100):
        size = np.exp(log_size)
        score = -n * np.log1p(mean / size)
        slope = n * mean / (size + mean)
        for j in range(n_above.size):
            inverse = 1.0 / (size + j)
            score += n_above[j] * inverse
            slope -= n_above[j] * size * inverse * inverse
        if score > 0:
            lower = log_size
        else:
            upper = log_size
        step = log_size - score / slope
        if not (lower < step < upper and abs(step - log_size) < 1):
            step = (lower + upper) / 2 if np.isfinite(lower + upper) else log_size + (1.0 if score > 0 else -1.0)
        if abs(step - log_size) < 1e-12:
            return np.exp(step)
        log_size = step
    return np.exp(log_size)


@jit(nopython=True, parallel=True, cache=True)
def _nb_log_likelihoods(X: np.ndarray, Y: np.ndarray, epsilon: float) -> np.ndarray:
    """Mean log-likelihood of every column of `Y` under a negative binomial fitted to the same column of `X`.

    NaN for the columns of `X` that are not overdispersed.
    """
    log_likelihoods = np.empty(X.shape[1])
    for k in prange(X.shape[1]):
        counts = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            counts[i] = round(X[i, k])
        mean = counts.mean()
        size = _nb_size_mle(counts)
        log_size_mean = np.log(size + mean + epsilon)
        total = 0.0
        for y in Y[:, k]:
            total += (
                size * (np.log(size + epsilon) - log_size_mean)
                + y * (np.log(mean + epsilon) - log_size_mean)
                + math.lgamma(y + size)
                - math.lgamma(size)
                - math.lgamma(y + 1)
            )
        log_likelihoods[k] = total / Y.shape[0]
    return log_likelihoods


def pairwise_distance_mean(X: np.ndarray, Y: np.ndarray | None = None, metric: str = "euclidean", **kwargs) -> float:
    """Compute mean pairwise distance. Memory-efficient and fast for euclidean.

    If Y is None, computes within-group distances (X to X).

    Args:
        X: First array of shape (n_samples_X, n_features).
        Y: Second array of shape (n_samples_Y, n_features). If None, computes within-group distances.
        metric: Distance metric to use.
        kwargs: Additional keyword arguments passed to the metric function.

    Returns:
        Mean pairwise distance.
    """
    if metric == "euclidean":
        if len(kwargs) > 0:
            warnings.warn(
                "kwargs are not used for euclidean distance.",
                UserWarning,
                stacklevel=2,
            )
        if Y is None:
            # Within-group distance (X to X)
            return _pairwise_kernel_sum_within(X) / len(X) ** 2 if len(X) else 0.0
        else:
            # Between-group distance (X to Y)
            return _pairwise_kernel_sum_between(X, Y) / (len(X) * len(Y)) if len(X) and len(Y) else 0.0
    elif Y is None:
        return pairwise_distances(X, X, metric=metric, **kwargs).mean()
    else:
        return pairwise_distances(X, Y, metric=metric, **kwargs).mean()


def _column_block_width(*arrays: np.ndarray) -> int:
    """Number of columns whose block over the longest of `arrays` stays within a few million elements."""
    return max(1, 2**22 // max(len(array) for array in arrays))


def _column_blocks(X: np.ndarray, width: int, dtype: np.dtype | None = None) -> Iterator[np.ndarray]:
    """Column-major copies of `width` columns of `X` at a time, so every column is reduced with pairwise summation."""
    for start in range(0, X.shape[1], width):
        yield np.asfortranarray(X[:, start : start + width], dtype=dtype)


def _column_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of every column of `X`."""
    if X.shape[1] == 0:
        return np.empty(0), np.empty(0)
    means, stds = zip(
        *((block.mean(axis=0), block.std(axis=0)) for block in _column_blocks(X, _column_block_width(X))), strict=True
    )
    return np.concatenate(means), np.concatenate(stds)


def _sqeuclidean_std(X: np.ndarray, Y: np.ndarray) -> float:
    """Standard deviation of the squared euclidean distances between all rows of `X` and `Y`, without forming them."""
    x_sq, y_sq = (X**2).sum(axis=1), (Y**2).sum(axis=1)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    mean = x_sq.mean() + y_sq.mean() - 2 * x_mean @ y_mean
    width = max(1, 2**22 // max(X.shape[1], 1))
    gram_product = sum(
        np.sum((X[:, start : start + width].T @ X) * (Y[:, start : start + width].T @ Y))
        for start in range(0, X.shape[1], width)
    )
    mean_sq = (
        (x_sq**2).mean()
        + (y_sq**2).mean()
        + 2 * x_sq.mean() * y_sq.mean()
        + 4 * gram_product / (len(X) * len(Y))
        - 4 * (x_sq @ X / len(X)) @ y_mean
        - 4 * x_mean @ (y_sq @ Y / len(Y))
    )
    return float(np.sqrt(max(mean_sq - mean**2, 0.0)))


def _padded_uniform_weights(n: int) -> np.ndarray:
    """Uniform weights of `n` points, zero-padded to a multiple of a quarter of the largest power of two up to `n`."""
    step = 1 << max(n.bit_length() - 3, 0)
    weights = np.zeros(-(-n // step) * step)
    weights[:n] = 1 / n
    return weights


def _group_indices(grouping: pd.Series, groups: Iterable[Hashable]) -> list[np.ndarray]:
    """Ascending positions of the cells of each of `groups` in `grouping`."""
    indices = grouping.groupby(grouping, observed=True, sort=False).indices
    return [np.asarray(indices.get(group, ()), dtype=np.intp) for group in groups]


class MeanVar(NamedTuple):
    mean: float
    variance: float


Metric = Literal[
    "edistance",
    "euclidean",
    "root_mean_squared_error",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "spearman_distance",
    "kendalltau_distance",
    "cosine_distance",
    "r2_distance",
    "mean_pairwise",
    "mmd",
    "wasserstein",
    "sym_kldiv",
    "t_test",
    "ks_test",
    "nb_ll",
    "classifier_proba",
    "classifier_cp",
    "mean_var_distribution",
    "mahalanobis",
]


class Distance:
    """Distance class, used to compute distances between groups of cells.

    The distance metric can be specified by the user. This class also provides a
    method to compute the pairwise distances between all groups of cells.
    Currently available metrics:

    - "edistance": Energy distance (Default metric).
        In essence, it is twice the mean pairwise distance between cells of two
        groups minus the mean pairwise distance between cells within each group
        respectively. More information can be found in
        `Peidli et al. (2023) <https://doi.org/10.1101/2022.08.20.504663>`__.
    - "euclidean": euclidean distance.
        Euclidean distance between the means of cells from two groups.
    - "root_mean_squared_error": euclidean distance.
        Euclidean distance between the means of cells from two groups.
    - "mse": Pseudobulk mean squared error.
        mean squared distance between the means of cells from two groups.
    - "mean_absolute_error": Pseudobulk mean absolute distance.
        Mean absolute distance between the means of cells from two groups.
    - "pearson_distance": Pearson distance.
        Pearson distance between the means of cells from two groups.
    - "spearman_distance": Spearman distance.
        Spearman distance between the means of cells from two groups.
    - "kendalltau_distance": Kendall tau distance.
        Kendall tau distance between the means of cells from two groups.
    - "cosine_distance": Cosine distance.
        Cosine distance between the means of cells from two groups.
    - "r2_distance": coefficient of determination distance.
        Coefficient of determination distance between the means of cells from two groups.
    - "mean_pairwise": Mean pairwise distance.
        Mean of the pairwise euclidean distances between cells of two groups.
    - "mmd": Maximum mean discrepancy
        Maximum mean discrepancy between the cells of two groups.
        Here, uses linear, rbf, and quadratic polynomial MMD. For theory on MMD in single-cell applications, see
        `Lotfollahi et al. (2019) <https://doi.org/10.48550/arXiv.1910.01791>`__.
    - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        Wasserstein distance between the cells of two groups. Uses an
        OTT-JAX implementation of the Sinkhorn algorithm to compute the distance.
        For more information on the optimal transport solver, see
        `Cuturi et al. (2013) <https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>`__.
    - "sym_kldiv": symmetrized Kullback–Leibler divergence distance.
        Kullback–Leibler divergence of the gaussian distributions between cells of two groups.
        Here we fit a gaussian distribution over one group of cells and then calculate the KL divergence on the other, and vice versa.
    - "t_test": t-test statistic.
        T-test statistic measure between cells of two groups.
    - "ks_test": Kolmogorov-Smirnov test statistic.
        Kolmogorov-Smirnov test statistic measure between cells of two groups.
    - "nb_ll": log-likelihood over negative binomial
        Average of log-likelihoods of samples of the secondary group after fitting a negative binomial distribution
        over the samples of the first group.
    - "classifier_proba": probability of a binary classifier
        Average of the classification probability of the perturbation for a binary classifier.
    - "classifier_cp": classifier class projection
        Average of the class
    - "mean_var_distribution": Distance between mean-variance distributions between cells of 2 groups.
       Mean square distance between the mean-variance distributions of cells from 2 groups using Kernel Density Estimation (KDE).
    - "mahalanobis": Mahalanobis distance between the means of cells from two groups.
        It is originally used to measure distance between a point and a distribution.
        in this context, it quantifies the difference between the mean profiles of a target group and a reference group.

    Attributes:
        metric: Name of distance metric.
        layer_key: Name of the counts to use in adata.layers.
        obsm_key: Name of embedding in adata.obsm to use.
        cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.

    Examples:
        >>> import pertpy as pt
        >>> adata = pt.ds.distance_example()
        >>> Distance = pt.tools.Distance(metric="edistance")
        >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
        >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
        >>> D = Distance(X, Y)
    """

    def __init__(
        self,
        metric: Metric = "edistance",
        agg_fct: Callable = np.mean,
        layer_key: str | None = None,
        obsm_key: str | None = None,
        cell_wise_metric: str = "euclidean",
    ):
        """Initialize Distance class.

        Args:
            metric: Distance metric to use.
            agg_fct: Aggregation function to generate pseudobulk vectors.
            layer_key: Name of the counts layer containing raw counts to calculate distances for.
                              Mutually exclusive with 'obsm_key'.
                              Is not used if `None`.
            obsm_key: Name of embedding in adata.obsm to use.
                      Mutually exclusive with 'layer_key'.
                      Defaults to None, but is set to "X_pca" if not explicitly set internally.
            cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.
        """
        metric_fct: AbstractDistance
        self.aggregation_func = agg_fct
        if metric == "edistance":
            metric_fct = Edistance()
        elif metric in ("euclidean", "root_mean_squared_error"):
            metric_fct = EuclideanDistance(self.aggregation_func)
        elif metric == "mse":
            metric_fct = MeanSquaredDistance(self.aggregation_func)
        elif metric == "mean_absolute_error":
            metric_fct = MeanAbsoluteDistance(self.aggregation_func)
        elif metric == "pearson_distance":
            metric_fct = PearsonDistance(self.aggregation_func)
        elif metric == "spearman_distance":
            metric_fct = SpearmanDistance(self.aggregation_func)
        elif metric == "kendalltau_distance":
            metric_fct = KendallTauDistance(self.aggregation_func)
        elif metric == "cosine_distance":
            metric_fct = CosineDistance(self.aggregation_func)
        elif metric == "r2_distance":
            metric_fct = R2ScoreDistance(self.aggregation_func)
        elif metric == "mean_pairwise":
            metric_fct = MeanPairwiseDistance()
        elif metric == "mmd":
            metric_fct = MMD()
        elif metric == "wasserstein":
            metric_fct = WassersteinDistance()
        elif metric == "sym_kldiv":
            metric_fct = SymmetricKLDivergence()
        elif metric == "t_test":
            metric_fct = TTestDistance()
        elif metric == "ks_test":
            metric_fct = KSTestDistance()
        elif metric == "nb_ll":
            metric_fct = NBLL()
        elif metric == "classifier_proba":
            metric_fct = ClassifierProbaDistance()
        elif metric == "classifier_cp":
            metric_fct = ClassifierClassProjection()
        elif metric == "mean_var_distribution":
            metric_fct = MeanVarDistributionDistance()
        elif metric == "mahalanobis":
            metric_fct = MahalanobisDistance(self.aggregation_func)
        else:
            raise ValueError(f"Metric {metric} not recognized.")
        self.metric_fct = metric_fct

        if layer_key and obsm_key:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time.\nPlease provide only one of the two keys."
            )
        if not layer_key and not obsm_key:
            obsm_key = "X_pca"
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self.metric = metric
        self.cell_wise_metric = cell_wise_metric

    def __call__(
        self,
        X: np.ndarray | CSBase,
        Y: np.ndarray | CSBase,
        **kwargs,
    ) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            kwargs: Passed to the metric function.

        Returns:
            float: Distance between X and Y.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
            >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
            >>> D = Distance(X, Y)
        """
        X = to_dense(X)
        Y = to_dense(Y)

        if len(X) == 0 or len(Y) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        return self.metric_fct(X, Y, **kwargs)

    def bootstrap(
        self,
        X: np.ndarray | CSBase,
        Y: np.ndarray | CSBase,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
        **kwargs,
    ) -> MeanVar:
        """Bootstrap computation of mean and variance of the distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            **kwargs: Passed to the metric function.

        Returns:
            Mean and variance of distance between X and Y.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
            >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
            >>> D = Distance.bootstrap(X, Y)
        """
        return self._bootstrap_mode(
            X,
            Y,
            n_bootstraps=n_bootstrap,
            random_state=random_state,
            **kwargs,
        )

    def pairwise(
        self,
        adata: AnnData,
        *,
        groupby: str,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Get pairwise distances between groups of cells.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            groups: List of groups to compute pairwise distances for.
                    If None, uses all groups.
            bootstrap: Whether to bootstrap the distance.
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            show_progressbar: Whether to show progress bar.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            :class:`pandas.DataFrame`: Dataframe with pairwise distances.
            tuple[:class:`pandas.DataFrame`, :class:`pandas.DataFrame`]: Two Dataframes, one for the mean and one for the variance of pairwise distances.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.pairwise(adata, groupby="perturbation")
        """
        obs = adata.obs
        groups = cast("list[str]", obs[groupby].unique()) if groups is None else groups
        group_idx = _group_indices(cast_frame(obs)[groupby], groups)
        n_groups = len(groups)
        dist = np.zeros((n_groups, n_groups))
        dist_var = np.zeros((n_groups, n_groups))
        fct = track if show_progressbar else lambda iterable: iterable

        # Check if metric supports value caching (within/between distances) - more efficient than precomputed matrix
        # This mode is incompatible with bootstrap since cached values would be invalid
        use_value_cache = self.metric_fct.supports_value_cache() and not bootstrap

        if use_value_cache:
            # Value caching mode: precompute within distances per group and between distances per pair
            dense_embedding = cast(
                "np.ndarray",
                adata.layers[self.layer_key] if self.layer_key else adata.obsm[cast("str", self.obsm_key)],
            )
            dense_cells = [dense_embedding[idx] for idx in group_idx]
            within = [float(self.metric_fct.compute_within_distance(cells, **kwargs)) for cells in fct(dense_cells)]
            for index_x in fct(range(n_groups)):
                for index_y in range(index_x + 1, n_groups):
                    between = float(
                        self.metric_fct.compute_between_distance(dense_cells[index_x], dense_cells[index_y], **kwargs)
                    )
                    dist[index_x, index_y] = self.metric_fct.from_cached_values(
                        within[index_x], within[index_y], between, **kwargs
                    )
                    dist[index_y, index_x] = self.metric_fct.from_cached_values(
                        within[index_y], within[index_x], between, **kwargs
                    )

        elif self.metric_fct.accepts_precomputed:
            # Precomputed pairwise distance matrix mode
            # Precompute the pairwise distances if needed
            if f"{self.obsm_key}_{self.cell_wise_metric}_predistances" not in adata.obsp:
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = cast_dense(adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"])
            for index_x in fct(range(n_groups)):
                for index_y in range(index_x, n_groups):
                    if index_x == index_y and not bootstrap:
                        continue
                    # subset the pairwise distance matrix to the two groups
                    idx_xy = np.union1d(group_idx[index_x], group_idx[index_y])
                    sub_pwd = pwd[np.ix_(idx_xy, idx_xy)]
                    sub_idx = np.isin(idx_xy, group_idx[index_x])
                    if not bootstrap:
                        dist[index_x, index_y] = dist[index_y, index_x] = self.metric_fct.from_precomputed(
                            sub_pwd, sub_idx, **kwargs
                        )
                    else:
                        bootstrap_output = self._bootstrap_mode_precomputed(
                            sub_pwd,
                            sub_idx,
                            n_bootstraps=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                        dist[index_x, index_y] = dist[index_y, index_x] = bootstrap_output.mean
                        dist_var[index_x, index_y] = dist_var[index_y, index_x] = bootstrap_output.variance
        else:
            # Standard mode: compute distances directly
            embedding = (
                cast_matrix(adata.layers[self.layer_key])
                if self.layer_key
                else cast_matrix(adata.obsm[cast("str", self.obsm_key)])
            )
            cells = [embedding[idx] for idx in group_idx]
            for index_x in fct(range(n_groups)):
                cells_x = to_dense(cells[index_x])
                for index_y in range(index_x, n_groups):
                    cells_y = cells[index_y]
                    if not bootstrap:
                        # By distance axiom, the distance between a group and itself is 0
                        if index_x != index_y:
                            dist[index_x, index_y] = dist[index_y, index_x] = self(cells_x, cells_y, **kwargs)
                    else:
                        bootstrap_output = self.bootstrap(
                            cells_x,
                            cells_y,
                            n_bootstrap=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                        dist[index_x, index_y] = dist[index_y, index_x] = bootstrap_output.mean
                        dist_var[index_x, index_y] = dist_var[index_y, index_x] = bootstrap_output.variance

        df = pd.DataFrame(dist, index=groups, columns=groups)
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric}"  # type: ignore[attr-defined]

        if not bootstrap:
            return df
        else:
            df = df.fillna(0)
            df_var = pd.DataFrame(dist_var, index=groups, columns=groups)
            df_var.index.name = groupby
            df_var.columns.name = groupby
            df_var = df_var.fillna(0)
            df_var.name = f"pairwise {self.metric} variance"  # type: ignore[attr-defined]

            return df, df_var

    def onesided_distances(
        self,
        adata: AnnData,
        *,
        groupby: str,
        selected_group: str | None = None,
        groups: list[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.Series | tuple[pd.Series, pd.Series]:
        """Get distances between one selected cell group and the remaining other cell groups.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            selected_group: Group to compute pairwise distances to all other.
            groups: List of groups to compute distances to selected_group for.
                    If None, uses all groups.
            bootstrap: Whether to bootstrap the distance.
            n_bootstrap: Number of bootstrap samples.
            random_state: Random state for bootstrapping.
            show_progressbar: Whether to show progress bar.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            :class:`pandas.DataFrame`: Dataframe with distances of groups to selected_group.
            tuple[:class:`pandas.DataFrame`, :class:`pandas.DataFrame`]: Two Dataframes, one for the mean and one for the variance of distances of groups to selected_group.


        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.onesided_distances(adata, groupby="perturbation", selected_group="control")
        """
        if self.metric == "classifier_cp":
            if bootstrap:
                raise NotImplementedError("Currently, ClassifierClassProjection does not support bootstrapping.")
            return self.metric_fct.onesided_distances(  # type: ignore
                adata,
                groupby=groupby,
                selected_group=selected_group,
                groups=groups,
                show_progressbar=show_progressbar,
                n_jobs=n_jobs,
                **kwargs,
            )

        obs = cast_frame(adata.obs)
        groups = cast("list[str]", obs[groupby].unique()) if groups is None else groups
        *group_idx, selected_idx = _group_indices(obs[groupby], [*groups, selected_group])
        df = pd.Series(index=groups, dtype=float)
        if bootstrap:
            df_var = pd.Series(index=groups, dtype=float)
        fct = track if show_progressbar else lambda iterable: iterable

        # Check if metric supports value caching (within/between distances) - more efficient than precomputed matrix
        # This mode is incompatible with bootstrap since cached values would be invalid
        use_value_cache = self.metric_fct.supports_value_cache() and not bootstrap

        if use_value_cache:
            # Value caching mode: precompute within distances per group and between distances per pair
            dense_embedding = cast(
                "np.ndarray",
                adata.layers[self.layer_key] if self.layer_key else adata.obsm[cast("str", self.obsm_key)],
            )

            # Precompute within distance for selected_group (only need it once)
            cells_selected = dense_embedding[selected_idx]
            within_selected = self.metric_fct.compute_within_distance(cells_selected, **kwargs)

            # Precompute within distances for each group and between distances to selected_group
            for index_x, group_x in enumerate(fct(groups)):
                if group_x == selected_group:
                    df.loc[group_x] = 0.0  # by distance axiom
                else:
                    dense_cells_x = dense_embedding[group_idx[index_x]]

                    # Compute within distance for this group
                    within_x = self.metric_fct.compute_within_distance(dense_cells_x, **kwargs)

                    # Compute between distance to selected_group
                    between = self.metric_fct.compute_between_distance(dense_cells_x, cells_selected, **kwargs)

                    # Compute distance from cached values
                    dist = self.metric_fct.from_cached_values(within_x, within_selected, between, **kwargs)
                    df.loc[group_x] = dist

        elif self.metric_fct.accepts_precomputed:
            # Precomputed pairwise distance matrix mode
            # Precompute the pairwise distances if needed
            if f"{self.obsm_key}_{self.cell_wise_metric}_predistances" not in adata.obsp:
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = cast_dense(adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"])
            for index_x, group_x in enumerate(fct(groups)):
                if group_x == selected_group:
                    df.loc[group_x] = 0.0  # by distance axiom
                else:
                    # subset the pairwise distance matrix to the two groups
                    idx_xy = np.union1d(group_idx[index_x], selected_idx)
                    sub_pwd = pwd[np.ix_(idx_xy, idx_xy)]
                    sub_idx = np.isin(idx_xy, group_idx[index_x])
                    if not bootstrap:
                        dist = self.metric_fct.from_precomputed(sub_pwd, sub_idx, **kwargs)
                        df.loc[group_x] = dist
                    else:
                        bootstrap_output = self._bootstrap_mode_precomputed(
                            sub_pwd,
                            sub_idx,
                            n_bootstraps=n_bootstrap,
                            random_state=random_state,
                            **kwargs,
                        )
                        df.loc[group_x] = bootstrap_output.mean
                        df_var.loc[group_x] = bootstrap_output.variance
        else:
            # Standard mode: compute distances directly
            embedding = (
                cast_matrix(adata.layers[self.layer_key])
                if self.layer_key
                else cast_matrix(adata.obsm[cast("str", self.obsm_key)])
            )
            cells_y = to_dense(embedding[selected_idx])
            for index_x, group_x in enumerate(fct(groups)):
                cells_x = embedding[group_idx[index_x]]
                if not bootstrap:
                    # By distance axiom, the distance between a group and itself is 0
                    dist = 0.0 if group_x == selected_group else self(cells_x, cells_y, **kwargs)
                    df.loc[group_x] = dist
                else:
                    bootstrap_output = self.bootstrap(
                        cells_x,
                        cells_y,
                        n_bootstrap=n_bootstrap,
                        random_state=random_state,
                        **kwargs,
                    )
                    # In the bootstrap case, distance of group to itself is a mean and can be non-zero
                    df.loc[group_x] = bootstrap_output.mean
                    df_var.loc[group_x] = bootstrap_output.variance
        df.index.name = groupby
        df.name = f"{self.metric} to {selected_group}"
        if not bootstrap:
            return df
        else:
            df_var.index.name = groupby
            df_var = df_var.fillna(0)
            df_var.name = f"pairwise {self.metric} variance to {selected_group}"

            return df, df_var

    def precompute_distances(self, adata: AnnData, n_jobs: int = -1) -> None:
        """Precompute pairwise distances between all cells, writes to adata.obsp.

        The precomputed distances are stored in adata.obsp under the key
        '{self.obsm_key}_{cell_wise_metric}_predistances', as they depend on
        both the cell-wise metric and the embedding used.

        Args:
            adata: Annotated data matrix.
            n_jobs: Number of cores to use. Defaults to -1 (all).

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.distance_example()
            >>> distance = pt.tools.Distance(metric="edistance")
            >>> distance.precompute_distances(adata)
        """
        cells = adata.layers[self.layer_key] if self.layer_key else adata.obsm[cast("str", self.obsm_key)].copy()  # type: ignore[union-attr]
        pwd = pairwise_distances(cells, cells, metric=self.cell_wise_metric, n_jobs=n_jobs)
        adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"] = pwd

    def compare_distance(
        self,
        pert: np.ndarray,
        pred: np.ndarray,
        ctrl: np.ndarray,
        mode: Literal["simple", "scaled"] = "simple",
        fit_to_pert_and_ctrl: bool = False,
        **kwargs,
    ) -> float:
        """Compute the score of simulating a perturbation.

        Args:
            pert: Real perturbed data.
            pred: Simulated perturbed data.
            ctrl: Control data
            mode: Mode to use.
            fit_to_pert_and_ctrl: Scales data based on both `pert` and `ctrl` if True, otherwise only on `ctrl`.
            kwargs: Additional keyword arguments passed to the metric function.
        """
        if mode == "simple":
            pass  # nothing to be done
        elif mode == "scaled":
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler().fit(np.vstack((pert, ctrl)) if fit_to_pert_and_ctrl else ctrl)
            pred = scaler.transform(pred)
            pert = scaler.transform(pert)
        else:
            raise ValueError(f"Unknown mode {mode}. Please choose simple or scaled.")

        d1 = self.metric_fct(pert, pred, **kwargs)
        d2 = self.metric_fct(ctrl, pred, **kwargs)
        return d1 / d2

    def _bootstrap_mode(self, X, Y, n_bootstraps=100, random_state=0, **kwargs) -> MeanVar:
        rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstraps):
            X_bootstrapped = X[rng.choice(a=X.shape[0], size=X.shape[0], replace=True)]
            Y_bootstrapped = Y[rng.choice(a=Y.shape[0], size=Y.shape[0], replace=True)]

            distance = self(X_bootstrapped, Y_bootstrapped, **kwargs)
            distances.append(distance)

        return MeanVar(mean=float(np.mean(distances)), variance=float(np.var(distances)))

    def _bootstrap_mode_precomputed(self, sub_pwd, sub_idx, n_bootstraps=100, random_state=0, **kwargs) -> MeanVar:
        rng = np.random.default_rng(random_state)
        pos_idx = np.flatnonzero(sub_idx)
        neg_idx = np.flatnonzero(~sub_idx)

        distances = []
        for _ in range(n_bootstraps):
            # To maintain the number of cells for both groups (whatever balancing they may have),
            # we sample the positive and negative indices separately
            bootstrap_pos_idx = rng.choice(a=pos_idx, size=pos_idx.size, replace=True)
            bootstrap_neg_idx = rng.choice(a=neg_idx, size=neg_idx.size, replace=True)
            bootstrap_idx = np.concatenate([bootstrap_pos_idx, bootstrap_neg_idx])

            bootstrap_sub_idx = sub_idx[bootstrap_idx]
            bootstrap_sub_pwd = sub_pwd[np.ix_(bootstrap_idx, bootstrap_idx)]

            distance = self.metric_fct.from_precomputed(bootstrap_sub_pwd, bootstrap_sub_idx, **kwargs)
            distances.append(distance)

        return MeanVar(mean=float(np.mean(distances)), variance=float(np.var(distances)))


class AbstractDistance(ABC):
    """Abstract class of distance metrics between two sets of vectors."""

    @abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed: bool | None = None

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Compute distance between vectors X and Y.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            kwargs: Passed to the metrics function.

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    @abstractmethod
    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        """Compute a distance between vectors X and Y with precomputed distances.

        Args:
            P: Pairwise distance matrix of shape (n_samples, n_samples).
            idx: Boolean array of shape (n_samples,) indicating which samples belong to X (or Y, since each metric is symmetric).
            kwargs: Passed to the metrics function.

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric class is abstract.")

    def supports_value_cache(self) -> bool:
        """Whether this metric supports value-level caching (within/between distances).

        Returns:
            bool: True if value caching is supported, False otherwise.
        """
        return False

    def compute_within_distance(self, X: np.ndarray, **kwargs) -> float:
        """Compute within-group distance statistic for caching.

        Only called if supports_value_cache() returns True.
        This represents the mean pairwise distance within a single group.

        Args:
            X: Vector of shape (n_samples, n_features) for a single group.
            kwargs: Additional keyword arguments.

        Returns:
            float: Cached within-group distance statistic.
        """
        raise NotImplementedError("Metric does not support value caching.")

    def compute_between_distance(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Compute between-group distance statistic for caching.

        Only called if supports_value_cache() returns True.
        This represents the mean pairwise distance between two groups.

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            kwargs: Additional keyword arguments.

        Returns:
            float: Cached between-group distance statistic.
        """
        raise NotImplementedError("Metric does not support value caching.")

    def from_cached_values(self, within_X: float, within_Y: float, between: float, **kwargs) -> float:
        """Compute distance using precomputed cached values.

        Only called if supports_value_cache() returns True and values have been cached.

        Args:
            within_X: Precomputed within-group distance for group X.
            within_Y: Precomputed within-group distance for group Y.
            between: Precomputed between-group distance for pair (X, Y).
            kwargs: Additional keyword arguments.

        Returns:
            float: Distance between X and Y.
        """
        raise NotImplementedError("Metric does not support value caching.")


class Edistance(AbstractDistance):
    """Edistance metric."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True
        self.cell_wise_metric = "euclidean"

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        within_X = pairwise_distance_mean(X, metric=self.cell_wise_metric, **kwargs)
        within_Y = pairwise_distance_mean(Y, metric=self.cell_wise_metric, **kwargs)
        between = pairwise_distance_mean(X, Y, metric=self.cell_wise_metric, **kwargs)
        return 2 * between - within_X - within_Y

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, total: float | None = None, **kwargs) -> float:
        """Compute the edistance between the cells in and out of `idx` from their pairwise distances `P`.

        Only the rows of the smaller group are read when the sum of all distances is passed as `total`, which requires `P` to be symmetric.
        """
        idx = np.asarray(idx, dtype=bool)
        if total is None:
            n_x = np.count_nonzero(idx)
            n_y = idx.size - n_x
            within_x, between = _masked_row_sums(P, idx, np.flatnonzero(idx))
            within_y, _ = _masked_row_sums(P, ~idx, np.flatnonzero(~idx))
            return 2 * between.sum() / (n_x * n_y) - within_x.sum() / n_x**2 - within_y.sum() / n_y**2
        smaller = idx if 2 * np.count_nonzero(idx) <= idx.size else ~idx
        inside, outside = _masked_row_sums(P, smaller, np.flatnonzero(smaller))
        n_smaller = np.count_nonzero(smaller)
        n_larger = idx.size - n_smaller
        within_smaller = inside.sum() / n_smaller**2
        between = outside.sum() / (n_smaller * n_larger)
        within_larger = (total - inside.sum() - 2 * outside.sum()) / n_larger**2
        return 2 * between - within_smaller - within_larger

    def supports_value_cache(self) -> bool:
        """Edistance benefits from caching within and between distances."""
        return True

    def compute_within_distance(self, X: np.ndarray, **kwargs) -> float:
        """Compute within-group distance (mean pairwise distance within group)."""
        return pairwise_distance_mean(X, metric=self.cell_wise_metric, **kwargs)

    def compute_between_distance(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Compute between-group distance (mean pairwise distance between groups)."""
        return pairwise_distance_mean(X, Y, metric=self.cell_wise_metric, **kwargs)

    def from_cached_values(self, within_X: float, within_Y: float, between: float, **kwargs) -> float:
        """Compute edistance using cached within and between distances."""
        return 2 * between - within_X - within_Y


class MMD(AbstractDistance):
    """Linear Maximum Mean Discrepancy."""

    # Taken in parts from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(
        self, X: np.ndarray, Y: np.ndarray, *, kernel: str = "linear", gamma: float = 1.0, degree: float = 2, **kwargs
    ) -> float:
        return self.from_cached_values(
            self.compute_within_distance(X, kernel=kernel, gamma=gamma, degree=degree),
            self.compute_within_distance(Y, kernel=kernel, gamma=gamma, degree=degree),
            self.compute_between_distance(X, Y, kernel=kernel, gamma=gamma, degree=degree),
        )

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MMD cannot be called on a pairwise distance matrix.")

    def supports_value_cache(self) -> bool:
        """MMD benefits from caching within and between kernel means."""
        return True

    def compute_within_distance(
        self, X: np.ndarray, *, kernel: str = "linear", gamma: float = 1.0, degree: float = 2, **kwargs
    ) -> float:
        """Compute within-group kernel mean (mean of kernel matrix within group)."""
        return self._kernel_mean(X, None, kernel=kernel, gamma=gamma, degree=degree)

    def compute_between_distance(
        self, X: np.ndarray, Y: np.ndarray, *, kernel: str = "linear", gamma: float = 1.0, degree: float = 2, **kwargs
    ) -> float:
        """Compute between-group kernel mean (mean of kernel matrix between groups)."""
        return self._kernel_mean(X, Y, kernel=kernel, gamma=gamma, degree=degree)

    @staticmethod
    def _kernel_mean(X: np.ndarray, Y: np.ndarray | None, *, kernel: str, gamma: float, degree: float) -> float:
        """Mean of the kernel matrix between `X` and `Y`, or within `X` if `Y` is `None`."""
        X = np.asarray(to_dense(X), dtype=np.float64)
        Y = X if Y is None else np.asarray(to_dense(Y), dtype=np.float64)
        if kernel == "linear":
            return float(X.mean(axis=0) @ Y.mean(axis=0))
        kernels = {"rbf": _RBF, "poly": _POLYNOMIAL}
        if kernel not in kernels:
            raise ValueError(f"Kernel {kernel} not recognized.")
        if Y is X:
            return _pairwise_kernel_sum_within(X, kernels[kernel], gamma, degree) / len(X) ** 2
        return _pairwise_kernel_sum_between(X, Y, kernels[kernel], gamma, degree) / (len(X) * len(Y))

    def from_cached_values(self, within_X: float, within_Y: float, between: float, **kwargs) -> float:
        """Compute MMD using cached within and between kernel means."""
        return within_X + within_Y - 2 * between


class WassersteinDistance(AbstractDistance):
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False
        with jax_import("The 'wasserstein' distance metric"):
            import jax
            from ott.solvers.linear.sinkhorn import Sinkhorn
        self.solver = jax.jit(Sinkhorn())

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        import jax.numpy as jnp
        from ott.geometry.pointcloud import PointCloud

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        epsilon = 0.05 * _sqeuclidean_std(X, Y)
        a, b = _padded_uniform_weights(len(X)), _padded_uniform_weights(len(Y))
        X = np.pad(X, ((0, len(a) - len(X)), (0, 0)))
        Y = np.pad(Y, ((0, len(b) - len(Y)), (0, 0)))
        geom = PointCloud(jnp.asarray(X), jnp.asarray(Y), epsilon=epsilon)
        return self.solve_ot_problem(geom, a=jnp.asarray(a), b=jnp.asarray(b), **kwargs)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        import jax.numpy as jnp
        from ott.geometry.geometry import Geometry

        P = np.asarray(P, dtype=np.float64)
        geom = Geometry(cost_matrix=jnp.asarray(P[idx, :][:, ~idx]))
        return self.solve_ot_problem(geom, **kwargs)

    def solve_ot_problem(self, geom: Geometry, a: jax.Array | None = None, b: jax.Array | None = None, **kwargs):
        from ott.problems.linear.linear_problem import LinearProblem

        ot_prob = LinearProblem(geom, a=a, b=b)
        ot = self.solver(ot_prob, **kwargs)
        cost = float(ot.reg_ot_cost)

        # Check for NaN or invalid cost
        if not np.isfinite(cost):
            return 1.0
        else:
            return cost


class EuclideanDistance(AbstractDistance):
    """Euclidean distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(
            self.aggregation_func(X, axis=0) - self.aggregation_func(Y, axis=0),
            ord=2,
            **kwargs,
        )

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("EuclideanDistance cannot be called on a pairwise distance matrix.")


class MeanSquaredDistance(AbstractDistance):
    """Mean squared distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return (
            np.linalg.norm(
                self.aggregation_func(X, axis=0) - self.aggregation_func(Y, axis=0),
                ord=2,
                **kwargs,
            )
            ** 2
            / X.shape[1]
        )

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MeanSquaredDistance cannot be called on a pairwise distance matrix.")


class MeanAbsoluteDistance(AbstractDistance):
    """Absolute (Norm-1) distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return (
            np.linalg.norm(
                self.aggregation_func(X, axis=0) - self.aggregation_func(Y, axis=0),
                ord=1,
                **kwargs,
            )
            / X.shape[1]
        )

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MeanAbsoluteDistance cannot be called on a pairwise distance matrix.")


class MeanPairwiseDistance(AbstractDistance):
    """Mean of the pairwise euclidean distance between two groups of cells."""

    # NOTE: This is not a metric in the mathematical sense.

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distance_mean(X, Y, **kwargs)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()


class PearsonDistance(AbstractDistance):
    """Pearson distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - pearsonr(self.aggregation_func(X, axis=0), self.aggregation_func(Y, axis=0))[0]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("PearsonDistance cannot be called on a pairwise distance matrix.")


class SpearmanDistance(AbstractDistance):
    """Spearman distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - spearmanr(self.aggregation_func(X, axis=0), self.aggregation_func(Y, axis=0))[0]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("SpearmanDistance cannot be called on a pairwise distance matrix.")


class KendallTauDistance(AbstractDistance):
    """Kendall-tau distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        x, y = self.aggregation_func(X, axis=0), self.aggregation_func(Y, axis=0)
        n = len(x)
        tau_corr = kendalltau(x, y).statistic
        tau_dist = (1 - tau_corr) * n * (n - 1) / 4
        return tau_dist

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("KendallTauDistance cannot be called on a pairwise distance matrix.")


class CosineDistance(AbstractDistance):
    """Cosine distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return cosine(self.aggregation_func(X, axis=0), self.aggregation_func(Y, axis=0))

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("CosineDistance cannot be called on a pairwise distance matrix.")


class R2ScoreDistance(AbstractDistance):
    """Coefficient of determination across genes between pseudobulk vectors."""

    # NOTE: This is not a distance metric but a similarity metric.

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - r2_score(self.aggregation_func(X, axis=0), self.aggregation_func(Y, axis=0))

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("R2ScoreDistance cannot be called on a pairwise distance matrix.")


class SymmetricKLDivergence(AbstractDistance):
    """Average of symmetric KL divergence between gene distributions of two groups.

    Assuming a Gaussian distribution for each gene in each group, calculates
    the KL divergence between them and averages over all genes. Repeats this ABBA to get a symmetrized distance.
    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence.

    """

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        (x_means, x_stds), (y_means, y_stds) = _column_moments(X), _column_moments(Y)
        kl_all = []
        for x_mean, x_std, y_mean, y_std in zip(x_means, x_stds + epsilon, y_means, y_stds + epsilon, strict=True):
            kl = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean) ** 2) / (2 * y_std**2) - 1 / 2
            klr = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean) ** 2) / (2 * x_std**2) - 1 / 2
            kl_all.append(kl + klr)
        return sum(kl_all) / len(kl_all)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("SymmetricKLDivergence cannot be called on a pairwise distance matrix.")


class TTestDistance(AbstractDistance):
    """Average of T test statistic between two groups assuming unequal variances."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        t_test_all = []
        n1 = X.shape[0]
        n2 = Y.shape[0]
        (x_means, x_stds), (y_means, y_stds) = _column_moments(X), _column_moments(Y)
        for m1, s1, m2, s2 in zip(x_means, x_stds, y_means, y_stds, strict=True):
            v1 = s1**2 * n1 / (n1 - 1) + epsilon
            v2 = s2**2 * n2 / (n2 - 1) + epsilon
            vn1 = v1 / n1
            vn2 = v2 / n2
            t = (m1 - m2) / np.sqrt(vn1 + vn2)
            t_test_all.append(abs(t))
        return sum(t_test_all) / len(t_test_all)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("TTestDistance cannot be called on a pairwise distance matrix.")


class KSTestDistance(AbstractDistance):
    """Average of two-sided KS test statistic between two groups."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        dtype = np.result_type(X, Y)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.dtype(np.float64)
        cdf_x, cdf_y = np.linspace(0, 1, X.shape[0] + 1, dtype=dtype), np.linspace(0, 1, Y.shape[0] + 1, dtype=dtype)
        width = _column_block_width(X, Y)
        stats = [
            np.abs(_ks_statistics(block_x, block_y, cdf_x, cdf_y)).tolist()
            for block_x, block_y in zip(_column_blocks(X, width, dtype), _column_blocks(Y, width, dtype), strict=True)
        ]
        return sum(sum(block) for block in stats) / X.shape[1]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("KSTestDistance cannot be called on a pairwise distance matrix.")


class NBLL(AbstractDistance):
    """Average of Log likelihood (scalar) of group B cells according to a NB distribution fitted over group A."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        def _is_count_matrix(matrix, tolerance=1e-6):
            return bool(matrix.dtype.kind == "i" or np.all(np.abs(matrix - np.round(matrix)) < tolerance))

        if not _is_count_matrix(matrix=X) or not _is_count_matrix(matrix=Y):
            raise ValueError("NBLL distance only works for raw counts.")

        nlls = _nb_log_likelihoods(X, Y, epsilon)
        unexpressed = ~X.any(axis=0)
        nlls[unexpressed] = np.where(Y[:, unexpressed].mean(axis=0) < 10, 0.0, np.nan)

        genes_skipped = np.count_nonzero(np.isnan(nlls))
        if genes_skipped > X.shape[1] / 2:
            raise AttributeError(f"{genes_skipped} genes could not be fit, which is over half.")

        return -np.nanmean(nlls)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("NBLL cannot be called on a pairwise distance matrix.")


def _sample(X, frac=None, n=None):
    """Returns subsample of cells in format (train, test)."""
    if frac and n:
        raise ValueError("Cannot pass both frac and n.")
    if frac:
        n_cells = max(1, int(X.shape[0] * frac))
    elif n:
        n_cells = n
    else:
        raise ValueError("Must pass either `frac` or `n`.")

    rng = np.random.default_rng()
    sampled_indices = rng.choice(X.shape[0], n_cells, replace=False)
    remaining_indices = np.setdiff1d(np.arange(X.shape[0]), sampled_indices)
    return X[remaining_indices, :], X[sampled_indices, :]


class ClassifierProbaDistance(AbstractDistance):
    """Average of classification probabilites of a binary classifier.

    Assumes the first condition is control and the second is perturbed.
    Always holds out 20% of the perturbed condition.
    """

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        Y_train, Y_test = _sample(Y, frac=0.2)
        label = ["c"] * X.shape[0] + ["p"] * Y_train.shape[0]
        train = np.concatenate([X, Y_train])

        reg = LogisticRegression()
        reg.fit(train, label)
        test_labels = reg.predict_proba(Y_test)
        return np.mean(test_labels[:, 1])

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("ClassifierProbaDistance cannot be called on a pairwise distance matrix.")


class ClassifierClassProjection(AbstractDistance):
    """Average of 1-(classification probability of control).

    Warning: unlike all other distances, this must also take a list of categorical labels the same length as X.
    """

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("ClassifierClassProjection can currently only be called with onesided.")

    def onesided_distances(
        self,
        adata: AnnData,
        *,
        groupby: str,
        selected_group: str | None = None,
        groups: list[str] | None = None,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> Series:
        """Unlike the parent function, all groups except the selected group are factored into the classifier.

        Similar to the parent function, the returned dataframe contains only the specified groups.
        """
        groups = cast("list[str]", adata.obs[groupby].unique()) if groups is None else groups
        fct = track if show_progressbar else lambda iterable: iterable

        X = adata[adata.obs[groupby] != selected_group].X
        labels = adata[adata.obs[groupby] != selected_group].obs[groupby].values
        Y = adata[adata.obs[groupby] == selected_group].X

        reg = LogisticRegression()
        reg.fit(X, labels)
        test_probas = reg.predict_proba(Y)

        df = pd.Series(index=groups, dtype=float)

        for group in fct(groups):
            if group == selected_group:
                df.loc[group] = 0
            else:
                class_idx = list(reg.classes_).index(group)
                df.loc[group] = 1 - np.mean(test_probas[:, class_idx])
        df.index.name = groupby
        df.name = f"classifier_cp to {selected_group}"

        return df

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("ClassifierClassProjection cannot be called on a pairwise distance matrix.")


class MeanVarDistributionDistance(AbstractDistance):
    """Distance between mean-var distributions of gene expression."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    @staticmethod
    def _mean_var(x, log: bool = False):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        positive = mean > 0
        mean = mean[positive]
        var = var[positive]
        if log:
            mean = np.log(mean)
            var = np.log(var)
        return mean, var

    @staticmethod
    def _prep_kde_data(x, y):
        return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    @staticmethod
    def _grid_points(d, n_points=100):
        # Make grid, add 1 bin on lower/upper end to get final n_points
        d_min = d.min()
        d_max = d.max()
        # Compute bin size
        d_bin = (d_max - d_min) / (n_points - 2)
        d_min = d_min - d_bin
        d_max = d_max + d_bin
        return np.arange(start=d_min + 0.5 * d_bin, stop=d_max, step=d_bin)

    @staticmethod
    def _kde(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Density at `grid` of the 2D exponential kernel density estimate of `points` with Silverman's bandwidth.

        Matches :class:`~sklearn.neighbors.KernelDensity` with `bandwidth="silverman"` and `kernel="exponential"`.
        """
        n_points = len(points)
        bandwidth = n_points ** (-1 / 6)
        sums = _exponential_kernel_sums(
            np.asarray(points, dtype=np.float64), np.asarray(grid, dtype=np.float64), bandwidth
        )
        return sums / (2 * np.pi * bandwidth**2 * n_points)

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        """Difference of mean-var distributions in 2 matrices.

        Args:
            X: Normalized and log transformed cells x genes count matrix.
            Y: Normalized and log transformed cells x genes count matrix.
            kwargs: Passed to the metrics function.
        """
        mean_x, var_x = self._mean_var(X, log=True)
        mean_y, var_y = self._mean_var(Y, log=True)

        x = self._prep_kde_data(mean_x, var_x)
        y = self._prep_kde_data(mean_y, var_y)

        # Gridpoints to eval KDE on
        mean_grid = self._grid_points(np.concatenate([mean_x, mean_y]))
        var_grid = self._grid_points(np.concatenate([var_x, var_y]))
        grid = np.array(np.meshgrid(mean_grid, var_grid)).T.reshape(-1, 2)

        return ((self._kde(x, grid) - self._kde(y, grid)) ** 2).mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MeanVarDistributionDistance cannot be called on a pairwise distance matrix.")


class MahalanobisDistance(AbstractDistance):
    """Mahalanobis distance between pseudobulk vectors."""

    def __init__(self, aggregation_func: Callable = np.mean) -> None:
        super().__init__()
        self.accepts_precomputed = False
        self.aggregation_func = aggregation_func

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return mahalanobis(
            self.aggregation_func(X, axis=0),
            self.aggregation_func(Y, axis=0),
            np.linalg.inv(np.cov(X.T)),
        )

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("Mahalanobis cannot be called on a pairwise distance matrix.")
