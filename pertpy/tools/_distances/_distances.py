from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numba
import numpy as np
import pandas as pd
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from pandas import Series
from rich.progress import track
from scipy.sparse import issparse
from scipy.spatial.distance import cosine
from scipy.special import gammaln
from scipy.stats import kendalltau, kstest, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from statsmodels.discrete.discrete_model import NegativeBinomialP

if TYPE_CHECKING:
    from anndata import AnnData


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

    Attributes:
        metric: Name of distance metric.
        layer_key: Name of the counts to use in adata.layers.
        obsm_key: Name of embedding in adata.obsm to use.
        cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.

    Examples:
        >>> import pertpy as pt
        >>> adata = pt.dt.distance_example()
        >>> Distance = pt.tools.Distance(metric="edistance")
        >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
        >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
        >>> D = Distance(X, Y)
    """

    def __init__(
        self,
        metric: str = "edistance",
        layer_key: str = None,
        obsm_key: str = None,
        cell_wise_metric: str = "euclidean",
    ):
        """Initialize Distance class.

        Args:
            metric: Distance metric to use. Defaults to "edistance".
            layer_key: Name of the counts layer containing raw counts to calculate distances for.
                              Mutually exclusive with 'obsm_key'.
                              Defaults to None and is then not used.
            obsm_key: Name of embedding in adata.obsm to use.
                      Mutually exclusive with 'counts_layer_key'.
                      Defaults to None, but is set to "X_pca" if not set explicitly internally.
            cell_wise_metric: Metric from scipy.spatial.distance to use for pairwise distances between single cells.
                                Defaults to "euclidean".
        """
        metric_fct: AbstractDistance = None
        if metric == "edistance":
            metric_fct = Edistance()
        elif metric == "euclidean":
            metric_fct = EuclideanDistance()
        elif metric == "root_mean_squared_error":
            metric_fct = EuclideanDistance()
        elif metric == "mse":
            metric_fct = MeanSquaredDistance()
        elif metric == "mean_absolute_error":
            metric_fct = MeanAbsoluteDistance()
        elif metric == "pearson_distance":
            metric_fct = PearsonDistance()
        elif metric == "spearman_distance":
            metric_fct = SpearmanDistance()
        elif metric == "kendalltau_distance":
            metric_fct = KendallTauDistance()
        elif metric == "cosine_distance":
            metric_fct = CosineDistance()
        elif metric == "r2_distance":
            metric_fct = R2ScoreDistance()
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
        else:
            raise ValueError(f"Metric {metric} not recognized.")
        self.metric_fct = metric_fct

        if layer_key and obsm_key:
            raise ValueError(
                "Cannot use 'counts_layer_key' and 'obsm_key' at the same time.\n"
                "Please provide only one of the two keys."
            )
        if not layer_key and not obsm_key:
            obsm_key = "X_pca"
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self.metric = metric
        self.cell_wise_metric = cell_wise_metric

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

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> X = adata.obsm["X_pca"][adata.obs["perturbation"] == "p-sgCREB1-2"]
            >>> Y = adata.obsm["X_pca"][adata.obs["perturbation"] == "control"]
            >>> D = Distance(X, Y)
        """
        if issparse(X):
            X = X.A
        if issparse(Y):
            Y = Y.A

        if len(X) == 0 or len(Y) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        return self.metric_fct(X, Y, **kwargs)

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        groups: list[str] | None = None,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame:
        """Get pairwise distances between groups of cells.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            groups: List of groups to compute pairwise distances for.
                    If None, uses all groups. Defaults to None.
            show_progressbar: Whether to show progress bar. Defaults to True.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            pd.DataFrame: Dataframe with pairwise distances.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.pairwise(adata, groupby="perturbation")
        """
        groups = adata.obs[groupby].unique() if groups is None else groups
        grouping = adata.obs[groupby].copy()
        df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        fct = track if show_progressbar else lambda iterable: iterable

        # Some metrics are able to handle precomputed distances. This means that
        # the pairwise distances between all cells are computed once and then
        # passed to the metric function. This is much faster than computing the
        # pairwise distances for each group separately. Other metrics are not
        # able to handle precomputed distances such as the PseudobulkDistance.
        if self.metric_fct.accepts_precomputed:
            # Precompute the pairwise distances if needed
            if f"{self.obsm_key}_{self.cell_wise_metric}_predistances" not in adata.obsp.keys():
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"]
            for index_x, group_x in enumerate(fct(groups)):
                idx_x = grouping == group_x
                for group_y in groups[index_x:]:  # type: ignore
                    if group_x == group_y:
                        dist = 0.0  # by distance axiom
                    else:
                        idx_y = grouping == group_y
                        # subset the pairwise distance matrix to the two groups
                        sub_pwd = pwd[idx_x | idx_y, :][:, idx_x | idx_y]
                        sub_idx = grouping[idx_x | idx_y] == group_x
                        dist = self.metric_fct.from_precomputed(sub_pwd, sub_idx, **kwargs)
                    df.loc[group_x, group_y] = dist
                    df.loc[group_y, group_x] = dist
        else:
            if self.layer_key:
                embedding = adata.layers[self.layer_key]
            else:
                embedding = adata.obsm[self.obsm_key].copy()
            for index_x, group_x in enumerate(fct(groups)):
                cells_x = embedding[grouping == group_x].copy()
                for group_y in groups[index_x:]:  # type: ignore
                    if group_x == group_y:
                        dist = 0.0
                    else:
                        cells_y = embedding[grouping == group_y].copy()
                        dist = self(cells_x, cells_y, **kwargs)
                    df.loc[group_x, group_y] = dist
                    df.loc[group_y, group_x] = dist
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric}"

        return df

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | None = None,
        groups: list[str] | None = None,
        show_progressbar: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ) -> pd.DataFrame:
        """Get distances between one selected cell group and the remaining other cell groups.

        Args:
            adata: Annotated data matrix.
            groupby: Column name in adata.obs.
            selected_group: Group to compute pairwise distances to all other.
            groups: List of groups to compute distances to selected_group for.
                    If None, uses all groups. Defaults to None.
            show_progressbar: Whether to show progress bar. Defaults to True.
            n_jobs: Number of cores to use. Defaults to -1 (all).
            kwargs: Additional keyword arguments passed to the metric function.

        Returns:
            pd.DataFrame: Dataframe with distances of groups to selected_group.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.distance_example()
            >>> Distance = pt.tools.Distance(metric="edistance")
            >>> pairwise_df = Distance.onesided_distances(adata, groupby="perturbation", selected_group="control")
        """
        if self.metric == "classifier_cp":
            return self.metric_fct.onesided_distances(  # type: ignore
                adata, groupby, selected_group, groups, show_progressbar, n_jobs, **kwargs
            )

        groups = adata.obs[groupby].unique() if groups is None else groups
        grouping = adata.obs[groupby].copy()
        df = pd.Series(index=groups, dtype=float)
        fct = track if show_progressbar else lambda iterable: iterable

        # Some metrics are able to handle precomputed distances. This means that
        # the pairwise distances between all cells are computed once and then
        # passed to the metric function. This is much faster than computing the
        # pairwise distances for each group separately. Other metrics are not
        # able to handle precomputed distances such as the PsuedobulkDistance.
        if self.metric_fct.accepts_precomputed:
            # Precompute the pairwise distances if needed
            if f"{self.obsm_key}_{self.cell_wise_metric}_predistances" not in adata.obsp.keys():
                self.precompute_distances(adata, n_jobs=n_jobs, **kwargs)
            pwd = adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"]
            for group_x in fct(groups):
                idx_x = grouping == group_x
                group_y = selected_group
                if group_x == group_y:
                    dist = 0.0  # by distance axiom
                else:
                    idx_y = grouping == group_y
                    # subset the pairwise distance matrix to the two groups
                    sub_pwd = pwd[idx_x | idx_y, :][:, idx_x | idx_y]
                    sub_idx = grouping[idx_x | idx_y] == group_x
                    dist = self.metric_fct.from_precomputed(sub_pwd, sub_idx, **kwargs)
                df.loc[group_x] = dist
        else:
            if self.layer_key:
                embedding = adata.layers[self.layer_key]
            else:
                embedding = adata.obsm[self.obsm_key].copy()
            for group_x in fct(groups):
                cells_x = embedding[grouping == group_x].copy()
                group_y = selected_group
                if group_x == group_y:
                    dist = 0.0
                else:
                    cells_y = embedding[grouping == group_y].copy()
                    dist = self(cells_x, cells_y, **kwargs)
                df.loc[group_x] = dist
        df.index.name = groupby
        df.name = f"{self.metric} to {selected_group}"
        return df

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
            >>> adata = pt.dt.distance_example()
            >>> distance = pt.tools.Distance(metric="edistance")
            >>> distance.precompute_distances(adata)
        """
        if self.layer_key:
            cells = adata.layers[self.layer_key]
        else:
            cells = adata.obsm[self.obsm_key].copy()
        pwd = pairwise_distances(cells, cells, metric=self.cell_wise_metric, n_jobs=n_jobs)
        adata.obsp[f"{self.obsm_key}_{self.cell_wise_metric}_predistances"] = pwd


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


class Edistance(AbstractDistance):
    """Edistance metric."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True
        self.cell_wise_metric = "sqeuclidean"

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        sigma_X = pairwise_distances(X, X, metric="sqeuclidean").mean()
        sigma_Y = pairwise_distances(Y, Y, metric="sqeuclidean").mean()
        delta = pairwise_distances(X, Y, metric="sqeuclidean").mean()
        return 2 * delta - sigma_X - sigma_Y

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        sigma_X = P[idx, :][:, idx].mean()
        sigma_Y = P[~idx, :][:, ~idx].mean()
        delta = P[idx, :][:, ~idx].mean()
        return 2 * delta - sigma_X - sigma_Y


class MMD(AbstractDistance):
    """Linear Maximum Mean Discrepancy."""

    # Taken in parts from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, kernel="linear", **kwargs) -> float:
        if kernel == "linear":
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
        elif kernel == "rbf":
            XX = rbf_kernel(X, X, gamma=1.0)
            YY = rbf_kernel(Y, Y, gamma=1.0)
            XY = rbf_kernel(X, Y, gamma=1.0)
        elif kernel == "poly":
            XX = polynomial_kernel(X, X, degree=2, gamma=1.0, coef0=0)
            YY = polynomial_kernel(Y, Y, degree=2, gamma=1.0, coef0=0)
            XY = polynomial_kernel(X, Y, degree=2, gamma=1.0, coef0=0)
        else:
            raise ValueError(f"Kernel {kernel} not recognized.")

        return XX.mean() + YY.mean() - 2 * XY.mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MMD cannot be called on a pairwise distance matrix.")


class WassersteinDistance(AbstractDistance):
    """Wasserstein distance metric (solved with entropy regularized Sinkhorn)."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        geom = PointCloud(X, Y)
        return self.solve_ot_problem(geom, **kwargs)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        geom = Geometry(cost_matrix=P[idx, :][:, ~idx])
        return self.solve_ot_problem(geom, **kwargs)

    def solve_ot_problem(self, geom: Geometry, **kwargs):
        ot_prob = LinearProblem(geom)
        solver = Sinkhorn()
        ot = solver(ot_prob, **kwargs)
        return ot.reg_ot_cost.item()


class EuclideanDistance(AbstractDistance):
    """Euclidean distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0), ord=2, **kwargs)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("EuclideanDistance cannot be called on a pairwise distance matrix.")


class MeanSquaredDistance(AbstractDistance):
    """Mean squared distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0), ord=2, **kwargs) ** 2 / X.shape[1]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MeanSquaredDistance cannot be called on a pairwise distance matrix.")


class MeanAbsoluteDistance(AbstractDistance):
    """Absolute (Norm-1) distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0), ord=1, **kwargs) / X.shape[1]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("MeanAbsoluteDistance cannot be called on a pairwise distance matrix.")


class MeanPairwiseDistance(AbstractDistance):
    """Mean of the pairwise euclidean distance between two groups of cells."""

    # NOTE: This is not a metric in the mathematical sense.

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distances(X, Y, **kwargs).mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()


class PearsonDistance(AbstractDistance):
    """Pearson distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - pearsonr(X.mean(axis=0), Y.mean(axis=0))[0]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("PearsonDistance cannot be called on a pairwise distance matrix.")


class SpearmanDistance(AbstractDistance):
    """Spearman distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - spearmanr(X.mean(axis=0), Y.mean(axis=0))[0]

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("SpearmanDistance cannot be called on a pairwise distance matrix.")


class KendallTauDistance(AbstractDistance):
    """Kendall-tau distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        x, y = X.mean(axis=0), Y.mean(axis=0)
        n = len(x)
        tau_corr = kendalltau(x, y).statistic
        tau_dist = (1 - tau_corr) * n * (n - 1) / 4
        return tau_dist

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("KendallTauDistance cannot be called on a pairwise distance matrix.")


class CosineDistance(AbstractDistance):
    """Cosine distance between pseudobulk vectors."""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return cosine(X.mean(axis=0), Y.mean(axis=0))

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("CosineDistance cannot be called on a pairwise distance matrix.")


class R2ScoreDistance(AbstractDistance):
    """Coefficient of determination across genes between pseudobulk vectors."""

    # NOTE: This is not a distance metric but a similarity metric.

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return 1 - r2_score(X.mean(axis=0), Y.mean(axis=0))

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("R2ScoreDistance cannot be called on a pairwise distance matrix.")


class SymmetricKLDivergence(AbstractDistance):
    """Average of symmetric KL divergence between gene distributions of two groups

    Assuming a Gaussian distribution for each gene in each group, calculates
    the KL divergence between them and averages over all genes. Repeats this ABBA to get a symmetrized distance.
    See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence.

    """

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        kl_all = []
        for i in range(X.shape[1]):
            x_mean, x_std = X[:, i].mean(), X[:, i].std() + epsilon
            y_mean, y_std = Y[:, i].mean(), Y[:, i].std() + epsilon
            kl = np.log(y_std / x_std) + (x_std**2 + (x_mean - y_mean) ** 2) / (2 * y_std**2) - 1 / 2
            klr = np.log(x_std / y_std) + (y_std**2 + (y_mean - x_mean) ** 2) / (2 * x_std**2) - 1 / 2
            kl_all.append(kl + klr)
        return sum(kl_all) / len(kl_all)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("SymmetricKLDivergence cannot be called on a pairwise distance matrix.")


class TTestDistance(AbstractDistance):
    """Average of T test statistic between two groups assuming unequal variances"""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        t_test_all = []
        n1 = X.shape[0]
        n2 = Y.shape[0]
        for i in range(X.shape[1]):
            m1, v1 = X[:, i].mean(), X[:, i].std() ** 2 * n1 / (n1 - 1) + epsilon
            m2, v2 = Y[:, i].mean(), Y[:, i].std() ** 2 * n2 / (n2 - 1) + epsilon
            vn1 = v1 / n1
            vn2 = v2 / n2
            t = (m1 - m2) / np.sqrt(vn1 + vn2)
            t_test_all.append(abs(t))
        return sum(t_test_all) / len(t_test_all)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("TTestDistance cannot be called on a pairwise distance matrix.")


class KSTestDistance(AbstractDistance):
    """Average of two-sided KS test statistic between two groups"""

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        stats = []
        for i in range(X.shape[1]):
            stats.append(abs(kstest(X[:, i], Y[:, i])[0]))
        return sum(stats) / len(stats)

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        raise NotImplementedError("KSTestDistance cannot be called on a pairwise distance matrix.")


class NBLL(AbstractDistance):
    """
    Average of Log likelihood (scalar) of group B cells
    according to a NB distribution fitted over group A
    """

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = False

    def __call__(self, X: np.ndarray, Y: np.ndarray, epsilon=1e-8, **kwargs) -> float:
        def _is_count_matrix(matrix, tolerance=1e-6):
            if matrix.dtype.kind == "i" or np.all(np.abs(matrix - np.round(matrix)) < tolerance):
                return True
            else:
                return False

        if not _is_count_matrix(matrix=X) or not _is_count_matrix(matrix=Y):
            raise ValueError("NBLL distance only works for raw counts.")

        @numba.jit(forceobj=True)
        def _compute_nll(y: np.ndarray, nb_params: tuple[float, float], epsilon: float) -> float:
            mu = np.exp(nb_params[0])
            theta = 1 / nb_params[1]
            eps = epsilon

            log_theta_mu_eps = np.log(theta + mu + eps)
            nll = (
                theta * (np.log(theta + eps) - log_theta_mu_eps)
                + y * (np.log(mu + eps) - log_theta_mu_eps)
                + gammaln(y + theta)
                - gammaln(theta)
                - gammaln(y + 1)
            )
            return nll.mean()

        def _process_gene(x: np.ndarray, y: np.ndarray, epsilon: float) -> float:
            try:
                nb_params = NegativeBinomialP(x, np.ones_like(x)).fit(disp=False).params
                return _compute_nll(y, nb_params, epsilon)
            except np.linalg.linalg.LinAlgError:
                if x.mean() < 10 and y.mean() < 10:
                    return 0.0
                else:
                    return np.nan  # Use NaN to indicate skipped genes

        nlls = []
        genes_skipped = 0

        for i in range(X.shape[1]):
            nll = _process_gene(X[:, i], Y[:, i], epsilon)
            if np.isnan(nll):
                genes_skipped += 1
            else:
                nlls.append(nll)

        if genes_skipped > X.shape[1] / 2:
            raise AttributeError(f"{genes_skipped} genes could not be fit, which is over half.")

        return -np.sum(nlls) / len(nlls)

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
        groups = adata.obs[groupby].unique() if groups is None else groups

        X = adata[adata.obs[groupby] != selected_group].X
        labels = adata[adata.obs[groupby] != selected_group].obs[groupby].values
        Y = adata[adata.obs[groupby] == selected_group].X

        reg = LogisticRegression()
        reg.fit(X, labels)
        test_probas = reg.predict_proba(Y)

        df = pd.Series(index=groups, dtype=float)
        for group in groups:
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
