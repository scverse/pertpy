from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from anndata import AnnData
from ott.geometry.geometry import Geometry
from ott.geometry.pointcloud import PointCloud
from ott.problems.linear.linear_problem import LinearProblem
from ott.solvers.linear.sinkhorn import Sinkhorn
from rich.progress import track
from sklearn.metrics import pairwise_distances


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
    - "mmd": Maximum mean discrepancy
        This is the maximum mean discrepancy between the cells of two groups.
        Here, uses linear MMD. For theory on MMD in single-cell applications, see
        `Lotfollahi et al. (2019) <https://doi.org/10.48550/arXiv.1910.01791>`__.
    - "wasserstein": Wasserstein distance (Earth Mover's Distance)
        This is the Wasserstein distance between the cells of two groups. Uses an
        OTT-JAX implementation of the Sinkhorn algorithm to compute the distance.
        For more information on the optimal transport solver, see
        `Cuturi et al. (2013) <https://proceedings.neurips.cc/paper/2013/file/af21d0c97db2e27e13572cbf59eb343d-Paper.pdf>`__.
    - "pseudobulk": Pseudobulk distance.
        This is the euclidean distance between the means of cells from two groups.
    - "mean_pairwise": Mean pairwise distance.
        This is the mean of the pairwise euclidean distances between cells of two groups.

    Attributes:
        metric: Name of distance metric.
        obsm_key: Name of embedding in adata.obsm to use.
        metric_fct: Distance metric function.

    Example:
        .. code-block:: python

            import pertpy as pt

            adata = pt.dt.distance_example_data()
            Distance = pt.tools.Distance(metric='edistance')
            X = adata.obsm['X_pca'][adata.obs['perturbation'] == 'p-sgCREB1-2']
            Y = adata.obsm['X_pca'][adata.obs['perturbation'] == 'control']
            D = Distance(X, Y)
    """

    def __init__(
        self,
        metric: str = "edistance",
        obsm_key: str = "X_pca",
    ):
        """Initialize Distance class.

        Args:
            metric: Distance metric to use. Defaults to "edistance".
            obsm_key: Name of embedding in adata.obsm to use. Defaults to "X_pca".
        """
        metric_fct: AbstractDistance = None
        if metric == "edistance":
            metric_fct = Edistance()
        elif metric == "pseudobulk":
            metric_fct = PseudobulkDistance()
        elif metric == "mean_pairwise":
            metric_fct = MeanPairwiseDistance()
        elif metric == "mmd":
            metric_fct = MMD()
        elif metric == "wasserstein":
            metric_fct = WassersteinDistance()
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

        Example:
            .. code-block:: python

                import pertpy as pt

                adata = pt.dt.distance_example_data()
                Distance = pt.tools.Distance(metric='edistance')
                X = adata.obsm['X_pca'][adata.obs['perturbation'] == 'p-sgCREB1-2']
                Y = adata.obsm['X_pca'][adata.obs['perturbation'] == 'control']
                D = Distance(X, Y)
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
                    If None, uses all groups. Defaults to None.
            verbose: Whether to show progress bar. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe with pairwise distances.

        Example:
            .. code-block:: python

                import pertpy as pt

                adata = pt.dt.distance_example_data()
                Distance = pt.tools.Distance(metric='edistance')
                pairwise_df = distance.pairwise(adata, groupby='perturbation')
        """
        groups = adata.obs[groupby].unique() if groups is None else groups
        grouping = adata.obs[groupby].copy()
        df = pd.DataFrame(index=groups, columns=groups, dtype=float)
        fct = track if verbose else lambda iterable: iterable

        if self.metric_fct.accepts_precomputed:
            # Precompute the pairwise distances if needed
            if f"{self.obsm_key}_predistances" not in adata.obsp.keys():
                self.precompute_distances(adata, **kwargs)
            pwd = adata.obsp[f"{self.obsm_key}_predistances"]
            for index_x, group_x in enumerate(fct(groups)):
                idx_x = grouping == group_x
                for group_y in groups[index_x:]:
                    if group_x == group_y:
                        dist = 0.0
                    else:
                        idx_y = grouping == group_y
                        sub_pwd = pwd[idx_x | idx_y, :][:, idx_x | idx_y]
                        sub_idx = grouping[idx_x | idx_y] == group_x
                        dist = self.metric_fct.from_precomputed(sub_pwd, sub_idx, **kwargs)
                    df.loc[group_x, group_y] = dist
                    df.loc[group_y, group_x] = dist
        else:
            embedding = adata.obsm[self.obsm_key].copy()
            for index_x, group_x in enumerate(fct(groups)):
                cells_x = embedding[grouping == group_x].copy()
                for group_y in groups[index_x:]:
                    if group_x == group_y:
                        dist = 0.0
                    else:
                        cells_y = embedding[grouping == group_y].copy()
                        dist = self.metric_fct(cells_x, cells_y, **kwargs)
                    df.loc[group_x, group_y] = dist
                    df.loc[group_y, group_x] = dist
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric}"
        return df

    def precompute_distances(self, adata: AnnData, cell_wise_metric: str = "euclidean") -> None:
        """Precompute pairwise distances between all cells, writes to adata.obsp.

        The precomputed distances are stored in adata.obsp under the key
        '{self.obsm_key}_predistances', as they depend on the embedding used.

        Args:
            adata: Annotated data matrix.
            obs_key: Column name in adata.obs.
            cell_wise_metric: Metric to use for pairwise distances.
        """
        # Precompute the pairwise distances
        cells = adata.obsm[self.obsm_key].copy()
        pwd = pairwise_distances(cells, cells, metric=cell_wise_metric)
        # Write to adata.obsp
        adata.obsp[f"{self.obsm_key}_predistances"] = pwd


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

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        sigma_X = pairwise_distances(X, X, metric="euclidean").mean()
        sigma_Y = pairwise_distances(Y, Y, metric="euclidean").mean()
        delta = pairwise_distances(X, Y, metric="euclidean").mean()
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

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        delta = X.mean(0) - Y.mean(0)
        return delta.dot(delta.T)

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
        # Define a linear problem with that cost structure.
        ot_prob = LinearProblem(geom)
        # Create a Sinkhorn solver
        solver = Sinkhorn()
        # Solve OT problem
        ot = solver(ot_prob, **kwargs)
        return ot.reg_ot_cost


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

    # NOTE: This is not a metric in the mathematical sense.

    def __init__(self) -> None:
        super().__init__()
        self.accepts_precomputed = True

    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
        return pairwise_distances(X, Y, **kwargs).mean()

    def from_precomputed(self, P: np.ndarray, idx: np.ndarray, **kwargs) -> float:
        return P[idx, :][:, ~idx].mean()
