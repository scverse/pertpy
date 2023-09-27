from typing import Any, Literal

import numpy as np
from collections.abc import Mapping
from types import MappingProxyType

from pertpy.tools._distances._distances import AbstractDistance, Distance

# class DistanceBootstrapper:
#     def __init__(self, distance_metric, n_bootstraps=1000):
#         self.distance_metric = distance_metric
#         self.n_bootstraps = n_bootstraps

#     def calculate_variance(self, X, Y, **kwargs):
#         distances = []
#         for _ in range(self.n_bootstraps):
#             # Generate bootstrapped samples
#             X_bootstrapped = np.random.choice(X, len(X), replace=True)
#             Y_bootstrapped = np.random.choice(Y, len(Y), replace=True)

#             # Calculate the distance using the provided distance metric
#             distance = self.distance_metric(X_bootstrapped, Y_bootstrapped, **kwargs)
#             distances.append(distance)

#         # Calculate the variance of the distances
#         variance = np.var(distances)
#         return variance


class ThreeWayComparison:
    """First draft for a ThreeWayComparison class.
    TODOs:
    - evaluate if something like this is doing what we want
    - if yes, integrate into pairwise comparison (and one-sided)
    """

    metric_fct: AbstractDistance | None

    def __init__(
        self,
        metric: str,
        # TODO: might pass things here
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
        """
        try:
            dist = Distance(metric)
        except ValueError:
            self.metric_fct = None
        else:
            self.metric_fct = dist.metric_fct

    @property
    def is_pertpy_metric(self) -> bool:
        return self.metric_fct is None

    # call(X,Y,Z) -> (X,Y), (Y,Z)
    # one-sided distance (adata, selected_group=None, groupby, groups=None)
    # pairwise distance (adata, groupby, groups=None) - dont do as this is doing just "all" the pairwise ones?
    # precomputed (adata, cell_wise_metric="euclidean")

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        /,
        *,
        bootstrap: bool = False,
        n_bootstraps: int = 100,
        random_state: int = 0,
        **kwargs,
    ):
        """_summary_

        Args:
            X: First vector of shape (n_samples, n_features).
            Y: Second vector of shape (n_samples, n_features).
            Z: Third vector of shape (n_samples, n_features).
            bootstrap (bool, optional): Defaults to False.
            n_bootstrap: Number of bootstraps to use. Defaults to 100.
            random_state: Random state to use for bootstrapping. Defaults to 0.

        Returns:
            _type_: _description_
        """
        a = self.metric_fct(X, Y, **kwargs)
        b = self.metric_fct(Y, Z, **kwargs)

        return a, b

    def precomputed(self, X, Y, Z, /, **kwargs):
        a = self.metric_fct.precomputed(X, Y, **kwargs)
        b = self.metric_fct.precomputed(Y, Z, **kwargs)

        return a, b

    # less efficient as it computes all pairwise distances
    def alternative(self, adata, groupby, *, selected_group=None, groups=None):
        dists = self.metric_fct.pairwise(adata, groupby, groups=groups)
        dist_XY = dists.iloc[1]  # get XY
        dist_YZ = dists.iloc[2]  # get YZ

    # note to me - this is called on an adata which is already subsetted for a cell type
    # second note - this is all we need?
    # def pairwise(self, adata, groupby, groups=None, **kwargs):
    #    a = self.pairwise(adata, groupby, groups=groups, **kwargs)
    #    b = self.pairwise(adata, groupby, groups=groups, **kwargs)
    #    return self.compare_X_to_Y(X, Y, pairwise, **kwargs), \
    #           self.compare_X_to_Y(Y, Z, pairwise, **kwargs)
    # def _compare_X_to_Y(self, X, Y=None, pairwise=False, **kwargs):
    #     # Calculate the distance using the provided distance metric
    #     if not pairwise:
    #         if Y is None:
    #             raise ValueError("If pairwise is False, Y must be provided.")
    #         distance = self.metric(X, Y, **kwargs)
    #     else:
    #         if isinstance(X, sc.AnnData):
    #             raise ValueError("If pairwise is True, X must be an AnnData.")
    #         distance = self.distance_metric.pairwise(X, **kwargs)
    #     return distance



def score(
    *,
    ctrl: np.ndarray,
    pred: np.ndarray,
    pert: np.ndarray,
    metric: str = 'euclidean',
    kind: Literal['simple', 'scaled', 'nearest'] = 'simple',
    # TODO: pass metric_kwds directly to this function instead?
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
):
    dist = Distance(metric)
    if kind == 'simple':
        d1 = dist.metric_fct(pert, pred, **metric_kwds)
        d2 = dist.metric_fct(ctrl, pred, **metric_kwds)
        return d1 / d2
    elif kind in {'scaled', 'nearest'}:
        raise NotImplementedError(f'kind {kind} not implemented yet')
    else:
        raise ValueError(f'Unknown kind {kind}')
