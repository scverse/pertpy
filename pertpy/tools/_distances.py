
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from anndata import AnnData

class Distance:
    """Distance class.
    Distances between groups of cells.

    Parameters
    ----------
    metric : str
        Distance metric to use. E.g. 'edistance', 'wasserstein', 'MMD'.

    Attributes
    ----------
    metric : str
        Distance metric to use. E.g. 'edistance', 'wasserstein', 'MMD'.

    Methods
    -------
    __call__(X, Y=None)
        Compute distance between X and Y.
    """

    def __init__(self, metric: str):
        self.metric = metric

    def __call__(self, X: np.ndarray, Y: np.ndarray):
        """Compute distance between vectors X and Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            First data set.

        Y : array-like, shape (n_samples, n_features)
            Second data set.

        Returns
        -------
        dist : array-like, shape (n_samples, n_samples)
            Distance matrix between X and Y.
        """
        if metric == "edistance":
            return self.edistance(X, Y)
        elif metric == "wasserstein":
            return self.wasserstein(X, Y)
        elif metric == "MMD":
            return self.MMD(X, Y)
        else:
            raise ValueError("Unknown distance metric. Must be one of 'edistance', 'wasserstein', 'MMD'.")
    
    def edistance():
        pass
    
    def MMD():
        # e.g.
        # https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
        # https://github.com/calico/scmmd
        pass
    
    def wasserstein(self, x, y):
        return wasserstein_distance(x,y)
    
    def permutation_test(self, adata : AnnData, groupby : str, contrast : str, n_perms : int = 1000):
        """Runs permutation test for all groups of cells against a contrast group (control).
        Uses the distance metric defined in the class.
        
        COMMENT: This could be done in a separate class, e.g. PermutationTest.
        """
        # either use raw version implemented with the etest in scperturb
        # or go for scipy.stats.permutation_test
        pass
