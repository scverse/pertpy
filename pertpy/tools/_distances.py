
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from anndata import AnnData
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import pairwise_distances

'''
User interface:
- get distances between any two or more groups of cells with a distance metric of choice
- get p-values for permutation test with a distance metric of choice for all or 
    some of their groups
    
Internally:
- There will be two types of PermutationTest classes:
    - One that recomputes the pairwise distance matrix every time and works for
      Dist 

Enhancments:
- TODO make etest allow for multiple controls (accept list of controls)

Dev Notes:
- Problem:
    PermutationTest is waaaaaay quicker when we precompute pairwise distances 
    and pass them to the metric after permutation, instead of computing them
    every time. However, this is not possible for all metrics, e.g. wasserstein
    from scipy.stats.
    The way we define the Metric class makes it a callable on X and Y cell coordinates,
    BUT for PermutationTest we need to pass the pairwise distance matrix P instead.
    
- Solutions:
    1. Have the Metric class with two function: one of them on X and Y, the other on P.
    2. Have another Metric class for pairwise distances, e.g. P_Edistence.
    3. Have a wrapper function that takes a metric and returns a new metric that
    is callable on X and Y, and uses the pairwise distance matrix P internally.
''' 

# wrapper fct accessable to user
def distance(x, y, metric='edistance'):
    if metric=='edistance':
        P = pairwise_distances([x,y], [x,y])
        idx = [x,y] == x
        return Edistance(P, idx)
    
class Metric:
    """Metric class.
    Distance metrics between groups of cells.

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

    def __call__(self, X: np.ndarray, Y: np.ndarray, P, **kwargs):
        """Compute distance between vectors X and Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            First data set.

        Y : array-like, shape (n_samples, n_features)
            Second data set.
        
        P : array-like, shape (n_samples, n_samples), optional (default: None)
            Paiwise distance matrix.

        Returns
        -------
        dist : array-like, shape (n_samples, n_samples)
            Distance between X and Y.
        """
        dist = None
        return dist



class Edistance(Metric):
    def __init__(self, n_pc : int = 50):
        super().__init__("edistance")
        self.n_pc = n_pc
        
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        N = X.shape[0]
        M = Y.shape[0]
        # 
        sigma_X = pairwise_distances(X, X) / (M**2)
        sigma_Y = pairwise_distances(Y, Y) / (N**2)
        delta = pairwise_distances(X, Y) / (N*M)
        return delta - sigma_X - sigma_Y
    

class MMD(Metric):
    # e.g.
    # https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    # https://github.com/calico/scmmd
    pass

class Wasserstein(Metric):
    """First wasserstein distance metric.
    Uses scipy.stats.wasserstein_distance.
    """
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        return wasserstein_distance(X, Y, **kwargs)

class PseudobulkDistance(Metric):
    pass

class MeanPairwiseDistance(Metric):
    def __call__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        return pairwise_distances(X, Y).mean()

class PermutationTest:
    """Runs permutation test for all groups of cells against a contrast group ("control").
    
    Parameters
    ----------
    metric : Metric
        Distance metric to use.
    n_perms : int, optional (default: 1000)
        Number of permutations to run.
    embedding : str, optional (default: "X_pca")
        Embedding to use for distance computation.
    groups : list, optional (default: None)
        Defines groups of cells for testing.
    alpha : float, optional (default: 0.05)
        Significance level.
    correction : str, optional (default: "holm-sidak")
        Correction method for multiple testing.
    control : str, optional (default: "control")
        Name of the control group.
    verbose : bool, optional (default: True)
        Print progress.
    
    Returns
    -------
    pandas.DataFrame
        Results of the permutation test. Has columns:
        - distance: distance between the contrast group and the group
        - pvalue: p-value of the permutation test
        - significant: whether the group is significantly different from the contrast group
        - pvalue_adj: p-value after multiple testing correction
        - significant_adj: whether the group is significantly different from the contrast group after multiple testing correction
    """
    
    def __init__(self, metric: Metric, n_perms: int = 1000,
                 embedding: str = "X_pca", groups: Optional[List[str]] = None,
                 alpha: float = 0.05, correction: str = "holm-sidak",
                 contrast: str = "control",
                 verbose: bool = True, 
                 ):
        self.metric = metric
        self.n_perms = n_perms
        self.embedding = embedding
        self.groups = groups
        self.alpha = alpha
        self.correction = correction
        self.contrast = contrast
        self.verbose = verbose
        
    def __call__(self, adata: AnnData, groupby: str, contrast: str):
        groups = adata.obs[groupby].unique()
        if self.contrast not in groups:
            raise ValueError(f"Contrast group '{self.contrast}' not found in '{groupby}' of adata.obs.")
        fct = tqdm if self.verbose else lambda x: x
        emb = adata.obsm[self.embedding]
        res = []
        
        # Generate the null distribution
        for i in fct(range(self.n_perms)):
            # per perturbation, shuffle with control and compute e-distance
            df = pd.DataFrame(index=groups, columns=['distance'], dtype=float)
            for group in fct(groups):
                if group == self.contrast:
                    continue
                # Shuffle the labels of the groups
                labels = adata.obs[groupby].values[adata.obs[groupby].isin([group, contrast])]
                shuffled_labels = np.random.permutation(labels)
                idx = shuffled_labels==group
                
                X = emb[idx]  # shuffled group
                Y = emb[~idx]  # shuffled contrast
                dist = self.metric(X, Y)
                dist = distance(X, Y, metric=self.metric)
                
                # quicker
                idx = ...
                pwd = ...
                
                dist = self.clever_metric(pwd, idx)
                
                df.loc[group, 'distance'] = dist
            res.append(df.sort_index())
        
        # Generate the empirical distribution
        for group in fct(groups):
            if group == self.contrast:
                continue
            X = emb[adata.obs[groupby]==group]
            Y = emb[adata.obs[groupby]==contrast]
            df.loc[group, 'distance'] = self.metric(X, Y)
        
        # Evaluate the test
        # count times shuffling resulted in larger distance
        results = np.array(pd.concat([r['distance'] - df['distance'] for r in res], axis=1) > 0, dtype=int)
        n_failures = pd.Series(np.clip(np.sum(results, axis=1), 1, np.inf), index=df.index)
        pvalues = n_failures / self.n_perms

        # Apply multiple testing correction
        significant_adj, pvalue_adj, _, _ = multipletests(pvalues.values, 
                                                          alpha=self.alpha, 
                                                          method=self.correction
                                                          )

        # Aggregate results
        tab = pd.DataFrame({'distance': df['distance'], 
                            'pvalue': pvalues, 
                            'significant': pvalues < self.alpha, 
                            'pvalue_adj': pvalue_adj, 
                            'significant_adj': significant_adj}, 
                           index=df.index)

def etest():
    # calls PermutationTest with edistance metric
    pass

class Etest(PermutationTest):    
    def __init__(self, n_perms: int = 1000, embedding: str = "X_pca", 
                 groups: Optional[List[str]] = None, alpha: float = 0.05, 
                 correction: str = "holm-sidak", contrast: str = "control", 
                 verbose: bool = True):
        super().__init__(Metric("edistance"), n_perms, embedding, groups, alpha, 
                         correction, contrast, verbose)