import pytest
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame

import pertpy as pt
from pertpy.tools._distances._distances import Metric

distances: tuple[Metric, ...] = (
    "edistance",
    "euclidean",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "spearman_distance",
    "kendalltau_distance",
    "cosine_distance",
    "wasserstein",
    "mean_pairwise",
    "mmd",
    "r2_distance",
    "sym_kldiv",
    "t_test",
    "ks_test",
    "classifier_proba",
    # "classifier_cp",
    # "nbll",
    "mahalanobis",
    "mean_var_distribution",
)

count_distances = ["nb_ll"]


@pytest.fixture
def adata() -> AnnData:
    adata = pt.dt.distance_example()
    adata = sc.pp.subsample(adata, 0.1, copy=True)

    return adata


@pytest.mark.parametrize("distance", distances)
def test_distancetest(adata: AnnData, distance: Metric) -> None:
    etest = pt.tl.DistanceTest(distance, n_perms=10, obsm_key="X_pca", alpha=0.05, correction="holm-sidak")
    tab = etest(adata, groupby="perturbation", contrast="control")

    # Well-defined output
    assert tab.shape[1] == 5
    assert isinstance(tab, DataFrame)

    # p-values are in [0,1]
    assert tab["pvalue"].min() >= 0
    assert tab["pvalue"].max() <= 1
    assert tab["pvalue_adj"].min() >= 0
    assert tab["pvalue_adj"].max() <= 1
