import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame
from scipy import sparse

import pertpy as pt
from pertpy.tools._distances._distance_tests import _permutation_pvalues
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
    adata = pt.ds.distance_example()
    adata = sc.pp.sample(adata, 0.1, copy=True, rng=0)

    return adata


def test_permutation_pvalues_follow_group_labels() -> None:
    """The permutations are stored in alphabetical order while the observed distances keep the order of appearance."""
    index = pd.Index(["control", "zeta", "alpha"])
    df = DataFrame({"distance": [np.nan, 10.0, 0.1]}, index=index)
    results = [DataFrame({"distance": [np.nan, 0.5, 0.2]}, index=index).sort_index() for _ in range(4)]

    pvalues = _permutation_pvalues(results, df, 4)

    assert pvalues["zeta"] == 0.25
    assert pvalues["alpha"] == 1.0


@pytest.mark.parametrize("distance", distances)
def test_distancetest(adata: AnnData, distance: Metric) -> None:
    etest = pt.tl.DistanceTest(distance, n_perms=10, obsm_key="X_pca", padj_threshold=0.05, correction="holm-sidak")
    tab = etest(adata, groupby="perturbation", contrast="control")

    # Well-defined output
    assert tab.shape[1] == 5
    assert isinstance(tab, DataFrame)

    # p-values are in [0,1]
    assert tab["pvalue"].min() >= 0
    assert tab["pvalue"].max() <= 1
    assert tab["pvalue_adj"].min() >= 0
    assert tab["pvalue_adj"].max() <= 1


@pytest.mark.parametrize("metric", ["euclidean", "mse", "mean_absolute_error"])
@pytest.mark.parametrize("matrix_type", ["dense", "csr_matrix", "csr_array"])
def test_distance_test_xy_accepts_layer_key(metric: Metric, matrix_type: str) -> None:
    """Direct distance metrics honor the documented layer_key without an obsm embedding."""
    rng = np.random.default_rng(3)
    labels = np.array(["control"] * 8 + ["KO"] * 8)
    x_layer = np.concatenate([rng.normal(0, 0.1, (8, 3)), rng.normal(3, 0.1, (8, 3))])
    expression: np.ndarray | sparse.csr_matrix | sparse.csr_array = x_layer
    if matrix_type == "csr_matrix":
        expression = sparse.csr_matrix(x_layer)
    elif matrix_type == "csr_array":
        expression = sparse.csr_array(x_layer)
    adata = AnnData(X=np.zeros_like(x_layer), obs=pd.DataFrame({"perturbation": labels}))
    adata.layers["signal"] = expression
    # Deliberately do not create X_pca: this must read the layer.
    distance_test = pt.tl.DistanceTest(metric, n_perms=9, layer_key="signal")

    result = distance_test(adata, groupby="perturbation", contrast="control", show_progressbar=False)
    expected = distance_test.distance(x_layer[labels == "KO"], x_layer[labels == "control"])

    assert expected > 0
    assert result.loc["KO", "distance"] == pytest.approx(expected)
