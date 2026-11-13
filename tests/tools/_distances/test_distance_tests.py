import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame
from statsmodels.stats.multitest import multipletests

import pertpy as pt
from pertpy.tools._distances._distance_tests import _correct_non_control_pvalues, _permutation_pvalues
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


@pytest.mark.parametrize("correction", ["holm-sidak", "fdr_bh", "bonferroni"])
def test_distance_test_correction_excludes_contrast(correction: str) -> None:
    # Keep the reference between targets to check index alignment, not just lengths.
    pvalues = pd.Series([0.04, 0.02, 0.02], index=["KO_B", "control", "KO_A"])
    significant, adjusted = _correct_non_control_pvalues(pvalues, "control", alpha=0.05, correction=correction)
    expected_rejected, expected_adj, _, _ = multipletests(
        pvalues.loc[["KO_B", "KO_A"]].to_numpy(), alpha=0.05, method=correction
    )

    np.testing.assert_allclose(adjusted.loc[["KO_B", "KO_A"]], expected_adj)
    np.testing.assert_array_equal(significant.loc[["KO_B", "KO_A"]], expected_rejected)
    assert adjusted.loc["control"] == 1.0
    assert not significant.loc["control"]


@pytest.mark.parametrize("metric", ["edistance", "euclidean"])
def test_distance_test_pvalue_correction_uses_only_tested_groups(metric: Metric) -> None:
    """Both precomputed and direct paths must exclude the control reference from multiplicity."""
    rng = np.random.default_rng(41)
    labels = ["control"] * 10 + ["KO_A"] * 10 + ["KO_B"] * 10
    embedding = np.concatenate([rng.normal(0, 0.2, (10, 3)), rng.normal(2, 0.2, (10, 3)), rng.normal(1, 0.2, (10, 3))])
    adata = AnnData(X=embedding, obs=pd.DataFrame({"perturbation": labels}))
    adata.obsm["X_pca"] = embedding
    test = pt.tl.DistanceTest(metric, n_perms=49, obsm_key="X_pca", correction="holm-sidak")

    tab = test(adata, groupby="perturbation", contrast="control", show_progressbar=False)
    actual = tab.loc[["KO_A", "KO_B"], "pvalue"]
    expected_significant, expected_adj, _, _ = multipletests(actual.to_numpy(), alpha=0.05, method="holm-sidak")

    np.testing.assert_allclose(tab.loc[actual.index, "pvalue_adj"], expected_adj)
    np.testing.assert_array_equal(tab.loc[actual.index, "significant_adj"], expected_significant)
    assert tab.loc["control", "pvalue_adj"] == 1.0
    assert not tab.loc["control", "significant_adj"]


def test_distance_test_correction_with_only_control() -> None:
    significant, adjusted = _correct_non_control_pvalues(
        pd.Series([1.0], index=["control"]), "control", alpha=0.05, correction="holm-sidak"
    )
    assert adjusted.loc["control"] == 1.0
    assert not significant.loc["control"]
