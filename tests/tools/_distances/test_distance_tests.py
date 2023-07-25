from pandas import DataFrame
from pytest import fixture, mark

import pertpy as pt

distances = [
    "edistance",
    "pseudobulk",
    "mean_absolute_distance",
    "pearson_distance",
    "spearman_distance",
    "cosine_distance",
    "wasserstein",
    "mean_pairwise",
    "mmd",
    "r2_distance",
    "kl_divergence",
    "t_test",
]


class TestPermutationTest:
    @fixture
    def adata(self):
        adata = pt.dt.distance_example_data()
        return adata

    @mark.parametrize("distance", distances)
    def test_distancetest(self, adata, distance):
        etest = pt.tl.DistanceTest(distance, n_perms=10, obsm_key="X_pca", alpha=0.05, correction="holm-sidak")
        tab = etest(adata, groupby="perturbation", contrast="control")
        # Well-defined output
        assert tab.shape[1] == 5
        assert type(tab) == DataFrame
        # p-values are in [0,1]
        assert tab["pvalue"].min() >= 0
        assert tab["pvalue"].max() <= 1
        assert tab["pvalue_adj"].min() >= 0
        assert tab["pvalue_adj"].max() <= 1
