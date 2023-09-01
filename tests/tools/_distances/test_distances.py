import numpy as np
import pertpy as pt
import scanpy as sc
from pandas import DataFrame
from pytest import fixture, mark

actual_distances = [
    "edistance",
    "pseudobulk",
    "mean_absolute_distance",
    "pearson_distance",
    "spearman_distance",
    "cosine_distance",
    "wasserstein",
]
pseudo_distances = ["mean_pairwise", "mmd", "r2_distance", "kl_divergence", "t_test"]


class TestDistances:
    @fixture
    def adata(self):
        adata = pt.dt.distance_example()
        adata_subsampled = sc.pp.subsample(adata, 0.1, copy=True)
        return adata_subsampled

    @mark.parametrize("distance", actual_distances)
    def test_distance_axioms(self, adata, distance):
        # Test if distances are well-defined in accordance with metric axioms
        Distance = pt.tl.Distance(distance, "X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", verbose=True)
        # (M1) Positiv definiteness
        assert all(np.diag(df.values) == 0)  # distance to self is 0
        assert len(df) == np.sum(df.values == 0)  # distance to other is not 0
        # (M2) Symmetry
        assert np.sum(df.values - df.values.T) == 0
        assert df.columns.equals(df.index)
        # (M3) Triangle inequality (we just probe this for a few random triplets)
        for _i in range(10):
            triplet = np.random.choice(df.index, size=3, replace=False)
            assert df.loc[triplet[0], triplet[1]] + df.loc[triplet[1], triplet[2]] >= df.loc[triplet[0], triplet[2]]

    @mark.parametrize("distance", actual_distances + pseudo_distances)
    def test_distance(self, adata, distance):
        Distance = pt.tl.Distance(distance, "X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", verbose=True)
        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry
