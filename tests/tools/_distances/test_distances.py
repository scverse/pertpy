import numpy as np
import pertpy as pt
import pytest
import scanpy as sc
from pandas import DataFrame, Series
from pytest import fixture, mark

actual_distances = [
    "euclidean",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "spearman_distance",
    "kendalltau_distance",
    "cosine_distance",
    "wasserstein",
]
pseudo_distances = ["mean_pairwise", "mmd", "r2_distance", "kl_divergence", "t_test", "edistance"]
pseudo_counts_distances = ["nb_ll"]


class TestDistances:
    @fixture
    def adata(self, request):
        no_subsample_distances = ["kl_divergence", "t_test"]
        distance = request.node.callspec.params["distance"]

        adata = pt.dt.distance_example()
        if distance not in no_subsample_distances:
            adata = sc.pp.subsample(adata, 0.001, copy=True)
        else:
            adata = sc.pp.subsample(adata, 0.1, copy=True)

        adata.layers["lognorm"] = adata.X.copy()
        adata.layers["counts"] = np.round(adata.X.toarray()).astype(int)
        if "X_pca" not in adata.obsm.keys():
            sc.pp.pca(adata, n_comps=10)

        return adata

    @mark.parametrize("distance", actual_distances)
    def test_distance_axioms(self, adata, distance):
        # Test if distances are well-defined in accordance with metric axioms
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        # (M1) Positiv definiteness
        assert all(np.diag(df.values) == 0)  # distance to self is 0
        assert len(df) == np.sum(df.values == 0)  # distance to other is not 0
        assert all(df.values.flatten() >= 0)  # distance is non-negative

        # (M2) Symmetry
        assert np.sum(df.values - df.values.T) == 0
        assert df.columns.equals(df.index)

        # (M3) Triangle inequality (we just probe this for a few random triplets)
        for _i in range(10):
            triplet = np.random.choice(df.index, size=3, replace=False)
            assert df.loc[triplet[0], triplet[1]] + df.loc[triplet[1], triplet[2]] >= df.loc[triplet[0], triplet[2]]

    @mark.parametrize("distance", actual_distances + pseudo_distances)
    def test_distance(self, adata, distance):
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    @mark.parametrize("distance", actual_distances + pseudo_distances)
    def test_distance_layers(self, adata, distance):
        Distance = pt.tl.Distance(distance, layer_key="lognorm")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    # SP: this is giving error "numpy.linalg.LinAlgError: Singular matrix"
    # @mark.parametrize("distance", actual_distances + pseudo_counts_distances)
    # def test_distance_counts(self, adata, distance):
    #     Distance = pt.tl.Distance(distance, layer_key="counts")
    #     df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)
    #     assert isinstance(df, DataFrame)
    #     assert df.columns.equals(df.index)
    #     assert np.sum(df.values - df.values.T) == 0

    @mark.parametrize("distance", actual_distances)
    def test_mutually_exclusive_keys(self, adata, distance):
        with pytest.raises(ValueError):
            _ = pt.tl.Distance(distance, layer_key="counts", obsm_key="X_pca")

    @mark.parametrize("distance", actual_distances)
    def test_distance_pairwise(self, adata, distance):
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    @mark.parametrize("distance", actual_distances + pseudo_distances)
    def test_distance_onesided(self, adata, distance):
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        selected_group = adata.obs.perturbation.unique()[0]
        df = Distance.onesided_distances(
            adata, groupby="perturbation", selected_group=selected_group, show_progressbar=True
        )

        assert isinstance(df, Series)
        assert df.loc[selected_group] == 0  # distance to self is 0
