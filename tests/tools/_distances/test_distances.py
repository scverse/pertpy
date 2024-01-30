import numpy as np
import pertpy as pt
import pytest
import scanpy as sc
from pandas import DataFrame, Series
from pytest import fixture, mark

actual_distances = [
    # Euclidean distances and related
    "euclidean",
    "mean_absolute_error",
    "mean_pairwise",
    "mse",
    "edistance",
    # Other
    "cosine_distance",
    "kendalltau_distance",
    "mmd",
    "pearson_distance",
    "spearman_distance",
    "t_test",
    "wasserstein",
]
semi_distances = ["r2_distance", "sym_kldiv", "ks_test"]
non_distances = ["classifier_proba"]
onesided_only = ["classifier_cp"]
pseudo_counts_distances = ["nb_ll"]
all_distances = actual_distances + semi_distances + non_distances  # + onesided_only + pseudo_counts_distances


class TestDistances:
    @fixture
    def adata(self, request):
        no_subsample_distances = ["sym_kldiv", "t_test", "ks_test", "classifier_proba", "classifier_cp"]
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

    @mark.parametrize("distance", all_distances)
    def test_distance(self, adata, distance):
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        assert isinstance(df, DataFrame)

    @mark.parametrize("distance", actual_distances + semi_distances)
    def test_distance_axioms(self, adata, distance):
        # This is equivalent to testing for a semimetric, defined as fulfilling all axioms except triangle inequality.
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        # (M1) Definiteness
        assert all(np.diag(df.values) == 0)  # distance to self is 0

        # (M2) Positivity
        assert len(df) == np.sum(df.values == 0)  # distance to other is not 0
        assert all(df.values.flatten() >= 0)  # distance is non-negative

        # (M3) Symmetry
        assert np.sum(df.values - df.values.T) == 0

    @mark.parametrize("distance", actual_distances)
    def test_triangle_inequality(self, adata, distance):
        # Test if distances are well-defined in accordance with metric axioms
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)

        # (M4) Triangle inequality (we just probe this for a few random triplets)
        for _i in range(10):
            rng = np.random.default_rng()
            triplet = rng.choice(df.index, size=3, replace=False)
            assert df.loc[triplet[0], triplet[1]] + df.loc[triplet[1], triplet[2]] >= df.loc[triplet[0], triplet[2]]

    @mark.parametrize("distance", all_distances)
    def test_distance_layers(self, adata, distance):
        Distance = pt.tl.Distance(distance, layer_key="lognorm")
        df = Distance.pairwise(adata, groupby="perturbation")

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    @mark.parametrize("distance", actual_distances + pseudo_counts_distances)
    def test_distance_counts(self, adata, distance):
        Distance = pt.tl.Distance(distance, layer_key="counts")
        df = Distance.pairwise(adata, groupby="perturbation")
        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0

    @mark.parametrize("distance", all_distances)
    def test_mutually_exclusive_keys(self, adata, distance):
        with pytest.raises(ValueError):
            _ = pt.tl.Distance(distance, layer_key="counts", obsm_key="X_pca")

    @mark.parametrize("distance", all_distances)
    def test_distance_output_type(self, distance):
        # Test if distances are outputting floats
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        rng = np.random.default_rng()
        X = rng.normal(size=(100, 10))
        Y = rng.normal(size=(100, 10))
        d = Distance(X, Y)
        assert isinstance(d, float)

    @mark.parametrize("distance", all_distances)
    def test_distance_pairwise(self, adata, distance):
        # Test consistency of pairwise distance results
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation")

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    @mark.parametrize("distance", all_distances + onesided_only)
    def test_distance_onesided(self, adata, distance):
        # Test consistency of one-sided distance results
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        selected_group = adata.obs.perturbation.unique()[0]
        df = Distance.onesided_distances(adata, groupby="perturbation", selected_group=selected_group)
        assert isinstance(df, Series)
        assert df.loc[selected_group] == 0  # distance to self is 0
