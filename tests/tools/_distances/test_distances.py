import numpy as np
import pandas as pd
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
    "mahalanobis",
]
semi_distances = ["r2_distance", "sym_kldiv", "ks_test"]
non_distances = ["classifier_proba"]
onesided_only = ["classifier_cp"]
pseudo_counts_distances = ["nb_ll"]
lognorm_counts_distances = ["mean_var_distribution"]
all_distances = (
    actual_distances + semi_distances + non_distances + lognorm_counts_distances + pseudo_counts_distances
)  # + onesided_only


class TestDistances:
    @fixture
    def adata(self, request):
        low_subsample_distances = [
            "sym_kldiv",
            "t_test",
            "ks_test",
            "classifier_proba",
            "classifier_cp",
            "mahalanobis",
            "mean_var_distribution",
        ]
        no_subsample_distances = ["mahalanobis"]  # mahalanobis only works on the full data without subsampling

        distance = request.node.callspec.params["distance"]

        adata = pt.dt.distance_example()
        if distance not in no_subsample_distances:
            if distance in low_subsample_distances:
                adata = sc.pp.subsample(adata, 0.1, copy=True)
            else:
                adata = sc.pp.subsample(adata, 0.001, copy=True)

        adata.layers["lognorm"] = adata.X.copy()
        adata.layers["counts"] = np.round(adata.X.toarray()).astype(int)
        if "X_pca" not in adata.obsm.keys():
            sc.pp.pca(adata, n_comps=5)
        if distance in lognorm_counts_distances:
            groups = np.unique(adata.obs["perturbation"])
            # KDE is slow, subset to 5 groups for speed up
            adata = adata[adata.obs["perturbation"].isin(groups[0:5])].copy()

        return adata

    @fixture
    def distance_obj(self, request):
        distance = request.node.callspec.params["distance"]
        if distance in lognorm_counts_distances:
            Distance = pt.tl.Distance(distance, layer_key="lognorm")
        elif distance in pseudo_counts_distances:
            Distance = pt.tl.Distance(distance, layer_key="counts")
        else:
            Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        return Distance

    @mark.parametrize("distance", actual_distances + semi_distances + non_distances)
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
        assert len(df) == np.sum(df.values == 0)  # distance to other is not 0 (TODO)
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
    def test_distance_layers(self, adata, distance_obj, distance):
        df = distance_obj.pairwise(adata, groupby="perturbation")

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
    def test_mutually_exclusive_keys(self, distance):
        with pytest.raises(ValueError):
            _ = pt.tl.Distance(distance, layer_key="counts", obsm_key="X_pca")

    @mark.parametrize("distance", actual_distances + semi_distances + non_distances)
    def test_distance_output_type(self, distance):
        # Test if distances are outputting floats
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        rng = np.random.default_rng()
        X = rng.normal(size=(100, 10))
        Y = rng.normal(size=(100, 10))
        d = Distance(X, Y)
        assert isinstance(d, float)

    @mark.parametrize("distance", all_distances)
    def test_distance_pairwise(self, adata, distance_obj, distance):
        # Test consistency of pairwise distance results
        df = distance_obj.pairwise(adata, groupby="perturbation")

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry

    @mark.parametrize("distance", all_distances + onesided_only)
    def test_distance_onesided(self, adata, distance_obj, distance):
        # Test consistency of one-sided distance results
        selected_group = adata.obs.perturbation.unique()[0]
        df = distance_obj.onesided_distances(adata, groupby="perturbation", selected_group=selected_group)
        assert isinstance(df, Series)
        assert df.loc[selected_group] == 0  # distance to self is 0

    @mark.parametrize("distance", actual_distances + semi_distances + non_distances)
    def test_bootstrap_distance_output_type(self, distance):
        # Test if distances are outputting floats
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        rng = np.random.default_rng()
        X = rng.normal(size=(100, 10))
        Y = rng.normal(size=(100, 10))
        d = Distance.bootstrap(X, Y, n_bootstrap=10)
        assert hasattr(d, "mean")
        assert hasattr(d, "variance")

    @mark.parametrize("distance", all_distances)
    def test_bootstrap_distance_pairwise(self, adata, distance_obj, distance):
        # Test consistency of pairwise distance results
        bootstrap_output = distance_obj.pairwise(adata, groupby="perturbation", bootstrap=True, n_bootstrap=10)
        assert isinstance(bootstrap_output, tuple)

        mean = bootstrap_output[0]
        var = bootstrap_output[1]

        assert mean.columns.equals(mean.index)
        assert np.sum(mean.values - mean.values.T) == 0  # symmetry
        assert np.sum(var.values - var.values.T) == 0  # symmetry

    @mark.parametrize("distance", all_distances)
    def test_bootstrap_distance_onesided(self, adata, distance_obj, distance):
        # Test consistency of one-sided distance results
        selected_group = adata.obs.perturbation.unique()[0]
        bootstrap_output = distance_obj.onesided_distances(
            adata,
            groupby="perturbation",
            selected_group=selected_group,
            bootstrap=True,
            n_bootstrap=10,
        )

        assert isinstance(bootstrap_output, tuple)
