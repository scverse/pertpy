import numpy as np
import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc
from pandas import DataFrame, Series

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
lognorm_counts_distances = ["mean_var_distn"]
all_distances = (
    actual_distances + semi_distances + non_distances + pseudo_counts_distances + lognorm_counts_distances
)  # + onesided_only


@pytest.fixture
def all_pairwise_distances():
    all_calulated_distances = {}
    low_subsample_distances = [
        "sym_kldiv",
        "t_test",
        "ks_test",
        "classifier_proba",
        "classifier_cp",
        "mahalanobis",
        "mean_var_distn",
    ]
    no_subsample_distances = ["mahalanobis"]  # mahalanobis only works on the full data without subsampling

    for distance in all_distances:
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
            Distance = pt.tl.Distance(distance, layer_key="lognorm")
        elif distance in pseudo_counts_distances:
            Distance = pt.tl.Distance(distance, layer_key="counts")
        else:
            Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.pairwise(adata, groupby="perturbation", show_progressbar=True)
        all_calulated_distances[distance] = df

    return all_calulated_distances


def test_distance_axioms(all_pairwise_distances):
    for distance in actual_distances + semi_distances:
        # This is equivalent to testing for a semimetric, defined as fulfilling all axioms except triangle inequality.
        df = all_pairwise_distances[distance]

        # (M1) Definiteness
        assert all(np.diag(df.values) == 0)  # distance to self is 0

        # (M2) Positivity
        assert len(df) == np.sum(df.values == 0)  # distance to other is not 0
        assert all(df.values.flatten() >= 0)  # distance is non-negative

        # (M3) Symmetry
        assert np.sum(df.values - df.values.T) == 0


def test_triangle_inequality(all_pairwise_distances):
    for distance in actual_distances:
        # Test if distances are well-defined in accordance with metric axioms
        df = all_pairwise_distances[distance]

        # (M4) Triangle inequality (we just probe this for a few random triplets)
        for _i in range(10):
            rng = np.random.default_rng()
            triplet = rng.choice(df.index, size=3, replace=False)
            assert df.loc[triplet[0], triplet[1]] + df.loc[triplet[1], triplet[2]] >= df.loc[triplet[0], triplet[2]]


def test_distance_layers(all_pairwise_distances):
    for distance in all_distances:
        df = all_pairwise_distances[distance]

    assert isinstance(df, DataFrame)
    assert df.columns.equals(df.index)
    assert np.sum(df.values - df.values.T) == 0  # symmetry


def test_distance_counts(all_pairwise_distances):
    for distance in actual_distances + pseudo_counts_distances:
        df = all_pairwise_distances[distance]
        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0


def test_mutually_exclusive_keys():
    for distance in all_distances:
        with pytest.raises(ValueError):
            _ = pt.tl.Distance(distance, layer_key="counts", obsm_key="X_pca")


def test_distance_output_type(all_pairwise_distances):
    # Test if distances are outputting floats
    for distance in all_distances:
        df = all_pairwise_distances[distance]
        assert df.apply(lambda col: pd.api.types.is_float_dtype(col)).all(), "Not all values are floats."


def test_distance_pairwise(all_pairwise_distances):
    # Test consistency of pairwise distance results
    for distance in all_distances:
        df = all_pairwise_distances[distance]

        assert isinstance(df, DataFrame)
        assert df.columns.equals(df.index)
        assert np.sum(df.values - df.values.T) == 0  # symmetry


def test_distance_onesided():
    # Test consistency of one-sided distance results
    adata = pt.dt.distance_example()
    adata = sc.pp.subsample(adata, 0.1, copy=True)
    selected_group = adata.obs.perturbation.unique()[0]

    for distance in onesided_only:
        Distance = pt.tl.Distance(distance, obsm_key="X_pca")
        df = Distance.onesided_distances(adata, groupby="perturbation", selected_group=selected_group)

        assert isinstance(df, Series)
        assert df.loc[selected_group] == 0  # distance to self is 0
