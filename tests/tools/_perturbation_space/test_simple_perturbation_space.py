import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pertpy as pt


def test_differential_response():
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(
        {
            "perturbations": [
                "control",
                "target1",
                "target1",
                "target2",
                "target2",
                "target1",
                "target1",
                "target2",
                "target2",
                "target2",
            ]
        }
    )
    adata = AnnData(X, obs=obs)

    # Compute the differential response
    ps = pt.tl.PseudobulkSpace()
    ps_adata = ps.compute_control_diff(adata, copy=True)

    # Test that the differential response was computed correctly
    expected_diff_matrix = adata.X - adata.X[0, :]
    np.testing.assert_allclose(ps_adata.X, expected_diff_matrix, rtol=1e-4)

    # Check that the function raises an error if the reference key is not found
    with pytest.raises(ValueError):
        ps.compute_control_diff(
            adata,
            target_col="perturbations",
            reference_key="not_found",
            layer_key="counts",
            new_layer_key="counts_diff",
            embedding_key="X_pca",
            new_embedding_key="pca_diff",
            copy=True,
        )


def test_pseudobulk_response():
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(
        {
            "perturbations": [
                "control",
                "target1",
                "target1",
                "target2",
                "target2",
                "target1",
                "target1",
                "target2",
                "target2",
                "target2",
            ]
        }
    )
    adata = AnnData(X, obs=obs)

    # Compute the pseudobulk
    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata, mode="mean", min_cells=0, min_counts=0)

    # Test that the pseudobulk response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].X.mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    # Test in UMAP space
    adata.obsm["X_umap"] = X

    # Compute the pseudobulk
    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata, embedding_key="X_umap", mode="mean", min_cells=0, min_counts=0)

    # Test that the pseudobulk response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    # Check that the function raises an error if the layer key is not found
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
            layer_key="not_found",
        )

    # Check that the function raises an error if the layer key and embedding key are used at the same time
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
            embedding_key="not_found",
            layer_key="not_found",
        )


def test_centroid_umap_response():
    X = np.zeros((10, 5))

    pert_index = [
        "control",
        "target1",
        "target1",
        "target2",
        "target2",
        "target1",
        "target1",
        "target2",
        "target2",
        "target2",
    ]

    for i, value in enumerate(pert_index):
        if value == "control":
            X[i, :] = 0
        elif value == "target1":
            X[i, :] = 10
        elif value == "target2":
            X[i, :] = 30

    obs = pd.DataFrame({"perturbations": pert_index})

    adata = AnnData(X, obs=obs)
    adata.obsm["X_umap"] = X

    # Compute the centroids
    ps = pt.tl.CentroidSpace()
    psadata = ps.compute(adata, embedding_key="X_umap")

    # Test that the centroids response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    ps = pt.tl.CentroidSpace()
    psadata = ps.compute(adata)  # if nothing specific, compute with X, and X and X_umap are the same

    # Test that the centroids response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    # Check that the function raises an error if the embedding key is not found
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
            embedding_key="not_found",
        )

    # Check that the function raises an error if the layer key and embedding key are used at the same time
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
            embedding_key="not_found",
            layer_key="not_found",
        )


def test_linear_operations():
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(
        {
            "perturbations": [
                "control",
                "target1",
                "target1",
                "target2",
                "target2",
                "target1",
                "target1",
                "target2",
                "target2",
                "target2",
            ]
        }
    )
    adata = AnnData(X, obs=obs)
    adata.obsm["X_umap"] = X

    # Compute the pseudobulk
    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata, mode="mean", min_cells=0, min_counts=0)

    psadata_umap = ps.compute(adata, mode="mean", min_cells=0, min_counts=0, embedding_key="X_umap")
    psadata.obsm["X_umap"] = psadata_umap.X

    # Do summation
    ps_adata, data_compare = ps.add(psadata, perturbations=["target1", "target2"], ensure_consistency=True)

    # Test in X
    test = data_compare["control"].X + data_compare["target1"].X + data_compare["target2"].X
    np.testing.assert_allclose(test, ps_adata["target1+target2"].X, rtol=1e-4)

    # Test in UMAP
    test = (
        data_compare["control"].obsm["X_umap_control_diff"]
        + data_compare["target1"].obsm["X_umap_control_diff"]
        + data_compare["target2"].obsm["X_umap_control_diff"]
    )
    np.testing.assert_allclose(test, ps_adata["target1+target2"].obsm["X_umap"], rtol=1e-4)

    # Do subtraction
    ps_adata, data_compare = ps.subtract(
        psadata, reference_key="target1", perturbations=["target2"], ensure_consistency=True
    )

    # Test in X
    test = data_compare["target1"].X - data_compare["target2"].X
    np.testing.assert_allclose(test, ps_adata["target1-target2"].X, rtol=1e-4)

    # Operations after differential expression, do the results match?
    ps_adata = ps.compute_control_diff(psadata, copy=True)

    # Do summation
    ps_adata2 = ps.add(ps_adata, perturbations=["target1", "target2"])

    # Test in X
    test = ps_adata["control"].X + ps_adata["target1"].X + ps_adata["target2"].X
    np.testing.assert_allclose(test, ps_adata2["target1+target2"].X, rtol=1e-4)

    # Do subtract
    ps_adata2 = ps.subtract(ps_adata, reference_key="target1", perturbations=["target1"])
    ps_vector = ps_adata2["target1-target1"].X
    np.testing.assert_allclose(ps_adata2["control"].X, ps_adata2["target1-target1"].X, rtol=1e-4)

    ps_adata2, data_compare = ps.subtract(
        ps_adata, reference_key="target1", perturbations=["target1"], ensure_consistency=True
    )
    ps_inner_vector = ps_adata2["target1-target1"].X

    # Compare process data vs pseudobulk before, should be the same
    np.testing.assert_allclose(ps_inner_vector, ps_vector, rtol=1e-4)

    # Check result in UMAP
    np.testing.assert_allclose(
        data_compare["control"].obsm["X_umap_control_diff"], ps_adata2["target1-target1"].obsm["X_umap"], rtol=1e-4
    )

    # Check that the function raises an error if the perturbation is not found
    with pytest.raises(ValueError):
        ps.add(
            ps_adata,
            perturbations=["target1", "target3"],
        )

    # Check that the function raises an error if some key is not found
    with pytest.raises(ValueError):
        ps.add(
            ps_adata,
            perturbations=["target1", "target3"],
        )
