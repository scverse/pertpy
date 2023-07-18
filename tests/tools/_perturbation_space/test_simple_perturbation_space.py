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
    adata = adata[:, :3]
    ps = pt.tl.PseudobulkSpace()
    adata = ps.compute_differential_expression(adata, copy=True)

    # Test that the differential response was computed correctly
    expected_diff_matrix = adata.X - np.mean(adata.X[1:, :], axis=0)
    np.testing.assert_allclose(adata.X, expected_diff_matrix, rtol=1e-4)

    # Check that the function raises an error if the reference key is not found
    with pytest.raises(ValueError):
        ps.compute_differential_expression(
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

    # Compute the differential response
    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata, mode='mean', min_cells=0, min_counts=0)

    # Test that the pseudobulk response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].X.mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    # Check that the function raises an error if the layer key is not found
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
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

    # Compute the differential response

    ps = pt.tl.CentroidSpace()
    psadata = ps.compute(adata, embedding_key='X_umap')

    # Test that the pseudobulk response was computed correctly
    adata_target1 = adata[adata.obs.perturbations == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    # Check that the function raises an error if the layer key is not found
    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbations",
            embedding_key="not_found",
        )
