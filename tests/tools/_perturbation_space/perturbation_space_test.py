import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pertpy as pt


def test_simple_differential_response():
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
    ps = pt.tl.PerturbationSpace()
    adata = ps.simple_differential_response(adata, copy=True)

    # Test that the differential response was computed correctly
    expected_diff_matrix = adata.X - np.mean(adata.X[1:, :], axis=0)
    np.testing.assert_allclose(adata.X, expected_diff_matrix, rtol=1e-4)

    # Check that the function raises an error if the reference key is not found
    with pytest.raises(ValueError):
        ps.simple_differential_response(
            adata,
            target_col="perturbations",
            reference_key="not_found",
            layer_key="counts",
            new_layer_key="counts_diff",
            embedding_key="X_pca",
            new_embedding_key="pca_diff",
            copy=True,
        )
