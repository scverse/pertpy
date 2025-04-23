import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from scipy import sparse


@pytest.fixture
def adata_simple():
    exp_matrix = np.array(
        [
            [9, 0, 1, 0, 1, 0, 0],
            [1, 5, 1, 7, 0, 0, 0],
            [2, 0, 1, 0, 0, 8, 0],
            [1, 1, 1, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 5, 0],
            [9, 0, 1, 7, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 6],
            [8, 0, 1, 0, 0, 0, 0],
        ]
    ).astype(np.float32)
    adata = ad.AnnData(
        exp_matrix,
        obs=pd.DataFrame(index=[f"cell_{i + 1}" for i in range(exp_matrix.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i + 1}" for i in range(exp_matrix.shape[1])]),
    )
    return adata


@pytest.fixture
def tiny_dense_adata():
    exp_matrix = np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]]).astype(np.float32)
    return ad.AnnData(
        exp_matrix,
        obs=pd.DataFrame(index=[f"cell_{i + 1}" for i in range(exp_matrix.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i + 1}" for i in range(exp_matrix.shape[1])]),
    )


@pytest.fixture
def tiny_sparse_adata():
    sparse_matrix = sparse.csr_matrix(np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]]).astype(np.float32))
    return ad.AnnData(
        sparse_matrix,
        obs=pd.DataFrame(index=[f"cell_{i + 1}" for i in range(sparse_matrix.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i + 1}" for i in range(sparse_matrix.shape[1])]),
    )


@pytest.mark.parametrize("adata_fixture", ["tiny_dense_adata", "tiny_sparse_adata"])
def test_grna_threshold_assignment(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)
    threshold = 5
    output_layer = "assigned_guides"
    assert output_layer not in adata.layers

    ga = pt.pp.GuideAssignment()
    ga.assign_by_threshold(adata, assignment_threshold=threshold, output_layer=output_layer)

    assert output_layer in adata.layers

    # Convert to dense for comparison if needed
    if sparse.issparse(adata.layers[output_layer]):
        result_matrix = adata.layers[output_layer].toarray()
    else:
        result_matrix = adata.layers[output_layer]

    # Convert original data to dense for comparison if needed
    original_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X

    assert np.all(np.logical_xor(original_matrix < threshold, result_matrix == 1))


@pytest.mark.parametrize("adata_fixture", ["tiny_dense_adata", "tiny_sparse_adata"])
def test_grna_max_assignment(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)
    threshold = 6
    obs_key = "assigned_guide"
    assert obs_key not in adata.obs

    ga = pt.pp.GuideAssignment()
    ga.assign_to_max_guide(adata, assignment_threshold=threshold, obs_key=obs_key)
    assert obs_key in adata.obs
    assert tuple(adata.obs[obs_key]) == ("guide_1", "Negative", "guide_3")


def test_grna_mixture_model(adata_simple):
    output_key = "assigned_guide"
    assert output_key not in adata_simple.obs

    ga = pt.pp.GuideAssignment()
    ga.assign_mixture_model(adata_simple)
    assert output_key in adata_simple.obs
    target = [f"guide_{i}" if i > 0 else "negative" for i in [1, 4, 6, 0, 6, 1, 7, 1, 0]]
    assert all(t in g for t, g in zip(target, adata_simple.obs[output_key], strict=False))
