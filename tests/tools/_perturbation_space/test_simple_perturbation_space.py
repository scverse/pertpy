import numpy as np
import pandas as pd
import pertpy as pt
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def adata(rng):
    X = rng.random((69, 50))
    adata = AnnData(X)
    perturbations = np.array(["control", "target1", "target2"] * 22 + ["unknown"] * 3)
    adata.obs["perturbation"] = perturbations
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)

    return adata


@pytest.fixture
def adata_simple(rng):
    X = rng.random(size=(10, 5))
    obs = pd.DataFrame(
        {
            "perturbation": [
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

    return adata


def test_differential_response(adata_simple):
    ps = pt.tl.PseudobulkSpace()
    ps_adata = ps.compute_control_diff(adata_simple, target_col="perturbation", copy=True)

    expected_diff_matrix = adata_simple.X - adata_simple.X[0, :]
    np.testing.assert_allclose(ps_adata.X, expected_diff_matrix, rtol=1e-4)

    with pytest.raises(ValueError):
        ps.compute_control_diff(
            adata_simple,
            target_col="perturbation",
            reference_key="not_found",
            layer_key="counts",
            new_layer_key="counts_diff",
            embedding_key="X_pca",
            new_embedding_key="pca_diff",
            copy=True,
        )


def test_pseudobulk_response(adata_simple):
    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata_simple, mode="mean", min_cells=0, min_counts=0)

    adata_target1 = adata_simple[adata_simple.obs.perturbation == "target1"].X.mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    adata_simple.obsm["X_umap"] = adata_simple.X

    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata_simple, embedding_key="X_umap", mode="mean", min_cells=0, min_counts=0)

    adata_target1 = adata_simple[adata_simple.obs.perturbation == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    with pytest.raises(ValueError):
        ps.compute(
            adata_simple,
            target_col="perturbation",
            layer_key="not_found",
        )

    with pytest.raises(ValueError):
        ps.compute(
            adata_simple,
            target_col="perturbation",
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

    obs = pd.DataFrame({"perturbation": pert_index})

    adata = AnnData(X, obs=obs)
    adata.obsm["X_umap"] = X

    ps = pt.tl.CentroidSpace()
    psadata = ps.compute(adata, embedding_key="X_umap")

    adata_target1 = adata[adata.obs.perturbation == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    ps = pt.tl.CentroidSpace()
    psadata = ps.compute(adata)

    adata_target1 = adata[adata.obs.perturbation == "target1"].obsm["X_umap"].mean(0)
    np.testing.assert_allclose(adata_target1, psadata["target1"].X[0], rtol=1e-4)

    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbation",
            embedding_key="not_found",
        )

    with pytest.raises(ValueError):
        ps.compute(
            adata,
            target_col="perturbation",
            embedding_key="not_found",
            layer_key="not_found",
        )


def test_linear_operations():
    """Tests add/subtract and other linear operations."""
    rng = np.random.default_rng()
    X = rng.random(size=(10, 5))
    obs = pd.DataFrame(
        {
            "perturbation": [
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

    ps = pt.tl.PseudobulkSpace()
    psadata = ps.compute(adata, mode="mean", min_cells=0, min_counts=0)

    psadata_umap = ps.compute(adata, mode="mean", min_cells=0, min_counts=0, embedding_key="X_umap")
    psadata.obsm["X_umap"] = psadata_umap.X

    ps_adata, data_compare = ps.add(psadata, perturbations=["target1", "target2"], ensure_consistency=True)

    test = data_compare["control"].X + data_compare["target1"].X + data_compare["target2"].X
    np.testing.assert_allclose(test, ps_adata["target1+target2"].X, rtol=1e-4)

    test = (
        data_compare["control"].obsm["X_umap_control_diff"]
        + data_compare["target1"].obsm["X_umap_control_diff"]
        + data_compare["target2"].obsm["X_umap_control_diff"]
    )
    np.testing.assert_allclose(test, ps_adata["target1+target2"].obsm["X_umap"], rtol=1e-4)

    ps_adata, data_compare = ps.subtract(
        psadata, reference_key="target1", perturbations=["target2"], ensure_consistency=True
    )

    test = data_compare["target1"].X - data_compare["target2"].X
    np.testing.assert_allclose(test, ps_adata["target1-target2"].X, rtol=1e-4)

    ps_adata = ps.compute_control_diff(psadata, copy=True)

    ps_adata2 = ps.add(ps_adata, perturbations=["target1", "target2"])

    test = ps_adata["control"].X + ps_adata["target1"].X + ps_adata["target2"].X
    np.testing.assert_allclose(test, ps_adata2["target1+target2"].X, rtol=1e-4)

    ps_adata2 = ps.subtract(ps_adata, reference_key="target1", perturbations=["target1"])
    ps_vector = ps_adata2["target1-target1"].X
    np.testing.assert_allclose(ps_adata2["control"].X, ps_adata2["target1-target1"].X, rtol=1e-4)

    ps_adata2, data_compare = ps.subtract(
        ps_adata, reference_key="target1", perturbations=["target1"], ensure_consistency=True
    )
    ps_inner_vector = ps_adata2["target1-target1"].X

    np.testing.assert_allclose(ps_inner_vector, ps_vector, rtol=1e-4)

    np.testing.assert_allclose(
        data_compare["control"].obsm["X_umap_control_diff"], ps_adata2["target1-target1"].obsm["X_umap"], rtol=1e-4
    )

    with pytest.raises(ValueError):
        ps.add(
            ps_adata,
            perturbations=["target1", "target3"],
        )

    with pytest.raises(ValueError):
        ps.add(
            ps_adata,
            perturbations=["target1", "target3"],
        )


def test_label_transfer():
    rng = np.random.default_rng()
    X = rng.standard_normal((69, 50))
    adata = AnnData(X)
    perturbations = np.array(["A", "B", "C"] * 22 + ["unknown"] * 3)
    adata.obs["perturbation"] = perturbations

    with pytest.raises(ValueError):
        ps = pt.tl.PseudobulkSpace()
        ps.label_transfer(adata)

    sc.pp.neighbors(adata, use_rep="X")
    sc.tl.umap(adata)
    ps = pt.tl.PseudobulkSpace()
    ps.label_transfer(adata)
    assert "unknown" not in adata.obs["perturbation"]
    assert all(adata.obs["perturbation_transfer_uncertainty"] >= 0)
    assert not all(adata.obs["perturbation_transfer_uncertainty"] == 0)
    is_known = perturbations != "unknown"
    assert all(adata.obs.loc[is_known, "perturbation_transfer_uncertainty"] == 0)
