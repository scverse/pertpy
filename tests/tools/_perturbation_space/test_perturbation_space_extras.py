import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def adata(rng):
    labels = np.array(["control", "A", "B"]).repeat(20)
    centers = {"control": 0.0, "A": 5.0, "B": -5.0}
    X = np.vstack([rng.normal(centers[label], 0.3, size=8) for label in labels])
    adata = AnnData(X, obs=pd.DataFrame({"perturbation": labels}))
    sc.pp.pca(adata, n_comps=5)
    return adata


def test_distance_space(adata):
    ds = pt.tl.DistanceSpace()
    ds_adata = ds.compute(adata, metric="euclidean", embedding_key="X_pca")

    assert ds_adata.shape == (3, 3)
    assert "distances" in ds_adata.obsp
    np.testing.assert_allclose(np.diag(ds_adata.obsp["distances"]), 0.0, atol=1e-6)
    # feeds directly into clustering
    clustered = pt.tl.KMeansSpace().compute(ds_adata, n_clusters=3, copy=True, random_state=0)
    assert clustered.obs["k-means"].nunique() == 3


def test_nearest_perturbations(adata):
    ds_adata = pt.tl.DistanceSpace().compute(adata, metric="euclidean", embedding_key="X_pca")
    neighbors = pt.tl.DistanceSpace().nearest_perturbations(ds_adata, "A", n_neighbors=2)
    assert list(neighbors.index) == ["control", "B"]  # control is closer to A than B


def test_embedding_space(adata):
    embedding = pd.DataFrame(np.eye(3), index=["A", "B", "control"])
    es_adata = pt.tl.EmbeddingSpace().compute(adata, embedding, target_col="perturbation")
    assert es_adata.shape == (3, 3)
    assert set(es_adata.obs_names) == {"A", "B", "control"}

    with pytest.raises(ValueError, match="No overlap"):
        pt.tl.EmbeddingSpace().compute(adata, pd.DataFrame(np.eye(2), index=["X", "Y"]))


def test_evaluate_combinations(rng):
    dim = 6
    eff = {"A": rng.normal(size=dim), "B": rng.normal(size=dim), "C": rng.normal(size=dim)}
    rows = {
        "control": np.zeros(dim),
        "A": eff["A"],
        "B": eff["B"],
        "C": eff["C"],
        "A+B": eff["A"] + eff["B"],  # perfectly additive
        "A+C": eff["A"] + eff["C"] + rng.normal(0, 3, dim),  # interaction
    }
    ps_adata = AnnData(X=np.vstack(list(rows.values())))
    ps_adata.obs_names = list(rows.keys())
    ps_adata.obs["perturbation"] = pd.Categorical(list(rows.keys()))

    result = pt.tl.PseudobulkSpace().evaluate_combinations(ps_adata, reference_key="control", metric="pearson")
    assert set(result.index) == {"A+B", "A+C"}
    assert result.loc["A+B", "distance"] < result.loc["A+C", "distance"]
    np.testing.assert_allclose(result.loc["A+B", "distance"], 0.0, atol=1e-6)


def test_dose_response(rng):
    groups, doses = [], []
    for pert in ["control", "drug"]:
        for dose in [0.0] if pert == "control" else [1.0, 10.0, 100.0]:
            groups += [pert] * 15
            doses += [dose] * 15
    groups = np.array(groups)
    doses = np.array(doses, dtype=float)
    X = rng.normal(0, 0.3, (len(groups), 8))
    X[groups == "drug"] += (doses[groups == "drug"] / 10.0)[:, None]
    adata = AnnData(X, obs=pd.DataFrame({"perturbation": groups, "dose": doses}))
    sc.pp.pca(adata, n_comps=5)

    curves = pt.tl.PseudobulkSpace().dose_response(adata, dose_col="dose", metric="euclidean", embedding_key="X_pca")
    drug = curves[curves["perturbation"] == "drug"].sort_values("dose")
    assert drug["distance"].is_monotonic_increasing
