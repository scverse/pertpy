import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import pertpy as pt


def _blobs(rng):
    labels = np.array(["control", "target1", "target2"]).repeat(8)
    centers = {"control": 0.0, "target1": 10.0, "target2": 30.0}
    X = np.vstack([rng.normal(centers[label], 0.1, size=5) for label in labels])
    return AnnData(X, obs=pd.DataFrame({"perturbations": labels}))


def test_kmeans(rng):
    adata = _blobs(rng)
    ps = pt.tl.KMeansSpace()
    adata = ps.compute(adata, n_clusters=3, copy=True, random_state=0)

    results = ps.evaluate_clustering(
        adata, true_label_col="perturbations", cluster_col="k-means", metrics=["nmi", "ari"]
    )
    np.testing.assert_allclose(results["nmi"], 1.0, rtol=0.1)
    np.testing.assert_allclose(results["ari"], 1.0, rtol=0.1)


def test_hdbscan(rng):
    adata = _blobs(rng)
    ps = pt.tl.HDBSCANSpace()
    adata = ps.compute(adata, min_cluster_size=3, copy=True)

    assert "hdbscan" in adata.obs
    results = ps.evaluate_clustering(
        adata, true_label_col="perturbations", cluster_col="hdbscan", metrics=["nmi", "ari"]
    )
    np.testing.assert_allclose(results["nmi"], 1.0, rtol=0.1)
    np.testing.assert_allclose(results["ari"], 1.0, rtol=0.1)


def test_evaluate_clustering_asw_uses_cell_distances(rng):
    adata = _blobs(rng)
    adata.obs["k-means"] = adata.obs["perturbations"]

    results = pt.tl.KMeansSpace().evaluate_clustering(
        adata, true_label_col="perturbations", cluster_col="k-means", metrics=["asw"]
    )
    np.testing.assert_allclose(results["asw"], silhouette_score(adata.X, adata.obs["perturbations"]))


@pytest.mark.parametrize("input_source", ["X", "embedding", "precomputed"])
def test_evaluate_clustering_asw_uses_predicted_clusters(input_source: str) -> None:
    """ASW must score the clustering being evaluated, not ground-truth categories."""
    points = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])[:, None]
    truth = ["A", "A", "A", "B", "B", "B"]
    predictions = ["A", "B", "A", "B", "A", "B"]
    adata = AnnData(X=points, obs=pd.DataFrame({"truth": truth, "predicted": predictions}))
    options: dict = {}
    if input_source == "embedding":
        adata.obsm["X_test"] = points.copy()
        options["embedding_key"] = "X_test"
    elif input_source == "precomputed":
        options["distances"] = cdist(points, points)

    observed = pt.tl.KMeansSpace().evaluate_clustering(
        adata, true_label_col="truth", cluster_col="predicted", metrics=["asw"], **options
    )["asw"]
    expected = silhouette_score(points, predictions)

    np.testing.assert_allclose(observed, expected)
    assert not np.isclose(expected, silhouette_score(points, truth))


def test_evaluate_clustering_asw_rejects_single_predicted_cluster() -> None:
    """A single predicted cluster has undefined ASW even if the truth has multiple groups."""
    points = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])[:, None]
    adata = AnnData(
        X=points,
        obs=pd.DataFrame({"truth": ["A"] * 3 + ["B"] * 3, "predicted": ["one_cluster"] * 6}),
    )

    with pytest.raises(ValueError, match="Number of labels is 1"):
        pt.tl.KMeansSpace().evaluate_clustering(adata, true_label_col="truth", cluster_col="predicted", metrics=["asw"])
