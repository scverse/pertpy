import numpy as np
import pandas as pd
from anndata import AnnData

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
