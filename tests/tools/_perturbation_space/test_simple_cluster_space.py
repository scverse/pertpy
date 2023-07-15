import numpy as np
import pandas as pd
from anndata import AnnData

import pertpy as pt


def test_clustering():
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
            
    obs = pd.DataFrame(
        {
            "perturbations": pert_index
        }
    )

    adata = AnnData(X, obs=obs)

    # Compute clustering at observation level
    ps = pt.tl.KMeansSpace()
    adata = ps(adata, n_clusters=3, copy=True)

    ps = pt.tl.DBSCANSpace()
    adata = ps(adata, min_samples=1, copy=True)

    results = ps.evaluate_clustering(adata, true_label_col="perturbations", cluster_col="k-means")
    np.testing.assert_equal(len(results), 3)
    np.testing.assert_allclose(results['nmi'], 0.99, rtol=0.1)
    np.testing.assert_allclose(results['ari'], 0.99, rtol=0.1)
    np.testing.assert_allclose(results['asw'], 0.99, rtol=0.1)

    results = ps.evaluate_clustering(adata, true_label_col="perturbations", cluster_col="dbscan")
    np.testing.assert_equal(len(results), 3)
    np.testing.assert_allclose(results['nmi'], 0.99, rtol=0.1)
    np.testing.assert_allclose(results['ari'], 0.99, rtol=0.1)
    np.testing.assert_allclose(results['asw'], 0.99, rtol=0.1)
