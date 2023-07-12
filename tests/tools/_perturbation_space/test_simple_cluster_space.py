import numpy as np
import pandas as pd
from anndata import AnnData

import pertpy as pt


def test_clustering():
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

    # Compute clustering at observation level
    ps = pt.tl.ClusteringSpace()
    adata = ps(adata, n_cluster=4, copy=True)

    ps = pt.tl.ClusteringSpace()
    adata = ps(adata, copy=True)

    results = ps.evaluate(adata, true_label_col="perturbations", cluster_col="k-means")
    np.testing.assert_equal(len(results), 3)

    results = ps.evaluate(adata, true_label_col="perturbations", cluster_col="dbscan")
    np.testing.assert_equal(len(results), 3)
