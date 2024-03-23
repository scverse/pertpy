import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from anndata import AnnData


@pytest.fixture
def adata():
    X = np.zeros((20, 5), dtype=np.float32)

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

    # Add a obs annotations to the adata
    adata.obs["MoA"] = ["Growth" if pert == "target1" else "Unknown" for pert in adata.obs["perturbations"]]
    adata.obs["Partial Annotation"] = ["Anno1" if pert == "target2" else np.nan for pert in adata.obs["perturbations"]]

    return adata


def test_mlp_classifier_space(adata):
    classifier_ps = pt.tl.MLPClassifierSpace()
    pert_embeddings = classifier_ps.compute(adata, hidden_dim=[128], max_epochs=2)

    # The embeddings should cluster in 3 perfects clusters since the perturbations are easily separable
    ps = pt.tl.KMeansSpace()
    adata = ps.compute(pert_embeddings, n_clusters=3, copy=True)
    results = ps.evaluate_clustering(adata, true_label_col="perturbations", cluster_col="k-means")
    np.testing.assert_equal(len(results), 3)
    np.testing.assert_allclose(results["nmi"], 0.99, rtol=0.1)
    np.testing.assert_allclose(results["ari"], 0.99, rtol=0.1)
    np.testing.assert_allclose(results["asw"], 0.99, rtol=0.1)


def test_regression_classifier_space(adata):
    ps = pt.tl.LRClassifierSpace()
    pert_embeddings = ps.compute(adata)

    assert pert_embeddings.shape == (3, 5)
    assert pert_embeddings.obs[pert_embeddings.obs["perturbations"] == "target1"]["MoA"].values == "Growth"
    assert "Partial Annotation" not in pert_embeddings.obs_names
    # The classifier should be able to distinguish control and target2 from the respective other two classes
    assert np.all(
        pert_embeddings.obs[pert_embeddings.obs["perturbations"].isin(["control", "target2"])][
            "classifier_score"
        ].values
        == 1.0
    )
