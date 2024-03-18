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

    return adata

def test_mlp_discriminator_classifier(adata):
    # Compute the embeddings using the MLP classifier
    ps = pt.tl.DiscriminatorClassifierSpace("mlp")
    classifier_ps = ps.load(adata, hidden_dim=[128])
    classifier_ps.train(max_epochs=2)
    pert_embeddings = classifier_ps.get_embeddings()

    # The embeddings should cluster in 3 perfects clusters since the perturbations are easily separable
    ps = pt.tl.KMeansSpace()
    adata = ps.compute(pert_embeddings, n_clusters=3, copy=True)
    results = ps.evaluate_clustering(adata, true_label_col="perturbations", cluster_col="k-means")
    np.testing.assert_equal(len(results), 3)
    np.testing.assert_allclose(results["nmi"], 0.99, rtol=0.1)
    np.testing.assert_allclose(results["ari"], 0.99, rtol=0.1)
    np.testing.assert_allclose(results["asw"], 0.99, rtol=0.1)

def test_regression_discriminator_classifier(adata):
    # Compute the embeddings using the regression classifier
    ps = pt.tl.DiscriminatorClassifierSpace("regression")
    classifier_ps = ps.load(adata)
    classifier_ps.train()
    pert_embeddings = classifier_ps.get_embeddings()

    assert pert_embeddings.shape == (3, 5)
    # The classifier should be able to distinguish control and target2 from the respective other two classes
    assert np.all(pert_embeddings.obs[pert_embeddings.obs["perturbations"].isin(["control", "target2"])]["classifier_score"].values == 1.0)
