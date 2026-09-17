import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData

import pertpy as pt


@pytest.fixture
def adata(rng):
    labels = np.array(["control", "target1", "target2"]).repeat(30)
    centers = {"control": 0.0, "target1": 5.0, "target2": -5.0}
    X = np.vstack([rng.normal(centers[label], 0.4, size=10) for label in labels]).astype(np.float32)
    obs = pd.DataFrame(
        {
            "perturbation": labels,
            "batch": "b0",  # constant within each perturbation -> carried over
            "MoA": ["Growth" if label == "target1" else "Unknown" for label in labels],  # constant within group
            "partial": [np.nan if label == "control" else "annotated" for label in labels],  # varies -> dropped
        }
    )
    adata = AnnData(X, obs=obs)
    sc.pp.pca(adata, n_comps=5)
    return adata


def test_lr_classifier_space_is_perturbation_level(adata):
    ps = pt.tl.LRClassifierSpace()
    pert_adata = ps.compute(adata, embedding_key="X_pca", target_col="perturbation")

    assert pert_adata.n_obs == 3
    assert set(pert_adata.obs_names) == {"control", "target1", "target2"}
    assert "classifier_score" in pert_adata.obs
    # constant-within-group obs are carried over, ambiguous ones dropped
    assert pert_adata.obs.loc["target1", "MoA"] == "Growth"
    assert "partial" not in pert_adata.obs


def test_lr_classifier_space_reproducible_and_accepts_generator(adata):
    ps = pt.tl.LRClassifierSpace()
    first = ps.compute(adata, embedding_key="X_pca", random_state=0)
    second = ps.compute(adata, embedding_key="X_pca", random_state=0)
    np.testing.assert_allclose(first.X, second.X)

    from_generator = ps.compute(adata, embedding_key="X_pca", random_state=np.random.default_rng(0))
    assert from_generator.n_obs == 3


def test_mlp_classifier_space_is_perturbation_level(adata):
    ps = pt.tl.MLPClassifierSpace()
    pert_adata = ps.compute(adata, target_col="perturbation", max_epochs=3, hidden_dim=[16])

    assert pert_adata.n_obs == 3
    assert set(pert_adata.obs_names) == {"control", "target1", "target2"}
    assert pert_adata.obs.loc["target1", "MoA"] == "Growth"


def test_mlp_classifier_space_sparse(adata):
    adata.X = sp.csr_matrix(adata.X)
    ps = pt.tl.MLPClassifierSpace()
    pert_adata = ps.compute(adata, target_col="perturbation", max_epochs=3, hidden_dim=[16])
    assert pert_adata.n_obs == 3
