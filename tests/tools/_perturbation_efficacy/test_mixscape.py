import anndata as ad
import numpy as np
import pandas as pd

import pertpy as pt
from pertpy.tools._perturbation_efficacy._mixscape import MixscapeGaussianMixture

NUM_CELLS_PER_GROUP = 10
ACCURACY_THRESHOLD = 0.8


def test_mixscape(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")

    np_result = adata.obs["mixscape_class_global"] == "NP"
    np_result_correct = np_result[NUM_CELLS_PER_GROUP : NUM_CELLS_PER_GROUP * 2]
    ko_result = adata.obs["mixscape_class_global"] == "KO"
    ko_result_correct = ko_result[NUM_CELLS_PER_GROUP * 2 : NUM_CELLS_PER_GROUP * 3]

    assert "mixscape_class" in adata.obs
    assert "mixscape_class_global" in adata.obs
    assert "mixscape_class_p_ko" in adata.obs
    assert sum(np_result_correct) > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert sum(ko_result_correct) > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP


def test_perturbation_signature(adata):
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(adata, pert_key="label", control="control")

    assert "X_pert" in adata.layers


def test_lda(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")
    mixscape_identifier.lda(adata=adata, pert_key="gene_target", control="NT", test_method="t-test")

    assert "mixscape_lda" in adata.uns


def test_deterministic_perturbation_signature():
    n_genes = 5
    n_cells_per_class = 50
    cell_classes = ["NT", "KO", "NP"]
    groups = ["Group1", "Group2"]

    cell_classes_array = np.repeat(cell_classes, n_cells_per_class)
    groups_array = np.tile(np.repeat(groups, n_cells_per_class // 2), len(cell_classes))
    obs = pd.DataFrame(
        {
            "cell_class": cell_classes_array,
            "group": groups_array,
            "perturbation": ["control" if cell_class == "NT" else "pert1" for cell_class in cell_classes_array],
        }
    )

    data = np.zeros((len(obs), n_genes))
    pert_effect = np.random.default_rng().uniform(-1, 1, size=(n_cells_per_class // len(groups), n_genes))
    for group in groups:
        baseline_expr = 2 if group == "Group1" else 10
        group_mask = obs["group"] == group
        data[(obs["cell_class"] == "NT") & group_mask] = baseline_expr
        data[(obs["cell_class"] == "KO") & group_mask] = baseline_expr + pert_effect
        data[(obs["cell_class"] == "NP") & group_mask] = baseline_expr

    var = pd.DataFrame(index=[f"Gene{i + 1}" for i in range(n_genes)])
    adata = ad.AnnData(X=data, obs=obs, var=var)

    for ref_selection_mode in ("nn", "split_by"):
        adata.layers.pop("X_pert", None)
        pt.tl.Mixscape().perturbation_signature(
            adata,
            pert_key="perturbation",
            control="control",
            ref_selection_mode=ref_selection_mode,
            n_neighbors=5,
            split_by="group",
        )
        assert "X_pert" in adata.layers
        assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NT"], 0)
        assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NP"], 0)
        assert np.allclose(
            adata.layers["X_pert"][obs["cell_class"] == "KO"], -np.concatenate([pert_effect] * len(groups), axis=0)
        )


def test_mixscape_gaussian_mixture():
    X = np.random.default_rng().random(100)
    fixed_means = [0.2, None]
    fixed_covariances = [None, 0.1]

    model = MixscapeGaussianMixture(n_components=2, fixed_means=fixed_means, fixed_covariances=fixed_covariances)
    model.fit(X.reshape(-1, 1))

    assert np.allclose(model.means_[0], fixed_means[0])
    assert np.allclose(model.covariances_[1], fixed_covariances[1])
