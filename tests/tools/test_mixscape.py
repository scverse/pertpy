from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from scipy import sparse

from pertpy.tools._mixscape import MixscapeGaussianMixture

CWD = Path(__file__).parent.resolve()

# Random generate data settings
num_cells_per_group = 10
num_not_de = 10
num_de = 10
accuracy_threshold = 0.8


@pytest.fixture
def adata():
    rng = np.random.default_rng(seed=1)
    # generate not differentially expressed genes
    for i in range(num_not_de):
        NT = rng.normal(0, 1, num_cells_per_group)
        NT = np.where(NT < 0, 0, NT)
        NP = rng.normal(0, 1, num_cells_per_group)
        NP = np.where(NP < 0, 0, NP)
        KO = rng.normal(0, 1, num_cells_per_group)
        KO = np.where(KO < 0, 0, KO)
        gene_i = np.concatenate((NT, NP, KO))
        gene_i = np.expand_dims(gene_i, axis=1)
        if i == 0:
            X = gene_i
        else:
            X = np.concatenate((X, gene_i), axis=1)

    # generate differentially expressed genes
    for i in range(num_de):
        NT = rng.normal(i + 2, 0.5 + 0.05 * i, num_cells_per_group)
        NT = np.where(NT < 0, 0, NT)
        NP = rng.normal(i + 2, 0.5 + 0.05 * i, num_cells_per_group)
        NP = np.where(NP < 0, 0, NP)
        KO = rng.normal(i + 4, 0.5 + 0.1 * i, num_cells_per_group)
        KO = np.where(KO < 0, 0, KO)
        gene_i = np.concatenate((NT, NP, KO))
        gene_i = np.expand_dims(gene_i, axis=1)
        X = np.concatenate((X, gene_i), axis=1)

    # obs for random AnnData
    gene_target = {"gene_target": ["NT"] * num_cells_per_group + ["target_gene_a"] * num_cells_per_group * 2}
    gene_target = pd.DataFrame(gene_target)
    label = {"label": ["control"] * num_cells_per_group + ["treatment"] * num_cells_per_group* 2 }
    label = pd.DataFrame(label)
    obs = pd.concat([gene_target, label], axis=1)
    obs = obs.set_index(np.arange(num_cells_per_group * 3))
    obs.index.rename("index", inplace=True)

    # var for random AnnData
    var_data = {"name": ["gene" + str(i) for i in range(1, num_not_de + num_de + 1)]}
    var = pd.DataFrame(var_data)
    var = var.set_index("name", drop=False)
    var.index.rename("index", inplace=True)

    X = sparse.csr_matrix(X)
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return adata


def test_mixscape(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, labels="gene_target", control="NT", test_method="t-test")
    np_result = adata.obs["mixscape_class_global"] == "NP"
    np_result_correct = np_result[num_cells_per_group : num_cells_per_group * 2]

    ko_result = adata.obs["mixscape_class_global"] == "KO"
    ko_result_correct = ko_result[num_cells_per_group * 2 : num_cells_per_group * 3]

    assert "mixscape_class" in adata.obs
    assert "mixscape_class_global" in adata.obs
    assert "mixscape_class_p_ko" in adata.obs
    assert sum(np_result_correct) > accuracy_threshold * num_cells_per_group
    assert sum(ko_result_correct) > accuracy_threshold * num_cells_per_group


def test_perturbation_signature(adata):
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(adata, pert_key="label", control="control")

    assert "X_pert" in adata.layers


def test_lda(adata):
    adata.layers["X_pert"] = adata.X
    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.mixscape(adata=adata, labels="gene_target", control="NT", test_method="t-test")
    mixscape_identifier.lda(adata=adata, labels="gene_target", control="NT", test_method="t-test")

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
    for _, group in enumerate(groups):
        baseline_expr = 2 if group == "Group1" else 10
        group_mask = obs["group"] == group

        nt_mask = (obs["cell_class"] == "NT") & group_mask
        data[nt_mask] = baseline_expr

        ko_mask = (obs["cell_class"] == "KO") & group_mask
        data[ko_mask] = baseline_expr + pert_effect

        np_mask = (obs["cell_class"] == "NP") & group_mask
        data[np_mask] = baseline_expr

    var = pd.DataFrame(index=[f"Gene{i + 1}" for i in range(n_genes)])
    adata = anndata.AnnData(X=data, obs=obs, var=var)

    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(
        adata, pert_key="perturbation", control="control", n_neighbors=5, split_by="group"
    )

    assert "X_pert" in adata.layers
    assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NT"], 0)
    assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NP"], 0)
    assert np.allclose(
        adata.layers["X_pert"][obs["cell_class"] == "KO"], -np.concatenate([pert_effect] * len(groups), axis=0)
    )

    del adata.layers["X_pert"]

    mixscape_identifier = pt.tl.Mixscape()
    mixscape_identifier.perturbation_signature(
        adata, pert_key="perturbation", control="control", ref_selection_mode="split_by", split_by="group"
    )

    assert "X_pert" in adata.layers
    assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NT"], 0)
    assert np.allclose(adata.layers["X_pert"][obs["cell_class"] == "NP"], 0)
    assert np.allclose(
        adata.layers["X_pert"][obs["cell_class"] == "KO"], -np.concatenate([pert_effect] * len(groups), axis=0)
    )

    
def test_mixscape_gaussian_mixture():
    X = np.random.rand(100)

    fixed_means = [0.2, None]
    fixed_covariances = [None, 0.1]

    model = MixscapeGaussianMixture(n_components=2, fixed_means=fixed_means, fixed_covariances=fixed_covariances)
    model.fit(X.reshape(-1, 1))

    assert np.allclose(model.means_[0], fixed_means[0])
    assert np.allclose(model.covariances_[1], fixed_covariances[1])
