from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from scipy import sparse

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
    label = {"label": ["control", "treatment", "treatment"] * num_cells_per_group}
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
    mixscape_identifier = pt.tl.Mixscape()
    adata.layers["X_pert"] = adata.X
    mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target")
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
    mixscape_identifier.mixscape(adata=adata, control="NT", labels="gene_target")
    mixscape_identifier.lda(adata=adata, labels="gene_target", control="NT")

    assert "mixscape_lda" in adata.uns
