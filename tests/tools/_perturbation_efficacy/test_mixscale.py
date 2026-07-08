"""Tests for Mixscale continuous perturbation scoring.

The `R_GOLDEN` values were produced by `RunMixscale` from https://github.com/satijalab/Mixscale on the dataset built by the `parity_adata` fixture.
The same DE genes are supplied via its `DE.gene` argument so the comparison isolates the scoring algorithm from the DE test, with matching `scale` (`slot`) and `split.by` settings.
The pertpy and R scores agree to within floating-point round-off (~1e-13); the values below are rounded.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pertpy as pt

NUM_CELLS_PER_GROUP = 10

DE_GENES_BY_TARGET = {
    "GeneA": [f"Gene{i}" for i in range(8)],
    "GeneB": [f"Gene{i}" for i in range(8, 13)],
}

# Per-cell scores from R's RunMixscale; cells are Cell0..Cell23 in order (the first 8 are NT controls).
R_GOLDEN = {
    ("nosplit", False): [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.514691542,
        7.547581559,
        8.308927689,
        7.027790116,
        8.856137701,
        10.27242998,
        7.472028438,
        10.44919624,
        8.222717541,
        6.790026264,
        3.645112649,
        3.338441385,
        4.270389556,
        4.392104461,
        3.291505505,
        2.442656334,
    ],
    ("replicate", False): [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.269530405,
        8.259218669,
        8.242553632,
        7.988511579,
        8.608806982,
        11.32851433,
        7.475598635,
        11.64795266,
        8.041098786,
        7.6848051,
        4.474797496,
        7.19368398,
        4.838105457,
        7.512732638,
        4.26093846,
        4.971528379,
    ],
    ("nosplit", True): [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8.571392513,
        7.646342068,
        8.572557144,
        7.028868448,
        8.726570745,
        9.937827644,
        7.709038712,
        10.97100301,
        7.509618135,
        6.579441777,
        3.681654505,
        3.205923984,
        4.255066398,
        4.093707024,
        3.681669612,
        2.618569012,
    ],
}


@pytest.fixture(params=[np.asarray, sparse.csr_array], ids=["dense", "csr_array"])
def parity_adata(request):
    """The exact deterministic dataset the R golden scores were computed on, over each supported array type."""
    rng = np.random.default_rng(0)
    n_genes = 16
    labels = ["NT"] * 8 + ["GeneA"] * 10 + ["GeneB"] * 6
    n_cells = len(labels)
    x_pert = rng.normal(0, 1, (n_cells, n_genes))
    x_pert[8:18, :8] += 2.5
    x_pert[18:24, 8:13] -= 1.7

    adata = ad.AnnData(
        X=np.zeros((n_cells, n_genes)),
        obs=pd.DataFrame(
            {"gene_target": labels, "replicate": ["rep1" if i % 2 == 0 else "rep2" for i in range(n_cells)]},
            index=[f"Cell{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"Gene{i}" for i in range(n_genes)]),
    )
    adata.layers["X_pert"] = request.param(x_pert)
    return adata


@pytest.fixture
def scored_adata(adata):
    """The shared mixscape fixture scored with Mixscale (dense signature, t-test DE)."""
    adata.layers["X_pert"] = adata.X.toarray()
    pt.tl.Mixscale().mixscale(adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=3)
    return adata


def test_control_cells_score_zero(scored_adata):
    nt = scored_adata.obs["gene_target"] == "NT"
    assert (scored_adata.obs.loc[nt, "mixscale_score"] == 0).all()


def test_perturbed_cells_score_nonzero(scored_adata):
    perturbed = scored_adata.obs["gene_target"] != "NT"
    assert scored_adata.obs.loc[perturbed, "mixscale_score"].abs().sum() > 0


def test_knockout_scores_exceed_non_perturbed(scored_adata):
    scores = scored_adata.obs["mixscale_score"].to_numpy()
    non_perturbed = np.abs(scores[NUM_CELLS_PER_GROUP : NUM_CELLS_PER_GROUP * 2]).mean()
    knockout = np.abs(scores[NUM_CELLS_PER_GROUP * 2 :]).mean()
    assert knockout > non_perturbed


def test_score_dtype_and_stored_results(scored_adata):
    assert scored_adata.obs["mixscale_score"].dtype == float
    assert "mixscale" in scored_adata.uns
    assert "mixscale_de_genes" in scored_adata.uns


def test_requires_perturbation_signature(adata):
    with pytest.raises(KeyError, match="X_pert"):
        pt.tl.Mixscale().mixscale(adata, pert_key="gene_target", control="NT")


def test_custom_score_column(adata):
    adata.layers["X_pert"] = adata.X.toarray()
    pt.tl.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=3, new_class_name="efficacy"
    )
    assert "efficacy" in adata.obs
    assert "mixscale_score" not in adata.obs


def test_copy_does_not_mutate_input(adata):
    adata.layers["X_pert"] = adata.X.toarray()
    result = pt.tl.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=3, copy=True
    )
    assert result is not adata
    assert "mixscale_score" in result.obs
    assert "mixscale_score" not in adata.obs


def test_de_genes_by_target_override(adata):
    adata.layers["X_pert"] = adata.X.toarray()
    de_genes = {"target_gene_a": [f"gene{i}" for i in range(11, 19)]}
    pt.tl.Mixscale().mixscale(adata, pert_key="gene_target", control="NT", de_genes_by_target=de_genes, min_de_genes=1)
    assert list(adata.uns["mixscale_de_genes"]["target_gene_a"]) == de_genes["target_gene_a"]


def test_too_few_de_genes_fall_back_to_one(adata):
    adata.layers["X_pert"] = adata.X.toarray()
    pt.tl.Mixscale().mixscale(adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=1000)
    perturbed = adata.obs["gene_target"] != "NT"
    assert (adata.obs.loc[perturbed, "mixscale_score"] == 1.0).all()
    assert (adata.obs.loc[~perturbed, "mixscale_score"] == 0.0).all()


@pytest.mark.parametrize("array_type", [np.asarray, sparse.csr_array], ids=["dense", "csr_array"])
def test_signature_and_score_end_to_end(adata, array_type):
    adata.X = array_type(adata.X.toarray())
    ms = pt.tl.Mixscale()
    ms.perturbation_signature(adata, pert_key="gene_target", control="NT")
    ms.mixscale(adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=3)

    scores = adata.obs["mixscale_score"]
    assert not scores.isna().any()
    assert (scores[adata.obs["gene_target"] == "NT"] == 0).all()
    assert scores[adata.obs["gene_target"] != "NT"].abs().sum() > 0


@pytest.mark.parametrize(("split_by", "scale"), list(R_GOLDEN))
def test_matches_r_reference(parity_adata, split_by, scale):
    pt.tl.Mixscale().mixscale(
        parity_adata,
        pert_key="gene_target",
        control="NT",
        layer="X_pert",
        de_genes_by_target=DE_GENES_BY_TARGET,
        min_de_genes=1,
        max_de_genes=1000,
        split_by=None if split_by == "nosplit" else split_by,
        scale=scale,
    )
    np.testing.assert_allclose(
        parity_adata.obs["mixscale_score"].to_numpy(),
        np.array(R_GOLDEN[(split_by, scale)]),
        atol=1e-6,
    )
