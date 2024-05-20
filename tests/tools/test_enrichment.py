import numpy as np
import pertpy as pt
import pytest
import scanpy as sc
from anndata import AnnData


@pytest.fixture
def dummy_adata():
    n_obs = 10
    n_vars = 5
    rng = np.random.default_rng()
    X = rng.random((n_obs, n_vars))
    adata = AnnData(X)
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    adata.obs["cluster"] = ["group_1"] * 5 + ["group_2"] * 5
    sc.tl.rank_genes_groups(adata, groupby="cluster", method="t-test")

    return adata


@pytest.fixture(scope="module")
def enricher():
    return pt.tl.Enrichment()


def test_score_basic(dummy_adata, enricher):
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enricher.score(adata=dummy_adata, targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_different_layers(dummy_adata, enricher):
    rng = np.random.default_rng()
    dummy_adata.layers["layer"] = rng.random((10, 5))
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enricher.score(adata=dummy_adata, layer="layer", targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_nested_targets(dummy_adata, enricher):
    targets = {"category1": {"group1": ["gene1", "gene2"]}, "category2": {"group2": ["gene3", "gene4"]}}
    enricher.score(adata=dummy_adata, targets=targets, nested=True)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_hypergeometric_basic(dummy_adata, enricher):
    targets = {"group1": ["gene1", "gene2"]}
    results = enricher.hypergeometric(dummy_adata, targets)
    assert isinstance(results, dict)


def test_hypergeometric_with_nested_targets(dummy_adata, enricher):
    targets = {"category1": {"group1": ["gene1", "gene2"]}}
    results = enricher.hypergeometric(dummy_adata, targets, nested=True)
    assert isinstance(results, dict)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_hypergeometric_with_different_directions(dummy_adata, enricher, direction):
    targets = {"group1": ["gene1", "gene2"]}
    results = enricher.hypergeometric(dummy_adata, targets, direction=direction)
    assert isinstance(results, dict)
