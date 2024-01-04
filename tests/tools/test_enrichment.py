import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from pertpy.tools import Enrichment


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


@pytest.fixture
def enrichment_instance():
    return Enrichment()


def test_score_basic(dummy_adata, enrichment_instance):
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enrichment_instance.score(adata=dummy_adata, targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_different_layers(dummy_adata, enrichment_instance):
    rng = np.random.default_rng()
    dummy_adata.layers["layer"] = rng.random((10, 5))
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enrichment_instance.score(adata=dummy_adata, layer="layer", targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_nested_targets(dummy_adata, enrichment_instance):
    targets = {"category1": {"group1": ["gene1", "gene2"]}, "category2": {"group2": ["gene3", "gene4"]}}
    enrichment_instance.score(adata=dummy_adata, targets=targets, nested=True)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_hypergeometric_basic(dummy_adata, enrichment_instance):
    targets = {"group1": ["gene1", "gene2"]}
    results = enrichment_instance.hypergeometric(dummy_adata, targets)
    assert isinstance(results, dict)


def test_hypergeometric_with_nested_targets(dummy_adata, enrichment_instance):
    targets = {"category1": {"group1": ["gene1", "gene2"]}}
    results = enrichment_instance.hypergeometric(dummy_adata, targets, nested=True)
    assert isinstance(results, dict)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_hypergeometric_with_different_directions(dummy_adata, enrichment_instance, direction):
    targets = {"group1": ["gene1", "gene2"]}
    results = enrichment_instance.hypergeometric(dummy_adata, targets, direction=direction)
    assert isinstance(results, dict)
