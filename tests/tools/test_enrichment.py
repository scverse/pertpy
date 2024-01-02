import numpy as np
import pytest
from anndata import AnnData
from pertpy.tools import Enrichment


@pytest.fixture
def dummy_adata():
    """Fixture to create a dummy AnnData object."""
    n_obs = 10
    n_vars = 5
    rng = np.random.default_rng()
    X = rng.random((n_obs, n_vars))
    adata = AnnData(X)
    adata.var_names = [f"gene{i}" for i in range(n_vars)]
    return adata


@pytest.fixture
def enrichment_instance():
    """Fixture to create an instance of the Enrichment class."""
    return Enrichment()


def test_score_basic(dummy_adata, enrichment_instance):
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enrichment_instance.score(adata=dummy_adata, targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_different_layers(dummy_adata, enrichment_instance):
    rng = np.random.default_rng()
    dummy_adata.layers["layer"] = rng((10, 5))
    targets = {"group1": ["gene1", "gene2"], "group2": ["gene3", "gene4"]}
    enrichment_instance.score(adata=dummy_adata, layer="layer", targets=targets)
    assert "pertpy_enrichment_score" in dummy_adata.uns


def test_score_with_nested_targets(dummy_adata, enrichment_instance):
    targets = {"category1": {"group1": ["gene1", "gene2"]}, "category2": {"group2": ["gene3", "gene4"]}}
    enrichment_instance.score(adata=dummy_adata, targets=targets, nested=True)
    assert "pertpy_enrichment_score" in dummy_adata.uns
