import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse

import pertpy as pt
from pertpy._types import cast_frame


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
    results = enricher.hypergeometric(dummy_adata, targets=targets)
    assert isinstance(results, dict)


def test_hypergeometric_with_nested_targets(dummy_adata, enricher):
    targets = {"category1": {"group1": ["gene1", "gene2"]}}
    results = enricher.hypergeometric(dummy_adata, targets=targets, nested=True)
    assert isinstance(results, dict)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_hypergeometric_with_different_directions(dummy_adata, enricher, direction):
    targets = {"group1": ["gene1", "gene2"]}
    results = enricher.hypergeometric(dummy_adata, targets=targets, direction=direction)
    assert isinstance(results, dict)


def test_signature_reversal_cmap_writes_scores_to_adata(enricher):
    labels = ["reverse", "mimic", "control"]
    adata = AnnData(
        X=np.array(
            [
                [-3.0, -2.0, 3.0, 2.0, 1.0, -1.0],
                [3.0, 2.0, -3.0, -2.0, 1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        obs=pd.DataFrame({"perturbation": labels}, index=labels),
    )
    adata.var_names = ["up_1", "up_2", "down_1", "down_2", "other_1", "other_2"]

    enricher.signature_reversal(
        adata,
        up_genes=["up_1", "up_2"],
        down_genes=["down_1", "down_2"],
    )

    assert adata.obs["signature_reversal_rank"].idxmin() == "reverse"
    np.testing.assert_allclose(adata.obs.loc["reverse", "signature_reversal_score"], 1.0)
    np.testing.assert_allclose(adata.obs.loc["mimic", "signature_reversal_score"], -1.0)
    np.testing.assert_allclose(adata.obs.loc["control", "signature_reversal_connectivity"], 0.0)
    assert adata.uns["signature_reversal"]["method"] == "cmap_wtcs"
    assert adata.uns["signature_reversal"]["n_matched_up_genes"] == 2
    assert adata.uns["signature_reversal"]["n_matched_down_genes"] == 2


def test_signature_reversal_matches_hand_calculated_weighted_score(enricher):
    adata = AnnData(X=np.array([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]), obs=pd.DataFrame(index=["perturbation"]))
    adata.var_names = [f"g{i}" for i in range(1, 7)]

    enricher.signature_reversal(adata, up_genes=["g2", "g5"], down_genes=["g4", "g6"])

    # ES_up = 13/28 and ES_down = -3/4, so WTCS = (ES_up - ES_down) / 2 = 17/28.
    np.testing.assert_allclose(adata.obs.loc["perturbation", "signature_reversal_connectivity"], 17 / 28)
    np.testing.assert_allclose(adata.obs.loc["perturbation", "signature_reversal_score"], -17 / 28)


def test_signature_reversal_accepts_signed_query(enricher):
    adata = AnnData(
        X=np.array([[-1.0, 1.0], [1.0, -1.0]]),
        obs=pd.DataFrame(index=["reverse", "mimic"]),
    )
    adata.var_names = ["up", "down"]

    enricher.signature_reversal(
        adata,
        pd.Series({"up": 100.0, "down": -0.01}),
    )

    assert adata.obs["signature_reversal_rank"].idxmin() == "reverse"
    np.testing.assert_allclose(adata.obs.loc["reverse", "signature_reversal_score"], 1.0)
    np.testing.assert_allclose(adata.obs.loc["mimic", "signature_reversal_score"], -1.0)


def test_signature_reversal_uses_layer_and_gene_symbols(enricher):
    adata = AnnData(
        X=np.zeros((2, 2)),
        obs=pd.DataFrame(index=["reverse", "mimic"]),
        var=pd.DataFrame({"symbol": ["up", "down"]}, index=["ens1", "ens2"]),
    )
    adata.layers["signatures"] = np.array([[-1.0, 1.0], [1.0, -1.0]])

    enricher.signature_reversal(
        adata,
        up_genes=["up"],
        down_genes=["down"],
        layer="signatures",
        gene_symbols_key="symbol",
        key_added="sr",
    )

    assert adata.obs["sr_rank"].idxmin() == "reverse"
    assert adata.uns["sr"]["layer"] == "signatures"
    assert adata.uns["sr"]["gene_symbols_key"] == "symbol"


@pytest.mark.parametrize(
    ("query", "values"),
    [
        ({"up_genes": "gene"}, [[-1.0, 1.0], [1.0, -1.0]]),
        ({"down_genes": ["gene"]}, [[1.0, -1.0], [-1.0, 1.0]]),
    ],
)
def test_signature_reversal_accepts_one_sided_query(enricher, query, values):
    adata = AnnData(
        X=np.array(values),
        obs=pd.DataFrame(index=["reverse", "mimic"]),
    )
    adata.var_names = ["gene", "other"]

    enricher.signature_reversal(adata, **query)

    assert adata.obs["signature_reversal_rank"].idxmin() == "reverse"


def test_signature_reversal_returns_zero_when_query_sets_move_together(enricher):
    adata = AnnData(
        X=np.array([[3.0, 2.0, 0.0]]),
        obs=pd.DataFrame(index=["same_direction"]),
    )
    adata.var_names = ["up", "down", "other"]

    enricher.signature_reversal(adata, up_genes=["up"], down_genes=["down"])

    np.testing.assert_allclose(adata.obs.loc["same_direction", "signature_reversal_connectivity"], 0.0)


def test_signature_reversal_rejects_query_without_non_hit_genes(enricher):
    adata = AnnData(
        X=np.array([[1.0]]),
        obs=pd.DataFrame(index=["all_hits"]),
    )
    adata.var_names = ["gene"]

    with pytest.raises(ValueError, match="one non-hit gene"):
        enricher.signature_reversal(adata, up_genes=["gene"])


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ({"query_signature": {"gene1": 1.0}, "up_genes": ["gene1"]}, "Pass either"),
        ({"up_genes": ["gene1"], "down_genes": ["gene1"]}, "occurs in both"),
        ({"query_signature": {"gene1": 0.0}}, "at least one non-zero"),
        ({"up_genes": ["missing"]}, "Only 0 query genes"),
        ({"up_genes": ["gene1"], "min_genes": 0}, "`min_genes` must be at least 1"),
        ({"up_genes": ["gene1"], "layer": "missing"}, "Layer 'missing' does not exist"),
        ({"up_genes": ["gene1"], "gene_symbols_key": "missing"}, "Column 'missing' does not exist"),
        ({"query_signature": {"gene1": np.inf}}, "must be finite"),
        ({"up_genes": ["gene1", "gene1"]}, "must not contain duplicate"),
        ({"down_genes": ["gene1", "gene1"]}, "must not contain duplicate"),
        (
            {"query_signature": pd.Series([1.0, -1.0], index=["gene1", "gene1"])},
            "must be unique",
        ),
        ({"query_signature": pd.Series([1.0], index=[None])}, "must not be missing"),
    ],
)
def test_signature_reversal_rejects_invalid_input(enricher, query, match):
    adata = AnnData(
        X=np.ones((1, 2)),
        obs=pd.DataFrame(index=["perturbation"]),
        var=pd.DataFrame({"symbol": ["gene1", "gene2"]}),
    )
    adata.var_names = ["gene1", "gene2"]

    with pytest.raises(ValueError, match=match):
        enricher.signature_reversal(adata, **query)


def test_signature_reversal_is_invariant_to_scale(enricher):
    profile = np.array([-3.0, 2.0, 1.0, 0.5])
    adata = AnnData(X=np.vstack([profile, profile * 1e-10]), obs=pd.DataFrame(index=["unit", "scaled"]))
    adata.var_names = ["gene", "other_1", "other_2", "other_3"]

    enricher.signature_reversal(adata, up_genes=["gene"])

    np.testing.assert_allclose(
        adata.obs.loc[["unit", "scaled"], "signature_reversal_connectivity"],
        [-1.0, -1.0],
    )


def test_signature_reversal_groups_tied_values(enricher):
    def score(genes: list[str]) -> float:
        values = {"query": 1.0, "tied": 1.0, "high": 2.0, "low": 0.0}
        adata = AnnData(X=np.array([[values[gene] for gene in genes]]), obs=pd.DataFrame(index=["perturbation"]))
        adata.var_names = genes
        enricher.signature_reversal(adata, up_genes=["query"])
        return float(cast_frame(adata.obs)["signature_reversal_connectivity"].to_numpy(dtype=float)[0])

    np.testing.assert_allclose(score(["query", "tied", "high", "low"]), 1 / 3)
    np.testing.assert_allclose(score(["tied", "query", "high", "low"]), 1 / 3)


def test_signature_reversal_returns_zero_for_constant_profile(enricher):
    adata = AnnData(X=np.ones((1, 4)), obs=pd.DataFrame(index=["constant"]))
    adata.var_names = ["gene", "other_1", "other_2", "other_3"]

    enricher.signature_reversal(adata, up_genes=["gene"])

    np.testing.assert_allclose(adata.obs.loc["constant", "signature_reversal_connectivity"], 0.0)


def test_signature_reversal_sparse_matches_dense(enricher):
    values = np.array([[-3.0, 2.0, 1.0, 0.5], [3.0, -2.0, -1.0, -0.5]])
    scores = []
    for matrix in (values, sparse.csr_matrix(values)):
        adata = AnnData(X=matrix, obs=pd.DataFrame(index=["reverse", "mimic"]))
        adata.var_names = ["gene", "other_1", "other_2", "other_3"]
        enricher.signature_reversal(adata, up_genes=["gene"])
        scores.append(adata.obs["signature_reversal_score"].to_numpy())

    np.testing.assert_allclose(scores[0], scores[1])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_signature_reversal_rejects_non_finite_profiles(enricher, invalid_value):
    adata = AnnData(X=np.array([[invalid_value, 1.0]]), obs=pd.DataFrame(index=["perturbation"]))
    adata.var_names = ["gene", "other"]

    with pytest.raises(ValueError, match="contains non-finite values"):
        enricher.signature_reversal(adata, up_genes=["gene"])


@pytest.mark.parametrize("gene_symbols_key", [None, "symbol"])
def test_signature_reversal_rejects_duplicate_gene_identifiers(enricher, gene_symbols_key):
    adata = AnnData(
        X=np.ones((1, 3)),
        obs=pd.DataFrame(index=["perturbation"]),
        var=pd.DataFrame({"symbol": ["gene", "gene", "other"]}),
    )
    adata.var_names = ["gene", "gene", "other"] if gene_symbols_key is None else ["id1", "id2", "id3"]

    with pytest.raises(ValueError, match="must be unique"):
        enricher.signature_reversal(adata, up_genes=["gene"], gene_symbols_key=gene_symbols_key)


def test_signature_reversal_rejects_missing_gene_identifiers(enricher):
    adata = AnnData(
        X=np.ones((1, 3)),
        obs=pd.DataFrame(index=["perturbation"]),
        var=pd.DataFrame({"symbol": ["gene", None, "other"]}),
    )

    with pytest.raises(ValueError, match="must not be missing"):
        enricher.signature_reversal(adata, up_genes=["gene"], gene_symbols_key="symbol")


@pytest.mark.parametrize(
    ("up_genes", "down_genes", "match"),
    [
        (["missing"], ["down"], "None of the query-up genes"),
        (["up"], ["missing"], "None of the query-down genes"),
    ],
)
def test_signature_reversal_rejects_missing_query_arm(enricher, up_genes, down_genes, match):
    adata = AnnData(X=np.array([[-1.0, 1.0, 0.0]]), obs=pd.DataFrame(index=["perturbation"]))
    adata.var_names = ["up", "down", "other"]

    with pytest.raises(ValueError, match=match):
        enricher.signature_reversal(adata, up_genes=up_genes, down_genes=down_genes)
