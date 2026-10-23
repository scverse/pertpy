from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

if find_spec("formulaic_contrasts") is None or find_spec("formulaic") is None:
    pytestmark = pytest.mark.skip(reason="formulaic_contrasts and formulaic not available")


def _cells_of_donors(n_genes: int, *, seed: int = 0) -> ad.AnnData:
    """Log-scale expression of cells from eight donors, with the condition assigned per donor and a cell-level covariate."""
    rng = np.random.default_rng(seed)
    donors = np.repeat(np.arange(8), [12, 20, 15, 30, 9, 25, 18, 22])
    covariate = rng.normal(size=donors.size)
    donor_effect = rng.normal(scale=0.8, size=(8, n_genes))
    donor_effect[:, 2::3] = 0
    X = rng.normal(size=(donors.size, n_genes)) + donor_effect[donors] + 0.3 * covariate[:, None]
    obs = pd.DataFrame(
        {
            "donor": [f"D{donor}" for donor in donors],
            "condition": np.where(donors % 2 == 0, "ctrl", "stim"),
            "covariate": covariate,
        },
        index=[f"cell{i}" for i in range(donors.size)],
    )
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))


@pytest.mark.parametrize(
    "kwargs", [{}, {"regression_model": sm.GLM, "family": sm.families.NegativeBinomial(alpha=1.0)}]
)
def test_statsmodels(test_adata, kwargs):
    """Check that the method can be initialized and fitted, and perform basic checks on the result of test_contrasts."""
    from pertpy.tools._differential_gene_expression import Statsmodels

    method = Statsmodels(adata=test_adata, design="~condition")
    assert not method.is_fitted
    method.fit(**kwargs)
    assert method.is_fitted
    res_df = method.test_contrasts(np.array([0, 1]))
    # Check that the result has the correct number of rows
    assert len(res_df) == test_adata.n_vars


# TODO: there should be a test checking if, for a concrete example, the output p-values and effect sizes are what
# we expect (-> frozen snapshot, that way we also get a heads-up if something changes upstream)


def test_random_intercept_matches_lmertest():
    """Reference values from lmerTest 3.2.1 (REML, Satterthwaite) on this exact data, including a fit on the boundary."""
    from pertpy.tools._differential_gene_expression import Statsmodels

    adata = _cells_of_donors(n_genes=3)
    method = Statsmodels(adata, design="~condition + covariate + (1 | donor)")
    method.fit()
    contrasts = {
        "stim": method.contrast(column="condition", baseline="ctrl", group_to_compare="stim"),
        "covariate": np.array([0.0, 0.0, 1.0]),
        "both": method.cond(condition="stim", covariate=1.0) - method.cond(condition="ctrl", covariate=0.0),
    }
    res = method.test_contrasts(contrasts).set_index(["contrast", "variable"]).sort_index()
    expected = pd.DataFrame(
        [
            ("stim", "gene0", 0.0479424000659, 0.40487594424, 0.909669910934),
            ("covariate", "gene0", 0.206583084507, 0.0930645872892, 0.0280035797359),
            ("both", "gene0", 0.254525484573, 0.418902673772, 0.563310062404),
            ("stim", "gene1", -0.536172106102, 0.696514357941, 0.471370865701),
            ("covariate", "gene1", 0.39318982076, 0.0850237207331, 8.35745118657e-06),
            ("both", "gene1", -0.142982285342, 0.703391140188, 0.845551760376),
            ("stim", "gene2", 0.0209764095948, 0.168433420581, 0.90105807902),
            ("covariate", "gene2", 0.286895356907, 0.0844553833793, 0.000875172448353),
            ("both", "gene2", 0.307871766502, 0.194457306979, 0.115502310835),
        ],
        columns=["contrast", "variable", "log_fc", "sd", "p_value"],
    ).set_index(["contrast", "variable"])
    np.testing.assert_allclose(res[expected.columns].to_numpy(), expected.sort_index().to_numpy(), rtol=1e-5)


def test_random_intercept_is_calibrated():
    """A condition assigned per donor is judged by the eight donors, not the cells, so null genes are not called."""
    from pertpy.tools._differential_gene_expression import Statsmodels

    adata = _cells_of_donors(n_genes=300, seed=1)
    method = Statsmodels(adata, design="~condition + (1 | donor)")
    method.fit()
    res = method.test_contrasts(method.contrast(column="condition", baseline="ctrl", group_to_compare="stim"))
    assert (res["p_value"] < 0.05).mean() < 0.1


def test_random_intercept_constant_variable():
    """Variables whose mixed model is degenerate get NaN statistics and are excluded from the FDR correction."""
    from pertpy.tools._differential_gene_expression import Statsmodels

    adata = _cells_of_donors(n_genes=4)
    adata.X[:, 0] = 1.0
    method = Statsmodels(adata, design="~condition + (1 | donor)")
    method.fit()
    res = method.test_contrasts(np.array([0.0, 1.0])).set_index("variable")
    assert res.loc["gene0", ["log_fc", "sd", "p_value", "adj_p_value"]].isna().all()
    assert res.drop(index="gene0")["adj_p_value"].notna().all()


@pytest.mark.parametrize(
    ("design", "match"),
    [
        ("~condition + (1 | donor) + (1 | batch)", "single random intercept"),
        ("~condition + (1 | missing)", "not a column"),
        ("~condition + (1 | batch)", "single level"),
        ("~condition + (1 | incomplete)", "missing values"),
    ],
)
def test_random_intercept_rejects_invalid_designs(design, match):
    from pertpy.tools._differential_gene_expression import Statsmodels

    adata = _cells_of_donors(n_genes=2)
    adata.obs["batch"] = "b1"
    adata.obs["incomplete"] = adata.obs["donor"].where(adata.obs.index != "cell0")
    with pytest.raises(ValueError, match=match):
        Statsmodels(adata, design=design)


@pytest.mark.skipif(find_spec("pydeseq2") is None, reason="pydeseq2 not available")
def test_random_intercept_rejected_by_other_methods():
    from pertpy.tools._differential_gene_expression import PyDESeq2

    adata = _cells_of_donors(n_genes=2)
    adata.X = np.ones_like(adata.X, dtype=int)
    with pytest.raises(ValueError, match="does not support random effects"):
        PyDESeq2(adata, design="~condition + (1 | donor)")
