from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

if find_spec("formulaic_contrasts") is None or find_spec("formulaic") is None:
    pytestmark = pytest.mark.skip(reason="formulaic_contrasts and formulaic not available")


def _donor_level_adata(n_donors: int = 6, n_cells: int = 30, n_genes: int = 5, *, seed: int = 0) -> ad.AnnData:
    """Log-scale expression with a random donor effect; the condition is assigned per donor (between-donor)."""
    rng = np.random.default_rng(seed)
    donor = np.repeat([f"D{i}" for i in range(n_donors)], n_cells)
    condition = np.where(np.repeat(np.arange(n_donors), n_cells) < n_donors // 2, "A", "B")
    donor_effect = rng.normal(0, 0.7, (n_donors, n_genes))
    X = rng.normal(size=(n_donors * n_cells, n_genes)) + np.repeat(donor_effect, n_cells, axis=0)
    obs = pd.DataFrame({"donor": donor, "condition": condition}, index=[f"c{i}" for i in range(len(donor))])
    return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)]))


def test_linear_mixed_model(test_adata_minimal):
    """Check that the method can be initialized and fitted, and perform basic checks on the result of test_contrasts."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    method = LinearMixedModel(adata=test_adata_minimal, design="~condition + (1 | donor)")
    assert not method.is_fitted
    method.fit()
    assert method.is_fitted
    res_df = method.test_contrasts(method.contrast(column="condition", baseline="A", group_to_compare="B"))
    assert len(res_df) == test_adata_minimal.n_vars
    assert {"variable", "p_value", "t_value", "sd", "log_fc", "adj_p_value", "converged"} <= set(res_df.columns)
    # gene1 has a large difference between conditions
    assert res_df.set_index("variable").loc["gene1", "p_value"] < 0.05


def test_matches_statsmodels():
    """Effect sizes come from statsmodels' MixedLM; p-values use the between-group degrees of freedom."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata(n_genes=3)
    method = LinearMixedModel(adata, design="~condition + (1 | donor)")
    method.fit()
    contrast = method.contrast(column="condition", baseline="A", group_to_compare="B")
    res = method.test_contrasts(contrast).set_index("variable")

    exog = np.column_stack([np.ones(adata.n_obs), (adata.obs["condition"] == "B").to_numpy(float)])
    for i, gene in enumerate(adata.var_names):
        expected = sm.MixedLM(adata.X[:, i], exog, groups=adata.obs["donor"].to_numpy()).fit(reml=True)
        np.testing.assert_allclose(res.loc[gene, "log_fc"], expected.fe_params[1], rtol=1e-6)
        np.testing.assert_allclose(res.loc[gene, "sd"], expected.bse_fe[1], rtol=1e-6)
        t_value = expected.fe_params[1] / expected.bse_fe[1]
        # 6 donors minus intercept and condition, both constant within donors
        np.testing.assert_allclose(res.loc[gene, "p_value"], 2 * stats.t.sf(abs(t_value), df=4), rtol=1e-6)


def test_between_within_df():
    from pertpy.tools._differential_gene_expression._linear_mixed_model import _between_within_df

    groups = np.repeat(["D0", "D1", "D2", "D3"], 5)
    between = np.repeat([0.0, 0.0, 1.0, 1.0], 5)  # constant within each donor
    within = np.tile([0.0, 1.0, 0.0, 1.0, 1.0], 4)  # varies within donors
    design = np.column_stack([np.ones(20), between, within])

    is_between, df_between, df_within = _between_within_df(design, groups)

    np.testing.assert_array_equal(is_between, [True, True, False])
    assert df_between == 4 - 2  # donors minus intercept and between-donor effect
    assert df_within == 20 - 4 - 1  # cells minus donor indicators and within-donor effect


def test_type_i_error_between_donor_effect():
    """With few donors, the default normal approximation would be anti-conservative; the t-test must not be."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata(n_genes=200, seed=1)
    method = LinearMixedModel(adata, design="~condition + (1 | donor)")
    method.fit()
    res = method.test_contrasts(method.contrast(column="condition", baseline="A", group_to_compare="B"))
    assert (res["p_value"] < 0.05).mean() < 0.1


def test_optimizer_fallback_converges(recwarn):
    """Fits that fail statsmodels' gradient check are refit with derivative-free optimizers, so clean data converges."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata(n_genes=200, seed=1)
    method = LinearMixedModel(adata, design="~condition + (1 | donor)")
    method.fit()
    res = method.test_contrasts(method.contrast(column="condition", baseline="A", group_to_compare="B"))
    assert res["converged"].all()
    assert not [w for w in recwarn if "did not converge" in str(w.message)]


def test_compare_groups():
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata()
    res = LinearMixedModel.compare_groups(
        adata, column="condition", baseline="A", groups_to_compare="B", paired_by="donor"
    )
    assert len(res) == adata.n_vars
    assert set(res["contrast"]) == {"B"}

    with pytest.raises(ValueError, match="requires `paired_by`"):
        LinearMixedModel.compare_groups(adata, column="condition", baseline="A", groups_to_compare="B")


@pytest.mark.parametrize(
    ("design", "error", "match"),
    [
        ("~condition", ValueError, "exactly one random intercept"),
        ("~condition + (1 | donor) + (1 | batch)", ValueError, "exactly one random intercept"),
        ("~condition + (1 | missing)", ValueError, "not a column of adata.obs"),
        (np.ones((180, 1)), TypeError, "requires a formula"),
    ],
)
def test_invalid_design(design, error, match):
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata()
    adata.obs["batch"] = "b1"
    with pytest.raises(error, match=match):
        LinearMixedModel(adata, design=design)


def test_missing_random_effect_values():
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata()
    adata.obs["donor"] = adata.obs["donor"].astype(object)
    adata.obs.iloc[0, adata.obs.columns.get_loc("donor")] = None
    with pytest.raises(ValueError, match="missing values"):
        LinearMixedModel(adata, design="~condition + (1 | donor)")


def test_untestable_contrast():
    """One donor per condition leaves no degrees of freedom for a between-donor effect."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata(n_donors=2)
    method = LinearMixedModel(adata, design="~condition + (1 | donor)")
    method.fit()
    with pytest.raises(ValueError, match="cannot be tested"):
        method.test_contrasts(method.contrast(column="condition", baseline="A", group_to_compare="B"))


def test_constant_variable():
    """Variables that cannot be fit get NaN statistics and are excluded from the FDR correction."""
    from pertpy.tools._differential_gene_expression import LinearMixedModel

    adata = _donor_level_adata(n_genes=4)
    adata.X[:, 0] = 0
    method = LinearMixedModel(adata, design="~condition + (1 | donor)")
    with pytest.warns(UserWarning, match="did not converge"):
        method.fit()
    res = method.test_contrasts(method.contrast(column="condition", baseline="A", group_to_compare="B"))
    res = res.set_index("variable")

    assert np.isnan(res.loc["gene0", "p_value"])
    assert np.isnan(res.loc["gene0", "adj_p_value"])
    assert not res.loc["gene0", "converged"]
    assert res.drop(index="gene0")["adj_p_value"].notna().all()
