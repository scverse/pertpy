"""Unit tests for the DIALOGUE building blocks.

End-to-end / R-reference tests are added in follow-up commits alongside the public
``Dialogue.fit_programs`` / ``test_celltype_pairs`` / ``refine_scores`` methods.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from pertpy.tools._dialogue import (
    _anova_filter_features,
    _center_scale_winsorize,
    _column_anova,
    _fisher_combine_by_label,
    _hlm_pvalue_per_row,
    _iterative_nnls,
    _partial_spearman,
    _pseudobulk_per_sample,
    _residualize,
    _zscores_from_signed_pvalues,
)


@pytest.fixture
def tiny_adata_dense():
    rng = np.random.default_rng(0)
    n_obs, n_vars = 12, 4
    X = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(["S1", "S1", "S1", "S2", "S2", "S2", "S3", "S3", "S3", "S4", "S4", "S4"]),
        },
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return ad.AnnData(X, obs=obs, var=var)


@pytest.fixture
def tiny_adata_sparse(tiny_adata_dense):
    adata = tiny_adata_dense.copy()
    adata.X = sparse.csr_matrix(adata.X)
    return adata


# ---------------------------------------------------------------------------
# _pseudobulk_per_sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", ["tiny_adata_dense", "tiny_adata_sparse"])
def test_pseudobulk_per_sample_matches_groupby_median(request, fixture):
    adata = request.getfixturevalue(fixture)
    pb = _pseudobulk_per_sample(adata, sample_key="sample", agg="median")
    X_dense = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    expected = (
        pd.DataFrame(X_dense, columns=adata.var_names).groupby(adata.obs["sample"].to_numpy()).median().sort_index()
    )
    pd.testing.assert_frame_equal(pb.loc[expected.index].astype(float), expected.astype(float), check_dtype=False)


def test_pseudobulk_per_sample_mean_matches_groupby_mean(tiny_adata_sparse):
    adata = tiny_adata_sparse
    pb = _pseudobulk_per_sample(adata, sample_key="sample", agg="mean")
    X = adata.X.toarray()
    expected = pd.DataFrame(X, columns=adata.var_names).groupby(adata.obs["sample"].to_numpy()).mean().sort_index()
    pd.testing.assert_frame_equal(pb.loc[expected.index].astype(float), expected.astype(float), check_dtype=False)


# ---------------------------------------------------------------------------
# _column_anova / _anova_filter_features
# ---------------------------------------------------------------------------


def test_column_anova_finds_signal():
    rng = np.random.default_rng(0)
    groups = np.array(["A"] * 50 + ["B"] * 50 + ["C"] * 50)
    informative = np.concatenate([rng.normal(0, 1, 50), rng.normal(3, 1, 50), rng.normal(-2, 1, 50)])[:, None]
    noise = rng.normal(0, 1, (150, 3))
    matrix = np.column_stack([informative, noise])
    p = _column_anova(matrix, groups)
    assert p[0] < 1e-10
    assert np.all(p[1:] > 0.05)


def test_column_anova_sparse_matches_dense():
    rng = np.random.default_rng(1)
    groups = np.array(["A"] * 30 + ["B"] * 30)
    matrix = rng.normal(0, 1, (60, 5))
    matrix[:30, 0] += 4
    p_dense = _column_anova(matrix, groups)
    p_sparse = _column_anova(sparse.csr_matrix(matrix), groups)
    np.testing.assert_allclose(p_dense, p_sparse, atol=1e-12)


def test_anova_filter_features_drops_uninformative():
    rng = np.random.default_rng(2)
    groups = np.array(["A"] * 25 + ["B"] * 25)
    informative = np.concatenate([rng.normal(0, 1, 25), rng.normal(5, 1, 25)])[:, None]
    noise = rng.normal(0, 1, (50, 9))
    matrix = np.column_stack([informative, noise])
    mask = _anova_filter_features(matrix, groups, alpha=0.05)
    assert mask[0]
    assert mask[1:].sum() < 3


# ---------------------------------------------------------------------------
# _center_scale_winsorize
# ---------------------------------------------------------------------------


def test_center_scale_winsorize_centered_and_capped():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (200, 4))
    x[0, 0] = 100.0
    out = _center_scale_winsorize(x, cap=0.05)
    np.testing.assert_allclose(out.mean(0), 0.0, atol=0.2)
    assert np.abs(out[0, 0]) < 5.0
    raw_max = ((x - x.mean(0)) / x.std(0, ddof=1)).max()
    assert out.max() < raw_max


def test_center_scale_winsorize_zero_var_column_is_zero():
    x = np.ones((10, 2))
    x[:, 1] = np.arange(10)
    out = _center_scale_winsorize(x, cap=0.0)
    np.testing.assert_allclose(out[:, 0], 0.0)


# ---------------------------------------------------------------------------
# _residualize
# ---------------------------------------------------------------------------


def test_residualize_known_regression():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 1, 100)
    epsilon = rng.normal(0, 0.1, 100)
    y = 2.0 + 3.0 * z + epsilon
    resid = _residualize(y, z)
    assert abs(resid.mean()) < 0.05
    assert np.corrcoef(resid.ravel(), z)[0, 1] < 0.1


def test_residualize_multi_target():
    rng = np.random.default_rng(5)
    z = rng.normal(0, 1, (200, 1))
    Y = np.column_stack([z[:, 0] + rng.normal(0, 0.1, 200), 2 * z[:, 0] + rng.normal(0, 0.1, 200)])
    resid = _residualize(Y, z)
    for col in range(resid.shape[1]):
        assert abs(np.corrcoef(resid[:, col], z[:, 0])[0, 1]) < 0.1


# ---------------------------------------------------------------------------
# _partial_spearman
# ---------------------------------------------------------------------------


def test_partial_spearman_removes_confound():
    rng = np.random.default_rng(6)
    n = 400
    z = rng.normal(0, 1, n)
    x = z + rng.normal(0, 0.1, n)
    y = z + rng.normal(0, 0.1, n)
    raw_R = np.corrcoef(x, y)[0, 1]
    R, _ = _partial_spearman(x[:, None], y[:, None], z[:, None])
    assert raw_R > 0.95
    assert abs(R[0, 0]) < 0.2


def test_partial_spearman_keeps_unconfounded_correlation():
    rng = np.random.default_rng(7)
    n = 400
    z = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    y = x + rng.normal(0, 0.1, n)
    R, P = _partial_spearman(x[:, None], y[:, None], z[:, None])
    assert R[0, 0] > 0.8
    assert P[0, 0] < 1e-30


# ---------------------------------------------------------------------------
# _zscores_from_signed_pvalues
# ---------------------------------------------------------------------------


def test_zscores_from_signed_pvalues_sign_follows_estimate():
    estimate = np.array([1.0, -1.0, 1.0, -1.0])
    pvalue = np.array([0.001, 0.001, 0.5, 0.5])
    z = _zscores_from_signed_pvalues(estimate, pvalue)
    # Significant p with positive estimate -> large positive z, and vice versa.
    assert z[0] > 1.0
    assert z[1] < -1.0
    # At p=0.5 the one-sided p is 0.25, so |z| sits a bit above zero but the sign still follows estimate.
    assert z[2] > 0
    assert z[3] < 0
    assert abs(z[2]) < 1.0
    assert abs(z[3]) < 1.0


def test_zscores_from_signed_pvalues_handles_zero_pvalue():
    """Zero p-values are floored to the smallest positive p / 2 (matches R's behaviour)."""
    estimate = np.array([1.0, -1.0, 1.0])
    pvalue = np.array([0.0, 0.0, 1e-50])
    z = _zscores_from_signed_pvalues(estimate, pvalue)
    assert np.isfinite(z).all()
    assert z[0] > 20
    assert z[1] < -20
    assert z[2] > 20


# ---------------------------------------------------------------------------
# _fisher_combine_by_label
# ---------------------------------------------------------------------------


def test_fisher_combine_strong_signal_combines_to_significant():
    pvalues = np.array(
        [
            [1e-5, 1e-5, 1e-5],
            [0.5, 0.5, 0.5],
        ]
    )
    labels = np.array(["MCP1", "MCP1"])
    combined = _fisher_combine_by_label(pvalues, labels)
    assert combined[0] < 0.05
    assert combined[1] > 0.05


def test_fisher_combine_per_label_independence():
    pvalues = np.array(
        [
            [1e-8, 1e-8],
            [0.6, 0.6],
            [1e-8, 1e-8],
            [0.6, 0.6],
        ]
    )
    labels = np.array(["A", "A", "B", "B"])
    combined = _fisher_combine_by_label(pvalues, labels)
    assert combined[0] < 0.05
    assert combined[2] < 0.05
    assert combined[1] > 0.05
    assert combined[3] > 0.05


# ---------------------------------------------------------------------------
# _iterative_nnls
# ---------------------------------------------------------------------------


def test_iterative_nnls_recovers_positive_weights():
    """Each rank bucket needs >= minimum_features (default 5) entries to be fit, matching R's gating."""
    rng = np.random.default_rng(8)
    n = 200
    n_high = 6
    n_low = 6
    n_feat = n_high + n_low
    X = rng.normal(0, 1, (n, n_feat))
    true_coef = np.concatenate([np.array([2.0, 1.5, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])])
    y = X @ true_coef + rng.normal(0, 0.05, n)
    rank = np.concatenate([np.full(n_high, 1.0), np.full(n_low, 0.5)])
    coef = _iterative_nnls(X, y, rank)
    assert (coef >= 0).all()
    assert coef[0] > 1.0
    assert coef[1] > 1.0
    fit = X @ coef
    assert np.corrcoef(fit, y)[0, 1] > 0.95


# ---------------------------------------------------------------------------
# _hlm_pvalue_per_row
# ---------------------------------------------------------------------------


def test_hlm_pvalue_per_row_detects_planted_signal():
    rng = np.random.default_rng(9)
    n_samples = 6
    n_cells_per_sample = 30
    n = n_samples * n_cells_per_sample
    sample = np.repeat([f"S{i}" for i in range(n_samples)], n_cells_per_sample)
    sample_re = rng.normal(0, 1.0, n_samples).repeat(n_cells_per_sample)
    cellQ = rng.normal(0, 1, n)
    signal = rng.normal(0, 1, n)
    score = signal * 1.5 + sample_re + cellQ * 0.3 + rng.normal(0, 0.5, n)
    noise = rng.normal(0, 1, n)
    expression = np.vstack([signal, noise])
    covariates = pd.DataFrame({"cellQ": cellQ})
    res = _hlm_pvalue_per_row(expression, score, covariates, sample)
    assert res["pvalue"].iloc[0] < 1e-5
    assert res["pvalue"].iloc[1] > 0.05
    assert res["estimate"].iloc[0] > 0.5


def test_hlm_pvalue_per_row_handles_degenerate_row():
    n = 30
    sample = np.repeat(["A", "B", "C"], n // 3)
    expression = np.zeros((2, n))
    expression[1] = np.arange(n) / n
    covariates = pd.DataFrame({"cellQ": np.linspace(0, 1, n)})
    score = np.linspace(-1, 1, n)
    res = _hlm_pvalue_per_row(expression, score, covariates, sample)
    assert res.shape == (2, 2)
