"""Tests for DIALOGUE: unit tests for the building blocks plus an R-reference regression
test for ``Dialogue.fit_programs``.

The R-reference test is gated on the presence of ``tests/_data/dialogue_reference/``
so the suite still runs (with that test skipped) if the fixture is missing.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

import pertpy as pt
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

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "_data" / "dialogue_reference"


def _safe_name(ct: str) -> str:
    """Filename suffix used by the R reference exporter (`make.names` in R)."""
    return ct.replace("+ ", "..").replace("+", "")


def _sign_align(W_py: np.ndarray, W_r: np.ndarray) -> np.ndarray:
    """Flip sign of each column of ``W_py`` to maximize agreement with ``W_r``."""
    out = W_py.copy()
    for j in range(out.shape[1]):
        if np.dot(out[:, j], W_r[:, j]) < 0:
            out[:, j] *= -1.0
    return out


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


# ---------------------------------------------------------------------------
# Dialogue.fit_programs — end-to-end on dialogue_example with R reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dialogue_adata():
    """Same preprocessing as the R reference run: 30-PC PCA, drop CD8+ IL17+, keep samples present in every remaining cell type."""
    adata = pt.dt.dialogue_example()
    sc.pp.pca(adata, n_comps=30, random_state=0)
    adata = adata[adata.obs["cell.subtypes"] != "CD8+ IL17+"].copy()
    isecs = pd.crosstab(adata.obs["cell.subtypes"], adata.obs["sample"])
    keep_pts = list(isecs.loc[:, (isecs > 3).sum(axis=0) == isecs.shape[0]].columns.values)
    adata = adata[adata.obs["sample"].isin(keep_pts), :].copy()
    adata.obs["cell.subtypes"] = adata.obs["cell.subtypes"].astype("category").cat.remove_unused_categories()
    return adata


@pytest.fixture
def fitted_dialogue(dialogue_adata):
    adata = dialogue_adata.copy()
    pt.tl.Dialogue(
        celltype_key="cell.subtypes",
        sample_key="sample",
        cell_quality_key="cellQ",
        n_programs=3,
        n_components=30,
        n_permutations=20,
        min_cells_per_sample=5,
        random_state=1234,
    ).fit_programs(adata)
    return adata


def test_fit_programs_populates_uns(fitted_dialogue):
    state = fitted_dialogue.uns["dialogue"]
    for key in (
        "weights",
        "weights_index",
        "pseudobulk_features",
        "empirical_pvalues",
        "cca_correlations_R",
        "cca_correlations_P",
        "program_celltypes",
        "program_signatures",
        "cell_type_order",
        "shared_samples",
        "params",
    ):
        assert key in state, f"missing dialogue state key: {key}"
    assert "X_dialogue_cca" in fitted_dialogue.obsm
    assert fitted_dialogue.obsm["X_dialogue_cca"].shape[1] == 3
    assert np.isnan(fitted_dialogue.obsm["X_dialogue_cca"]).sum() == 0


@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="R reference fixture not available")
def test_fit_programs_weights_match_R(fitted_dialogue):
    """PMD weights should match the R reference up to per-column sign (cosine > 0.99 per program)."""
    state = fitted_dialogue.uns["dialogue"]
    for ct in state["cell_type_order"]:
        W_py = state["weights"][ct]
        index_py = state["weights_index"][ct]
        W_r = pd.read_csv(REFERENCE_DIR / f"weights_{_safe_name(ct)}.csv", index_col=0).to_numpy()
        # Embed Python weights at original PC indices (R weights span the full PC range).
        full_py = np.zeros_like(W_r)
        for i, name in enumerate(index_py):
            full_py[int(name[2:]) - 1] = W_py[i]
        aligned = _sign_align(full_py, W_r)
        cos = np.array(
            [
                np.dot(aligned[:, j], W_r[:, j]) / (np.linalg.norm(aligned[:, j]) * np.linalg.norm(W_r[:, j]) + 1e-30)
                for j in range(aligned.shape[1])
            ]
        )
        assert (cos > 0.99).all(), f"weights for {ct} drift from R: cos={cos}"


@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="R reference fixture not available")
def test_fit_programs_significant_program_matches_R(fitted_dialogue):
    """MCP1 is highly significant across every pair in R (empirical p ~0.04). Verify the Python run flags it as significant for all pairs too."""
    emp = fitted_dialogue.uns["dialogue"]["empirical_pvalues"]
    assert (emp.loc["MCP1"] < 0.1).all(), f"MCP1 empirical p > 0.1: {emp.loc['MCP1'].to_dict()}"


def test_fit_programs_dense_matches_sparse(dialogue_adata):
    """Dense and sparse X produce identical weights, scores, and signatures."""
    adata_dense = dialogue_adata.copy()
    adata_sparse = dialogue_adata.copy()
    adata_sparse.X = sparse.csr_matrix(adata_sparse.X)

    common_kwargs = {
        "celltype_key": "cell.subtypes",
        "sample_key": "sample",
        "cell_quality_key": "cellQ",
        "n_programs": 3,
        "n_components": 30,
        "n_permutations": 5,
        "random_state": 1234,
    }
    pt.tl.Dialogue(**common_kwargs).fit_programs(adata_dense)
    pt.tl.Dialogue(**common_kwargs).fit_programs(adata_sparse)

    for ct in adata_dense.uns["dialogue"]["cell_type_order"]:
        np.testing.assert_allclose(
            adata_dense.uns["dialogue"]["weights"][ct],
            adata_sparse.uns["dialogue"]["weights"][ct],
            atol=1e-8,
        )
    np.testing.assert_allclose(
        adata_dense.obsm["X_dialogue_cca"],
        adata_sparse.obsm["X_dialogue_cca"],
        atol=1e-8,
    )
    # Signature gene sets must be identical
    for prog, by_ct_d in adata_dense.uns["dialogue"]["program_signatures"].items():
        for ct, sig_d in by_ct_d.items():
            sig_s = adata_sparse.uns["dialogue"]["program_signatures"][prog][ct]
            assert sig_d["up"] == sig_s["up"], f"{prog}/{ct} up differ: dense vs sparse"
            assert sig_d["down"] == sig_s["down"], f"{prog}/{ct} down differ"


def test_fit_programs_raises_on_too_few_shared_samples(dialogue_adata):
    """Synthesizes a case with too few shared samples by restricting to two samples."""
    keep = list(dialogue_adata.obs["sample"].cat.categories[:2])
    a = dialogue_adata[dialogue_adata.obs["sample"].isin(keep)].copy()
    a.obs["sample"] = a.obs["sample"].astype("category").cat.remove_unused_categories()
    with pytest.raises(ValueError, match="at least 5"):
        pt.tl.Dialogue(
            celltype_key="cell.subtypes",
            sample_key="sample",
            n_programs=3,
            n_components=30,
            n_permutations=2,
            random_state=1234,
        ).fit_programs(a)


# ---------------------------------------------------------------------------
# Full pipeline: run / test_celltype_pairs / refine_scores / aux methods
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dialogue(dialogue_adata):
    """Module-scoped: run the full pipeline once and share the result across tests."""
    adata = dialogue_adata.copy()
    pt.tl.Dialogue(
        celltype_key="cell.subtypes",
        sample_key="sample",
        cell_quality_key="cellQ",
        n_programs=3,
        n_components=30,
        n_genes_per_signature=30,
        n_permutations=20,
        random_state=1234,
    ).run(adata)
    return adata


def test_run_populates_obsm_and_obs(run_dialogue):
    state = run_dialogue.uns["dialogue"]
    assert "X_dialogue" in run_dialogue.obsm
    assert "X_dialogue_cca" in run_dialogue.obsm
    assert run_dialogue.obsm["X_dialogue"].shape[1] == 3
    assert not np.isnan(run_dialogue.obsm["X_dialogue"]).any()
    for i in range(3):
        col = f"mcp_{i}"
        assert col in run_dialogue.obs.columns
        np.testing.assert_allclose(run_dialogue.obs[col].to_numpy(), run_dialogue.obsm["X_dialogue"][:, i])
    assert set(state["gene_pvalues"].keys()) == set(state["cell_type_order"])
    assert set(state["pair_results"].keys()) >= {"CD8+ IELs_CD8+ LP", "Macrophages_TA2"}


def test_run_program_signatures_have_content(run_dialogue):
    """The cross-cell-type-shared programs (MCP1, MCP3) should produce non-empty refined signatures."""
    sigs = run_dialogue.uns["dialogue"]["program_gene_signatures"]
    for program in ("MCP1", "MCP3"):
        for ct, info in sigs[program].items():
            assert len(info["up"]) + len(info["down"]) > 0, f"{program}/{ct} signature is empty"


def test_run_gene_pvalues_have_fisher_combined(run_dialogue):
    """`gene_pvalues[ct]` rows tagged 'up'==True must have finite Fisher-combined p_up."""
    for df in run_dialogue.uns["dialogue"]["gene_pvalues"].values():
        if df.empty:
            continue
        up_rows = df[df["up"].astype(bool)]
        assert (up_rows["p_up"] >= 0).all() and (up_rows["p_up"] <= 1).all()


@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="R reference fixture not available")
def test_refined_scores_correlate_with_R(run_dialogue):
    """The Python refined per-cell scores should agree with R's `R$scores` to within sign+rtol per program."""
    py_scores = run_dialogue.obsm["X_dialogue"]
    state = run_dialogue.uns["dialogue"]
    for ct in state["cell_type_order"]:
        mask = (run_dialogue.obs["cell.subtypes"] == ct).to_numpy()
        py = py_scores[mask]
        r_path = REFERENCE_DIR / f"final_scores_{_safe_name(ct)}.parquet"
        if not r_path.exists():
            continue
        r_df = pd.read_parquet(r_path)
        # R columns prefixed MCP1, MCP2, MCP3 -> match first three columns
        program_cols = [c for c in r_df.columns if c.startswith("MCP")][:3]
        r_df[program_cols].to_numpy(dtype=np.float64)
        # Align rows: r_df indexed by cell names, run_dialogue.obs is too
        cell_names = run_dialogue.obs_names[mask].astype(str)
        aligned = r_df.loc[cell_names, program_cols].to_numpy(dtype=np.float64)
        for p in range(py.shape[1]):
            x = py[:, p]
            y = aligned[:, p]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 10:
                continue
            r = abs(float(np.corrcoef(x[valid], y[valid])[0, 1]))
            assert r > 0.7, f"{ct}/MCP{p + 1}: corr with R refined scores {r:.3f} < 0.7"


def test_dense_matches_sparse_full_pipeline(dialogue_adata):
    """Identical state for dense vs sparse adata.X through the full pipeline."""
    common_kwargs = {
        "celltype_key": "cell.subtypes",
        "sample_key": "sample",
        "cell_quality_key": "cellQ",
        "n_programs": 3,
        "n_components": 30,
        "n_genes_per_signature": 20,
        "n_permutations": 10,
        "random_state": 1234,
    }
    dense = dialogue_adata.copy()
    sparse_ad = dialogue_adata.copy()
    sparse_ad.X = sparse.csr_matrix(sparse_ad.X)
    pt.tl.Dialogue(**common_kwargs).run(dense)
    pt.tl.Dialogue(**common_kwargs).run(sparse_ad)
    np.testing.assert_allclose(dense.obsm["X_dialogue"], sparse_ad.obsm["X_dialogue"], atol=1e-7)
    np.testing.assert_allclose(dense.obsm["X_dialogue_cca"], sparse_ad.obsm["X_dialogue_cca"], atol=1e-7)
    for ct in dense.uns["dialogue"]["cell_type_order"]:
        np.testing.assert_allclose(
            dense.uns["dialogue"]["weights"][ct],
            sparse_ad.uns["dialogue"]["weights"][ct],
            atol=1e-7,
        )


def test_get_program_genes(run_dialogue):
    dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
    per_ct = dl.get_program_genes(run_dialogue, program="MCP1", celltype="CD8+ IELs")
    assert "up" in per_ct and "down" in per_ct
    intersected = dl.get_program_genes(run_dialogue, program="MCP1")
    assert "up" in intersected and "down" in intersected


def test_find_extreme_score_genes_returns_rank_tables(run_dialogue):
    dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
    res = dl.find_extreme_score_genes(run_dialogue, program="MCP1", fraction=0.1)
    assert isinstance(res, dict)
    assert any(isinstance(df, pd.DataFrame) and not df.empty for df in res.values())


def test_test_phenotype_association_returns_zscore_table(run_dialogue):
    adata = run_dialogue.copy()
    dl = pt.tl.Dialogue(
        celltype_key="cell.subtypes",
        sample_key="sample",
        cell_quality_key="cellQ",
        n_programs=3,
    )
    z = dl.test_phenotype_association(adata, condition_key="path_str")
    assert isinstance(z, pd.DataFrame)
    assert z.shape == (len(adata.uns["dialogue"]["cell_type_order"]), 3)
    assert "phenotype_pvalues" in adata.uns["dialogue"]
