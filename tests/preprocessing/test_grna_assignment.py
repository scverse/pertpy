import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import pertpy as pt


@pytest.fixture
def tiny_dense_adata():
    exp_matrix = np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]]).astype(np.float32)
    return ad.AnnData(
        exp_matrix,
        obs=pd.DataFrame(index=[f"cell_{i + 1}" for i in range(exp_matrix.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i + 1}" for i in range(exp_matrix.shape[1])]),
    )


@pytest.fixture
def tiny_sparse_adata():
    sparse_matrix = sparse.csr_matrix(np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]]).astype(np.float32))
    return ad.AnnData(
        sparse_matrix,
        obs=pd.DataFrame(index=[f"cell_{i + 1}" for i in range(sparse_matrix.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i + 1}" for i in range(sparse_matrix.shape[1])]),
    )


def _make_synthetic_matrix():
    """Cell-by-guide matrix with a clear bimodal positive/background split per guide."""
    rng = np.random.default_rng(0)
    n_cells, n_guides = 800, 12
    frac_pos = 0.12
    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    truth = np.zeros((n_cells, n_guides), dtype=bool)
    for g in range(n_guides):
        is_pos = rng.random(n_cells) < frac_pos
        pos_counts = np.clip(np.round(2 ** rng.normal(5.0, 0.6, is_pos.sum())), 1, None)
        bg_counts = rng.poisson(0.8, (~is_pos).sum())
        X[is_pos, g] = pos_counts
        X[~is_pos, g] = bg_counts
        truth[is_pos, g] = True
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=[f"guide_{i}" for i in range(n_guides)])
    return X, truth, obs, var


@pytest.fixture
def synthetic_dense_adata():
    X, truth, obs, var = _make_synthetic_matrix()
    return ad.AnnData(X, obs=obs, var=var), truth


@pytest.fixture
def synthetic_sparse_adata():
    X, truth, obs, var = _make_synthetic_matrix()
    return ad.AnnData(sparse.csr_matrix(X), obs=obs, var=var), truth


@pytest.fixture
def low_count_dense_adata():
    """One all-zero guide, one with a single nonzero cell, one with max count 1, and two well-populated guides."""
    rng = np.random.default_rng(0)
    n_cells = 200
    n_guides = 5
    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    # column 0: all zero
    # column 1: a single non-zero cell
    X[0, 1] = 1
    # column 2: many non-zero cells but max count 1 -> should be skipped
    X[: n_cells // 4, 2] = 1
    # columns 3-4: standard bimodal positive/background guides that should fit successfully
    for g in (3, 4):
        is_pos = rng.random(n_cells) < 0.15
        pos_counts = np.clip(np.round(2 ** rng.normal(5.0, 0.6, is_pos.sum())), 1, None)
        bg_counts = rng.poisson(0.8, (~is_pos).sum())
        X[is_pos, g] = pos_counts
        X[~is_pos, g] = bg_counts
    return ad.AnnData(
        X,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=["zero", "one_nonzero", "max_below_two", "good_a", "good_b"]),
    )


@pytest.fixture
def low_count_sparse_adata(low_count_dense_adata):
    a = low_count_dense_adata.copy()
    a.X = sparse.csr_matrix(a.X)
    return a


def _x_as_dense(adata):
    return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)


@pytest.mark.parametrize("adata_fixture", ["tiny_dense_adata", "tiny_sparse_adata"])
def test_grna_threshold_assignment(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)
    threshold = 5
    output_layer = "assigned_guides"
    assert output_layer not in adata.layers

    ga = pt.pp.GuideAssignment()
    ga.assign_by_threshold(adata, assignment_threshold=threshold, output_layer=output_layer)

    assert output_layer in adata.layers

    result_matrix = (
        adata.layers[output_layer].toarray()
        if sparse.issparse(adata.layers[output_layer])
        else adata.layers[output_layer]
    )
    original_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    assert np.all((original_matrix >= threshold) == (result_matrix == 1))


@pytest.mark.parametrize("adata_fixture", ["tiny_dense_adata", "tiny_sparse_adata"])
def test_grna_max_assignment(request, adata_fixture):
    adata = request.getfixturevalue(adata_fixture)
    threshold = 6
    obs_key = "assigned_guide"
    assert obs_key not in adata.obs

    ga = pt.pp.GuideAssignment()
    ga.assign_to_max_guide(adata, assignment_threshold=threshold, obs_key=obs_key)
    assert obs_key in adata.obs
    assert tuple(adata.obs[obs_key]) == ("guide_1", "Negative", "guide_3")


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_recovers_known_positives(request, adata_fixture):
    """On synthetic data, per-cell-by-guide accuracy must be high and recall non-trivial."""
    adata, truth = request.getfixturevalue(adata_fixture)
    pt.pp.GuideAssignment().assign_mixture_model(adata)

    thresholds = adata.var["threshold"].to_numpy()
    X = _x_as_dense(adata)
    pred = np.zeros_like(truth)
    for g, thr in enumerate(thresholds):
        if not np.isnan(thr):
            pred[:, g] = X[:, g] >= thr

    accuracy = (pred == truth).mean()
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    assert accuracy > 0.98, f"per-cell-by-guide accuracy too low: {accuracy:.4f}"
    assert sensitivity > 0.9, f"sensitivity too low: {sensitivity:.4f}"
    assert specificity > 0.99, f"specificity too low: {specificity:.4f}"
    assert fp <= 0.005 * truth.size, f"too many false positives: {fp}"


def test_grna_mixture_model_dense_matches_sparse(synthetic_dense_adata, synthetic_sparse_adata):
    """Dense and sparse inputs must produce identical assignments, thresholds, and MAP estimates."""
    adata_dense, _ = synthetic_dense_adata
    adata_sparse, _ = synthetic_sparse_adata

    pt.pp.GuideAssignment().assign_mixture_model(adata_dense)
    pt.pp.GuideAssignment().assign_mixture_model(adata_sparse)

    assert list(adata_dense.obs["assigned_guide"]) == list(adata_sparse.obs["assigned_guide"])
    np.testing.assert_array_equal(adata_dense.var["threshold"].to_numpy(), adata_sparse.var["threshold"].to_numpy())
    for col in ("poisson_rate", "gaussian_mean", "gaussian_std", "mix_probs_0", "mix_probs_1"):
        np.testing.assert_allclose(
            adata_dense.var[col].to_numpy(),
            adata_sparse.var[col].to_numpy(),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_writes_var_columns(request, adata_fixture):
    adata, _ = request.getfixturevalue(adata_fixture)
    pt.pp.GuideAssignment().assign_mixture_model(adata)
    for col in (
        "poisson_rate",
        "gaussian_mean",
        "gaussian_std",
        "mix_probs_0",
        "mix_probs_1",
        "threshold",
        "final_loss",
    ):
        assert col in adata.var.columns


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_threshold_rule_holds(request, adata_fixture):
    """Cells assigned to a guide must have count >= its threshold, and vice versa."""
    adata, _ = request.getfixturevalue(adata_fixture)
    pt.pp.GuideAssignment().assign_mixture_model(adata)
    thresholds = adata.var["threshold"].to_numpy()
    X = _x_as_dense(adata)
    n_cells = X.shape[0]
    for g, thr in enumerate(thresholds):
        if np.isnan(thr):
            continue
        col = X[:, g]
        passing = col >= thr
        guide_name = adata.var_names[g]
        for cell_idx in range(n_cells):
            assigned_str = adata.obs["assigned_guide"].iloc[cell_idx]
            if assigned_str == "multiple":
                continue
            assigned = assigned_str != "negative" and guide_name in str(assigned_str).split("+")
            if assigned:
                assert passing[cell_idx], (
                    f"cell {cell_idx} assigned to {guide_name} but count {col[cell_idx]} < threshold {thr}"
                )
            elif passing[cell_idx]:
                assert guide_name in str(assigned_str).split("+"), (
                    f"cell {cell_idx} has count {col[cell_idx]} >= threshold {thr} for {guide_name} but assignment={assigned_str}"
                )


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_gaussian_above_poisson(request, adata_fixture):
    """For fittable guides with clear bimodality the Gaussian mean must lie above the Poisson rate."""
    adata, _ = request.getfixturevalue(adata_fixture)
    pt.pp.GuideAssignment().assign_mixture_model(adata)
    mu = adata.var["gaussian_mean"].to_numpy()
    lam = adata.var["poisson_rate"].to_numpy()
    valid = ~np.isnan(mu)
    assert valid.sum() >= adata.n_vars - 1
    assert np.all(mu[valid] > lam[valid])


@pytest.mark.parametrize("adata_fixture", ["low_count_dense_adata", "low_count_sparse_adata"])
def test_grna_mixture_model_skips_low_count_guides(request, adata_fixture):
    """Guides with <2 nonzero cells or max count <2 must be skipped (NaN params and a warning)."""
    adata = request.getfixturevalue(adata_fixture)
    with (
        pytest.warns(UserWarning, match="less than 2 cells"),
        pytest.warns(UserWarning, match="maximum count is below 2"),
    ):
        pt.pp.GuideAssignment().assign_mixture_model(adata)
    assert np.isnan(adata.var.loc["zero", "threshold"])
    assert np.isnan(adata.var.loc["one_nonzero", "threshold"])
    assert np.isnan(adata.var.loc["max_below_two", "threshold"])
    assert not np.isnan(adata.var.loc["good_a", "threshold"])
    assert not np.isnan(adata.var.loc["good_b", "threshold"])


@pytest.mark.parametrize("sparsify", [False, True])
def test_grna_mixture_model_rejects_negative_values(sparsify):
    X = np.array([[1.0, 2.0], [-0.5, 3.0]], dtype=np.float32)
    X_in = sparse.csr_matrix(X) if sparsify else X
    adata = ad.AnnData(
        X_in,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    with pytest.raises(ValueError, match="negative values"):
        pt.pp.GuideAssignment().assign_mixture_model(adata)


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_rejects_unknown_model(request, adata_fixture):
    adata, _ = request.getfixturevalue(adata_fixture)
    with pytest.raises(ValueError, match="Model not implemented"):
        pt.pp.GuideAssignment().assign_mixture_model(adata, model="not_a_model")


@pytest.mark.parametrize("adata_fixture", ["synthetic_dense_adata", "synthetic_sparse_adata"])
def test_grna_mixture_model_only_return_results(request, adata_fixture):
    adata, _ = request.getfixturevalue(adata_fixture)
    n_cells = adata.n_obs
    assignments = pt.pp.GuideAssignment().assign_mixture_model(adata, only_return_results=True)
    assert isinstance(assignments, np.ndarray)
    assert assignments.shape == (n_cells,)
    assert "assigned_guide" not in adata.obs


@pytest.mark.parametrize("matrix_type", ["numpy", "sparse"])
def test_grna_mixture_model_matrix_overloads_match_anndata(matrix_type, synthetic_dense_adata):
    """Calling the matrix overloads must yield the same per-cell strings as the AnnData overload."""
    adata, _ = synthetic_dense_adata
    X = np.asarray(adata.X) if matrix_type == "numpy" else sparse.csr_matrix(adata.X)
    out = pt.pp.GuideAssignment().assign_mixture_model(X, var=adata.var)
    assert isinstance(out, np.ndarray)
    assert out.shape == (adata.n_obs,)
    pt.pp.GuideAssignment().assign_mixture_model(adata)
    assert list(out) == list(adata.obs["assigned_guide"])
