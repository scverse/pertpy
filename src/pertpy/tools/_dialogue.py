"""DIALOGUE: cross-cell-type multicellular program discovery.

Re-implements the algorithm of Jerby-Arnon & Regev (2022), `livnatje/DIALOGUE <https://github.com/livnatje/DIALOGUE>`__, on AnnData.

The pipeline has three phases:
    1. ``fit_programs`` — pseudobulk per sample, filter informative features by ANOVA, center + winsorize, run penalized multiple-CCA, residualize on confounders, find program gene signatures by partial Spearman correlation.
    2. ``test_celltype_pairs`` — for every pair of cell types, fit a hierarchical linear model (``y ~ (1 | sample) + x + cell_quality + tme_qc``) of one cell type's program score against the partner cell type's pseudobulk expression of candidate genes by REML, producing signed log10 p-value z-scores from Satterthwaite t-tests as ``lmerTest`` does.
    3. ``refine_scores`` — aggregate per-gene HLM p-values across pairs via Fisher's method, fit a non-negative least-squares regression of CCA scores against retained genes, and write final per-cell program scores back to ``adata.obsm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from fast_array_utils.conv import to_dense
from numba import njit, prange
from scipy import sparse as sp
from scipy import stats
from scipy.linalg import svd
from scipy.optimize import nnls
from statsmodels.stats.multitest import multipletests

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._parallel import _numba_threads
from pertpy._types import CSBase, cast_dense, cast_frame, cast_matrix
from pertpy.tools._mixed_models import _random_intercept_fits

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from matplotlib.figure import Figure

_LOG2_PI = float(np.log(2.0 * np.pi))


def _collation_key(name: str) -> tuple[tuple[tuple[int, str], ...], str]:
    """Sort key approximating the ICU collation of R's ``sort``: punctuation before digits before letters, case-insensitive first and lowercase first on ties.

    The order of the shared samples decides the signs of the PMD's SVD initialization and with them which local optimum it reaches, so it has to follow R.
    """
    primary = tuple((0, c) if not c.isalnum() else (1, c) if c.isdigit() else (2, c.lower()) for c in name)
    return primary, name.swapcase()


def _pseudobulk_per_sample(
    adata: AnnData,
    sample_key: str,
    *,
    layer: str | None = None,
    agg: str = "median",
) -> pd.DataFrame:
    """Sample-level pseudobulk of ``adata``.

    Uses :func:`scanpy.get.aggregate` so sparse ``X`` stays sparse until per-group densification.

    Returns:
        DataFrame indexed by sample, columns are ``adata.var_names``.
    """
    aggregated = sc.get.aggregate(adata, by=sample_key, func=agg, layer=layer)  # type: ignore[arg-type]
    matrix = to_dense(aggregated.layers[agg])
    return pd.DataFrame(matrix, index=list(aggregated.obs_names), columns=list(aggregated.var_names))


@singledispatch
def _column_anova(matrix, groups: np.ndarray) -> np.ndarray:
    """One-way ANOVA p-value for each column of ``matrix``, grouping rows by ``groups``.

    Dispatches on dense ``np.ndarray`` and ``scipy.sparse`` matrices.
    """
    raise NotImplementedError(f"Unsupported matrix type: {type(matrix)!r}")


@_column_anova.register(np.ndarray)
def _column_anova_dense(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    n_features = matrix.shape[1]
    pvals = np.ones(n_features, dtype=np.float64)
    for j in range(n_features):
        samples = [matrix[groups == g, j] for g in unique]
        if any(s.size < 1 for s in samples) or len(samples) < 2:
            continue
        _, p = stats.f_oneway(*samples)
        pvals[j] = p if np.isfinite(p) else 1.0
    return pvals


@_column_anova.register(sp.spmatrix)
def _column_anova_sparse(matrix: CSBase, groups: np.ndarray) -> np.ndarray:
    return _column_anova_dense(matrix.toarray(), np.asarray(groups))


def _anova_filter_features(
    matrix: np.ndarray | sp.spmatrix,
    groups: np.ndarray,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Boolean mask of columns with one-way ANOVA p < ``alpha`` after BH adjustment."""
    raw = _column_anova(matrix, np.asarray(groups))
    if raw.size == 0:
        return np.zeros(0, dtype=bool)
    adjusted = multipletests(raw, method="fdr_bh")[1]
    return adjusted < alpha


def _center_scale_winsorize(matrix: np.ndarray, *, cap: float = 0.01) -> np.ndarray:
    """Column-wise center + unit-variance scaling, then clip to inner ``[cap, 1-cap]`` quantiles.

    Matches R's ``center.matrix`` + ``cap.mat``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    scaled = (arr - mean) / std
    if cap > 0:
        lo = np.quantile(scaled, cap, axis=0, keepdims=True)
        hi = np.quantile(scaled, 1.0 - cap, axis=0, keepdims=True)
        scaled = np.clip(scaled, lo, hi)
    return scaled


@njit(cache=True)
def _soft_threshold_unit(vector: np.ndarray, bound: float) -> np.ndarray:
    """Soft-threshold ``vector`` to unit L2 norm and L1 norm at most ``bound``, PMA's ``BinarySearch`` followed by ``soft``."""
    norm = np.sqrt(np.sum(vector**2))
    if norm == 0.0:
        norm = 0.05
    lam = 0.0
    if norm != 0.0 and np.sum(np.abs(vector / norm)) > bound:
        lam1, lam2 = 0.0, np.max(np.abs(vector)) - 1e-5
        for _ in range(150):
            soft = np.sign(vector) * np.maximum(np.abs(vector) - 0.5 * (lam1 + lam2), 0.0)
            soft_norm = np.sqrt(np.sum(soft**2))
            if soft_norm == 0.0:
                soft_norm = 0.05
            if np.sum(np.abs(soft / soft_norm)) < bound:
                lam2 = 0.5 * (lam1 + lam2)
            else:
                lam1 = 0.5 * (lam1 + lam2)
            if lam2 - lam1 < 1e-6:
                break
        lam = 0.5 * (lam1 + lam2)
    soft = np.sign(vector) * np.maximum(np.abs(vector) - lam, 0.0)
    soft_norm = np.sqrt(np.sum(soft**2))
    if soft_norm == 0.0:
        soft_norm = 0.05
    return soft / soft_norm


@njit(cache=True)
def _multicca_iterate(cross: np.ndarray, inits: np.ndarray, penalties: np.ndarray, n_iter: int) -> np.ndarray:
    """PMA's ``MultiCCA`` iterations for standardized data sets given their cross-products ``cross[i, j] = X_i' X_j`` and SVD initializations.

    Data sets are zero-padded to a common number of features, which leaves every update unchanged.
    """
    n_sets, _, p_max, _ = cross.shape
    n_components = inits.shape[2]
    final = np.zeros((n_sets, p_max, n_components))
    for comp in range(n_components):
        ws = np.zeros((n_sets, p_max))
        for i in range(n_sets):
            ws[i] = inits[i, :, comp]
        crit_old, crit, iteration = -10.0, -20.0, 0
        while iteration < n_iter and abs(crit_old - crit) / abs(crit_old) > 0.001 and crit_old != 0.0:
            crit_old = crit
            crit = 0.0
            for ii in range(1, n_sets):
                for jj in range(ii):
                    crit += ws[ii] @ (cross[ii, jj] @ ws[jj])
            iteration += 1
            for idx in range(n_sets):
                total = np.zeros(p_max)
                for jj in range(n_sets):
                    if jj == idx:
                        continue
                    projected = final[idx].T @ (cross[idx, jj] @ final[jj])
                    diagonal = np.zeros(n_components)
                    for k in range(n_components):
                        diagonal[k] = projected[k, k]
                    total += cross[idx, jj] @ ws[jj] - final[idx] @ (diagonal * (final[jj].T @ ws[jj]))
                ws[idx] = _soft_threshold_unit(total, penalties[idx])
        for i in range(n_sets):
            final[i, :, comp] = ws[i]
    return final


def _multicca(matrices: list[np.ndarray], penalties: np.ndarray, *, n_components: int, n_iter: int) -> list[np.ndarray]:
    """Sparse multiple CCA by penalized matrix decomposition, PMA's ``MultiCCA(type = "standard", standardize = TRUE)``.

    Returns one ``features x n_components`` weight matrix per data set.
    """
    standardized = []
    for matrix in matrices:
        centered = np.asarray(matrix, dtype=np.float64) - np.mean(matrix, axis=0)
        std = centered.std(axis=0, ddof=1)
        standardized.append(centered / np.where(std > 0, std, 1.0))
    sizes = [m.shape[1] for m in standardized]
    p_max = max(sizes)
    cross = np.zeros((len(matrices), len(matrices), p_max, p_max))
    inits = np.zeros((len(matrices), p_max, n_components))
    for i, a in enumerate(standardized):
        inits[i, : sizes[i]] = svd(a)[2][:n_components].T
        for j, b in enumerate(standardized):
            cross[i, j, : sizes[i], : sizes[j]] = a.T @ b
    final = _multicca_iterate(cross, inits, np.asarray(penalties, dtype=np.float64), n_iter)
    return [final[i, : sizes[i]] for i in range(len(matrices))]


def _residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS residuals of ``values`` regressed on ``covariates``.

    ``values`` is ``[n_obs, n_targets]``, ``covariates`` is ``[n_obs, n_covar]``.
    An intercept column is added automatically.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    covariates = np.asarray(covariates, dtype=np.float64)
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    n = values.shape[0]
    if covariates.shape[0] != n:
        raise ValueError("values and covariates must share the first dimension")
    design = np.column_stack([np.ones(n), covariates])
    beta, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ beta


@njit(parallel=True, cache=True)
def _rank_columns(x: np.ndarray) -> np.ndarray:
    """Average ranks within each column of ``x``, keeping NaNs like :meth:`pandas.DataFrame.rank`."""
    n_rows, n_cols = x.shape
    ranks = np.empty((n_rows, n_cols))
    for j in prange(n_cols):
        col = x[:, j].copy()
        order = np.argsort(col, kind="mergesort")
        start = 0
        while start < n_rows and not np.isnan(col[order[start]]):
            end = start + 1
            while end < n_rows and col[order[end]] == col[order[start]]:
                end += 1
            for k in range(start, end):
                ranks[order[k], j] = (start + end + 1) / 2
            start = end
        for k in range(start, n_rows):
            ranks[order[k], j] = np.nan
    return ranks


def _partial_spearman(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partial Spearman correlation of every column of ``X`` against every column of ``Y``, controlling for ``Z``.

    Returns ``(R, P)`` arrays of shape ``[X_cols, Y_cols]`` with the t-test p-values of ``ppcor::pcor.test``, as used by R's ``pcor.mat``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = Y.shape[0]
    if X.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y and Z must have the same number of rows")
    design = np.column_stack([np.ones(n), _rank_columns(np.ascontiguousarray(Z))])

    def residual_standardized(values: np.ndarray) -> np.ndarray:
        ranks = _rank_columns(np.ascontiguousarray(values))
        resid = ranks - design @ np.linalg.lstsq(design, ranks, rcond=None)[0]
        std = resid.std(0, ddof=1)
        return (resid - resid.mean(0)) / np.where(std > 0, std, 1.0)

    Xs = residual_standardized(X)
    Ys = residual_standardized(Y)
    df = max(n - 2 - Z.shape[1], 1)
    R = (Xs.T @ Ys) / (n - 1)
    t_stat = R * np.sqrt(df / np.clip(1 - R**2, 1e-30, None))
    return R, 2.0 * stats.t.sf(np.abs(t_stat), df=df)


def _pearson_columns(X: np.ndarray | CSBase, Y: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of ``X`` with every column of ``Y``, NaN for constant columns like R's ``cor``."""
    Y = np.asarray(Y, dtype=np.float64)
    n = Y.shape[0]
    Yc = Y - Y.mean(axis=0)
    if sp.issparse(X):
        mean_x = np.asarray(X.sum(axis=0, dtype=np.float64)).ravel() / n
        sum_sq = np.asarray(X.multiply(X).sum(axis=0, dtype=np.float64)).ravel()
        cross = np.asarray(X.T @ Yc, dtype=np.float64)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        mean_x = Xd.mean(axis=0)
        sum_sq = np.einsum("ij,ij->j", Xd, Xd)
        cross = Xd.T @ Yc
    var_x = sum_sq - n * mean_x**2
    constant = var_x <= 1e-10 * np.maximum(sum_sq, np.finfo(np.float64).tiny)
    var_x = np.where(constant, 1.0, var_x)
    r = cross / np.sqrt(np.outer(var_x, np.einsum("ij,ij->j", Yc, Yc)))
    r[constant] = np.nan
    return r


def _top_correlated(values: np.ndarray, *, q: int, min_ci: float) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks of the top ``q`` positive and the top ``q`` negative values with magnitude at least ``min_ci``, like R's ``get.top.cor``.

    Ties at the ``q``-th value are kept, as in R.
    """
    values = np.asarray(values, dtype=np.float64)

    def top(signed: np.ndarray) -> np.ndarray:
        finite = signed[np.isfinite(signed)]
        if finite.size == 0:
            return np.zeros(signed.shape, dtype=bool)
        cutoff = min(np.sort(finite)[min(q, finite.size) - 1], -abs(min_ci))
        return np.isfinite(signed) & (signed <= cutoff)

    return top(-values), top(values)


def _ranksum_greater_pvalue(value: float, reference: np.ndarray) -> float:
    """One-sided Wilcoxon rank-sum p-value that ``value`` exceeds ``reference``, choosing the exact or normal approximation like R's ``wilcox.test``."""
    reference = reference[np.isfinite(reference)]
    if not np.isfinite(value) or reference.size == 0:
        return np.nan
    exact = reference.size < 50 and np.unique(np.append(reference, value)).size == reference.size + 1
    result = stats.mannwhitneyu([value], reference, alternative="greater", method="exact" if exact else "asymptotic")
    return float(result.pvalue)


def _zscores_from_signed_pvalues(estimate: np.ndarray, pvalue: np.ndarray) -> np.ndarray:
    """Signed log10-style z-scores: positive when estimate>0 & p small, negative when estimate<0 & p small.

    Matches R's ``get.cor.zscores`` after converting one-sided p-values.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    pvalue = np.asarray(pvalue, dtype=np.float64)
    pos = np.where(pvalue == 0, 0.0, pvalue)
    smallest = pos[pos > 0]
    floor = smallest.min() / 2.0 if smallest.size else 1e-300
    pvalue = np.where(pvalue > 0, pvalue, floor)
    pos_half = np.where(estimate > 0, pvalue / 2.0, 1.0 - pvalue / 2.0)
    neg_half = np.where(-estimate > 0, pvalue / 2.0, 1.0 - pvalue / 2.0)
    z = np.where(pos_half > 0.5, np.log10(neg_half), -np.log10(pos_half))
    return z


def _adjust_pvalues(pvalues: np.ndarray, method: str) -> np.ndarray:
    """Adjust p-values with ``method``, leaving NaN entries in place like R's ``p.adjust``."""
    out = np.full(pvalues.shape, np.nan)
    valid = np.isfinite(pvalues)
    if valid.any():
        out[valid] = multipletests(pvalues[valid], method=method)[1]
    return out


def _nnls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative least squares solved on the Cholesky factor of the normal equations, which has the same minimizer as the full problem whenever ``X`` has full column rank."""
    gram = X.T @ X
    try:
        lower = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError:
        return nnls(X, y)[0]
    return nnls(lower.T, np.linalg.solve(lower, X.T @ y))[0]


def _iterative_nnls(
    X: np.ndarray,
    y: np.ndarray,
    feature_rank: np.ndarray,
    *,
    correlation_threshold: float = 0.95,
    minimum_features: int = 5,
) -> np.ndarray:
    """Iterative non-negative least squares matching R's ``DLG.iterative.nnls``.

    Features are bucketed by their normalized rank (``feature_rank in [0, 1]``). Within each bucket
    (largest first, down to a third), fit NNLS on that subset, accumulate the fit, then repeat with
    the residuals as the new target. Stop early when the cumulative fit correlates with the original
    target above ``correlation_threshold``.
    """
    X = np.asarray(X, dtype=np.float64)
    y0 = np.asarray(y, dtype=np.float64).ravel()
    y = y0.copy()
    coef = np.zeros(X.shape[1], dtype=np.float64)
    feature_rank = np.asarray(feature_rank, dtype=np.float64)
    y_fit = np.zeros_like(y0)
    buckets = sorted({float(r) for r in feature_rank if r >= 1.0 / 3.0}, reverse=True)

    for bucket in buckets:
        mask = feature_rank == bucket
        if mask.sum() < minimum_features:
            continue
        x_sel = X[:, mask]
        x_coef = _nnls(x_sel, y)
        coef[mask] = x_coef
        fit = x_sel @ x_coef
        y_fit = y_fit + fit
        y = y - fit
        if np.unique(y_fit).size > 10 and np.corrcoef(y0, y_fit)[0, 1] > correlation_threshold:
            return coef

    leftover = feature_rank < (buckets[-1] if buckets else 1.0)
    if leftover.sum() >= minimum_features:
        x_sel = X[:, leftover]
        coef[leftover] = _nnls(x_sel, y)
    return coef


def _hlm_pvalue_per_row(
    expression: np.ndarray | pd.DataFrame,
    score: np.ndarray,
    covariates: pd.DataFrame,
    sample_groups: np.ndarray,
) -> pd.DataFrame:
    """Hierarchical linear model per row: ``score ~ (1|sample) + x + covariates`` where ``x`` is each row of ``expression``.

    Fitted by REML with Satterthwaite t-tests, like ``lmerTest::lmer`` as DIALOGUE calls it.
    Returns a DataFrame with ``estimate``, ``std_error``, ``df`` and ``pvalue`` columns indexed like ``expression.index``.
    """
    if isinstance(expression, pd.DataFrame):
        gene_index = expression.index
        expression = expression.to_numpy()
    else:
        gene_index = pd.Index([f"gene_{i}" for i in range(expression.shape[0])])
    expression = np.asarray(expression, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n = score.shape[0]
    if covariates.shape[0] != n:
        raise ValueError("covariates rows must match score length")
    labels, groups = np.unique(np.asarray(sample_groups), return_inverse=True)
    dummies = np.empty((n, 0))
    if covariates.shape[1]:
        covariates = covariates.apply(
            lambda col: col.cat.remove_unused_categories() if isinstance(col.dtype, pd.CategoricalDtype) else col
        )
        dummies = pd.get_dummies(covariates, drop_first=True, dtype=np.float64).to_numpy(dtype=np.float64)
        dummies = dummies[:, np.ptp(dummies, axis=0) > 0]
    fixed = np.column_stack([np.ones(n), dummies, score])
    estimates = np.full(expression.shape[0], np.nan)
    std_errors = np.full(expression.shape[0], np.nan)
    dfs = np.full(expression.shape[0], np.nan)
    rows = ~np.isnan(expression).any(axis=1) & ~np.isnan(fixed).any()
    if rows.any():
        with _numba_threads(int(rows.sum())):
            estimates[rows], std_errors[rows], dfs[rows] = _random_intercept_fits(
                np.ascontiguousarray(expression[rows]), fixed, groups.astype(np.int64), len(labels)
            )
    pvalues = 2 * stats.t.sf(np.abs(estimates / std_errors), dfs)
    return pd.DataFrame(
        {"estimate": estimates, "std_error": std_errors, "df": dfs, "pvalue": pvalues}, index=gene_index
    )


@dataclass
class DialogueState:
    """Cached intermediates produced by ``fit_programs``."""

    cell_type_order: list[str]
    shared_samples: list[str]
    pseudobulk_features: dict[str, pd.DataFrame]
    weights: dict[str, np.ndarray]
    cca_scores: dict[str, np.ndarray]
    empirical_pvalues: pd.DataFrame
    cca_correlations: pd.DataFrame
    gene_signatures: dict[str, dict[str, dict[str, list[str]]]]


class Dialogue:
    """Multicellular program discovery (DIALOGUE).

    Args:
        celltype_key: Column of ``adata.obs`` with the cell-type assignment.
        sample_key: Column of ``adata.obs`` with the sample / niche identifier.
        cell_quality_key: Column of ``adata.obs`` with the per-cell QC value (typically log-counts) used as a confounder in residualization and in the per-pair HLM (R's ``cellQ``).
        n_programs: Number of multicellular programs to fit (``k`` in the paper).
        feature_space_key: ``adata.obsm`` key for the pre-computed feature space (typically PCA).
        n_components: Number of components of the feature space to use.
        n_genes_per_signature: Number of top-correlated genes kept per program signature and direction (R's ``n.genes``, 100 as in the DIALOGUE tutorial).
        anova_alpha: Per-feature ANOVA significance threshold for filtering uninformative components (R's ``p.anova``).
        winsorize_quantile: Tail clipping fraction applied to pseudobulk components (R's ``cap.mat`` parameter).
        n_permutations: Number of PMD fits in the empirical test, the observed one plus ``n_permutations - 1`` on sample-permuted matrices (R's ``n1`` in ``DIALOGUE1.PMD.empirical``).
            At least 11 are needed for any pair to pass the default ``empirical_alpha``.
        empirical_alpha: P-value threshold below which a program is considered shared between a pair of cell types (R's implicit ``< 0.1``).
        use_tme_qc: If True, add ``tme_qc`` (partner-celltype per-sample average of ``cell_quality_key``) as an additional HLM covariate (R default).
        additional_covariates: Extra ``adata.obs`` columns to include as HLM covariates.
        min_cells_per_sample: Minimum cells a sample needs for its cells to enter the ANOVA feature filter (R's ``abn.c``).
        random_state: Reproducibility seed for the permutation tests.
    """

    def __init__(
        self,
        *,
        celltype_key: str,
        sample_key: str,
        cell_quality_key: str = "cellQ",
        n_programs: int = 3,
        feature_space_key: str = "X_pca",
        n_components: int = 30,
        n_genes_per_signature: int = 100,
        anova_alpha: float = 0.05,
        winsorize_quantile: float = 0.01,
        n_permutations: int = 100,
        empirical_alpha: float = 0.1,
        use_tme_qc: bool = True,
        additional_covariates: Sequence[str] = (),
        min_cells_per_sample: int = 15,
        random_state: int = 1234,
    ) -> None:
        self.celltype_key = celltype_key
        self.sample_key = sample_key
        self.cell_quality_key = cell_quality_key
        self.n_programs = n_programs
        self.feature_space_key = feature_space_key
        self.n_components = n_components
        self.n_genes_per_signature = n_genes_per_signature
        self.anova_alpha = anova_alpha
        self.winsorize_quantile = winsorize_quantile
        self.n_permutations = n_permutations
        self.empirical_alpha = empirical_alpha
        self.use_tme_qc = use_tme_qc
        self.additional_covariates = tuple(additional_covariates)
        self.min_cells_per_sample = min_cells_per_sample
        self.random_state = int(random_state)

    def fit_programs(self, adata: AnnData) -> AnnData:
        """Identify multicellular programs across cell types via penalized multiple-CCA.

        Phase 1 of DIALOGUE.
        Pseudobulks each cell type per sample, filters uninformative components by ANOVA, centers and winsorizes them, then runs penalized multiple-CCA on the cell types' pseudobulk feature spaces to obtain weights and per-cell program scores.
        Empirical p-values for each program-by-pair are computed by repeating the PMD on permuted matrices.

        Stores the following on ``adata``:

        - ``adata.obsm["X_dialogue_cca"]`` — per-cell CCA scores (``n_obs × n_programs``), residualized on the cell-quality confounder, NaN-padded for cells of cell types skipped during fitting.
        - ``adata.uns["dialogue"]["weights"][celltype]`` — PMD weights (``n_components × n_programs``).
        - ``adata.uns["dialogue"]["pseudobulk_features"][celltype]`` — the post-filter, post-center pseudobulk matrices (samples × retained components).
        - ``adata.uns["dialogue"]["empirical_pvalues"]`` — programs × cell-type pairs.
        - ``adata.uns["dialogue"]["cca_correlations_R"]`` / ``"_P"`` — per-pair pairwise correlation and p-value of the cell types' CCA scores.
        - ``adata.uns["dialogue"]["program_celltypes"]`` — mapping of each program to the cell types retained by R's ``DIALOGUE.identify.cell.types`` pruning of the pairs passing ``empirical_alpha``.
        - ``adata.uns["dialogue"]["program_signatures"][program][celltype]`` — initial signature genes, the top Pearson-correlated genes re-ranked by partial Spearman correlation of the CCA scores with expression, controlling for cell quality.
        - ``adata.uns["dialogue"]["params"]`` — recorded hyperparameters.
        - ``adata.uns["dialogue"]["shared_samples"]`` — samples present in all cell types.
        - ``adata.uns["dialogue"]["cell_type_order"]`` — cell types fit (in stable order).

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
        """
        celltypes = self._cell_type_order(adata)
        pseudobulks_full, ct_views = self._per_celltype_pseudobulks(adata, celltypes)
        pseudobulks = self._anova_filter_per_celltype(pseudobulks_full, ct_views)

        for ct, pb in pseudobulks.items():
            if pb.shape[1] < self.n_programs:
                raise ValueError(
                    f"Cell type {ct!r} retained only {pb.shape[1]} components after the ANOVA "
                    f"filter (need >= n_programs={self.n_programs}). Loosen anova_alpha or use more PCs."
                )

        shared = self._shared_samples(pseudobulks)
        centered = {ct: self._center(pb).loc[shared] for ct, pb in pseudobulks.items()}

        matrices = [centered[ct].to_numpy() for ct in celltypes]
        weights = self._fit_pmd(matrices)
        ws_dict = {
            ct: pd.DataFrame(
                weights[i], index=centered[ct].columns, columns=[f"MCP{j + 1}" for j in range(weights[i].shape[1])]
            )
            for i, ct in enumerate(celltypes)
        }

        empirical_p = self._empirical_pmd_pvalues(matrices, celltypes)

        cca_correlations_R, cca_correlations_P = self._cca_correlations(matrices, weights, celltypes)

        cca_scores = {
            ct: ct_views[ct].obsm[self.feature_space_key][:, : self.n_components][  # type: ignore[call-overload, index]
                :, _retained_indices(pb_full, ws_dict[ct])  # type: ignore[index]
            ]
            @ ws_dict[ct].to_numpy()
            for ct, pb_full in pseudobulks_full.items()
        }
        cca_scores = self._residualize_cca_scores(cca_scores, ct_views)

        adata.obsm["X_dialogue_cca"] = self._broadcast_per_celltype(adata, cca_scores, ct_views, n_cols=self.n_programs)

        program_celltypes = self._program_celltypes(empirical_p, celltypes)
        program_signatures = self._initial_program_signatures(ct_views, cca_scores)

        adata.uns["dialogue"] = {
            "weights": {ct: ws_dict[ct].to_numpy() for ct in celltypes},
            "weights_index": {ct: list(ws_dict[ct].index) for ct in celltypes},
            "pseudobulk_features": {ct: centered[ct] for ct in celltypes},
            "empirical_pvalues": empirical_p,
            "cca_correlations_R": cca_correlations_R,
            "cca_correlations_P": cca_correlations_P,
            "program_celltypes": program_celltypes,
            "program_signatures": program_signatures,
            "cell_type_order": list(celltypes),
            "shared_samples": list(shared),
            "params": self._param_dict(),
        }
        return adata

    def _cell_type_order(self, adata: AnnData) -> list[str]:
        col = adata.obs[self.celltype_key]
        if hasattr(col, "cat"):
            return [str(c) for c in col.cat.categories if (col == c).any()]
        return sorted(map(str, col.unique()))

    def _per_celltype_pseudobulks(
        self, adata: AnnData, celltypes: list[str]
    ) -> tuple[dict[str, pd.DataFrame], dict[str, AnnData]]:
        """Per cell type, return (sample-level median PCA pseudobulk, cell-level AnnData view).

        R uses ``colMedians`` as ``param$averaging.function``; we match.
        """
        ct_views: dict[str, AnnData] = {}
        pseudobulks: dict[str, pd.DataFrame] = {}
        for ct in celltypes:
            mask = (adata.obs[self.celltype_key] == ct).to_numpy()
            sub = adata[mask].copy()
            ct_views[ct] = sub
            pcs = cast_dense(sub.obsm[self.feature_space_key])[:, : self.n_components]
            pb_df = (
                pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
                .assign(_sample=sub.obs[self.sample_key].astype(str).to_numpy())
                .groupby("_sample")
                .median()
                .sort_index()
            )
            pseudobulks[ct] = pb_df
        return pseudobulks, ct_views

    def _anova_filter_per_celltype(
        self, pseudobulks_full: dict[str, pd.DataFrame], ct_views: dict[str, AnnData]
    ) -> dict[str, pd.DataFrame]:
        """For each cell type, drop pseudobulk components whose per-cell ANOVA across samples is non-significant.

        Mirrors R's filter: the ANOVA runs on the per-cell feature values of samples with at least ``min_cells_per_sample`` cells and is BH-adjusted, and fewer than five retained components is an error.
        """
        out: dict[str, pd.DataFrame] = {}
        for ct, pb in pseudobulks_full.items():
            view = ct_views[ct]
            pcs = cast_dense(view.obsm[self.feature_space_key])[:, : self.n_components]
            samples = view.obs[self.sample_key].astype(str).to_numpy()
            counts = pd.Series(samples).value_counts()
            row_mask = np.isin(samples, counts.index[counts >= self.min_cells_per_sample].to_numpy())
            if np.unique(samples[row_mask]).size < 2:
                raise ValueError(
                    f"Fewer than two samples of {ct!r} have at least min_cells_per_sample={self.min_cells_per_sample} "
                    "cells, so the ANOVA feature filter cannot run."
                )
            mask = _anova_filter_features(pcs[row_mask], samples[row_mask], alpha=self.anova_alpha)
            if mask.sum() < 5:
                raise ValueError(
                    f"Only {int(mask.sum())} {ct!r} features passed the ANOVA filter. Try rerunning without {ct!r}."
                )
            out[ct] = pb.iloc[:, mask]
        return out

    def _shared_samples(self, pseudobulks: dict[str, pd.DataFrame]) -> list[str]:
        shared = set.intersection(*[set(pb.index) for pb in pseudobulks.values()])
        if len(shared) < 5:
            raise ValueError(f"Only {len(shared)} samples are present in all cell types; DIALOGUE needs at least 5.")
        return sorted(shared, key=_collation_key)

    def _center(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        scaled = _center_scale_winsorize(pseudobulk.to_numpy(), cap=self.winsorize_quantile)
        return pd.DataFrame(scaled, index=pseudobulk.index, columns=pseudobulk.columns)

    def _fit_pmd(self, matrices: list[np.ndarray]) -> list[np.ndarray]:
        """Penalized multi-CCA weights with R's penalty ``sqrt(ncol(X[[1]])) / 2`` for every matrix."""
        penalties = np.full(len(matrices), np.sqrt(matrices[0].shape[1]) / 2.0)
        return _multicca(matrices, penalties, n_components=self.n_programs, n_iter=100)

    def _empirical_pmd_pvalues(self, matrices: list[np.ndarray], celltypes: list[str]) -> pd.DataFrame:
        """Empirical p-values of the per-pair CCA score correlations, R's ``DIALOGUE1.PMD.empirical``.

        The observed correlation is compared with ``n_permutations - 1`` correlations from PMD fits on matrices whose columns were permuted independently, by a one-sided Wilcoxon rank-sum test.
        """
        if self.n_permutations < 2:
            raise ValueError("n_permutations must be at least 2.")
        rng = np.random.default_rng(self.random_state)
        observed = self._pmd_pair_correlations(matrices, celltypes)
        permuted = np.stack(
            [
                self._pmd_pair_correlations([_column_shuffle(m, rng) for m in matrices], celltypes).to_numpy()
                for _ in range(self.n_permutations - 1)
            ]
        )
        observed_values = observed.to_numpy(dtype=np.float64)
        pvalues = np.array(
            [
                [_ranksum_greater_pvalue(observed_values[i, j], permuted[:, i, j]) for j in range(observed.shape[1])]
                for i in range(observed.shape[0])
            ]
        )
        return pd.DataFrame(pvalues, index=observed.index, columns=observed.columns)

    def _pmd_pair_correlations(self, matrices: list[np.ndarray], celltypes: list[str]) -> pd.DataFrame:
        weights = self._fit_pmd(matrices)
        scores = [matrices[i] @ weights[i] for i in range(len(matrices))]
        names = [f"{celltypes[i]}_{celltypes[j]}" for i, j in _pair_indices(len(matrices))]
        cor = np.zeros((self.n_programs, len(names)))
        for col, (i, j) in enumerate(_pair_indices(len(matrices))):
            for k in range(self.n_programs):
                a = scores[i][:, k]
                b = scores[j][:, k]
                denom = (a.std(ddof=1) * b.std(ddof=1)) or 1.0
                cor[k, col] = float(np.cov(a, b, ddof=1)[0, 1] / denom)
        return pd.DataFrame(cor, columns=names, index=[f"MCP{i + 1}" for i in range(self.n_programs)])

    def _cca_correlations(
        self, matrices: list[np.ndarray], weights: list[np.ndarray], celltypes: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores = [matrices[i] @ weights[i] for i in range(len(matrices))]
        names = [f"{celltypes[i]}_{celltypes[j]}" for i, j in _pair_indices(len(matrices))]
        n = matrices[0].shape[0]
        df = max(n - 2, 1)
        R = np.zeros((self.n_programs, len(names)))
        P = np.zeros((self.n_programs, len(names)))
        for col, (i, j) in enumerate(_pair_indices(len(matrices))):
            for k in range(self.n_programs):
                a = scores[i][:, k]
                b = scores[j][:, k]
                denom = (a.std(ddof=1) * b.std(ddof=1)) or 1.0
                r = float(np.cov(a, b, ddof=1)[0, 1] / denom)
                R[k, col] = r
                t_stat = r * np.sqrt(df / np.clip(1 - r**2, 1e-30, None))
                P[k, col] = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
        index = [f"MCP{i + 1}" for i in range(self.n_programs)]
        return (
            pd.DataFrame(R, index=index, columns=names),
            pd.DataFrame(P, index=index, columns=names),
        )

    def _residualize_cca_scores(
        self, cca_scores: dict[str, np.ndarray], ct_views: dict[str, AnnData]
    ) -> dict[str, np.ndarray]:
        out = {}
        for ct, scores in cca_scores.items():
            conf = cast_frame(ct_views[ct].obs)[self.cell_quality_key].to_numpy(dtype=np.float64)
            out[ct] = _residualize(scores, conf)
        return out

    def _broadcast_per_celltype(
        self,
        adata: AnnData,
        per_ct: dict[str, np.ndarray],
        ct_views: dict[str, AnnData],
        *,
        n_cols: int,
    ) -> np.ndarray:
        out = np.full((adata.n_obs, n_cols), np.nan, dtype=np.float64)
        cell_index = pd.Index(adata.obs_names)
        for ct, mat in per_ct.items():
            view = ct_views[ct]
            positions = cell_index.get_indexer(view.obs_names)
            out[positions] = mat
        return out

    def _program_celltypes(self, empirical_p: pd.DataFrame, celltypes: list[str]) -> dict[str, list[str]]:
        """Cell types participating in each program, R's ``DIALOGUE.identify.cell.types``.

        Cell types are pruned one at a time, the first with the fewest significant pairs (counting itself) going first, until every remaining cell type is significant with at least half of the remaining ones.
        Like R, the first pruning step reads the counts in alphabetical order but removes by input position.
        """
        out: dict[str, list[str]] = {}
        if len(celltypes) == 2:
            for program in empirical_p.index:
                out[program] = list(celltypes) if empirical_p.loc[program].iloc[0] < self.empirical_alpha else []
            return out
        for program in empirical_p.index:
            members = list(celltypes)
            ordered = sorted(celltypes, key=_collation_key)
            significant = pd.DataFrame(np.eye(len(celltypes), dtype=bool), index=ordered, columns=ordered)
            program_p = empirical_p.loc[program].to_numpy(dtype=np.float64)
            for value, (i, j) in zip(program_p, _pair_indices(len(celltypes)), strict=True):
                hit = bool(value < self.empirical_alpha)
                significant.loc[celltypes[i], celltypes[j]] = hit
                significant.loc[celltypes[j], celltypes[i]] = hit
            counts = significant.sum(axis=1).to_numpy()
            while counts.min() < int(np.ceil(len(members) / 2)):
                members.pop(int(np.flatnonzero(counts <= counts.min())[0]))
                significant = significant.loc[members, members]
                counts = significant.sum(axis=1).to_numpy()
            out[program] = members
        return out

    def _initial_program_signatures(
        self, ct_views: dict[str, AnnData], cca_scores: dict[str, np.ndarray]
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        """Per program and cell type, the genes most correlated with the program scores, following R's ``DIALOGUE1``.

        The top ``n_genes_per_signature`` Pearson-correlated genes per program and direction are re-ranked by partial Spearman correlation controlling for cell quality, keeping Bonferroni-significant genes with ``|rho| >= 0.05``.
        """
        q = self.n_genes_per_signature
        programs = [f"MCP{i + 1}" for i in range(self.n_programs)]
        out: dict[str, dict[str, dict[str, list[str]]]] = {program: {} for program in programs}
        for ct, scores in cca_scores.items():
            view = ct_views[ct]
            pearson = _pearson_columns(cast_matrix(view.X), scores)
            candidate = np.zeros(view.n_vars, dtype=bool)
            for program_idx in range(scores.shape[1]):
                up, down = _top_correlated(pearson[:, program_idx], q=q, min_ci=0.05)
                candidate |= up | down
            genes = np.asarray(view.var_names[candidate], dtype=object)
            if genes.size == 0:
                for program in programs:
                    out[program][ct] = {"up": [], "down": []}
                continue
            cellQ = cast_frame(view.obs)[self.cell_quality_key].to_numpy(dtype=np.float64)
            expression = _select_dense_gene_columns(view.X, view.var_names, genes.tolist())
            R, P = _partial_spearman(expression, scores, cellQ)
            R[0.05 / view.n_vars < P] = 0.0
            for program_idx, program in enumerate(programs):
                up, down = _top_correlated(R[:, program_idx], q=q, min_ci=0.05)
                out[program][ct] = {"up": sorted(genes[up]), "down": sorted(genes[down])}
        return out

    def test_celltype_pairs(self, adata: AnnData, *, show_progress: bool = False) -> AnnData:
        """For every ordered pair of cell types, fit a hierarchical linear model of one cell type's program score against the partner cell type's pseudobulk expression of candidate genes.

        Phase 2 of DIALOGUE.
        Builds per-pair, per-program tables of (estimate, standard error, degrees of freedom, p-value, z-score) for each candidate gene from ``fit_programs``' signatures and keeps the top 100 genes per direction with ``|z| >= 1`` as refined signatures, like R.
        Only samples with at least two cells in both cell types of the pair enter a pair's models, as in R.

        Stores on ``adata.uns["dialogue"]["pair_results"]`` a nested dict.
        ``pair_results[pair_name][program][celltype]`` is a DataFrame with one row per gene tested (in that cell type's signature) and columns ``estimate, std_error, df, pvalue, zscore, pvalue_adj, up``.
        Refined per-pair signatures live at ``pair_results[pair_name][program]["refined_signatures"][celltype] = {"up": [...], "down": [...]}``.

        Args:
            adata: AnnData previously processed by :meth:`fit_programs`.
            show_progress: If True, print one line per pair while running.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
        """
        if "dialogue" not in adata.uns:
            raise RuntimeError("Run fit_programs(adata) before test_celltype_pairs(adata).")
        state = adata.uns["dialogue"]
        celltypes = state["cell_type_order"]
        ct_views = self._rebuild_celltype_views(adata, celltypes)
        cca_scores = self._extract_cca_scores(adata, ct_views)
        gene_pseudobulks = self._build_gene_pseudobulks(ct_views)
        per_sample_quality = self._build_per_sample_quality(ct_views)

        pair_results: dict[str, dict[str, dict[str, object]]] = {}
        for i, j in _pair_indices(len(celltypes)):
            ct1, ct2 = celltypes[i], celltypes[j]
            pair_name = f"{ct1}_{ct2}"
            shared = self._shared_abundant_samples(ct_views, ct1, ct2)
            if len(shared) < 2:
                if show_progress:
                    print(f"  skip {pair_name}: only {len(shared)} shared samples with at least two cells")
                pair_results[pair_name] = {}
                continue

            ct1_cells = ct_views[ct1].obs[self.sample_key].astype(str).isin(shared).to_numpy()
            ct2_cells = ct_views[ct2].obs[self.sample_key].astype(str).isin(shared).to_numpy()
            ct1_scores = cca_scores[ct1][ct1_cells]
            ct2_scores = cca_scores[ct2][ct2_cells]
            ct1_samples = ct_views[ct1].obs[self.sample_key].astype(str).to_numpy()[ct1_cells]
            ct2_samples = ct_views[ct2].obs[self.sample_key].astype(str).to_numpy()[ct2_cells]
            ct1_quality = cast_frame(ct_views[ct1].obs)[self.cell_quality_key].to_numpy(dtype=np.float64)[ct1_cells]
            ct2_quality = cast_frame(ct_views[ct2].obs)[self.cell_quality_key].to_numpy(dtype=np.float64)[ct2_cells]
            ct1_covariates = cast_frame(ct_views[ct1].obs)[list(self.additional_covariates)][ct1_cells]
            ct2_covariates = cast_frame(ct_views[ct2].obs)[list(self.additional_covariates)][ct2_cells]
            ct1_tme_qc = per_sample_quality[ct2].reindex(ct1_samples).to_numpy()
            ct2_tme_qc = per_sample_quality[ct1].reindex(ct2_samples).to_numpy()

            shared_mcps = [
                program for program, members in state["program_celltypes"].items() if ct1 in members and ct2 in members
            ]
            if show_progress:
                print(f"  pair {pair_name}: {len(shared_mcps)} shared programs, {len(shared)} samples")
            pair_results[pair_name] = {}
            for program in shared_mcps:
                program_idx = int(program.replace("MCP", "")) - 1
                sig1 = state["program_signatures"][program][ct1]
                sig2 = state["program_signatures"][program][ct2]
                sig1_up = self._intersect_genes(sig1["up"], gene_pseudobulks[ct1].columns)
                sig1_down = self._intersect_genes(sig1["down"], gene_pseudobulks[ct1].columns)
                sig2_up = self._intersect_genes(sig2["up"], gene_pseudobulks[ct2].columns)
                sig2_down = self._intersect_genes(sig2["down"], gene_pseudobulks[ct2].columns)
                ct1_genes_to_test = sig1_up + sig1_down
                ct2_genes_to_test = sig2_up + sig2_down

                # ct2's program score vs ct1's pseudobulk expression at ct2's cells (R's p1).
                ct2_tme_for_ct1_genes = np.asarray(gene_pseudobulks[ct1].loc[list(ct2_samples), ct1_genes_to_test])
                df_ct1 = self._hlm_block(
                    ct2_scores[:, program_idx],
                    ct2_tme_for_ct1_genes,
                    gene_names=ct1_genes_to_test,
                    up_set=sig1_up,
                    cell_quality=ct2_quality,
                    tme_qc=ct2_tme_qc,
                    additional_covariates=ct2_covariates,
                    sample_groups=ct2_samples,
                )

                # ct1's program score vs ct2's pseudobulk expression at ct1's cells (R's p2).
                ct1_tme_for_ct2_genes = np.asarray(gene_pseudobulks[ct2].loc[list(ct1_samples), ct2_genes_to_test])
                df_ct2 = self._hlm_block(
                    ct1_scores[:, program_idx],
                    ct1_tme_for_ct2_genes,
                    gene_names=ct2_genes_to_test,
                    up_set=sig2_up,
                    cell_quality=ct1_quality,
                    tme_qc=ct1_tme_qc,
                    additional_covariates=ct1_covariates,
                    sample_groups=ct1_samples,
                )

                refined_ct1 = self._top_by_zscore(df_ct1)
                refined_ct2 = self._top_by_zscore(df_ct2)
                pair_results[pair_name][program] = {
                    ct1: df_ct1,
                    ct2: df_ct2,
                    "refined_signatures": {ct1: refined_ct1, ct2: refined_ct2},
                }

        state["pair_results"] = pair_results
        state["gene_pseudobulks"] = {ct: gene_pseudobulks[ct] for ct in celltypes}
        state["per_sample_quality"] = {ct: per_sample_quality[ct] for ct in celltypes}
        return adata

    def _rebuild_celltype_views(self, adata: AnnData, celltypes: list[str]) -> dict[str, AnnData]:
        return {ct: adata[(adata.obs[self.celltype_key] == ct).to_numpy()].copy() for ct in celltypes}

    def _extract_cca_scores(self, adata: AnnData, ct_views: dict[str, AnnData]) -> dict[str, np.ndarray]:
        full = adata.obsm["X_dialogue_cca"]
        idx = pd.Index(adata.obs_names)
        out = {}
        for ct, view in ct_views.items():
            positions = idx.get_indexer(view.obs_names)
            out[ct] = np.asarray(full[positions], dtype=np.float64)
        return out

    def _build_gene_pseudobulks(self, ct_views: dict[str, AnnData]) -> dict[str, pd.DataFrame]:
        """Per cell type, return a sample × gene mean-pseudobulk DataFrame (matches R's ``tpmAv``)."""
        out: dict[str, pd.DataFrame] = {}
        for ct, view in ct_views.items():
            out[ct] = _pseudobulk_per_sample(view, sample_key=self.sample_key, agg="mean")
        return out

    def _build_per_sample_quality(self, ct_views: dict[str, AnnData]) -> dict[str, pd.Series]:
        """Per cell type, per-sample mean of ``cell_quality_key`` (matches R's ``qcAv``)."""
        out: dict[str, pd.Series] = {}
        for ct, view in ct_views.items():
            samples = view.obs[self.sample_key].astype(str).to_numpy()
            quality = cast_frame(view.obs)[self.cell_quality_key].to_numpy(dtype=np.float64)
            out[ct] = pd.Series(quality).groupby(samples).mean().rename("qcAv")
        return out

    def _shared_abundant_samples(self, ct_views: dict[str, AnnData], ct1: str, ct2: str) -> list[str]:
        """Samples with at least two cells in both cell types, R's ``get.abundant`` default in ``DIALOGUE2.pair``."""

        def _abundant(ct: str) -> set[str]:
            counts = ct_views[ct].obs[self.sample_key].astype(str).value_counts()
            return set(counts[counts >= 2].index)

        return sorted(_abundant(ct1) & _abundant(ct2))

    def _intersect_genes(self, candidate: list[str], present: pd.Index) -> list[str]:
        present_set = set(present)
        return [g for g in candidate if g in present_set]

    def _hlm_block(
        self,
        score: np.ndarray,
        expression: np.ndarray,
        *,
        gene_names: list[str],
        up_set: list[str],
        cell_quality: np.ndarray,
        tme_qc: np.ndarray,
        additional_covariates: pd.DataFrame,
        sample_groups: np.ndarray,
    ) -> pd.DataFrame:
        if len(gene_names) == 0:
            return pd.DataFrame(columns=["estimate", "std_error", "df", "pvalue", "zscore", "pvalue_adj", "up"])
        covariate_dict = {self.cell_quality_key: cell_quality}
        if self.use_tme_qc:
            covariate_dict["tme_qc"] = tme_qc
        for col in self.additional_covariates:
            covariate_dict[col] = additional_covariates[col].to_numpy()
        covariates = pd.DataFrame(covariate_dict)
        # expression rows -> genes, columns -> cells. Transpose to genes-by-cells for our helper.
        expression_arr = pd.DataFrame(expression.T, index=gene_names).to_numpy()
        res = _hlm_pvalue_per_row(
            pd.DataFrame(expression_arr, index=gene_names),
            score,
            covariates,
            sample_groups,
        )
        res["zscore"] = _zscores_from_signed_pvalues(res["estimate"].to_numpy(), res["pvalue"].to_numpy())
        res["pvalue_adj"] = _adjust_pvalues(res["pvalue"].to_numpy(), "fdr_bh")
        up_lookup = set(up_set)
        res["up"] = [g in up_lookup for g in res.index]
        return res

    def _top_by_zscore(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """Top 100 genes per direction with ``|z| >= 1``, R's ``get.top.cor(p, q = 100, idx = "Z", min.ci = 1)``."""
        if df.empty:
            return {"up": [], "down": []}
        up, down = _top_correlated(df["zscore"].to_numpy(dtype=np.float64), q=100, min_ci=1.0)
        return {"up": sorted(df.index[up]), "down": sorted(df.index[down])}

    def refine_scores(self, adata: AnnData) -> AnnData:
        """Aggregate per-pair HLM evidence and fit final per-cell program scores via iterative non-negative least squares.

        Phase 3 of DIALOGUE.
        For every cell type, gather the per-gene z-scores produced by :meth:`test_celltype_pairs` across every pair the cell type appears in, BH-adjust within each program-by-direction, Fisher-combine across pairs, then run iterative NNLS to fit per-cell program scores against the resulting candidate gene set (sign-flipping down-regulated columns).
        The fitted scores are residualized on the cell-quality confounder and written back to ``adata.obsm["X_dialogue"]``.

        Stores on ``adata.uns["dialogue"]``:

        - ``gene_pvalues[celltype]`` — R's ``gene.pval`` table with per-partner z-scores, Fisher-combined one-sided ``p_up``/``p_down``, support counts ``n_up``/``n_down`` and fractions ``nf_up``/``nf_down`` at R's fixed 0.1 threshold, program label, ``up`` direction, and the fitted ``coef`` from NNLS.
        - ``program_gene_signatures[program][celltype] = {"up": [...], "down": [...]}`` — refined gene signatures (R's ``sig1`` from ``DLG.find.scoring``).
        - ``program_gene_signatures_strict[program][celltype] = {"up": [...], "down": [...]}`` — stricter set (R's ``sig2``).
        - ``pair_refined_correlations[pair][program]`` — R's ``pref``: the correlation ``R`` of the per-sample median refined scores of the two cell types, and the p-values ``hlm_p1``/``hlm_p2`` of each cell type's scores against the partner's per-sample median score in the pair HLM.

        Updates ``adata.obsm["X_dialogue"]`` with the refined per-cell program scores.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
        """
        if "pair_results" not in adata.uns.get("dialogue", {}):
            raise RuntimeError("Run test_celltype_pairs(adata) before refine_scores(adata).")
        state = adata.uns["dialogue"]
        celltypes = state["cell_type_order"]
        ct_views = self._rebuild_celltype_views(adata, celltypes)

        gene_pvalues: dict[str, pd.DataFrame] = {}
        for ct in celltypes:
            gene_pvalues[ct] = self._aggregate_gene_pvalues_for_celltype(state["pair_results"], celltypes, ct)

        nnls_scores: dict[str, np.ndarray] = {}
        refined_signatures: dict[str, dict[str, dict[str, list[str]]]] = {
            f"MCP{p + 1}": {} for p in range(self.n_programs)
        }
        strict_signatures: dict[str, dict[str, dict[str, list[str]]]] = {
            f"MCP{p + 1}": {} for p in range(self.n_programs)
        }
        for ct in celltypes:
            view = ct_views[ct]
            cca0 = self._cca_scores_unresidualized(view, state, ct)

            program_columns = [f"MCP{p + 1}" for p in range(self.n_programs)]
            ct_scores = np.zeros((view.n_obs, self.n_programs))
            gene_pval = gene_pvalues[ct]
            gene_pval["coef"] = 0.0

            # Densify only the candidate gene columns; for a typical run that bounds the dense
            # working set at a few hundred columns rather than the per-celltype full gene matrix.
            candidate_genes_per_program: dict[str, np.ndarray] = {}
            all_candidates: list[str] = []
            for program in program_columns:
                program_rows = gene_pval[gene_pval["program"] == program]
                if program_rows.empty:
                    candidate_genes_per_program[program] = np.empty(0, dtype=object)
                    continue
                names = program_rows["gene"].to_numpy()
                candidate_genes_per_program[program] = names
                all_candidates.extend(names.tolist())
            unique_candidates = sorted(set(all_candidates))
            if unique_candidates:
                slim = _select_dense_gene_columns(view.X, view.var_names, unique_candidates)
                zscored_slim = _zscore_columns(slim)
                slim_name_to_idx = {name: i for i, name in enumerate(unique_candidates)}
            else:
                zscored_slim = np.empty((view.n_obs, 0), dtype=np.float64)
                slim_name_to_idx = {}

            for program_idx, program in enumerate(program_columns):
                y_target = cca0[:, program_idx]
                program_rows = gene_pval[gene_pval["program"] == program]
                if program_rows.empty:
                    ct_scores[:, program_idx] = y_target
                    continue
                gene_names = candidate_genes_per_program[program]
                slim_indices = np.array([slim_name_to_idx[g] for g in gene_names], dtype=np.int64)
                X_program = zscored_slim[:, slim_indices].copy()
                down_mask = ~program_rows["up"].to_numpy(dtype=bool)
                X_program[:, down_mask] *= -1.0
                ranks = program_rows["Nf"].to_numpy(dtype=np.float64)
                coefs = _iterative_nnls(X_program, y_target, ranks)
                ct_scores[:, program_idx] = X_program @ coefs
                gene_pval.loc[program_rows.index, "coef"] = coefs

            nnls_scores[ct] = ct_scores
            for program in program_columns:
                program_rows = gene_pval[gene_pval["program"] == program]
                if program_rows.empty:
                    refined_signatures[program][ct] = {"up": [], "down": []}
                    strict_signatures[program][ct] = {"up": [], "down": []}
                    continue
                n_cells_in_program = len(state["program_celltypes"].get(program, []))
                threshold_n = max(1, int(np.ceil(n_cells_in_program / 2)))
                strong_p = (program_rows["coef"].to_numpy() > 0) | (
                    ((program_rows["n_up"].to_numpy() >= threshold_n) & (program_rows["p_up"].to_numpy() < 1e-3))
                    | ((program_rows["n_down"].to_numpy() >= threshold_n) & (program_rows["p_down"].to_numpy() < 1e-3))
                )
                strict = (program_rows["Nf"].to_numpy() == 1.0) & (
                    (program_rows["p_up"].to_numpy() < 0.05) | (program_rows["p_down"].to_numpy() < 0.05)
                )
                refined_signatures[program][ct] = self._split_up_down(program_rows.loc[strong_p])
                strict_signatures[program][ct] = self._split_up_down(program_rows.loc[strict])

        for ct in celltypes:
            view = ct_views[ct]
            cellQ = cast_frame(view.obs)[self.cell_quality_key].to_numpy(dtype=np.float64)
            nnls_scores[ct] = _residualize(nnls_scores[ct], cellQ)

        adata.obsm["X_dialogue"] = self._broadcast_per_celltype(adata, nnls_scores, ct_views, n_cols=self.n_programs)
        for p in range(self.n_programs):
            adata.obs[f"mcp_{p}"] = cast_dense(adata.obsm["X_dialogue"])[:, p]

        pair_refined = self._refined_pair_correlations(nnls_scores, ct_views, celltypes, state["per_sample_quality"])

        state["gene_pvalues"] = gene_pvalues
        state["program_gene_signatures"] = refined_signatures
        state["program_gene_signatures_strict"] = strict_signatures
        state["pair_refined_correlations"] = pair_refined
        return adata

    def _aggregate_gene_pvalues_for_celltype(
        self,
        pair_results: dict[str, dict[str, dict[str, object]]],
        celltypes: list[str],
        ct: str,
    ) -> pd.DataFrame:
        """Build R's per-program-by-gene ``gene.pval`` table for one cell type from the pair results.

        Follows ``DLG.multi.get.gene.pval`` and, like ``DLG.find.scoring``, drops genes without a z-score in the first partner column.
        """
        position = celltypes.index(ct)
        partner_tables: dict[str, dict[str, pd.DataFrame]] = {}
        for partner in celltypes[position + 1 :] + celltypes[:position]:
            i, j = sorted((position, celltypes.index(partner)))
            programs = pair_results.get(f"{celltypes[i]}_{celltypes[j]}", {})
            tables: dict[str, pd.DataFrame] = {}
            for program, info in programs.items():
                table = info.get(ct)
                if isinstance(table, pd.DataFrame) and not table.empty:
                    tables[program] = table
            if tables:
                partner_tables[partner] = tables
        records: dict[tuple[str, str, bool], dict[str, float]] = {}
        for partner, tables in partner_tables.items():
            for program, table in tables.items():
                for gene, row in table.iterrows():
                    if np.isfinite(row["zscore"]):
                        records.setdefault((program, str(gene), bool(row["up"])), {})[partner] = float(row["zscore"])
        partner_cols = list(partner_tables)
        columns = ["gene", "program", "up", "programF", *partner_cols]
        columns += ["p_up", "p_down", "n_up", "nf_up", "n_down", "nf_down", "N", "Nf"]
        if not records:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(
            [
                {
                    "program": program,
                    "gene": gene,
                    "up": up,
                    "programF": f"{program}.{'up' if up else 'down'}",
                    **{col: partners.get(col, np.nan) for col in partner_cols},
                }
                for (program, gene, up), partners in records.items()
            ]
        )
        labels = df["programF"].to_numpy()
        z = df[partner_cols].to_numpy(dtype=np.float64)
        p_up = self._adjust_per_label(self._pvals_from_zscores(z), labels)
        p_down = self._adjust_per_label(self._pvals_from_zscores(-z), labels)
        df["p_up"] = self._fisher_per_row(p_up)
        df["p_down"] = self._fisher_per_row(p_down)
        for side, adjusted in (("up", p_up), ("down", p_down)):
            hits = np.where(np.isnan(adjusted), np.nan, (adjusted < 0.1).astype(np.float64))
            df[f"n_{side}"] = np.nansum(hits, axis=1).astype(np.int64)
            df[f"nf_{side}"] = np.nanmean(hits, axis=1)
        up = df["up"].to_numpy(dtype=bool)
        df["N"] = np.where(up, df["n_up"], df["n_down"])
        df["Nf"] = np.where(up, df["nf_up"], df["nf_down"])
        df.loc[~up, "p_up"] = 1.0
        df.loc[up, "p_down"] = 1.0
        df = df[np.isfinite(df[partner_cols[0]].to_numpy(dtype=np.float64))]
        return df[columns].reset_index(drop=True)

    def _adjust_per_label(self, pvalues: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """BH-adjust each partner column within each program label, R's ``p.adjust.mat.per.label``.

        With a single partner column R falls through to ``p.adjust``'s default, Holm, and so does this.
        """
        method = "fdr_bh" if pvalues.shape[1] >= 2 else "holm"
        adjusted = np.full_like(pvalues, np.nan)
        for label in np.unique(labels):
            mask = labels == label
            for j in range(pvalues.shape[1]):
                adjusted[mask, j] = _adjust_pvalues(pvalues[mask, j], method)
        return adjusted

    def _fisher_per_row(self, pvalues: np.ndarray) -> np.ndarray:
        """Fisher-combine the finite p-values of each row, R's ``fisher.combine``."""
        out = np.full(pvalues.shape[0], 1.0)
        for i, row in enumerate(pvalues):
            finite = row[np.isfinite(row)]
            if finite.size == 0:
                continue
            if (finite == 0).any():
                out[i] = 0.0
                continue
            out[i] = float(stats.chi2.sf(-2.0 * np.log(finite).sum(), df=2 * finite.size))
        return out

    @staticmethod
    def _pvals_from_zscores(z: np.ndarray) -> np.ndarray:
        """One-sided p-values from signed log10 z-scores, the inverse of :func:`_zscores_from_signed_pvalues` (R's ``get.pval.from.zscores``)."""
        z = np.asarray(z, dtype=np.float64)
        p = 10.0 ** (-np.abs(z))
        return np.where(np.isfinite(z), np.where(z < 0, 1.0 - p, p), np.nan)

    def _split_up_down(self, rows: pd.DataFrame) -> dict[str, list[str]]:
        if rows.empty:
            return {"up": [], "down": []}
        up = rows.loc[rows["up"].astype(bool), "gene"].tolist()
        down = rows.loc[~rows["up"].astype(bool), "gene"].tolist()
        return {"up": up, "down": down}

    def _cca_scores_unresidualized(self, view: AnnData, state: dict, ct: str) -> np.ndarray:
        W = state["weights"][ct]
        idx_names = state["weights_index"][ct]
        kept = [int(name[2:]) - 1 for name in idx_names]
        pcs = view.obsm[self.feature_space_key][:, : self.n_components][:, kept]  # type: ignore[call-overload, index]
        return np.asarray(pcs, dtype=np.float64) @ W

    def _refined_pair_correlations(
        self,
        nnls_scores: dict[str, np.ndarray],
        ct_views: dict[str, AnnData],
        celltypes: list[str],
        per_sample_quality: dict[str, pd.Series],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """R's ``pref``: per pair and program, the correlation of the per-sample median refined scores over samples with at least two cells in both cell types, and the p-values of ``DLG.hlm.pval``."""
        samples = {ct: ct_views[ct].obs[self.sample_key].astype(str).to_numpy() for ct in celltypes}
        medians = {ct: pd.DataFrame(nnls_scores[ct], index=samples[ct]).groupby(level=0).median() for ct in celltypes}
        abundant = {ct: set(pd.Series(samples[ct]).value_counts().loc[lambda c: c >= 2].index) for ct in celltypes}
        out: dict[str, dict[str, dict[str, float]]] = {}
        for i, j in _pair_indices(len(celltypes)):
            ct1, ct2 = celltypes[i], celltypes[j]
            shared = sorted(abundant[ct1] & abundant[ct2])
            a = medians[ct1].loc[shared].to_numpy()
            b = medians[ct2].loc[shared].to_numpy()
            hlm_p1 = self._pair_score_hlm(
                nnls_scores[ct1], ct_views[ct1], samples[ct1], medians[ct2], per_sample_quality[ct2]
            )
            hlm_p2 = self._pair_score_hlm(
                nnls_scores[ct2], ct_views[ct2], samples[ct2], medians[ct1], per_sample_quality[ct1]
            )
            out[f"{ct1}_{ct2}"] = {}
            for p in range(self.n_programs):
                r = float(np.corrcoef(a[:, p], b[:, p])[0, 1]) if len(shared) >= 2 else np.nan
                out[f"{ct1}_{ct2}"][f"MCP{p + 1}"] = {"R": r, "hlm_p1": float(hlm_p1[p]), "hlm_p2": float(hlm_p2[p])}
        return out

    def _pair_score_hlm(
        self,
        scores: np.ndarray,
        view: AnnData,
        samples: np.ndarray,
        partner_medians: pd.DataFrame,
        partner_quality: pd.Series,
    ) -> np.ndarray:
        """P-values per program of ``score ~ (1 | sample) + partner_score + cell_quality + tme_qc``, with the partner's per-sample median score and mean cell quality broadcast to the cells."""
        x = partner_medians.reindex(samples).to_numpy(dtype=np.float64)
        tme_qc = partner_quality.reindex(samples).to_numpy(dtype=np.float64)
        keep = np.isfinite(x).all(axis=1) & np.isfinite(tme_qc)
        pvalues = np.full(self.n_programs, np.nan)
        if keep.sum() < 2:
            return pvalues
        cellQ = cast_frame(view.obs)[self.cell_quality_key].to_numpy(dtype=np.float64)
        covariates = pd.DataFrame({self.cell_quality_key: cellQ[keep], "tme_qc": tme_qc[keep]})
        for p in range(self.n_programs):
            res = _hlm_pvalue_per_row(x[keep, p][None, :], scores[keep, p], covariates, samples[keep])
            pvalues[p] = res["pvalue"].iloc[0]
        return pvalues

    def test_phenotype_association(
        self,
        adata: AnnData,
        condition_key: str,
        *,
        conditions: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """Test each program's association with a phenotype by hierarchical linear models, R's ``DIALOGUE.pheno``.

        For every cell type, and once more for all cells pooled (row ``"All"``), fits ``score ~ (1 | sample) + phenotype + cell_quality`` on the refined scores.
        ``phenotype`` indicates the first level of ``adata.obs[condition_key]`` in sorted order, or is the column itself when it is boolean, as in R.
        Returns signed log10 p-value z-scores (rows: cell types and ``"All"``, columns: programs), positive when that level has higher scores.

        Args:
            adata: AnnData after :meth:`refine_scores`.
            condition_key: Column of ``adata.obs`` with the phenotype labels.
            conditions: Optional two values of ``adata.obs[condition_key]`` to compare.
                Cells with other values are dropped and ``phenotype`` indicates ``conditions[0]``.

        Returns:
            ``zscores`` DataFrame (rows: cell types and ``"All"``, columns: programs), also stored on ``adata.uns["dialogue"]["phenotype_zscores"]``.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
            >>> dl.test_phenotype_association(adata, condition_key="pathology")
        """
        if "dialogue" not in adata.uns:
            raise RuntimeError("Run fit_programs/refine_scores before test_phenotype_association.")
        if "X_dialogue" not in adata.obsm:
            raise RuntimeError("Refined scores missing; run refine_scores(adata) first.")
        state = adata.uns["dialogue"]
        celltypes = state["cell_type_order"]
        obs = cast_frame(adata.obs)
        phenotype = obs[condition_key]
        if conditions is not None:
            labels = phenotype.astype(str).to_numpy()
            keep = np.isin(labels, [str(c) for c in conditions])
            indicator = labels == str(conditions[0])
        else:
            keep = phenotype.notna().to_numpy()
            if pd.api.types.is_bool_dtype(phenotype):
                indicator = phenotype.to_numpy(dtype=bool)
            else:
                labels = phenotype.astype(str).to_numpy()
                indicator = labels == sorted(set(labels[keep]), key=_collation_key)[0]
        scores = cast_dense(adata.obsm["X_dialogue"])
        keep = keep & np.isfinite(scores).all(axis=1)
        cellQ = obs[self.cell_quality_key].to_numpy(dtype=np.float64)
        samples = obs[self.sample_key].astype(str).to_numpy()
        ct_labels = obs[self.celltype_key].astype(str).to_numpy()
        program_cols = [f"MCP{p + 1}" for p in range(self.n_programs)]
        z_table = pd.DataFrame(np.nan, index=[*celltypes, "All"], columns=program_cols)
        for ct in z_table.index:
            mask = keep if ct == "All" else keep & (ct_labels == ct)
            if np.unique(samples[mask]).size < 2:
                continue
            covariates = pd.DataFrame({self.cell_quality_key: cellQ[mask]})
            for program_idx, program in enumerate(program_cols):
                res = _hlm_pvalue_per_row(
                    indicator[mask].astype(np.float64)[None, :], scores[mask, program_idx], covariates, samples[mask]
                )
                z_table.loc[ct, program] = _zscores_from_signed_pvalues(
                    res["estimate"].to_numpy(), res["pvalue"].to_numpy()
                )[0]
        state["phenotype_zscores"] = z_table
        return z_table

    def get_program_genes(
        self,
        adata: AnnData,
        *,
        program: str,
        celltype: str | None = None,
        strict: bool = False,
    ) -> dict[str, list[str]]:
        """Return the refined gene signature ``{"up": [...], "down": [...]}`` for a program.

        Args:
            adata: AnnData after :meth:`refine_scores`.
            program: Program label (e.g. ``"MCP1"``).
            celltype: If given, return only that cell type's signature.
                Otherwise return the cross-celltype intersection of consistently up/down genes.
            strict: Use the strict variant from ``program_gene_signatures_strict`` (genes flagged in every pair).

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
            >>> dl.get_program_genes(adata, program="MCP1", celltype="CD8+ IELs")
        """
        if "dialogue" not in adata.uns:
            raise RuntimeError("Run refine_scores before get_program_genes.")
        key = "program_gene_signatures_strict" if strict else "program_gene_signatures"
        store = adata.uns["dialogue"][key]
        if program not in store:
            raise KeyError(f"Unknown program {program!r}; available: {sorted(store)}")
        per_ct = store[program]
        if celltype is not None:
            if celltype not in per_ct:
                raise KeyError(f"Cell type {celltype!r} not found in program {program}.")
            return {k: list(v) for k, v in per_ct[celltype].items()}
        if not per_ct:
            return {"up": [], "down": []}
        common_up = set.intersection(*(set(v["up"]) for v in per_ct.values()))
        common_down = set.intersection(*(set(v["down"]) for v in per_ct.values()))
        return {"up": sorted(common_up), "down": sorted(common_down)}

    def find_extreme_score_genes(
        self,
        adata: AnnData,
        *,
        program: str = "MCP1",
        fraction: float = 0.1,
    ) -> dict[str, pd.DataFrame]:
        """Differential-expression scan between the highest- and lowest-scoring cells per cell type for one program.

        Args:
            adata: AnnData after :meth:`refine_scores`.
            program: Program to use (``"MCP1"`` by default).
            fraction: Fraction of cells at each tail to compare.
                Must lie in ``(0, 0.5)``.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.ds.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
            >>> dl.find_extreme_score_genes(adata, program="MCP1", fraction=0.1)
        """
        if "X_dialogue" not in adata.obsm:
            raise RuntimeError("Run refine_scores(adata) first.")
        if not 0 < fraction < 0.5:
            raise ValueError("fraction must be in (0, 0.5)")
        idx = int(program.replace("MCP", "")) - 1
        scores = cast_dense(adata.obsm["X_dialogue"])[:, idx]
        out: dict[str, pd.DataFrame] = {}
        for ct in adata.uns["dialogue"]["cell_type_order"]:
            mask = (adata.obs[self.celltype_key] == ct).to_numpy() & np.isfinite(scores)
            if mask.sum() < int(2 / fraction):
                continue
            ct_scores = scores[mask]
            lo_cut = np.quantile(ct_scores, fraction)
            hi_cut = np.quantile(ct_scores, 1 - fraction)
            sub = adata[mask].copy()
            sub.obs["_extreme"] = pd.Categorical(
                np.where(ct_scores >= hi_cut, "high", np.where(ct_scores <= lo_cut, "low", "mid")),
                categories=["low", "mid", "high"],
            )
            sc.tl.rank_genes_groups(sub, groupby="_extreme", groups=["high"], reference="low", use_raw=False)
            result = sc.get.rank_genes_groups_df(sub, group="high")
            out[ct] = result
        return out

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_split_violins(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        *,
        condition_key: str,
        program: str = "MCP1",
        conditions: tuple[str, str] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Per-celltype split violin of program scores stratified by a binary phenotype.

        Args:
            adata: AnnData processed by :meth:`refine_scores`.
            condition_key: Column of ``adata.obs`` with the binary phenotype to split on.
            program: Program label (``"MCP1"``, ``"MCP2"``, ...).
            conditions: Pick which two values of ``adata.obs[condition_key]`` to plot.
                Required when the column has more than two levels.
            {common_plot_args}

        Returns:
            If ``return_fig`` is ``True``, returns the figure, otherwise ``None``.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.dialogue_example()
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
            >>> dl.plot_split_violins(adata, condition_key="path_str", program="MCP1")

        Preview:
            .. image:: /_static/docstring_previews/dialogue_violin.png
        """
        score_col = f"mcp_{int(program.replace('MCP', '')) - 1}"
        if score_col not in adata.obs.columns:
            raise RuntimeError(f"{score_col!r} not in adata.obs; run refine_scores(adata) first.")
        df = cast_frame(adata.obs)[[self.celltype_key, score_col, condition_key]].copy()
        if conditions is not None:
            df = df[df[condition_key].isin(conditions)]
        else:
            unique = pd.Series(df[condition_key]).dropna().unique()
            if len(unique) != 2:
                raise ValueError(f"adata.obs[{condition_key!r}] has {len(unique)} levels; pass `conditions=(a, b)`.")
        df[condition_key] = pd.Categorical(df[condition_key]).remove_unused_categories()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=df, x=self.celltype_key, y=score_col, hue=condition_key, split=True, ax=ax)
        ax.set_ylabel(program)
        ax.tick_params(axis="x", rotation=90)
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()
        plt.close(fig)
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_pairplot(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        *,
        color: str,
        program: str = "MCP1",
        return_fig: bool = False,
    ) -> Figure | None:
        """Cross-celltype pairplot of sample-level program scores, colored by a phenotype.

        Aggregates each (sample, cell type) to the mean program score, pivots into a sample-by-celltype matrix, and runs ``seaborn.pairplot`` with the sample-level ``color`` annotation as the hue.

        Args:
            adata: AnnData processed by :meth:`refine_scores`.
            color: Column of ``adata.obs`` with the per-sample annotation used as the pairplot hue.
            program: Program label.
            {common_plot_args}

        Returns:
            If ``return_fig`` is ``True``, returns the figure, otherwise ``None``.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.ds.dialogue_example()
            >>> dl = pt.tl.Dialogue(celltype_key="cell.subtypes", sample_key="sample", n_programs=3)
            >>> dl.fit_programs(adata)
            >>> dl.test_celltype_pairs(adata)
            >>> dl.refine_scores(adata)
            >>> dl.plot_pairplot(adata, color="clinical.status", program="MCP1")

        Preview:
            .. image:: /_static/docstring_previews/dialogue_pairplot.png
        """
        score_col = f"mcp_{int(program.replace('MCP', '')) - 1}"
        if score_col not in adata.obs.columns:
            raise RuntimeError(f"{score_col!r} not in adata.obs; run refine_scores(adata) first.")
        sample_means = (
            cast_frame(adata.obs)
            .groupby([self.sample_key, self.celltype_key], observed=True)[score_col]
            .mean()
            .unstack()
        )
        sample_color = cast_frame(adata.obs).groupby(self.sample_key, observed=True)[color].first()
        df = sample_means.copy()
        df[color] = sample_color
        grid = sns.pairplot(df, hue=color, corner=True)
        if return_fig:
            return grid.fig
        plt.show()
        return None

    def _param_dict(self) -> dict[str, object]:
        return {
            "celltype_key": self.celltype_key,
            "sample_key": self.sample_key,
            "cell_quality_key": self.cell_quality_key,
            "n_programs": self.n_programs,
            "feature_space_key": self.feature_space_key,
            "n_components": self.n_components,
            "n_genes_per_signature": self.n_genes_per_signature,
            "anova_alpha": self.anova_alpha,
            "winsorize_quantile": self.winsorize_quantile,
            "n_permutations": self.n_permutations,
            "empirical_alpha": self.empirical_alpha,
            "use_tme_qc": self.use_tme_qc,
            "additional_covariates": list(self.additional_covariates),
            "min_cells_per_sample": self.min_cells_per_sample,
            "random_state": self.random_state,
        }


def _retained_indices(pseudobulk_full: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    """Position indices in ``pseudobulk_full`` columns of the components retained in ``weights``."""
    full_cols = list(pseudobulk_full.columns)
    return np.asarray([full_cols.index(c) for c in weights.index], dtype=np.int64)


def _pair_indices(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _column_shuffle(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = matrix.copy()
    n = matrix.shape[0]
    for j in range(matrix.shape[1]):
        perm = rng.permutation(n)
        out[:, j] = matrix[perm, j]
    return out


@singledispatch
def _select_dense_gene_columns(X, var_names, gene_names: list[str]) -> np.ndarray:
    """Return the dense ``cells × len(gene_names)`` slice of ``X`` for the requested genes.

    Dispatches on dense ``np.ndarray`` and sparse ``scipy.sparse`` matrices so that a sparse adata never has to be densified beyond the candidate-gene subset (typically a few hundred columns).
    """
    raise NotImplementedError(f"Unsupported X type: {type(X)!r}")


@_select_dense_gene_columns.register(np.ndarray)
def _select_dense_gene_columns_dense(X: np.ndarray, var_names, gene_names: list[str]) -> np.ndarray:
    lookup = {g: i for i, g in enumerate(var_names)}
    idx = np.array([lookup[g] for g in gene_names], dtype=np.int64)
    return np.asarray(X[:, idx], dtype=np.float64)


@_select_dense_gene_columns.register(sp.spmatrix)
def _select_dense_gene_columns_sparse(X: CSBase, var_names, gene_names: list[str]) -> np.ndarray:
    lookup = {g: i for i, g in enumerate(var_names)}
    idx = np.array([lookup[g] for g in gene_names], dtype=np.int64)
    return np.asarray(X.tocsc()[:, idx].toarray(), dtype=np.float64)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, ddof=1, keepdims=True)
    std = np.where(std > 0, std, 1.0)
    return (arr - mean) / std
