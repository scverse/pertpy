"""DIALOGUE: cross-cell-type multicellular program discovery.

Re-implements the algorithm of Jerby-Arnon & Regev (2022),
`livnatje/DIALOGUE <https://github.com/livnatje/DIALOGUE>`__, on AnnData.

The pipeline has three phases:
    1. ``fit_programs``        Pseudobulk per sample, filter informative features by ANOVA, center + winsorize, run penalized multiple-CCA, residualize on confounders, find program gene signatures by partial Spearman correlation.
    2. ``test_celltype_pairs`` For every ordered pair of cell types, fit a hierarchical linear model (``y ~ (1 | sample) + x + cell_quality + tme_qc``) of one cell type's program score against the partner cell type's pseudobulk expression of candidate genes, producing signed z-scores.
    3. ``refine_scores``       Aggregate per-gene HLM p-values across pairs via Fisher's method, fit a non-negative least-squares regression of CCA scores against retained genes, and write final per-cell program scores back to ``adata.obsm``.

Sparse ``adata.X`` is supported end-to-end: per-celltype sub-AnnDatas keep their sparse representation, pseudobulks use ``scanpy.get.aggregate``, and the only dense materialization is on the per-sample × n_features pseudobulk matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.formula.api as smf
from scipy import sparse as sp
from scipy import stats
from scipy.optimize import nnls
from sparsecca import multicca_permute, multicca_pmd
from statsmodels.stats.multitest import multipletests

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

_LOG2_PI = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


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
    aggregated = sc.get.aggregate(adata, by=sample_key, func=agg, layer=layer)
    matrix = aggregated.layers[agg]
    if sp.issparse(matrix):
        matrix = matrix.toarray()
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
def _column_anova_sparse(matrix: sp.spmatrix, groups: np.ndarray) -> np.ndarray:
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


def _partial_spearman(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partial Spearman correlation of every column of X against every column of Y, controlling for Z.

    Returns ``(R, P)`` arrays of shape ``[X_cols, Y_cols]``. Matches R's ``ppcor::pcor.mat`` with the
    Spearman method as used by ``DIALOGUE::pcor.mat``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    n = X.shape[0]
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError("X, Y, Z must have the same number of rows")

    X_rank = pd.DataFrame(X).rank().to_numpy()
    Y_rank = pd.DataFrame(Y).rank().to_numpy()
    Z_rank = pd.DataFrame(Z).rank().to_numpy()

    design = np.column_stack([np.ones(n), Z_rank])
    Xr = X_rank - design @ np.linalg.lstsq(design, X_rank, rcond=None)[0]
    Yr = Y_rank - design @ np.linalg.lstsq(design, Y_rank, rcond=None)[0]

    Xs = (Xr - Xr.mean(0)) / np.where(Xr.std(0, ddof=1) > 0, Xr.std(0, ddof=1), 1.0)
    Ys = (Yr - Yr.mean(0)) / np.where(Yr.std(0, ddof=1) > 0, Yr.std(0, ddof=1), 1.0)
    R = (Xs.T @ Ys) / (n - 1)
    df = max(n - 2 - Z_rank.shape[1], 1)
    t_stat = R * np.sqrt(df / np.clip(1 - R**2, 1e-30, None))
    P = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    return R, P


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


def _fisher_combine_by_label(pvalues: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fisher-combine p-values within each label group, returning one combined p-value per row.

    ``pvalues`` shape ``[n_rows, n_columns]``; ``labels`` shape ``[n_rows]`` (program label per row).
    Mirrors R's ``fisher.combine`` applied after ``p.adjust.mat.per.label`` over programs.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    labels = np.asarray(labels)
    adjusted = np.full_like(pvalues, np.nan)
    for label in np.unique(labels):
        mask = labels == label
        block = pvalues[mask]
        valid = ~np.isnan(block)
        for j in range(block.shape[1]):
            col = block[:, j]
            mask_j = valid[:, j]
            if mask_j.sum() < 1:
                continue
            adj = multipletests(col[mask_j], method="fdr_bh")[1]
            full = np.full(col.shape, np.nan)
            full[mask_j] = adj
            block[:, j] = full
        adjusted[mask] = block
    combined = np.empty(adjusted.shape[0], dtype=np.float64)
    for i, row in enumerate(adjusted):
        finite = row[np.isfinite(row) & (row > 0)]
        if finite.size == 0:
            combined[i] = 1.0
            continue
        stat = -2.0 * np.log(finite).sum()
        df = 2 * finite.size
        combined[i] = float(stats.chi2.sf(stat, df=df))
    return combined


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
        x_coef, _ = nnls(x_sel, y)
        coef[mask] = x_coef
        fit = x_sel @ x_coef
        y_fit = y_fit + fit
        y = y - fit
        if np.unique(y_fit).size > 10 and np.corrcoef(y0, y_fit)[0, 1] > correlation_threshold:
            return coef

    leftover = feature_rank < (buckets[-1] if buckets else 1.0)
    if leftover.sum() >= minimum_features:
        x_sel = X[:, leftover]
        x_coef, _ = nnls(x_sel, y)
        coef[leftover] = x_coef
    return coef


def _hlm_pvalue_per_row(
    expression: np.ndarray,
    score: np.ndarray,
    covariates: pd.DataFrame,
    sample_groups: np.ndarray,
) -> pd.DataFrame:
    """Hierarchical linear model per row: ``score ~ (1|sample) + x + covariates`` where ``x`` is each row of ``expression``.

    Returns a DataFrame with ``estimate`` and ``pvalue`` columns indexed like ``expression.index``.
    """
    if isinstance(expression, pd.DataFrame):
        gene_index = expression.index
        expression = expression.to_numpy()
    else:
        gene_index = pd.Index([f"gene_{i}" for i in range(expression.shape[0])])
    score = np.asarray(score, dtype=np.float64)
    n = score.shape[0]
    covariates = covariates.reset_index(drop=True).copy()
    if covariates.shape[0] != n:
        raise ValueError("covariates rows must match score length")
    groups = pd.Series(sample_groups, name="_sample_").reset_index(drop=True)
    base = pd.concat([covariates, groups], axis=1)
    base["_y_"] = score
    extra_terms = " + ".join(f"Q('{col}')" for col in covariates.columns)
    formula = "_y_ ~ _x_" + (f" + {extra_terms}" if extra_terms else "")
    estimates = np.full(expression.shape[0], np.nan)
    pvalues = np.full(expression.shape[0], np.nan)
    for i in range(expression.shape[0]):
        base["_x_"] = expression[i]
        try:
            fit = smf.mixedlm(formula, base, groups=base["_sample_"]).fit(method="bfgs", reml=False, disp=False)
            estimates[i] = float(fit.params.get("_x_", np.nan))
            pvalues[i] = float(fit.pvalues.get("_x_", np.nan))
        except Exception:  # noqa: BLE001 — model may fail on degenerate covariates; record NaN
            continue
    return pd.DataFrame({"estimate": estimates, "pvalue": pvalues}, index=gene_index)


# ---------------------------------------------------------------------------
# Public class skeleton (methods filled in subsequent commits)
# ---------------------------------------------------------------------------


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
        celltype_key: ``adata.obs`` column with the cell-type assignment.
        sample_key: ``adata.obs`` column with the sample / niche identifier.
        cell_quality_key: ``adata.obs`` column with the per-cell QC value (typically log-counts), used as a confounder in residualization and in the per-pair HLM. Matches R's ``cellQ``.
        n_programs: Number of multicellular programs to fit (``k`` in the paper).
        feature_space_key: ``adata.obsm`` key for the pre-computed feature space (typically PCA).
        n_components: Number of components from the feature space to use.
        n_genes_per_signature: Number of top-correlated genes kept per program signature (R's ``n.genes``).
        anova_alpha: Per-feature ANOVA significance threshold for filtering uninformative components (R's ``p.anova``).
        winsorize_quantile: Tail clipping fraction applied to pseudobulk components (R's ``cap.mat`` parameter).
        n_permutations: Permutations used to derive empirical PMD p-values (R's ``n1`` in ``DIALOGUE1.PMD.empirical``).
        empirical_alpha: p-value threshold below which a program is considered shared between a pair of cell types (R's implicit ``< 0.1``).
        use_tme_qc: If True, add ``tme_qc`` (partner-celltype per-sample average of ``cell_quality_key``) as an additional HLM covariate (R default).
        additional_covariates: Extra ``adata.obs`` columns to include as HLM covariates.
        min_cells_per_sample: Minimum cells per sample required for a cell type to be considered in the pair-level HLM (R's ``abn.c``).
        random_state: Reproducibility seed for permutation tests and PMD permute search.
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
        min_cells_per_sample: int = 5,
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

    # The fit_programs / test_celltype_pairs / refine_scores methods are added in
    # subsequent commits — this file currently exposes only the helper functions.

    def run(self, adata: AnnData) -> AnnData:  # pragma: no cover - placeholder
        raise NotImplementedError("Dialogue.run is implemented in a follow-up commit.")
