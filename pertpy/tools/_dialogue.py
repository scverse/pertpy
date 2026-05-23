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

    def fit_programs(self, adata: AnnData) -> AnnData:
        """Identify multicellular programs across cell types via penalized multiple-CCA.

        Phase 1 of DIALOGUE. Pseudobulks each cell type per sample, filters uninformative components by ANOVA, centers and winsorizes them, then runs penalized multiple-CCA on the cell types' pseudobulk feature spaces to obtain weights and per-cell program scores. Empirical p-values for each program × pair are computed by repeating the PMD on permuted matrices.

        Stores the following on ``adata``:

        - ``adata.obsm["X_dialogue_cca"]`` — per-cell CCA scores (``n_obs × n_programs``), residualized on the cell-quality confounder, NaN-padded for cells of cell types skipped during fitting.
        - ``adata.uns["dialogue"]["weights"][celltype]`` — PMD weights (``n_components × n_programs``).
        - ``adata.uns["dialogue"]["pseudobulk_features"][celltype]`` — the post-filter, post-center pseudobulk matrices (samples × retained components).
        - ``adata.uns["dialogue"]["empirical_pvalues"]`` — programs × cell-type pairs.
        - ``adata.uns["dialogue"]["cca_correlations_R"]`` / ``"_P"`` — per-pair pairwise correlation and p-value of the cell types' CCA scores.
        - ``adata.uns["dialogue"]["program_celltypes"]`` — mapping of each program to the cell types whose pair passed ``empirical_alpha``.
        - ``adata.uns["dialogue"]["program_signatures"][program][celltype]`` — initial signature genes from partial Spearman correlation of CCA scores against the cell type's expression matrix.
        - ``adata.uns["dialogue"]["params"]`` — recorded hyperparameters.
        - ``adata.uns["dialogue"]["shared_samples"]`` — samples present in all cell types.
        - ``adata.uns["dialogue"]["cell_type_order"]`` — cell types fit (in stable order).
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
        pseudobulks = {ct: pb.loc[shared] for ct, pb in pseudobulks.items()}
        centered = {ct: self._center(pb) for ct, pb in pseudobulks.items()}

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
            ct: ct_views[ct].obsm[self.feature_space_key][:, : self.n_components][
                :, _retained_indices(pb_full, ws_dict[ct])
            ]
            @ ws_dict[ct].to_numpy()
            for ct, pb_full in pseudobulks_full.items()
        }
        cca_scores = self._residualize_cca_scores(cca_scores, ct_views)

        adata.obsm["X_dialogue_cca"] = self._broadcast_per_celltype(adata, cca_scores, ct_views, n_cols=self.n_programs)

        program_celltypes = self._program_celltypes(empirical_p, celltypes)
        program_signatures = self._initial_program_signatures(ct_views, cca_scores, ws_dict)

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

    # ------------------------------------------------------------------
    # fit_programs implementation helpers
    # ------------------------------------------------------------------

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
            pcs = sub.obsm[self.feature_space_key][:, : self.n_components]
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

        Mirrors R's filter: the ANOVA is run on the per-cell PCA values (not the sample-level pseudobulks), restricting to samples with ``>= min_cells_per_sample`` cells, and BH-adjusted.
        """
        out: dict[str, pd.DataFrame] = {}
        for ct, pb in pseudobulks_full.items():
            view = ct_views[ct]
            pcs = view.obsm[self.feature_space_key][:, : self.n_components]
            samples = view.obs[self.sample_key].astype(str).to_numpy()
            counts = pd.Series(samples).value_counts()
            abundant = counts[counts >= self.min_cells_per_sample].index
            row_mask = np.isin(samples, abundant.to_numpy())
            if row_mask.sum() == 0 or np.unique(samples[row_mask]).size < 2:
                out[ct] = pb
                continue
            mask = _anova_filter_features(pcs[row_mask], samples[row_mask], alpha=self.anova_alpha)
            if mask.sum() < self.n_programs:
                # R also keeps the original components when too few pass; we propagate that.
                out[ct] = pb
                continue
            out[ct] = pb.iloc[:, mask]
        return out

    def _shared_samples(self, pseudobulks: dict[str, pd.DataFrame]) -> list[str]:
        shared = set.intersection(*[set(pb.index) for pb in pseudobulks.values()])
        if len(shared) < 5:
            raise ValueError(f"Only {len(shared)} samples are present in all cell types; DIALOGUE needs at least 5.")
        return sorted(shared)

    def _center(self, pseudobulk: pd.DataFrame) -> pd.DataFrame:
        scaled = _center_scale_winsorize(pseudobulk.to_numpy(), cap=self.winsorize_quantile)
        return pd.DataFrame(scaled, index=pseudobulk.index, columns=pseudobulk.columns)

    def _fit_pmd(self, matrices: list[np.ndarray]) -> list[np.ndarray]:
        n_samples = matrices[0].shape[0]
        penalties = multicca_permute(
            matrices,
            penalties=float(np.sqrt(n_samples) / 2.0),
            nperms=10,
            niter=50,
            standardize=True,
        )["bestpenalties"]
        weights, _ = multicca_pmd(
            matrices,
            penalties,
            K=self.n_programs,
            standardize=True,
            niter=100,
            mimic_R=True,
        )
        return weights

    def _empirical_pmd_pvalues(self, matrices: list[np.ndarray], celltypes: list[str]) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        baseline = self._pmd_pair_correlations(matrices, celltypes)
        pair_names = baseline.columns.tolist()
        better = np.zeros((self.n_programs, len(pair_names)), dtype=np.float64)
        for _ in range(self.n_permutations):
            permuted = [_column_shuffle(m, rng) for m in matrices]
            try:
                perm_cor = self._pmd_pair_correlations(permuted, celltypes)
            except Exception:  # noqa: BLE001 - degenerate permutation; treat as exceeded
                better += 1.0
                continue
            better += (np.abs(perm_cor.to_numpy()) >= np.abs(baseline.to_numpy())).astype(np.float64)
        empirical = (better + 1.0) / (self.n_permutations + 1.0)
        index = [f"MCP{i + 1}" for i in range(self.n_programs)]
        return pd.DataFrame(empirical, index=index, columns=pair_names)

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
            conf = ct_views[ct].obs[self.cell_quality_key].to_numpy(dtype=np.float64)
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
        pair_to_celltypes = {
            f"{celltypes[i]}_{celltypes[j]}": (celltypes[i], celltypes[j]) for i, j in _pair_indices(len(celltypes))
        }
        out: dict[str, list[str]] = {}
        for program in empirical_p.index:
            members: set[str] = set()
            for col, value in empirical_p.loc[program].items():
                if value < self.empirical_alpha:
                    a, b = pair_to_celltypes[col]
                    members.update({a, b})
            out[program] = sorted(members) if members else []
        return out

    def _initial_program_signatures(
        self,
        ct_views: dict[str, AnnData],
        cca_scores: dict[str, np.ndarray],
        ws_dict: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, dict[str, list[str]]]]:
        out: dict[str, dict[str, dict[str, list[str]]]] = {f"MCP{i + 1}": {} for i in range(self.n_programs)}
        for ct, scores in cca_scores.items():
            view = ct_views[ct]
            X = view.X.toarray() if sp.issparse(view.X) else np.asarray(view.X)
            cellQ = view.obs[self.cell_quality_key].to_numpy(dtype=np.float64)
            R, P = _partial_spearman(X, scores, cellQ)
            for program_idx in range(scores.shape[1]):
                program_name = f"MCP{program_idx + 1}"
                col_R = R[:, program_idx]
                col_P = P[:, program_idx]
                bonferroni = 0.05 / max(X.shape[1], 1)
                ranked = np.argsort(-np.abs(col_R))
                up: list[str] = []
                down: list[str] = []
                for gene_idx in ranked:
                    if len(up) + len(down) >= self.n_genes_per_signature * 2:
                        break
                    if col_P[gene_idx] > bonferroni:
                        continue
                    name = view.var_names[gene_idx]
                    if col_R[gene_idx] > 0 and len(up) < self.n_genes_per_signature:
                        up.append(name)
                    elif col_R[gene_idx] < 0 and len(down) < self.n_genes_per_signature:
                        down.append(name)
                out[program_name][ct] = {"up": up, "down": down}
        return out

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


# ---------------------------------------------------------------------------
# Module-level helpers used by fit_programs that are not part of the public API
# ---------------------------------------------------------------------------


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
