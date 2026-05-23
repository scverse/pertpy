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
    # statsmodels' mixedlm raises ConvergenceWarning whenever the optimizer doesn't
    # hit a tiny gradient tolerance; that happens routinely on degenerate genes and
    # the recorded pvalue is still usable, so silence the noise in bulk loops.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=Warning)
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

    # ------------------------------------------------------------------
    # test_celltype_pairs
    # ------------------------------------------------------------------

    def test_celltype_pairs(self, adata: AnnData, *, show_progress: bool = False) -> AnnData:
        """For every ordered pair of cell types, fit a hierarchical linear model of one cell type's program score against the partner cell type's pseudobulk expression of candidate genes.

        Phase 2 of DIALOGUE. Builds per-pair, per-program tables of (estimate, pvalue, z-score) for each candidate gene from ``fit_programs``' signatures and prunes them to the top ``n_genes_per_signature`` per direction. Pair-level "shared abundant" samples are those with at least ``min_cells_per_sample`` cells in both cell types of the pair.

        Stores on ``adata.uns["dialogue"]["pair_results"]`` a nested dict:
        ``pair_results[pair_name][program][celltype]`` -> DataFrame with one row per gene tested (in that cell type's signature) and columns ``estimate, pvalue, zscore, up``.
        Also writes refined per-pair signatures at ``pair_results[pair_name][program]["refined_signatures"][celltype] = {"up": [...], "down": [...]}``.

        Args:
            adata: AnnData previously processed by :meth:`fit_programs`.
            show_progress: If True, print one line per pair while running.
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
            if len(shared) < 5:
                if show_progress:
                    print(f"  skip {pair_name}: only {len(shared)} shared abundant samples")
                pair_results[pair_name] = {}
                continue

            ct1_cells = ct_views[ct1].obs[self.sample_key].astype(str).isin(shared).to_numpy()
            ct2_cells = ct_views[ct2].obs[self.sample_key].astype(str).isin(shared).to_numpy()
            ct1_scores = cca_scores[ct1][ct1_cells]
            ct2_scores = cca_scores[ct2][ct2_cells]
            ct1_samples = ct_views[ct1].obs[self.sample_key].astype(str).to_numpy()[ct1_cells]
            ct2_samples = ct_views[ct2].obs[self.sample_key].astype(str).to_numpy()[ct2_cells]
            ct1_quality = ct_views[ct1].obs[self.cell_quality_key].to_numpy(dtype=np.float64)[ct1_cells]
            ct2_quality = ct_views[ct2].obs[self.cell_quality_key].to_numpy(dtype=np.float64)[ct2_cells]
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
                ct2_tme_for_ct1_genes = gene_pseudobulks[ct1].loc[ct2_samples, ct1_genes_to_test].to_numpy()
                df_ct1 = self._hlm_block(
                    ct2_scores[:, program_idx],
                    ct2_tme_for_ct1_genes,
                    ct1_genes_to_test,
                    sig1_up,
                    ct2_quality,
                    ct2_tme_qc,
                    ct2_samples,
                )

                # ct1's program score vs ct2's pseudobulk expression at ct1's cells (R's p2).
                ct1_tme_for_ct2_genes = gene_pseudobulks[ct2].loc[ct1_samples, ct2_genes_to_test].to_numpy()
                df_ct2 = self._hlm_block(
                    ct1_scores[:, program_idx],
                    ct1_tme_for_ct2_genes,
                    ct2_genes_to_test,
                    sig2_up,
                    ct1_quality,
                    ct1_tme_qc,
                    ct1_samples,
                )

                refined_ct1 = self._top_by_zscore(df_ct1, n=self.n_genes_per_signature)
                refined_ct2 = self._top_by_zscore(df_ct2, n=self.n_genes_per_signature)
                pair_results[pair_name][program] = {
                    ct1: df_ct1,
                    ct2: df_ct2,
                    "refined_signatures": {ct1: refined_ct1, ct2: refined_ct2},
                }

        state["pair_results"] = pair_results
        state["gene_pseudobulks"] = {ct: gene_pseudobulks[ct] for ct in celltypes}
        state["per_sample_quality"] = {ct: per_sample_quality[ct] for ct in celltypes}
        return adata

    # ------------------------------------------------------------------
    # test_celltype_pairs implementation helpers
    # ------------------------------------------------------------------

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
            quality = view.obs[self.cell_quality_key].to_numpy(dtype=np.float64)
            out[ct] = pd.Series(quality).groupby(samples).mean().rename("qcAv")
        return out

    def _shared_abundant_samples(self, ct_views: dict[str, AnnData], ct1: str, ct2: str) -> list[str]:
        def _abundant(ct: str) -> set[str]:
            samples = ct_views[ct].obs[self.sample_key].astype(str)
            counts = samples.value_counts()
            return set(counts[counts >= self.min_cells_per_sample].index)

        return sorted(_abundant(ct1) & _abundant(ct2))

    def _intersect_genes(self, candidate: list[str], present: pd.Index) -> list[str]:
        present_set = set(present)
        return [g for g in candidate if g in present_set]

    def _hlm_block(
        self,
        score: np.ndarray,
        expression: np.ndarray,
        gene_names: list[str],
        up_set: list[str],
        cell_quality: np.ndarray,
        tme_qc: np.ndarray,
        sample_groups: np.ndarray,
    ) -> pd.DataFrame:
        if len(gene_names) == 0:
            return pd.DataFrame(columns=["estimate", "pvalue", "zscore", "up"])
        covariate_dict = {self.cell_quality_key: cell_quality}
        if self.use_tme_qc:
            covariate_dict["tme_qc"] = tme_qc
        for col in self.additional_covariates:
            covariate_dict[col] = np.zeros_like(
                cell_quality
            )  # placeholder; user-provided covariate handling reserved for run()
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
        up_lookup = set(up_set)
        res["up"] = [g in up_lookup for g in res.index]
        return res

    def _top_by_zscore(self, df: pd.DataFrame, *, n: int) -> dict[str, list[str]]:
        if df.empty:
            return {"up": [], "down": []}
        finite = df.dropna(subset=["zscore"])
        up_candidates = finite.loc[finite["zscore"] > 0].sort_values("zscore", ascending=False)
        down_candidates = finite.loc[finite["zscore"] < 0].sort_values("zscore", ascending=True)
        return {
            "up": up_candidates.head(n).index.tolist(),
            "down": down_candidates.head(n).index.tolist(),
        }

    # ------------------------------------------------------------------
    # refine_scores
    # ------------------------------------------------------------------

    def refine_scores(self, adata: AnnData) -> AnnData:
        """Aggregate per-pair HLM evidence and fit final per-cell program scores via iterative non-negative least squares.

        Phase 3 of DIALOGUE. For every cell type, gather the per-gene z-scores produced by :meth:`test_celltype_pairs` across every pair the cell type appears in, BH-adjust within each program × direction, Fisher-combine across pairs, then run iterative NNLS to fit per-cell program scores against the resulting candidate gene set (sign-flipping down-regulated columns). The fitted scores are residualized on the cell-quality confounder and written back to ``adata.obsm["X_dialogue"]``.

        Stores on ``adata.uns["dialogue"]``:

        - ``gene_pvalues[celltype]`` — combined gene table with per-pair z-scores, Fisher-combined ``p_up``/``p_down``, support counts ``n_up``/``n_down``, fractions ``nf_up``/``nf_down``, program label, ``up`` direction, and the fitted ``coef`` from NNLS.
        - ``program_gene_signatures[program][celltype] = {"up": [...], "down": [...]}`` — refined gene signatures (R's ``sig1`` from ``DLG.find.scoring``).
        - ``program_gene_signatures_strict[program][celltype] = {"up": [...], "down": [...]}`` — stricter set (R's ``sig2``).
        - ``pair_refined_correlations[pair][program]`` — per-pair sample-average correlation R of the refined scores plus the HLM p-value for the same pair.

        Updates ``adata.obsm["X_dialogue"]`` with the refined per-cell program scores.
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
            X_dense = view.X.toarray() if sp.issparse(view.X) else np.asarray(view.X)
            cellQ = view.obs[self.cell_quality_key].to_numpy(dtype=np.float64)

            # Standardize expression
            zscored = self._zscore_expression(X_dense)
            # Compute initial CCA scores (unresidualized) for NNLS targets
            cca0 = self._cca_scores_unresidualized(view, state, ct)

            program_columns = [f"MCP{p + 1}" for p in range(self.n_programs)]
            ct_scores = np.zeros((view.n_obs, self.n_programs))
            gene_pval = gene_pvalues[ct]
            gene_pval["coef"] = 0.0

            for program_idx, program in enumerate(program_columns):
                y_target = cca0[:, program_idx]
                program_rows = gene_pval[gene_pval["program"] == program]
                if program_rows.empty:
                    ct_scores[:, program_idx] = y_target
                    continue
                gene_names = program_rows["gene"].to_numpy()
                gene_indices = self._gene_indices(view.var_names, gene_names)
                X_program = zscored[:, gene_indices].copy()
                down_mask = ~program_rows["up"].to_numpy(dtype=bool)
                X_program[:, down_mask] *= -1.0
                ranks = program_rows["Nf"].to_numpy(dtype=np.float64)
                coefs = _iterative_nnls(X_program, y_target, ranks)
                ct_scores[:, program_idx] = X_program @ coefs
                gene_pval.loc[program_rows.index, "coef"] = coefs

            nnls_scores[ct] = ct_scores
            # Refined signatures
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

        # Residualize on confounders
        for ct in celltypes:
            view = ct_views[ct]
            cellQ = view.obs[self.cell_quality_key].to_numpy(dtype=np.float64)
            nnls_scores[ct] = _residualize(nnls_scores[ct], cellQ)

        adata.obsm["X_dialogue"] = self._broadcast_per_celltype(adata, nnls_scores, ct_views, n_cols=self.n_programs)
        for p in range(self.n_programs):
            adata.obs[f"mcp_{p}"] = adata.obsm["X_dialogue"][:, p]

        # Pair-level refined correlations on sample-averaged refined scores
        pair_refined = self._refined_pair_correlations(adata, nnls_scores, ct_views, celltypes)

        state["gene_pvalues"] = gene_pvalues
        state["program_gene_signatures"] = refined_signatures
        state["program_gene_signatures_strict"] = strict_signatures
        state["pair_refined_correlations"] = pair_refined
        return adata

    # ------------------------------------------------------------------
    # refine_scores implementation helpers
    # ------------------------------------------------------------------

    def _aggregate_gene_pvalues_for_celltype(
        self,
        pair_results: dict[str, dict[str, dict[str, object]]],
        celltypes: list[str],
        ct: str,
    ) -> pd.DataFrame:
        """For a given cell type, build R's per-program-x-gene gene_pval DataFrame from pair results."""
        gene_records: dict[tuple[str, str, bool], dict[str, float]] = {}
        partner_cols: list[str] = []
        for pair_name, programs in pair_results.items():
            ct1, ct2 = self._pair_split(pair_name, celltypes)
            if ct not in (ct1, ct2):
                continue
            partner = ct2 if ct == ct1 else ct1
            colname = f"{partner}"
            if colname not in partner_cols:
                partner_cols.append(colname)
            for program, info in programs.items():
                df = info.get(ct)
                if df is None or df.empty:
                    continue
                for gene_name, row in df.iterrows():
                    if not np.isfinite(row["zscore"]):
                        continue
                    key = (program, gene_name, bool(row["up"]))
                    gene_records.setdefault(key, {})[colname] = float(row["zscore"])

        if not gene_records:
            return pd.DataFrame(
                columns=[
                    "gene",
                    "program",
                    "up",
                    "programF",
                    "p_up",
                    "p_down",
                    "n_up",
                    "nf_up",
                    "n_down",
                    "nf_down",
                    "N",
                    "Nf",
                ]
            )

        records = []
        for (program, gene, up), partners in gene_records.items():
            row = {"program": program, "gene": gene, "up": up, "programF": f"{program}.{'up' if up else 'down'}"}
            row.update({col: partners.get(col, np.nan) for col in partner_cols})
            records.append(row)
        df = pd.DataFrame(records)
        df.index = [f"{r['programF']}_{r['gene']}" for _, r in df.iterrows()]

        z = df[partner_cols].to_numpy()
        # Two-sided p from z, then BH-adjust within (program, direction) and Fisher-combine.
        p_up_partner = self._adjust_per_label(self._pvals_from_zscores(z), df["programF"].to_numpy())
        p_down_partner = self._adjust_per_label(self._pvals_from_zscores(-z), df["programF"].to_numpy())
        df["p_up"] = self._fisher_per_row(p_up_partner)
        df["p_down"] = self._fisher_per_row(p_down_partner)
        df["n_up"] = (p_up_partner < self.empirical_alpha).sum(axis=1)
        df["nf_up"] = (p_up_partner < self.empirical_alpha).mean(axis=1)
        df["n_down"] = (p_down_partner < self.empirical_alpha).sum(axis=1)
        df["nf_down"] = (p_down_partner < self.empirical_alpha).mean(axis=1)
        df["N"] = np.where(df["up"], df["n_up"], df["n_down"])
        df["Nf"] = np.where(df["up"], df["nf_up"], df["nf_down"])
        # Override p_up/p_down: when the gene is "up" we keep p_up; otherwise the down side carries the signal.
        df.loc[~df["up"].astype(bool), "p_up"] = 1.0
        df.loc[df["up"].astype(bool), "p_down"] = 1.0
        return df.reset_index(drop=True)

    def _pair_split(self, pair_name: str, celltypes: list[str]) -> tuple[str, str]:
        for i, j in _pair_indices(len(celltypes)):
            if f"{celltypes[i]}_{celltypes[j]}" == pair_name:
                return celltypes[i], celltypes[j]
        raise KeyError(pair_name)

    def _adjust_per_label(self, pvalues: np.ndarray, labels: np.ndarray) -> np.ndarray:
        adjusted = np.full_like(pvalues, np.nan)
        for label in np.unique(labels):
            mask = labels == label
            block = pvalues[mask]
            for j in range(block.shape[1]):
                col = block[:, j]
                valid = np.isfinite(col)
                if valid.sum() < 1:
                    continue
                adj = multipletests(col[valid], method="fdr_bh")[1]
                column_full = np.full_like(col, np.nan)
                column_full[valid] = adj
                block[:, j] = column_full
            adjusted[mask] = block
        return adjusted

    def _fisher_per_row(self, pvalues: np.ndarray) -> np.ndarray:
        out = np.full(pvalues.shape[0], 1.0)
        for i, row in enumerate(pvalues):
            finite = row[np.isfinite(row) & (row > 0)]
            if finite.size == 0:
                continue
            stat = -2.0 * np.log(finite).sum()
            out[i] = float(stats.chi2.sf(stat, df=2 * finite.size))
        return out

    @staticmethod
    def _pvals_from_zscores(z: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(z), 2.0 * stats.norm.sf(np.abs(z)), np.nan)

    def _split_up_down(self, rows: pd.DataFrame) -> dict[str, list[str]]:
        if rows.empty:
            return {"up": [], "down": []}
        up = rows.loc[rows["up"].astype(bool), "gene"].tolist()
        down = rows.loc[~rows["up"].astype(bool), "gene"].tolist()
        return {"up": up, "down": down}

    @staticmethod
    def _zscore_expression(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, ddof=1, keepdims=True)
        std = np.where(std > 0, std, 1.0)
        return (arr - mean) / std

    def _cca_scores_unresidualized(self, view: AnnData, state: dict, ct: str) -> np.ndarray:
        W = state["weights"][ct]
        idx_names = state["weights_index"][ct]
        kept = [int(name[2:]) - 1 for name in idx_names]
        pcs = view.obsm[self.feature_space_key][:, : self.n_components][:, kept]
        return np.asarray(pcs, dtype=np.float64) @ W

    @staticmethod
    def _gene_indices(var_names: pd.Index, genes: np.ndarray) -> np.ndarray:
        lookup = {g: i for i, g in enumerate(var_names)}
        return np.array([lookup[g] for g in genes], dtype=np.int64)

    def _refined_pair_correlations(
        self,
        adata: AnnData,
        nnls_scores: dict[str, np.ndarray],
        ct_views: dict[str, AnnData],
        celltypes: list[str],
    ) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        sample_avg = {
            ct: pd.DataFrame(nnls_scores[ct], index=ct_views[ct].obs[self.sample_key].astype(str).to_numpy())
            .groupby(level=0)
            .median()
            for ct in celltypes
        }
        for i, j in _pair_indices(len(celltypes)):
            ct1, ct2 = celltypes[i], celltypes[j]
            shared = sorted(set(sample_avg[ct1].index) & set(sample_avg[ct2].index))
            if len(shared) < 3:
                continue
            a = sample_avg[ct1].loc[shared].to_numpy()
            b = sample_avg[ct2].loc[shared].to_numpy()
            pair_name = f"{ct1}_{ct2}"
            out[pair_name] = {}
            for p in range(self.n_programs):
                ap = a[:, p]
                bp = b[:, p]
                denom = (ap.std(ddof=1) * bp.std(ddof=1)) or 1.0
                r = float(np.cov(ap, bp, ddof=1)[0, 1] / denom)
                out[pair_name][f"MCP{p + 1}"] = {"R": r}
        return out

    # ------------------------------------------------------------------
    # End-user convenience entry points
    # ------------------------------------------------------------------

    def run(self, adata: AnnData) -> AnnData:
        """Run all three DIALOGUE phases in order on ``adata`` (in-place)."""
        self.fit_programs(adata)
        self.test_celltype_pairs(adata)
        self.refine_scores(adata)
        return adata

    def test_phenotype_association(
        self,
        adata: AnnData,
        condition_key: str,
        *,
        conditions: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """Test each program's association with a binary phenotype using per-celltype hierarchical models.

        For every (program, cell type), fits ``score ~ phenotype + cell_quality + (1 | sample)`` on the cells of that cell type, where ``phenotype`` is a binary indicator coded from ``adata.obs[condition_key]``. Returns a DataFrame of signed z-scores (rows = cell types, columns = programs) plus a Fisher-combined p-value column across cell types per program.

        Args:
            adata: AnnData after :meth:`run` / :meth:`refine_scores`.
            condition_key: ``adata.obs`` column with the phenotype labels (categorical with exactly two levels, or pass ``conditions`` to pick which two to compare).
            conditions: Optional two-element tuple selecting which two values of ``adata.obs[condition_key]`` are compared.

        Returns:
            ``zscores`` DataFrame (rows: cell types, columns: programs).
            The combined p-values are stored on ``adata.uns["dialogue"]["phenotype_pvalues"]``.
        """
        if "dialogue" not in adata.uns:
            raise RuntimeError("Run fit_programs/refine_scores before test_phenotype_association.")
        if "X_dialogue" not in adata.obsm:
            raise RuntimeError("Refined scores missing; run refine_scores(adata) first.")
        state = adata.uns["dialogue"]
        celltypes = state["cell_type_order"]
        if conditions is None:
            labels = pd.Series(adata.obs[condition_key]).astype("category").cat.categories.tolist()
            if len(labels) != 2:
                raise ValueError(
                    f"adata.obs[{condition_key!r}] has {len(labels)} levels; pass `conditions` to pick two."
                )
            conditions = (labels[0], labels[1])
        scores = adata.obsm["X_dialogue"]
        obs = adata.obs
        program_cols = [f"MCP{p + 1}" for p in range(self.n_programs)]
        z_table = pd.DataFrame(np.nan, index=celltypes, columns=program_cols)
        p_table = pd.DataFrame(np.nan, index=celltypes, columns=program_cols)
        for ct in celltypes:
            mask = (obs[self.celltype_key] == ct).to_numpy()
            sub_scores = scores[mask]
            sub_obs = obs.loc[mask]
            condition = sub_obs[condition_key].astype(str).to_numpy()
            keep = np.isin(condition, list(conditions))
            if keep.sum() < 5:
                continue
            x = (condition[keep] == conditions[1]).astype(float)
            covariates = pd.DataFrame({self.cell_quality_key: sub_obs[self.cell_quality_key].to_numpy()[keep]})
            sample_groups = sub_obs[self.sample_key].astype(str).to_numpy()[keep]
            for program_idx, program in enumerate(program_cols):
                y = sub_scores[keep, program_idx]
                if not np.isfinite(y).any():
                    continue
                df_one = _hlm_pvalue_per_row(
                    np.asarray(x[None, :]),
                    y,
                    covariates,
                    sample_groups,
                )
                est = float(df_one["estimate"].iloc[0])
                pval = float(df_one["pvalue"].iloc[0])
                z_table.loc[ct, program] = float(_zscores_from_signed_pvalues(np.array([est]), np.array([pval]))[0])
                p_table.loc[ct, program] = pval
        # Combine across cell types for each program (the rows are cell types so we transpose).
        combined = self._fisher_per_row(p_table.to_numpy().T)
        state["phenotype_pvalues"] = pd.DataFrame({"combined_p": combined}, index=program_cols)
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
            celltype: If given, return only that cell type's signature; otherwise return the cross-celltype intersection of consistently up/down genes.
            strict: Use the strict variant from ``program_gene_signatures_strict`` (genes flagged in every pair).
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
            fraction: Fraction of cells at each tail to compare; must lie in ``(0, 0.5)``.
        """
        if "X_dialogue" not in adata.obsm:
            raise RuntimeError("Run refine_scores(adata) first.")
        if not 0 < fraction < 0.5:
            raise ValueError("fraction must be in (0, 0.5)")
        idx = int(program.replace("MCP", "")) - 1
        scores = adata.obsm["X_dialogue"][:, idx]
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
