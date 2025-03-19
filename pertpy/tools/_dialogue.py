from __future__ import annotations

import itertools
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as ssm
from lamin_utils import logger
from pandas import DataFrame
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from scipy import stats
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr, t
from seaborn import PairGrid
from sklearn.linear_model import LinearRegression
from sparsecca import lp_pmd, multicca_permute, multicca_pmd
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multitest import fdrcorrection

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Dialogue:
    """Python implementation of DIALOGUE"""

    def __init__(
        self,
        sample_id: str,
        celltype_key: str,
        n_counts_key: str,
        n_mpcs: int,
        conf: str | None = None,
        cov: list | None = None,
    ):
        """Constructor for Dialogue.

        Args:
            sample_id: The sample ID key in AnnData.obs which is used for pseudobulk determination.
            celltype_key: The key in AnnData.obs which contains the cell type column.
            n_counts_key: The key of the number of counts in Anndata.obs . Also commonly the size factor.
            n_mpcs: Number of PMD components which corresponds to the number of determined MCPs.
        """
        self.sample_id = sample_id
        self.celltype_key = celltype_key
        if " " in n_counts_key:
            raise ValueError(
                "Patsy, which we use for formulas, does not allow for spaces in this key.\n"
                "Please replace spaces with underscores and ensure that the key is in your object."
            )
        self.n_counts_key = n_counts_key
        self.n_mcps = n_mpcs
        self.conf = conf
        self.cov = cov

    def _pseudobulk(
        self,
        adata: AnnData,
        groupby: str,
        mode: str = "counts",
        strategy: str = "mean",
        n_components: int = 30,
        agg_func=np.median,
    ) -> pd.DataFrame:
        """
        Aggregate cell-level data into pseudobulk values based on a grouping variable.

        In "counts" mode, the function aggregates raw count/expression data from `adata.X` by grouping
        cells by the specified key and computing either the median or mean for each gene. In any other mode,
        the function assumes that `mode` is a key in `adata.obsm` representing a feature space (e.g. "X_pca")
        and aggregates the first `n_components` features using the provided aggregation function.

        Args:
            adata (AnnData): The AnnData object containing cell-level data.
            groupby (str): The key in `adata.obs` used for grouping (e.g. a sample or cell type identifier).
            mode (str, optional): Aggregation mode. If "counts", aggregate raw counts from `adata.X`;
                otherwise, treat mode as the key in `adata.obsm` to aggregate. Defaults to "counts".
            strategy (str, optional): For counts mode, either "median" or "mean". Defaults to "mean".
            n_components (int, optional): Number of components to aggregate in feature mode. Defaults to 30.
            agg_func (Callable, optional): Aggregation function for feature mode (e.g. np.median or np.mean).
                Defaults to np.median.

        Returns:
            pd.DataFrame: The pseudobulk aggregated data.
                - In "counts" mode, the DataFrame has genes as rows (indexed by `adata.var_names`)
                and groups as columns.
                - In feature mode, the DataFrame has groups as rows and aggregated features as columns.

        Raises:
            ValueError: If an invalid aggregation strategy is provided in counts mode or if mode is unrecognized.
        """
        if mode == "counts":
            pseudobulk = {"Genes": adata.var_names.values}
            for category in adata.obs[groupby].cat.categories:
                temp = adata.obs[groupby] == category
                if strategy == "median":
                    pseudobulk[category] = np.median(adata[temp].X, axis=0)
                elif strategy == "mean":
                    pseudobulk[category] = adata[temp].X.mean(axis=0)
                else:
                    raise ValueError("strategy must be either 'median' or 'mean'")
            return pd.DataFrame(pseudobulk).set_index("Genes")
        else:
            # Assume mode is a key in adata.obsm representing the feature space.
            aggr = {}
            for category in adata.obs[groupby].cat.categories:
                temp = adata.obs[groupby] == category
                # Aggregate the first n_components from the specified feature space.
                aggr[category] = agg_func(adata[temp].obsm[mode][:, :n_components], axis=0)
            aggr_df = pd.DataFrame(aggr)
            # Transpose so that rows correspond to groups and columns to features.
            return aggr_df.T

    def _scale_data(self, pseudobulks: pd.DataFrame, normalize: bool = True, cap: float = 0.01) -> np.ndarray:
        """Row-wise mean center and scale by the standard deviation,
        and then cap extreme values based on quantiles.

        This mimics the following R function (excluding row subsetting):

            f <- function(X1){
            if(param$center.flag){
                X1 <- center.matrix(X1, dim = 2, sd.flag = TRUE)
                X1 <- cap.mat(X1, cap = 0.01, MARGIN = 2)
            }
            X1 <- X1[samplesU, ]
            return(X1)
            }

        Args:
            pseudobulks: The pseudobulk PCA components as a DataFrame (samples as rows, features as columns).
            normalize: Whether to perform centering, scaling, and capping.
            cap: The quantile threshold for capping. For example, cap=0.01 means that for each column, values
                above the 99th percentile are set to the 99th percentile, and values below the 1st percentile are set to the 1st percentile.

        Returns:
            The processed (scaled and capped) matrix as a NumPy array.
        """
        if normalize:
            # Center and scale (column-wise: subtract the column mean and divide by the column std)
            scaled = (pseudobulks - pseudobulks.mean()) / pseudobulks.std()

            # Apply quantile-based capping column-wise.
            capped = scaled.copy()
            for col in scaled.columns:
                lower = scaled[col].quantile(cap)  # lower quantile (e.g., 1st percentile)
                upper = scaled[col].quantile(1 - cap)  # upper quantile (e.g., 99th percentile)
                capped[col] = scaled[col].clip(lower=lower, upper=upper)

            return capped.to_numpy()
        else:
            return pseudobulks.to_numpy()

    def _get_abundant_elements_from_series(self, series: pd.Series, min_count: int = 2) -> list[str]:
        """Returns a list from `elements` that occur more than `min_count` times.

        Args:
            series: To extract the top most frequent elements included in the final output
                      (i.e. the index in the computed frequency table) from
            min_count: Threshold specifying the minimum element count for an element in the frequency table (inclusive)

        Returns:
            A list of elements that occur more than `min_count` times.
        """
        frequency = series.value_counts()
        abundant_elements = frequency[frequency >= min_count].index.tolist()

        return abundant_elements

    def _get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Mimics DIALOGUE.get.residuals.

        Args:
            X: Covariate matrix, shape (n_samples, n_features). For example, adata.obsm['pseudobulk_feature_space'].
            y: Response matrix, shape (n_response, n_samples). Typically, each row corresponds to a MCP.

        Returns:
            Array of residuals with the same shape as y.
        """
        resid = []
        # Add constant (intercept) to the covariate matrix
        X_const = sm.add_constant(X)
        for y_sub in y:
            model = sm.OLS(y_sub, X_const).fit()
            resid.append(model.resid)
        return np.array(resid)

    # def _iterative_nnls(self, A_orig: np.ndarray, y_orig: np.ndarray, feature_ranks: list[int], n_iter: int = 1000):
    #     """Solves non-negative least-squares separately for different feature categories.

    #     Mimics DLG.iterative.nnls.
    #     Variables are notated according to:

    #         `argmin|Ax - y|`

    #     Returns:
    #         Returns the aggregated coefficients from nnls.
    #     """
    #     # TODO: Consider moving this internally to cca_sig
    #     y = y_orig.copy()

    #     sig_ranks = sorted(set(feature_ranks), reverse=True)
    #     sig_ranks = [rank for rank in sig_ranks if rank >= 1 / 3]  # code coverage only with n_mcps > 3
    #     masks = [feature_ranks == r for r in sig_ranks if sum(feature_ranks == r) >= 5]  # type: ignore

    #     # TODO: The few type ignores are dangerous and should be checked! They could be bugs.
    #     insig_mask = feature_ranks < sig_ranks[-1]  # type: ignore # TODO: rename variable after better understanding
    #     if sum(insig_mask) >= 5:  # such as genes with 0 rank, or those below 1/3
    #         masks.append(insig_mask)
    #         sig_ranks.append("insig")  # type: ignore

    #     x_final = np.zeros(A_orig.shape[0])
    #     Ax = np.zeros(A_orig.shape[1])
    #     for _, mask in zip(sig_ranks, masks, strict=False):
    #         A = A_orig[mask].T
    #         coef_nnls, _ = nnls(A, y, maxiter=n_iter)
    #         y = y - A @ coef_nnls  # residuals
    #         Ax += A @ coef_nnls
    #         x_final[mask] = coef_nnls

    #     return x_final

    def _corr2_coeff(self, A, B):
        # Rowwise mean of input arrays & subtract from input arrays themselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    def _load(
        self,
        adata: AnnData,
        ct_order: list[str],
        agg_feature: bool = True,
        normalize: bool = True,
        n_components: int = 30,
        feature_space_key="X_pca",
        subset_common: bool = True,  # new optional parameter
    ) -> tuple[list, dict, dict]:
        """Separates cells into AnnDatas by celltype_key and creates the multifactor PMD input.

        Mimics DIALOGUE's `make.cell.types` and the pre-processing that occurs in DIALOGUE1.

        Args:
            adata: AnnData object to generate celltype objects for.
            ct_order: The order of cell types.
            agg_feature: Whether to aggregate pseudobulks with some embeddings or not.
            normalize: Whether to mimic DIALOGUE behavior or not.
            subset_common: If True, restrict output to common samples across cell types.

        Returns:
            A tuple with:
            - mcca_in: A list of pseudobulk matrices (one per cell type), with rows corresponding to sample IDs (if subset_common is True).
            - ct_subs: A dictionary mapping each cell type to its corresponding AnnData subset (restricted to common samples if subset_common is True).
        """
        # 1. Split the AnnData into cell-type–specific subsets.
        ct_subs = {ct: adata[adata.obs[self.celltype_key] == ct].copy() for ct in ct_order}

        original_ct_subs = ct_subs.copy()

        # 2. Choose the aggregation function based on the flag.
        if agg_feature:
            mode_val = feature_space_key  # e.g., "X_pca"
        else:
            mode_val = "counts"

        ct_aggr = {
            ct: self._pseudobulk(adata, self.sample_id, mode=mode_val, n_components=n_components)
            for ct, adata in ct_subs.items()
        }

        # 4. Apply scaling/normalization to the aggregated data.
        #    We wrap the output back in a DataFrame to preserve the sample IDs.
        ct_scaled = {
            ct: pd.DataFrame(self._scale_data(df, normalize=normalize), index=df.index, columns=df.columns)
            for ct, df in ct_aggr.items()
        }

        if subset_common:
            # 5. Determine the set of common samples across all cell types (using the scaled data).
            common_samples = set(ct_scaled[ct_order[0]].index)
            for ct in ct_order[1:]:
                common_samples = common_samples.intersection(set(ct_scaled[ct].index))
            common_samples_sorted = sorted(common_samples)

            # Check if there are at least 5 common samples.
            if len(common_samples_sorted) < 5:
                raise ValueError("Cannot run DIALOGUE with less than 5 common samples across cell types.")

            # 6. Subset each scaled pseudobulk DataFrame to only the common samples.
            ct_scaled = {ct: df.loc[common_samples_sorted] for ct, df in ct_scaled.items()}

            # 7. Also, restrict each cell-type AnnData to cells belonging to one of the common samples.
            for ct in ct_subs:
                ct_subs[ct] = ct_subs[ct][ct_subs[ct].obs[self.sample_id].isin(common_samples_sorted)].copy()

        # 8. Order the preprocessed pseudobulk matrices as a list in the order specified by ct_order.
        mcca_in = [ct_scaled[ct] for ct in ct_order]

        return mcca_in, ct_subs, original_ct_subs

    def _compute_cca(self, pre_r_scores: dict[str, pd.DataFrame], cell_types: list[str]) -> pd.DataFrame:
        """
        Computes pairwise Pearson correlations between pre-residual scores for the specified cell types.

        Args:
            pre_r_scores: Dictionary mapping cell types to their pre-residual score DataFrames.
            cell_types: List of cell type names.

        Returns:
            A DataFrame of pairwise Pearson correlations (rows: MCPs, columns: cell type pairs).
        """
        pairs = list(itertools.combinations(cell_types, 2))
        cca_cor = {}
        for ct1, ct2 in pairs:
            common_samples = pre_r_scores[ct1].index.intersection(pre_r_scores[ct2].index)
            df1 = pre_r_scores[ct1].loc[common_samples]
            df2 = pre_r_scores[ct2].loc[common_samples]
            cor_vals = []
            for col in df1.columns:
                r_val, _ = pearsonr(df1[col].values, df2[col].values)
                cor_vals.append(r_val)
            key = f"{ct1}_{ct2}"
            cca_cor[key] = np.array(cor_vals)
        return pd.DataFrame(cca_cor, index=df1.columns)

    def _partial_corr_test(self, x1: np.ndarray, x2: np.ndarray, cov: np.ndarray, method="spearman"):
        """
        Mimics 'pcor.test(x1, x2, cov, method=...)' from R's 'ppcor' package
        in a simplified manner (single covariate, pairwise-complete, using a
        t-based formula for significance).

        Returns (estimate, p_value).
        """
        # 1) Drop any rows that have NaNs in x1, x2, or cov if we mimic pairwise.complete.obs.
        mask = (~np.isnan(x1)) & (~np.isnan(x2)) & (~np.isnan(cov))
        x1_sub = x1[mask]
        x2_sub = x2[mask]
        z_sub = cov[mask]
        n_obs = len(x1_sub)

        # If insufficient data, return NA
        if n_obs < 3:
            return (np.nan, np.nan)

        # 2) Compute the Spearman correlations r_xy, r_xz, r_yz
        if method == "spearman":
            r_xy, _ = spearmanr(x1_sub, x2_sub, nan_policy="omit")
            r_xz, _ = spearmanr(x1_sub, z_sub, nan_policy="omit")
            r_yz, _ = spearmanr(x2_sub, z_sub, nan_policy="omit")
        else:
            raise ValueError("Only spearman is implemented in this example.")

        # 3) Partial correlation formula for a single covariate
        #    r_{x,y|z} = (r_xy - r_xz*r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1.0 - r_xz**2) * (1.0 - r_yz**2))
        # if denominator < 1e-12:
        # Degenerate or collinear => partial correlation is undefined
        #    estimate = np.nan
        # else:
        estimate = numerator / denominator

        # 4) Approximate p-value using t distribution with df = n - 3
        #    (like ppcor would do for a single covariate).
        #    t = r * sqrt( (n - 2 - k) / (1 - r^2 ) ) for partial correlation with k=1.
        #    df = n - k - 2 => n - 3
        if np.isnan(estimate) or abs(estimate) >= 1.0 or n_obs < 4:
            p_val = np.nan
        else:
            df = n_obs - 3
            # t-value
            t_stat = estimate * np.sqrt((df) / (1.0 - estimate**2))
            # two-sided p-value
            p_val = 2.0 * (1.0 - t.cdf(abs(t_stat), df=df))

        return (estimate, p_val)

    def _p_adjust_bh(self, p_vals: np.ndarray) -> np.ndarray:
        """
        Benjamini-Hochberg p-value adjustment for a 2D array of p-values.
        We'll flatten them, adjust, then reshape. This is one approach
        (global BH across all entries).
        """
        shape_orig = p_vals.shape
        flat = p_vals.ravel()
        n = len(flat)

        # Sort p-values, compute rank, etc.
        order = np.argsort(flat)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)

        adjusted = flat * (n / ranks)
        adjusted[adjusted > 1.0] = 1.0

        # Ensure monotonic
        # We go from largest index to smallest
        for i in range(n - 2, -1, -1):
            if adjusted[i] > adjusted[i + 1]:
                adjusted[i] = adjusted[i + 1]

        # Reshape back
        return adjusted.reshape(shape_orig)

    def _pcor_mat(
        self,
        v1: pd.DataFrame,
        v2: pd.DataFrame,
        v3: np.ndarray | None,
        method: str = "spearman",
        use: str = "pairwise.complete.obs",
    ) -> dict[str, pd.DataFrame]:
        """
        Computes a correlation matrix between columns of v1 and v2, controlling for covariates in v3.
        If v3 is None, computes ordinary correlation (Spearman or Pearson) without adjusting for covariates.

        Args:
            v1 (pd.DataFrame): DataFrame of shape (n_samples, n_genes) or (n_samples, features).
            v2 (pd.DataFrame): DataFrame of shape (n_samples, n_mcps).
            v3 (np.ndarray | None): Covariate array of shape (n_samples,). If None, no covariate adjustment is performed.
            method (str, optional): Correlation method to use ("spearman" or "pearson"). Defaults to "spearman".
            use (str, optional): How to handle missing data. Defaults to "pairwise.complete.obs".

        Returns:
            dict[str, pd.DataFrame]: A dictionary with the following keys:
                - "R": DataFrame of correlation estimates (shape: genes x MCPs).
                - "P": DataFrame of p-values (shape: genes x MCPs).
                - "padj": DataFrame of adjusted p-values using Benjamini-Hochberg (shape: genes x MCPs).
        """
        genes = v1.columns
        mcps = v2.columns
        X = v1.values  # shape (n_samples, n_genes)
        Y = v2.values  # shape (n_samples, n_mcps)

        R = np.zeros((len(genes), len(mcps)), dtype=float)
        P = np.zeros((len(genes), len(mcps)), dtype=float)

        for j, _mcp_name in enumerate(mcps):
            x2 = Y[:, j]
            for i, _gene_name in enumerate(genes):
                x1 = X[:, i]
                if v3 is None:
                    if method.lower() == "spearman":
                        from scipy.stats import spearmanr

                        est, pval = spearmanr(x1, x2)
                    else:
                        from scipy.stats import pearsonr

                        est, pval = pearsonr(x1, x2)
                else:
                    est, pval = self._partial_corr_test(x1, x2, v3, method=method)
                R[i, j] = est
                P[i, j] = pval

        padj_array = self._p_adjust_bh(P)

        R_df = pd.DataFrame(R, index=genes, columns=mcps)
        P_df = pd.DataFrame(P, index=genes, columns=mcps)
        padj_df = pd.DataFrame(padj_array, index=genes, columns=mcps)

        return {"R": R_df, "P": P_df, "padj": padj_df}

    def _get_top_elements(self, m: pd.DataFrame, q: int = 100, min_ci: float = None, main: str = "") -> dict:
        """
        Python version of R's get.top.elements(m, q=100, min.ci=NULL, main="").

        Args:
            m (pd.DataFrame): shape (genes, columns). row index = gene names,
                            columns = something like MCP1.up, MCP2.down, etc.
            q (int): top threshold
            min_ci (float or None): used for ci = min(ci, min_ci) if provided
            main (str): prefix to apply to the final column names

        Returns:
            A dictionary mapping column_name -> list_of_genes
            (sorted by rowname) that pass m[:,col] <= ci.

        This matches R's logic:
        top.l <- list()
        v <- rownames(m)
        for(i in 1:ncol(m)){
            mi <- m[,i]; mi <- mi[!is.na(mi)]
            idx <- order(mi,decreasing=F)
            ci <- mi[idx[min(q,length(mi))]]
            ci <- min(ci, min.ci)
            b <- m[,i] <= ci
            b[is.na(m[,i])] <- F
            top.l[[i]] <- sort(v[b])
        }
        names(top.l) <- paste0(main, colnames(m))  # if main !=""
        """
        # Convert to numeric just in case:

        m_numeric = m.apply(pd.to_numeric, errors="coerce")

        top_l: dict[str, list[str]] = {}
        # We'll store rownames as an Index object
        row_names = m_numeric.index

        for i, col_name in enumerate(m_numeric.columns):
            # Extract the column: shape (genes,)
            col_data = m_numeric.iloc[:, i]

            # 1) Drop NAs
            col_no_na = col_data.dropna()

            # If there's no data after dropping, store empty
            if len(col_no_na) == 0:
                top_l[col_name] = []
                continue

            # 2) Sort ascending (like order(..., decreasing=F))
            col_sorted = col_no_na.sort_values(ascending=True)

            # 3) ci <- mi[idx[min(q,length(mi))]]
            pos = min(q, len(col_no_na))  # If length < q, use last
            ci_value = col_sorted.iloc[pos - 1]  # 0-based indexing

            # 4) ci_value <- min(ci_value, min_ci)
            if min_ci is not None:
                ci_value = min(ci_value, min_ci)

            # 5) b <- m[,i] <= ci_value
            # 6) b[is.na(m[,i])] <- F
            # We'll treat NA as inf so it won't pass <= ci_value
            col_filled = col_data.fillna(float("inf"))

            col_filled = col_filled.T

            mask = col_filled <= ci_value

            # 7) sort(v[mask])
            selected_genes = sorted(row_names[mask])

            top_l[col_name] = selected_genes

        # Finally, if main != "", we do:
        #   names(top.l) <- paste0(main, colnames(m))
        # In R, it does: if(main!=""){ main<-paste0(main,".") }
        # We'll replicate that:
        if main != "":
            main_str = main + "." if not main.endswith(".") else main
            renamed_top_l = {}
            for _, col_name in enumerate(m_numeric.columns):
                new_col_name = f"{main_str}{col_name}"
                renamed_top_l[new_col_name] = top_l[col_name]
            return renamed_top_l

        return top_l

    def _get_top_corr(
        self, m: pd.DataFrame, q: int = 100, min_ci: float = 0.0, idx: list = None, add_prefix: str = ""
    ) -> dict:
        """
        Python version of R's get.top.cor(m, q=100, min.ci=0, idx=NULL, add.prefix="").

        Steps:
        1) Convert m to numeric if needed.
        2) If colnames(m) is null in R, we'd assign 1..ncol(m). In Python, we skip if columns exist.
        3) m.pos = -m; m.neg = m
        4) colnames(m.pos) <- paste0(colnames(m.pos), ".up")
            colnames(m.neg) <- paste0(colnames(m.neg), ".down")
        5) combined = cbind(m.pos, m.neg)
        6) v = get_top_elements(combined, q, min_ci=-abs(min_ci))
        7) names(v) <- c(colnames(m.pos), colnames(m.neg))  [we replicate in a Python dict approach]
        8) if idx != None => v = v[paste(idx, c("up","down"), sep=".")]
        9) names(v) <- paste0(add.prefix, names(v))
        10) return v
        """
        # 1) Ensure numeric

        m_numeric = m.apply(pd.to_numeric, errors="coerce")

        # 2) If colnames(m) is null => in Python, we skip, as DataFrame typically has columns

        # 3) Build m.pos, m.neg
        m_pos = -m_numeric
        m_neg = m_numeric.copy()

        # 4) rename columns with .up and .down
        old_cols = list(m_numeric.columns)  # to keep track for final assignment
        m_pos.columns = [f"{c}.up" for c in old_cols]
        m_neg.columns = [f"{c}.down" for c in old_cols]

        # 5) cbind => axis=1
        combined = pd.concat([m_pos, m_neg], axis=1)

        # 6) get_top_elements with min_ci=-abs(min_ci)
        cutoff_val = -abs(min_ci)
        v = self._get_top_elements(combined, q=q, min_ci=cutoff_val)

        # 7) R does `names(v) <- c(colnames(m.pos), colnames(m.neg))`
        # in R, if m.pos had columns [A.up, B.up], and m.neg had columns [A.down, B.down],
        # then cbind => the final columns might be [A.up, B.up, A.down, B.down].
        # get_top_elements returns a dict with keys = those same columns.
        # So effectively it's already the same naming. If we want to reassign them exactly:
        # However, the dictionary 'v' is keyed by the actual columns in 'combined'.
        # So we skip this step because it is already the correct mapping.

        # 8) if idx => filter
        if idx is not None:
            wanted_cols = []
            for i_col in idx:
                wanted_cols.append(f"{i_col}.up")
                wanted_cols.append(f"{i_col}.down")
            new_dict = {}
            for ckey in wanted_cols:
                if ckey in v:
                    new_dict[ckey] = v[ckey]
            v = new_dict

        # 9) names(v) <- paste0(add.prefix, names(v))
        if add_prefix:
            renamed_dict = {}
            for k in v.keys():
                renamed_dict[f"{add_prefix}{k}"] = v[k]
            v = renamed_dict

        return v

    def calculate_multifactor_PMD(
        self,
        adata: Any,  # Replace with AnnData if available.
        penalties: list[int] = None,
        ct_order: list[str] = None,
        agg_feature: bool = True,
        solver: Literal["lp", "bs"] = "bs",
        normalize: bool = True,
        n_components: int = None,
        feature_space_key: str = "X_pca",
        conf: str = "cellQ",
        n_genes: int = 200,
    ) -> dict[str, Any]:
        """
        Runs multifactor PMD mimicking DIALOGUE1 with partial correlation steps.

        This function performs several steps to prepare data, compute canonical variates via PMD,
        and compute various correlation metrics. It returns a dictionary with multiple outputs
        including the modified AnnData, canonical variates, cell-type subsets, residual scores,
        gene correlations, partial correlations, and additional signatures.

        Args:
            adata (Any): AnnData object to be processed.
            penalties (list[int], optional): List of penalties. Defaults to None.
            ct_order (list[str], optional): Order of cell types. Defaults to None.
            agg_feature (bool, optional): Whether to aggregate pseudobulk features. Defaults to True.
            solver (Literal["lp", "bs"], optional): Which solver to use ('lp' or 'bs'). Defaults to "bs".
            normalize (bool, optional): Whether to standardize/normalize data. Defaults to True.
            n_components (int, optional): Number of components to use. Defaults to all components.
            feature_space_key (str, optional): Key in obsm for feature space. Defaults to "X_pca".
            conf (str, optional): Column name for the confounder. Defaults to "cellQ".
            n_genes (int, optional): Number of genes to select for correlation calculations. Defaults to 200.

        Returns:
            dict[str, Any]: A dictionary containing:
                - adata: AnnData with added MCP score columns in obs.
                - ws: A dictionary with the computed canonical variates.
                - original_ct_subs: A dictionary with the original cell-type–specific AnnData subsets.
                - ct_subs: A dictionary with cell-type–specific AnnData subsets.
                - pre_r_scores: A dictionary of pre-residual scores (DataFrames with sample IDs as index).
                - cca_cor_df: A DataFrame of pairwise Pearson correlations (rows: MCPs, columns: cell type pairs).
                - residual_scores: Residualized scores per cell type.
                - gene_correlations: Per cell type gene–MCP correlation DataFrames (ordinary correlation).
                - g1: A list of top genes.
                - partial_corr: A dictionary with partial correlation results for each cell type ("genes", "R", "P").
                - sig_genes: A dictionary with up/down gene signatures extracted from the partial correlation matrix.
                - redun_cor: Redundancy correlation matrices (using first k MCPs).
                - samples_cells: Per cell type, list of sample names.
        """

        # --- 1) Preliminary steps ---
        if ct_order is not None:
            cell_types = ct_order
        else:
            cell_types = adata.obs[self.celltype_key].astype("category").cat.categories.tolist()
            ct_order = cell_types

        # If n_components is not provided, use all available components from the feature space.
        if n_components is None:
            n_components = adata.obsm[feature_space_key].shape[1]

        # Load cell-type specific data (pseudobulk features, subsets, etc.)
        mcca_in, ct_subs, original_ct_subs = self._load(
            adata,
            ct_order=cell_types,
            agg_feature=agg_feature,
            normalize=normalize,
            n_components=n_components,
            feature_space_key=feature_space_key,
        )

        mcca_in_indices = [df.index for df in mcca_in]
        mcca_in = [df.to_numpy() if hasattr(df, "to_numpy") else df for df in mcca_in]

        n_samples = mcca_in[0].shape[1]
        if penalties is None:
            try:
                penalties = multicca_permute(
                    mcca_in, penalties=np.sqrt(n_samples) / 2, nperms=10, niter=50, standardize=True
                )["bestpenalties"]
            except ValueError as e:
                if "matmul: input operand 1 has a mismatch in its core dimension" in str(e):
                    raise ValueError("Please ensure that every cell type is represented in every sample.") from e
                else:
                    raise
        else:
            penalties = penalties

        if solver == "bs":
            ws, _ = multicca_pmd(mcca_in, penalties, K=self.n_mcps, standardize=True, niter=100, mimic_R=normalize)
        elif solver == "lp":
            ws, _ = lp_pmd(mcca_in, penalties, K=self.n_mcps, standardize=True, mimic_R=normalize)
        else:
            raise ValueError('Please select a valid solver. Must be one of "lp" or "bs".')

        ws_dict = {ct: ws[i] for i, ct in enumerate(ct_order)}

        # Compute sample-level scores
        pre_r_scores = {
            ct: pd.DataFrame(
                mcca_in[i] @ ws[i], index=mcca_in_indices[i], columns=[f"MCP{j+1}" for j in range(ws[i].shape[1])]
            )
            for i, ct in enumerate(cell_types)
        }

        # Pairwise Pearson correlations across cell types
        cca_cor_df = self._compute_cca(pre_r_scores, cell_types)

        # In R: y[[x]] <- r@X[, rownames(out$ws[[x]])] %*% out$ws[[x]]
        # Here we use 'feature_space_key' from the subsets to mimic that projection.

        y = {
            ct: pd.DataFrame(
                original_ct_subs[ct].obsm[feature_space_key] @ ws_dict[ct],
                index=original_ct_subs[ct].obs.index,
                columns=[f"MCP{j+1}" for j in range(ws_dict[ct].shape[1])],
            )
            for ct in cell_types
        }

        # --- 2) Residual scores ---
        residual_scores = {}
        for ct in cell_types:
            r = original_ct_subs[ct]
            scores0 = y[ct]
            if self.conf is None:
                residual_scores[ct] = scores0
            else:
                # Directly use conf_series without additional conversion.
                conf_series = r.obs[self.conf]
                conf_data = pd.DataFrame(conf_series, index=conf_series.index)
                # Duplicate the confounder data as in the original code.
                conf_m = pd.concat([conf_data, conf_data], axis=1)
                residual = self._get_residuals(conf_m.values, scores0.T.values).T
                residual_scores[ct] = pd.DataFrame(residual, index=scores0.index, columns=scores0.columns)

        # --- 3) Gene correlations, partial correlations, signatures ---
        gene_correlations = {}
        partial_corr = {}
        partial_sig = {}
        redun_cor = {}
        samples_cells = {}

        for ct in cell_types:
            r = original_ct_subs[ct]

            # Convert r.X to a DataFrame if needed
            if not isinstance(r.X, pd.DataFrame):
                tpm_df = pd.DataFrame(r.X, index=r.obs.index, columns=r.var_names)
            else:
                tpm_df = r.X

            # (a) ordinary correlation (R$cca.gene.cor1[[x]] <- cor(t(r@tpm), r@scores))
            # Here we do correlation of each gene vs each MCP using the residual scores
            scores_df = residual_scores[ct]
            gene_corr = {}
            for gene in tpm_df.columns:
                gene_vector = tpm_df[gene]
                gene_corr[gene] = scores_df.apply(lambda col, gv=gene_vector: gv.corr(col))
            gene_corr_df = pd.DataFrame(gene_corr).T  # (genes, MCPs)
            gene_correlations[ct] = gene_corr_df  # store it

            corr_sig = self._get_top_corr(
                m=gene_corr_df,
                q=n_genes,  # param$n.genes
                min_ci=0.05,
            )

            top_genes_set = set()
            for gene_list in corr_sig.values():
                top_genes_set.update(gene_list)
            g1 = sorted(top_genes_set)

            # (c) partial correlation controlling for r.obs[conf]
            if self.conf is None:
                conf_vector = None
            else:
                conf_vector = r.obs[self.conf].values

            pcor_res = self._pcor_mat(tpm_df[g1], scores_df, conf_vector)
            C1 = pcor_res["R"]  # partial correlation matrix (genes x MCPs)
            P1 = pcor_res["P"]  # p-value matrix

            # threshold partial correlation
            threshold = 0.07 / tpm_df.shape[1]  # or 0.05/nrow(r@tpm)
            mask = P1 > threshold
            C1[mask] = 0.0

            partial_corr[ct] = {
                "genes": g1,
                "R": C1,  # thresholded partial correlations
                "P": P1,
            }

            # (d) final signature genes from partial correlation
            # R$cca.sig[[x]] <- get.top.cor(C1, q=param$n.genes, min.ci=0.05)
            # C1 is DataFrame (genes x MCPs), so pass it to _get_top_corr again
            final_sig = self._get_top_corr(C1, q=n_genes, min_ci=0.05)
            partial_sig[ct] = final_sig

            # (e) redundancy correlation
            k_val = min(self.n_mcps, scores_df.shape[1])
            redun_cor[ct] = np.corrcoef(scores_df.values[:, :k_val], rowvar=False)

            # (f) sample names
            samples_cells = {f"MCP{j+1}": cell_types for j in range(self.n_mcps)}

        results_dict = {
            "adata": adata,
            "ws": ws_dict,
            "original_ct_subs": original_ct_subs,
            "ct_subs": ct_subs,
            "pre_r_scores": pre_r_scores,
            "cca_cor_df": cca_cor_df,
            "residual_scores": residual_scores,
            "gene_correlations": gene_correlations,
            "g1": g1,
            "partial_corr": partial_corr,
            "sig_genes": partial_sig,
            "redun_cor": redun_cor,
            "samples_cells": samples_cells,
        }
        return results_dict

    #### Here starts DIALOGUE 2 and ends DIALOGUE1

    def Dialogue2(self, main, dialogue1_output):
        """
        Mimics the R function DIALOGUE2.

        Parameters:
        main : str
            A string identifier for the analysis. If empty, it is set by joining the cell type names.
        dialogue1_output : dict
            The output dictionary from DIALOGUE1 (e.g. from calculate_multifactor_PMD).
        covariates : list of str, optional
            A list of covariate names to include in the formula. For example: ["cellQ", "tme.qc"]
            If None, no additional covariates are appended.

        Returns:
        tuple: (R, r1_obj, r2_obj)
            R is the updated results dictionary; r1_obj and r2_obj correspond to the last pair processed.
        """
        print("#************ DIALOGUE Step II: HLM ************#")

        # Get cell type names from the DIALOGUE1 output.
        cell_types = list(dialogue1_output["original_ct_subs"].keys())

        # If main is empty, build it by concatenating cell type names with underscores.
        if not main:
            main = "_".join(cell_types)

        if dialogue1_output is None:
            raise ValueError("You must provide the DIALOGUE1 output via the 'dialogue1_output' parameter.")
        R = dialogue1_output

        # Check if a modeling formula already exists; if not, create one.
        if R.get("param", {}).get("frm") is None:
            # Use provided covariates or default to empty list.
            if self.cov is None:
                pass
            # Construct default formula: fixed part plus any covariates.
            R["frm"] = "y ~ (1 | samples) + x"
            if self.cov:
                R["frm"] += " + " + " + ".join(self.cov)
        else:
            R["frm"] = R["param"]["frm"]
            print("Using input formula.")

        print("Modeling formula:", R["frm"])

        # Determine the number of MCP components from the first cell type's canonical variate matrix.
        first_ws = next(iter(R["ws"].values()))

        first_ws.shape[1]

        # Generate all pairwise combinations of cell types.
        from itertools import combinations

        pairs1 = list(combinations(cell_types, 2))

        # Process each pair sequentially.
        for pair in pairs1:
            x1, x2 = pair
            print(f"#************ DIALOGUE Step II (multilevel modeling): {x1} vs. {x2} ************#")
            pair_name = f"{x1}.vs.{x2}"

            # Call the pairwise modeling function.
            R[pair_name] = self._dialogue2_pair(R, R["original_ct_subs"][x1], R["original_ct_subs"][x2], self.cov)

        # Update the name field to indicate that this is a DIALOGUE2 result.
        R["name"] = f"DIALOGUE2_{main}"

        return R

    def _dialogue2_pair(self, R, r1, r2, cov):
        """
        Mimics the R function DIALOGUE2.pair.

        Parameters:
        - R: A dictionary containing the DIALOGUE1 results, including a key "sig_genes"
            with signature gene lists (e.g. keys like "MCP1.up", "MCP1.down", etc.).
        - r1, r2: AnnData objects for two cell types (e.g. original_ct_subs["A"] and original_ct_subs["B"]).
        - cell_types: List of all cell-type names.

        Returns:
        - A dictionary containing the pairwise modeling results.
        """
        # 1) Get cell-type names.

        x1 = r1.obs[self.celltype_key].iloc[0] if self.celltype_key in r1.obs.columns else r1.uns.get("name", "unknown")
        x2 = r2.obs[self.celltype_key].iloc[0] if self.celltype_key in r2.obs.columns else r2.uns.get("name", "unknown")

        # 2) Derive MCP names from the keys in R["sig_genes"].
        if "sig_genes" not in R:
            raise ValueError("The results dictionary does not contain 'sig_genes'.")
        # Assume x1 and x2 have been determined (e.g. "A" and "B")
        # Get the MCP names from the signatures of each cell type.

        mcp_names_r1 = {key.split(".")[0] for key in R.get("sig_genes", {}).get(x1, {}).keys()}
        mcp_names_r2 = {key.split(".")[0] for key in R.get("sig_genes", {}).get(x2, {}).keys()}

        # Take the intersection so that only MCP names present in both cell types are kept.
        MCP_names = sorted(mcp_names_r1.intersection(mcp_names_r2))

        if not MCP_names:
            print("No MCP signatures found in R['sig_genes'].")
            return None
        print(f"{len(MCP_names)} MCPs identified from sig_genes.")

        # 3) Extract signature gene information for each cell type.
        sig1 = R["sig_genes"].get(x1)
        sig2 = R["sig_genes"].get(x2)

        # 4) Use the shared samples already in the AnnData objects.
        samples_r1 = r1.obs[self.sample_id].tolist()
        samples_r2 = r2.obs[self.sample_id].tolist()

        idx = set(samples_r1).intersection(set(samples_r2))

        # 5) Subset r1 and r2 to cells with samples in idx.
        r1 = r1[r1.obs[self.sample_id].isin(list(idx)), :].copy()
        r2 = r2[r2.obs[self.sample_id].isin(list(idx)), :].copy()

        # Same thing here, we passed with the correct sample sizes, but without the correct genes
        # due to the input, so far no problem but this is subsetting a lot

        # 6) Subset the precomputed residual scores from R (assumed stored in R["residual_scores"]).

        r1.obsm["scores"] = R["residual_scores"][x1].loc[r1.obs_names]
        r2.obsm["scores"] = R["residual_scores"][x2].loc[r2.obs_names]

        # r1_scores = R["residual_scores"][x1].loc[r1.obs_names]
        # r2_scores = R["residual_scores"][x2].loc[r2.obs_names]

        # r1.uns["scores"] = r1_scores
        # r2.uns["scores"] = r2_scores

        # 7) Obtain overall expression (OE) information.
        oe_results = self._dlg_get_OE(r1, r2, compute_scores=False)
        r1 = oe_results["r1"]
        r2 = oe_results["r2"]

        # For cell type x2:
        adata_x2 = R["original_ct_subs"][x2]
        if "tpmAv_by_cell_type" in adata_x2.uns and x2 in adata_x2.uns["tpmAv_by_cell_type"]:
            # Use precomputed pseudobulk values (genes as rows, samples as columns)
            r1.uns["tme"] = adata_x2.uns["tpmAv_by_cell_type"][x2].loc[:, r1.obs[self.sample_id]]
        else:
            # Compute pseudobulk counts using self.sample_id as the grouping variable.
            pseudobulk_x2 = self._pseudobulk(adata_x2, groupby=self.sample_id, mode="counts", strategy="mean")
            # Our computed pseudobulk has samples as rows and genes as columns,
            # so we subset by selecting rows corresponding to the samples in r1.obs.
            r1.uns["tme"] = pseudobulk_x2.loc[:, r1.obs[self.sample_id]]

        # For cell type x1:
        adata_x1 = R["original_ct_subs"][x1]
        if "tpmAv_by_cell_type" in adata_x1.uns and x1 in adata_x1.uns["tpmAv_by_cell_type"]:
            r2.uns["tme"] = adata_x1.uns["tpmAv_by_cell_type"][x1].loc[:, r2.obs[self.sample_id]]
        else:
            pseudobulk_x1 = self._pseudobulk(adata_x1, groupby=self.sample_id, mode="counts", strategy="mean")
            r2.uns["tme"] = pseudobulk_x1.loc[:, r2.obs[self.sample_id]]

        # 10) Set tissue quality control (tme_qc) fields.
        # For r1, use the qcAv stored under cell type x2.
        # For r1: use the QC values stored under cell type x2, if available.
        if (
            "qcAv_by_cell_type" in R["original_ct_subs"][x2].uns
            and x2 in R["original_ct_subs"][x2].uns["qcAv_by_cell_type"]
        ):
            precomputed = R["original_ct_subs"][x2].uns["qcAv_by_cell_type"][x2]
            sample_ids_str = [str(s) for s in r1.obs[self.sample_id].tolist()]
            r1.uns["tme_qc"] = precomputed.loc[sample_ids_str, :].iloc[:, 1]
        else:
            r1.uns["tme_qc"] = None  # or leave it unchanged

        # For r2: use the QC values stored under cell type x1, if available.
        if (
            "qcAv_by_cell_type" in R["original_ct_subs"][x1].uns
            and x1 in R["original_ct_subs"][x1].uns["qcAv_by_cell_type"]
        ):
            precomputed = R["original_ct_subs"][x1].uns["qcAv_by_cell_type"][x1]
            sample_ids_str = [str(s) for s in r2.obs[self.sample_id].tolist()]
            r2.uns["tme_qc"] = precomputed.loc[sample_ids_str, :].iloc[:, 1]
        else:
            r2.uns["tme_qc"] = None  # or leave it unchanged

        # 11) Convert AnnData objects to list-like structures.
        r1a = self._cell_type_2_list(r1)
        r2a = self._cell_type_2_list(r2)

        # correct until here

        # 12) Define a helper function for mixed-effects modeling and signature extraction.
        def f1(mcp_name):
            p1 = self._dialogue2_mixed_effects(r2a, mcp_name, sig1, R["frm"])
            p2 = self._dialogue2_mixed_effects(r1a, mcp_name, sig2, R["frm"])
            sig1f = self._intersect_list1(
                self._get_top_corr(p1[p1["Z"].notna()], q=100, idx="Z", min_ci=1), list(r1.var_names)
            )
            sig2f = self._intersect_list1(
                self._get_top_corr(p2[p2["Z"].notna()], q=100, idx="Z", min_ci=1), list(r2.var_names)
            )
            # Rename keys by replacing "Z." with the current MCP name.

            sig1f = {k.replace("Z.", f"{mcp_name}."): v for k, v in sig1f.items()}
            sig2f = {k.replace("Z.", f"{mcp_name}."): v for k, v in sig2f.items()}
            p1["program"] = mcp_name
            p2["program"] = mcp_name
            p1["genes"] = p1.index
            p2["genes"] = p2.index
            return {"p1": p1, "p2": p2, "sig1f": sig1f, "sig2f": sig2f}

        # 13) Apply the helper function for each MCP name.

        R1 = {mcp: f1(mcp) for mcp in MCP_names}

        # 14) Combine the p1 and p2 results from each MCP.
        p1_list = [R1[mcp]["p1"] for mcp in MCP_names]
        p2_list = [R1[mcp]["p2"] for mcp in MCP_names]
        R1["p1"] = pd.concat(p1_list, axis=0)
        R1["p2"] = pd.concat(p2_list, axis=0)

        # 15) Extract signature gene lists from the DIALOGUE1 results using "sig_genes".
        R1["sig1"] = {mcp: R.get("sig_genes", {}).get(mcp, {}).get("sig1f") for mcp in MCP_names}
        R1["sig2"] = {mcp: R.get("sig_genes", {}).get(mcp, {}).get("sig2f") for mcp in MCP_names}

        # 16) Set the name of the pairwise result.
        R1["name"] = f"{x1}.vs.{x2}"

        return R1

    def _dlg_get_OE(self, r1, r2, plot_flag=False, compute_scores=True):
        """
        Mimics the R function DLG.get.OE.

        (Details omitted for brevity.)
        """

        # Compute averaged overall expression by grouping by the sample names.
        r1_scoresAv = r1.obsm["scores"].groupby(r1.obs[self.sample_id]).median()
        r2_scoresAv = r2.obsm["scores"].groupby(r2.obs[self.sample_id]).median()

        r1.uns["tme_OE"] = r2_scoresAv.reindex(r1.obs[self.sample_id])
        r2.uns["tme_OE"] = r1_scoresAv.reindex(r2.obs[self.sample_id])

        result = {"r1": r1, "r2": r2}

        if not plot_flag:
            return result

        result = {"r1": r1, "r2": r2}

        if not plot_flag:
            return result

        # If plotting is requested, call the correlation plotting function.
        self._dlg_cor_plot(r1, r2, sd_flag=False, q1=1 / 3, q2=2 / 3)
        return result

    def _dialogue2_mixed_effects(self, r1, mcp, sig2, frm="y ~ (1 | samples) + x + cellQ"):
        """
        Mimics the R function DIALOGUE2.mixed.effects using a cell-type object (converted to a dict).

        Parameters:
        r1 : dict
            A dictionary containing data for a cell type. Expected keys include:
                - "tme": a DataFrame of tissue expression data (rows: genes, columns: samples)
                - "uns": a dict that includes an entry "scores", a DataFrame of sample-level scores,
                        with columns corresponding to MCPs.
        mcp : str
            An identifier for the MCP (ideally something like "MCP1").
        sig2 : dict
            A dictionary with signature gene lists. It should contain keys like "MCP1.up" and "MCP1.down".
        frm : str, optional
            A formula string for the mixed-effects model.

        Returns:
        A pandas DataFrame with the mixed-effects modeling results.
        """
        # 1) Combine up and down signature gene lists for the given MCP.
        genes = list(sig2.get(f"{mcp}.up", [])) + list(sig2.get(f"{mcp}.down", []))

        # 2) Determine which of these genes are present in the tissue expression data.
        b = [gene in r1["uns"]["tme"].index for gene in genes]

        # 3) Build the list of available genes.
        available_genes = [gene for gene, present in zip(genes, b, strict=True) if present]

        # 4) Determine the response vector.
        scores_df = r1["obsm"]["scores"]

        # if mcp in scores_df.columns:
        response = scores_df[mcp]

        # elif mcp.startswith("MCP"):
        #    try:
        #        idx = int(mcp.replace("MCP", "")) - 1
        #        response = scores_df.iloc[:, idx]
        #    except Exception as e:
        #       raise ValueError(f"Could not determine the response for {mcp}: {e}")
        # else:
        # Fallback: use the first column of scores_df
        #    print(f"Warning: {mcp} not found; using the first column as response.")
        #    response = scores_df.iloc[:, 0]

        # 5) Apply the mixed-effects model.

        p = self._apply_formula_HLM(r1, X=r1["uns"]["tme"].loc[available_genes], Y=response, formula=frm)

        # 6) Adjust p-values using the Benjamini–Hochberg method.
        p["pval"] = self._p_adjust(p["P"], method="BH")

        # 7) Mark each gene as "up" if it is in the up-signature for the given MCP.
        p["up"] = p.index.isin(sig2.get(f"{mcp}.up", []))

        # 8) If all signature genes are available, return p directly.
        if all(b):
            # p.index = p.index.str.replace(".", "-", regex=False)
            return p

        # 9) Otherwise, create an empty DataFrame with all gene names from the full signature.
        P_df = pd.DataFrame(np.nan, index=genes, columns=p.columns)

        # 10) For available genes, copy the modeled results.
        available_gene_names = [gene for gene, present in zip(genes, b, strict=True) if present]
        P_df.loc[available_gene_names] = p.values

        # 11) Replace periods in the row labels with hyphens.
        P_df.index = P_df.index.str.replace(".", "-", regex=False)

        return P_df

    def _apply_formula_HLM(self, r, X, Y, margin=1, formula="y ~ (1 | samples) + x + cellQ", ttest_flag=False):
        """
        Mimics the R function apply.formula.HLM.

        Parameters:
        r : dict or similar
            A context object containing extra information (e.g. r["samples"]).
        X : array-like or DataFrame
            Predictor data (e.g., a DataFrame or numpy array).
        Y : array-like or DataFrame
            Response data. If Y is 2D (matrix or DataFrame) and margin==1,
            the model is applied row-wise.
        margin : int, default 1
            The axis along which to apply the model (1 means row-wise).
        formula : str, default "y ~ (1 | samples) + x"
            A formula string to be used in the mixed-effects model.
        ttest_flag : bool, default False
            If True, additional t-test–based filtering is performed (see details below).

        Returns:
        A pandas DataFrame with columns:
            - "Estimate": the estimated coefficient from the model,
            - "P": the corresponding p-value,
            - "Z": a computed Z-score from Estimate and P.
        If ttest_flag is True, additional t-test results may be appended.

        Procedure:
        (A) If Y is a 2D structure:
            - If ttest_flag is True, then:
                * Compute a pseudobulk average (mimicking R's average.mat.rows on t(Y))
                    and perform two t-tests (one on the pseudobulk and one on Y itself).
                * Create boolean masks (b1 and b2) based on the t-test p-values.
                * Combine them into a final filter mask, use that to subset Y.
            - Apply self.formula_HLM row-wise on the (possibly subset) Y DataFrame;
                each row is passed (as a Series) along with X and r to self.formula_HLM.
        (B) If Y is not 2D, then apply self.formula_HLM on rows of X instead.
        (C) The resulting list of (estimate, p-value) pairs is converted into a DataFrame,
            and a Z score is computed using self.get_cor_zscores.
        (D) If ttest_flag is True, t-test results are concatenated to the output.
        """

        # CASE A: Y is 2D
        if isinstance(Y, np.ndarray | pd.DataFrame) and getattr(Y, "ndim", 1) == 2:
            # in the end it should enter through here but we will see what happens

            if ttest_flag:
                # --- (A) T-test based filtering (this mimics the R code when ttest.flag is TRUE) ---
                # Compute pseudobulk averages from the transposed Y, grouping by samples (constructed from r["samples"] and X)
                m1 = self.average_mat_rows(Y.T, [f"{s}_{x}" for s, x in zip(r["samples"], X, strict=False)])
                # Perform t-tests on the pseudobulk and on Y directly.
                de1 = self.t_test_mat(m1, [self.get_strsplit(idx, "_", 2) for idx in m1.index])
                de2 = self.t_test_mat(Y, X)
                # Create boolean masks: b1 true if at least one of the first two columns of de1 is below 0.1, similarly for de2.
                b1 = (de1.iloc[:, :2] < 0.1).sum(axis=1) > 0
                b2 = (self.p_adjust_mat(de2.iloc[:, :2], method="BH") < 0.1).sum(axis=1) > 0
                b = b1 | b2
                # Combine t-test z-scores into a DataFrame for later (if desired).
                de_ttest = pd.concat([de1["zscores"], de2["zscores"]], axis=1)
                de_ttest = de_ttest[b]
                # Subset Y to only the rows passing the combined filter.
                if isinstance(Y, np.ndarray):
                    Y = Y[b, :]
                    Y_df = pd.DataFrame(Y)
                else:
                    Y_df = Y.loc[b].copy()
            else:
                # If no t-test filtering, ensure Y is a DataFrame.
                if isinstance(Y, np.ndarray):
                    Y_df = pd.DataFrame(Y)
                else:
                    Y_df = Y.copy()

            # --- (B) Apply the mixed-effects model row-wise on Y_df.
            # For each row (gene) in Y_df, apply self.formula_HLM which should return (Estimate, P).
            print(formula)
            results = Y_df.apply(lambda y: self._formula_HLM(y, X, r, formula=formula), axis=1)
            # Convert the list of tuples to a DataFrame with appropriate column names.
            m = pd.DataFrame(results.tolist(), index=Y_df.index, columns=["Estimate", "P"])
        else:
            # CASE B: Y is not 2D; assume we apply the model to rows of X.
            if isinstance(X, np.ndarray):
                X_df = pd.DataFrame(X)
            else:
                X_df = X.copy()
            results = X_df.apply(lambda x: self._formula_HLM(Y, x, r, formula=formula), axis=1)
            m = pd.DataFrame(results, index=X_df.index, columns=["Estimate", "P"])

        # --- (C) Compute the Z-score from the Estimate and P columns.
        m["Z"] = self._get_cor_zscores(m["Estimate"], m["P"])

        # --- (D) If ttest_flag is True, append the t-test results.
        if ttest_flag:
            m = pd.concat([m, de_ttest], axis=1)

        return m

    def _formula_HLM(self, y, x, r0, formula="y ~ (1 | samples) + x", val=None, return_all=False):
        """
        Mimics the R function formula.HLM.

        This function sets up a mixed-effects model using statsmodels' MixedLM to mimic the behavior of R's
        formula.HLM function. It assigns the predictor (x) and response (y) variables to a copy of the provided
        DataFrame (r0) and fits the model with a specified formula (after removing the random-effects specification).
        The function returns the coefficient for the fixed effect "x" (optionally modified by `val`) or the full fixed-effects
        coefficient table if `return_all` is True. In case of an error, it returns a Series with NaN values.

        Args:
            y (array-like): The response variable.
            x (array-like): The predictor variable.
            r0 (pd.DataFrame): DataFrame that will be updated with columns "x" and "y". It should contain a "samples" column.
            formula (str, optional): Model formula (default: "y ~ (1 | samples) + x").
            val (str or None, optional): Suffix appended to "x" for selecting the coefficient of interest. If None,
                defaults to "" if x is numeric or "TRUE" otherwise.
            return_all (bool, optional): If True, return the full fixed-effects coefficient table;
                otherwise, return only the row for "x" (modified by `val`).

        Returns:
            pd.Series or pd.DataFrame: The fixed-effect coefficient(s) with 'Estimate' and 'P' values.
        """
        # Determine default for 'val'
        try:
            float(x)
            default_val = ""
        except (ValueError, TypeError):
            default_val = "TRUE"
        if val is None:
            val = default_val

        # Work on a copy of r0 so we do not modify the original.
        r0 = r0.copy()

        # In our case, we want to store the predictor and response in r0.
        r0["x"] = x
        r0["y"] = y

        if self.conf is not None:
            r0[self.conf] = pd.Series(r0[self.conf], index=r0["x"].index)

        r0["y"] = pd.Series(r0["y"].values, index=r0["x"].index)

        groups = r0[self.sample_id]

        # Remove the random effects term from the formula.
        if "(1 | samples)" in formula:
            formula_fixed = formula.replace("(1 | samples) + ", "").replace("(1 | samples)", "")
        else:
            formula_fixed = formula

        # Clean extra spaces.
        formula_fixed = " ".join(formula_fixed.split())

        try:
            # Fit the mixed-effects model using statsmodels MixedLM.
            model = sm.MixedLM.from_formula(formula_fixed, data=r0, groups=groups)
            result = model.fit()
            # Get fixed effects estimates and p-values.
            coefs = result.params
            pvalues = result.pvalues
            coef_df = pd.DataFrame({"Estimate": coefs, "P": pvalues})

            if return_all:
                c1 = coef_df.loc[:, ["Estimate", "P"]]
            else:
                # Build the key for the coefficient of interest.
                var_name = "x" + val
                if var_name not in coef_df.index:
                    var_name = "x"
                c1 = coef_df.loc[var_name, ["Estimate", "P"]]
            return c1
        except (ValueError, TypeError, np.linalg.LinAlgError, IndexError):
            return pd.Series([float("nan"), float("nan")], index=["Estimate", "P"])

    def _get_onesided_p_value(self, c, p):
        """
        Mimics R's get.onesided.p.value:

        For each entry, if c > 0 then returns p/2, and if c <= 0 then returns 1 - (p/2).
        Before doing so, any zeros in p are replaced with the minimum positive p value.

        Parameters:
        c : numpy array
            Array of coefficients.
        p : numpy array
            Array of p-values (same shape as c).

        Returns:
        numpy array with the one-sided p-values.
        """
        # Ensure inputs are numpy arrays of floats.
        c = np.array(c, dtype=float)
        p = np.array(p, dtype=float)

        # Replace any zeros in p with the minimum positive value (if any exist)
        positive = p > 0
        if np.any(positive):
            min_positive = np.min(p[positive])
            p[p == 0] = min_positive

        # Initialize output array with NaNs.
        p_one_side = np.full(p.shape, np.nan)

        # For values where c > 0, one-sided p-value is p/2.
        mask_positive = (c > 0) & (~np.isnan(c))
        p_one_side[mask_positive] = p[mask_positive] / 2.0

        # For values where c <= 0, one-sided p-value is 1 - (p/2).
        mask_nonpositive = (c <= 0) & (~np.isnan(c))
        p_one_side[mask_nonpositive] = 1 - (p[mask_nonpositive] / 2.0)

        return p_one_side

    def _get_p_zscores(self, p_matrix):
        """
        Mimics R's get.p.zscores.

        Given a 2D array p_matrix (n x 2) where the first column is one‐sided p-values for c
        and the second column is for -c, compute a z-score vector as follows:
        - For each row where the first column is ≤ 0.5, z = -log10(p1)
        - For rows where the first column is > 0.5, z = log10(p2)

        Parameters:
        p_matrix : numpy array of shape (n, 2)
            The two-sided matrix of one-sided p-values.

        Returns:
        numpy array of z-scores.
        """
        # Create a boolean mask where the first column is > 0.5. Replace NaN with False.
        b = np.where(np.isnan(p_matrix[:, 0]), False, p_matrix[:, 0] > 0.5)

        # Compute negative log10 of the first column.
        zscores = -np.log10(p_matrix[:, 0])

        # For indices where b is True, set zscore to log10 of the second column.
        zscores[b] = np.log10(p_matrix[b, 1])

        return zscores

    def _get_cor_zscores(self, c, p):
        """
        Mimics R's get.cor.zscores.

        For each element, it first computes a two‐column matrix where:
        - First column = get.onesided.p.value(c, p)
        - Second column = get.onesided.p.value(-c, p)
        Then it computes z-scores using get.p.zscores.

        Parameters:
        c : numpy array
            Array of coefficient values.
        p : numpy array
            Array of p-values corresponding to c.

        Returns:
        numpy array of z-scores.
        """
        # Compute one-sided p-values for c and for -c.
        col1 = self._get_onesided_p_value(c, p)
        col2 = self._get_onesided_p_value(-c, p)

        # Stack as a two-column matrix.
        v = np.column_stack((col1, col2))

        # Compute z-scores from v.
        z = self._get_p_zscores(v)
        return z

    def _adjust_series(self, s: pd.Series) -> pd.Series:
        """
        Adjusts a pandas Series of p-values using the Benjamini–Hochberg (BH) procedure.

        Parameters:
        s : pandas Series of p-values.

        Returns:
        A pandas Series of adjusted p-values with the same index.
        """
        pvals = s.values
        # fdrcorrection returns a boolean array and the adjusted p-values.
        _, pvals_adj = fdrcorrection(pvals, alpha=0.05, method="indep")
        return pd.Series(pvals_adj, index=s.index)

    def _p_adjust(self, p: pd.DataFrame, v=None, method="BH") -> pd.DataFrame:
        """
        Mimics R's p.adjust.mat.per.label.

        If v is provided (an array-like of the same length as the number of rows of p),
        then for each unique label in v, the function adjusts the p-values for the rows
        corresponding to that label.

        If v is None, then:
        - If p is a Series, the entire series is adjusted.
        - If p is a DataFrame, each column is adjusted independently.

        Parameters:
        p : pandas DataFrame (or Series) of p-values.
        v : array-like of labels (one per row of p) or None.
        method : str, adjustment method; currently only "BH" is supported.

        Returns:
        A DataFrame (or Series) of adjusted p-values with the same shape as p.
        """

        # Helper: adjust an entire Series.
        def adjust_series_local(s: pd.Series) -> pd.Series:
            return self._adjust_series(s)

        if v is not None:
            # v is provided; adjust rows by label.
            v = np.array(v)
            p_adj = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
            for label in np.unique(v):
                mask = v == label
                if p.shape[1] < 2:
                    # p is essentially one-dimensional (a Series)
                    p_adj.loc[mask] = adjust_series_local(p.loc[mask])
                else:
                    # For each column in the DataFrame, adjust the rows corresponding to mask.
                    for col in p.columns:
                        p_adj.loc[mask, col] = adjust_series_local(p[col].loc[mask])
            return p_adj
        else:
            # No grouping provided: adjust the entire input.
            if isinstance(p, pd.Series):
                return adjust_series_local(p)
            elif isinstance(p, pd.DataFrame):
                adjusted_cols = {}
                for col in p.columns:
                    adjusted_cols[col] = adjust_series_local(p[col])
                return pd.DataFrame(adjusted_cols, index=p.index)
            else:
                raise ValueError("Input p must be a pandas Series or DataFrame.")

    def _intersect_list1(self, l: dict, g: list, n1: int = 0, HG_universe=None, prf: str = "") -> dict:
        """
        Mimics the R function intersect.list1.

        Parameters:
        l : dict
            A dictionary where keys are signature identifiers and values are lists of genes.
        g : list
            A list of genes with which to intersect each gene list in l.
        n1 : int, optional (default 0)
            Minimum number of genes required in the intersection for that key to be kept.
        HG_universe : any, optional
            (Optional) A gene universe for enrichment. (Not implemented here.)
        prf : str, optional (default "")
            If provided and non-empty, a prefix to add (with a period) to each key in the output.

        Returns:
        A dictionary mapping keys to the intersected gene lists (filtered so that only lists with
        more than n1 genes are retained). If prf is provided, the keys will be prefixed accordingly.
        """
        # For each key in l, compute the intersection with g.
        l1 = {k: list(set(v).intersection(g)) for k, v in l.items()}

        # Filter out any keys where the intersection has length <= n1.
        l1 = {k: v for k, v in l1.items() if len(v) > n1}

        # If a prefix is provided, update the keys by prepending it.
        if prf:
            l1 = {f"{prf}.{k}": v for k, v in l1.items()}

        # If HG_universe is provided, you could call a GO enrichment function here.
        # (This part is omitted because it requires additional implementation.)

        return l1

    def _dlg_cor_plot(self, r1, r2, idx=None, q1=0.25, q2=0.75, sd_flag=False):
        """
        Mimics the R function DLG.cor.plot.

        Parameters:
        - r1, r2: Cell-type objects. They should have:
                * r1.scoresAv, r2.scoresAv: DataFrames with row index = sample names and columns = MCP identifiers.
                * r1.scores, r2.scores: Raw score DataFrames (rows = cells; r.obs["samples"] indicates sample for each row).
        - idx: A string specifying the MCP column to plot (e.g., "MCP1"). If None, it is derived from r1.scores' column names.
        - q1, q2: Quantile thresholds (default 0.25 and 0.75).
        - sd_flag: If True, use half the standard deviation as error bars instead of quantiles.

        Returns:
        A tuple (corr, p_value) corresponding to the Spearman correlation between the averaged scores.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import spearmanr

        # If idx is not provided, extract unique first parts of the column names of r1.scores.
        if idx is None:
            idx_candidates = sorted({col.split(".")[0] for col in r1.scores.columns})
            if len(idx_candidates) > 1:
                # If more than one candidate exists, plot each in a separate figure.
                for candidate in idx_candidates:
                    self.DLG_cor_plot(r1, r2, idx=candidate, q1=q1, q2=q2, sd_flag=sd_flag)
                return
            else:
                idx = idx_candidates[0]

        # 1) Determine common samples between r1.scoresAv and r2.scoresAv.
        common_samples = r1.scoresAv.index.intersection(r2.scoresAv.index)
        # x and y are the averaged scores (from scoresAv) for the given MCP (idx).
        x = r1.scoresAv.loc[common_samples, idx]
        y = r2.scoresAv.loc[common_samples, idx]

        # 2) For each common sample, compute error bar values.
        x0, x1, y0, y1 = [], [], [], []
        for sample in common_samples:
            # Subset the raw scores for the sample.
            r1_sample = r1.scores[r1.obs["samples"] == sample][idx].dropna()
            r2_sample = r2.scores[r2.obs["samples"] == sample][idx].dropna()
            if len(r1_sample) == 0:
                x0.append(np.nan)
                x1.append(np.nan)
            else:
                # Quantile-based error bars.
                x0.append(np.quantile(r1_sample, q1))
                x1.append(np.quantile(r1_sample, q2))
            if len(r2_sample) == 0:
                y0.append(np.nan)
                y1.append(np.nan)
            else:
                y0.append(np.quantile(r2_sample, q1))
                y1.append(np.quantile(r2_sample, q2))

        x0 = np.array(x0)
        x1 = np.array(x1)
        y0 = np.array(y0)
        y1 = np.array(y1)

        # If sd_flag is True, compute half the standard deviation for each sample.
        if sd_flag:
            xd, yd = [], []
            for sample in common_samples:
                r1_sample = r1.scores[r1.obs["samples"] == sample][idx].dropna()
                r2_sample = r2.scores[r2.obs["samples"] == sample][idx].dropna()
                sd_x = np.std(r1_sample) / 2 if len(r1_sample) > 0 else 0
                sd_y = np.std(r2_sample) / 2 if len(r2_sample) > 0 else 0
                xd.append(sd_x)
                yd.append(sd_y)
            xd = np.array(xd)
            yd = np.array(yd)
            x0 = x.values - xd
            x1 = x.values + xd
            y0 = y.values - yd
            y1 = y.values + yd
        else:
            x = x.copy()
            y = y.copy()
            # Ensure x and y are numpy arrays.
            x = x.values
            y = y.values

        # 3) Plot the data.
        plt.figure(figsize=(6, 6))
        # Set limits to encompass both the scores and their error bars.
        xlim = (min(np.concatenate([x, x0])), max(np.concatenate([x, x1])))
        ylim = (min(np.concatenate([y, y0])), max(np.concatenate([y, y1])))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.title(f"Program {idx}")
        plt.xlabel(f"r1 scoresAv {idx}")
        plt.ylabel(f"r2 scoresAv {idx}")

        # Plot error bars for each sample.
        for i in range(len(common_samples)):
            # Horizontal error bars (for x).
            plt.hlines(y[i], x0[i], x1[i], colors="grey", lw=2)
            # Vertical error bars (for y).
            plt.vlines(x[i], y0[i], y1[i], colors="grey", lw=2)

        # Plot the points.
        plt.scatter(x, y, color="black", marker="o")
        # Plot the identity line.
        line_vals = np.linspace(xlim[0], xlim[1], 100)
        plt.plot(line_vals, line_vals, "k--", label="y = x")
        plt.legend()
        plt.grid(True)
        plt.show()

        # 4) Compute Spearman correlation between x and y.
        corr, p_value = spearmanr(x, y)
        return (corr, p_value)

    def _cell_type_2_list(self, r, idx=None):
        """
        Mimics the R function cell.type.2.list by converting an AnnData object
        to a dictionary that includes key metadata slots.

        Parameters:
        r : AnnData
            The AnnData object for a given cell type.
        idx : list or None
            Optional list of attribute names to extract. If None, we include the main slots.

        Returns:
        dict
            A dictionary containing:
                - 'obs': r.obs
                - 'var': r.var
                - 'obsm': r.obsm
                - 'varm': r.varm
                - 'uns': r.uns
            and also each column of r.obs as a separate key.
        """
        r_list = {}
        # Explicitly include the main AnnData slots.
        r_list["obs"] = r.obs
        r_list["var"] = r.var
        r_list["obsm"] = r.obsm
        r_list["varm"] = r.varm
        r_list["uns"] = r.uns  # <-- This is essential to avoid the KeyError.
        # Optionally, add each column of obs.

        for col in r.obs.columns:
            r_list[col] = r.obs[col].values  # Stores only the values as a NumPy array

        if r.uns is not None:
            for key, value in r.uns.items():
                r_list[f"uns_{key}"] = value

        # (Optional) If you want to add keys from obsm or varm as well, you can do so:
        # for key in r.obsm.keys():
        #     r_list[f"obsm_{key}"] = r.obsm[key]
        # for key in r.varm.keys():
        #     r_list[f"varm_{key}"] = r.varm[key]

        return r_list

    ########### Coding DIALOGUE3 now ########## Up Until here we had DIALOGUE1 and DIALOGUE2

    def Dialogue3(self, rA, main, dialogue2_output, k=2):
        """
        Mimics the R function DIALOGUE3, finalizing scores and computing several
        downstream metrics. This version ignores file reading/writing and instead
        uses the provided dialogue2_output (R dictionary) along with the cell-type
        objects in rA (a dictionary) to update and compute additional components.

        Parameters:
        rA : dict
            A dictionary mapping cell-type names to cell-type objects (e.g. from DIALOGUE1).
        main : str
            An identifier for the analysis; if empty, it is constructed by joining cell-type names.
        dialogue2_output : dict
            The result of the DIALOGUE2 step (assumed to be a dictionary with required keys).

        Returns:
        dict:
            A reduced dictionary (R1) containing selected components.
        """
        print("#************ Finalizing the scores ************#")

        # 1. Determine cell-type names.
        cell_types = list(rA.keys())
        if not main:
            main = "_".join(cell_types)

        # 2. Use the passed dialogue2_output (R) instead of reading from a file.
        R = dialogue2_output

        if self.conf is None:
            R["frm"] = "y ~ (1 | samples) + x"
        elif isinstance(self.conf, (list | tuple)):
            conf_str = " + ".join(self.conf)
            R["frm"] = "y ~ (1 | samples) + x + " + conf_str
        else:
            R["frm"] = "y ~ (1 | samples) + x + " + self.conf

        # 3. Compute gene-level p-values for each cell type.
        # (Assumes self.DLG_multi_get_gene_pval(ct, R) is implemented.)
        R["gene.pval"] = {ct: self._dlg_multi_get_gene_pval(ct, R) for ct in R.get("cell.types", cell_types)}
        R["cell.types"] = cell_types

        # 4. Update each cell-type object using DLG.find.scoring.
        rA = {ct: self._dlg_find_scoring(rA[ct], R) for ct in cell_types}

        # 5. Initialize the preference dictionary.
        R["pref"] = {}

        # 6. Generate all pairwise combinations of cell types.
        pairs = list(combinations(cell_types, 2))
        for x1, x2 in pairs:
            pair_key = f"{x1}.vs.{x2}"
            print(f"#************ DIALOGUE Step II (multilevel modeling): {x1} vs. {x2} ************#")

            # Get the two cell-type objects.
            r1 = rA[x1]
            r2 = rA[x2]

            r1.obsm["scores"] = pd.DataFrame(r1.obsm["scores"], index=r1.obs_names, columns=["MCP1", "MCP2"])

            r2.obsm["scores"] = pd.DataFrame(r2.obsm["scores"], index=r2.obs_names, columns=["MCP1", "MCP2"])

            # 7. Obtain overall expression (OE) information.
            # (Assumes self.DLG_get_OE returns a dict with keys "r1" and "r2".)
            oe = self._dlg_get_OE(r1, r2, plot_flag=False, compute_scores=False)
            r1, r2 = oe["r1"], oe["r2"]

            # 8. Determine the common (abundant) samples.
            samples_r1 = r1.obs[self.sample_id].tolist()
            samples_r2 = r2.obs[self.sample_id].tolist()
            idx = set(samples_r1).intersection(set(samples_r2))

            # 10. Compute the diagonal correlation between the pseudobulk averages.
            scoresAv1 = r1.uns["scoresAv"].loc[list(idx)]
            scoresAv2 = r2.uns["scoresAv"].loc[list(idx)]

            # Compute the full correlation matrix between the two sets.
            # Transpose so that variables (columns) become rows.
            combined = np.corrcoef(scoresAv1.T, scoresAv2.T)
            p = scoresAv1.shape[1]
            corr_mat = combined[:p, p:]
            diag_corr = np.diag(corr_mat)

            hlm_pval = self._dlg_hlm_pval(r1, r2, formula=R["frm"])
            # 12. Combine into a DataFrame and store in R["pref"].

            # Create a DataFrame for diag_corr with the MCP names as index and column name "R"
            diag_corr_df = pd.DataFrame({"R": diag_corr}, index=[f"MCP{i+1}" for i in range(diag_corr.shape[0])])

            # Rename the columns in hlm_pval if needed.
            # (If hlm_pval already has the desired names, you can skip this step.)
            hlm_pval = hlm_pval.rename(columns={"r1_p": "hlm.p12", "r2_p": "hlm.p22"})

            # Now concatenate diag_corr_df and hlm_pval along columns
            pref_df = pd.concat([diag_corr_df, hlm_pval], axis=1)

            R["pref"][pair_key] = pref_df

        R["gene.pval"] = {key: r1.uns["gene_pval"] for key, r1 in rA.items()}
        R["sig1"] = {key: r1.uns["sig"]["sig1"] for key, r1 in rA.items()}
        R["sig2"] = {key: r1.uns["sig"]["sig2"] for key, r1 in rA.items()}

        R["scores"] = {
            ct: pd.concat(
                [
                    # Convert the "scores" matrix (from obsm) to a DataFrame.
                    pd.DataFrame(rA[ct].obsm["scores"], index=rA[ct].obs.index),
                    # Create a DataFrame with additional information.
                    pd.DataFrame(
                        {
                            "samples": rA[ct].obs[self.sample_id],
                            "cells": rA[ct].obs.index,  # cell identifiers
                            "cell.type": ct,  # using the key as the cell type
                        },
                        index=rA[ct].obs.index,
                    ),
                    # Optionally, if you have cell-specific metadata (e.g. rA[ct].uns["metadata"]) that is a DataFrame:
                    # , pd.DataFrame(rA[ct].uns["metadata"], index=rA[ct].obs.index)
                ],
                axis=1,
            )
            for ct in cell_types
        }

        # 14. Set output name and compute MCPs from signatures.
        R["name"] = "DLG.output_" + main
        R["MCPs.full"] = self._sig2MCP(R["sig1"])
        R["MCPs"] = self._sig2MCP(R["sig2"])

        R["cca_fit"] = {
            ct: np.array(
                [
                    np.corrcoef(R["residual_scores"][ct].iloc[:, j], R["scores"][ct].iloc[:, :k].to_numpy()[:, j])[0, 1]
                    for j in range(min(R["residual_scores"][ct].shape[1], R["scores"][ct].iloc[:, :k].shape[1]))
                ]
            )
            for ct in cell_types
        }

        # Optionally, if you want a DataFrame with cell_types as index:
        R["cca_fit"] = pd.DataFrame(R["cca_fit"]).T
        R["cca_fit"].index = cell_types

        # 16. If a phenotype parameter is provided, compute phenoZ.
        # if R["param"].get("pheno") is not None:
        #    R["phenoZ"] = self.DIALOGUE_pheno(R, pheno=R["param"]["pheno"])

        # 17. Build a reduced dictionary containing selected keys.
        # keys_to_keep = ["cell.types", "scores", "gene.pval", "param", "MCP.cell.types", "MCPs",
        #                "MCPs.full", "emp.p", "pref", "k", "name", "phenoZ"]
        # R1 = {k: R[k] for k in R.keys() if k in keys_to_keep}

        # 18. (Ignoring file saving and cleanup in this version.)

        return R

    def _dlg_multi_get_gene_pval(self, cell_type, R):
        """
        Mimics the R function DLG.multi.get.gene.pval.

        Parameters:
        cell_type : str
            The cell type of interest.
        R : dict
            A dictionary containing results from previous steps.
            Expected to have keys such as "cca.sig", "MCP.cell.types", etc.

        Returns:
        pd.DataFrame or None:
            A DataFrame with gene-level p-value information and additional metrics,
            or None if no comparisons are found.
        """

        # 1. Identify the keys in R that correspond to pairwise comparisons.
        all_keys = list(R.keys())

        # 3. Identify keys where cell_type appears as the first part and as the second part.
        b1 = [("vs." in key) and (key.split(".vs.")[0] == cell_type) for key in all_keys]
        b2 = [("vs." in key) and (key.split(".vs.")[1] == cell_type) for key in all_keys]

        if (sum(b1) + sum(b2)) == 0:
            return None

        # 4. Define helper function f1.
        def f1(m1, pi="p1"):
            """
            Given a result element m1 (a dictionary or similar structure),
            extract the component indicated by pi ("p1" or "p2") and then modify its row labels.
            We assume that m1[pi] is a pandas DataFrame with columns including 'program', 'up', 'genes', and 'Z'.
            """
            x = m1[pi].copy()  # copy the DataFrame
            # Set the index to a string concatenation of program, a suffix, and genes.
            # (This replicates: paste0(x$program, ifelse(x$up, ".up_", ".down_"), x$genes) in R.)
            new_index = x.apply(
                lambda row: f"{row['program']}{'.up_' if row['up'] else '.down_'}{row['genes']}", axis=1
            )
            x.index = new_index
            return x

        # 5. Apply f1 to each relevant component in R.
        # For keys where cell_type is the first part, use pi = "p1".
        m_list_1 = [f1(R[key]) for key in all_keys if ("vs." in key and key.split(".vs.")[0] == cell_type)]
        # For keys where cell_type is the second part, use pi = "p2".
        m_list_2 = [f1(R[key], pi="p2") for key in all_keys if ("vs." in key and key.split(".vs.")[1] == cell_type)]
        # Combine the lists.
        m_list = m_list_1 + m_list_2

        # 6. Get all unique row labels from all data frames in m_list.
        g = pd.unique(np.concatenate([df.index.values for df in m_list]))

        # 7. Use a helper function get_strsplit to split each gene identifier in g by "_" and keep the first two parts.
        # For example, if an element of g is "MCP1.up_AC009501.4", we want ["MCP1.up", "AC009501.4"].
        # (Assume self.get_strsplit returns a pandas DataFrame with two columns.)
        def split_identifier(identifier):
            parts = identifier.split("_")
            return (parts[0], parts[1] if len(parts) >= 2 else "")

        split_list = [split_identifier(name) for name in g]
        p = pd.DataFrame(split_list, columns=["programF", "genes"], index=g)

        # 8. Add columns to p:
        # 'program': the first part of g when splitting by ".", and
        # 'up': whether "up" is in g.
        p["program"] = [name.split(".")[0] if "." in name else name for name in g]
        p["up"] = [("up" in name) for name in g]

        # 9. Modify the names of m_list elements:
        # Create a dictionary mapping a modified key (removing cell_type from the key) to the corresponding DataFrame.
        m_dict = {}
        for key in all_keys:
            if "vs." in key:
                parts = key.split(".vs.")
                if parts[0] == cell_type:
                    new_key = parts[1]
                elif parts[1] == cell_type:
                    new_key = parts[0]
                else:
                    continue
                # If key already exists, you might want to handle duplicates appropriately.
                m_dict[new_key] = f1(R[key]) if parts[0] == cell_type else f1(R[key], pi="p2")

        # 10. For each key in m_dict, update p by adding a new column with the Z-values.
        for key, df in m_dict.items():
            # Construct g1 for each row of df in the same way.
            g1 = df.apply(lambda row: f"{row['program']}{'.up_' if row['up'] else '.down_'}{row['genes']}", axis=1)
            # Create a mapping from these constructed names to the Z values.
            z_mapping = dict(zip(g1, df["Z"], strict=True))
            # For each row in p, map its index to a Z value if available.
            p[key] = p.index.to_series().map(lambda name, zm=z_mapping: zm.get(name, np.nan))

        # 11. Next, compute adjusted p-values.
        # Extract the columns from p that correspond to the m_dict keys (assumed to start at column 5 in R).
        # (If p has fewer than 5 columns, adjust accordingly.)
        cols_for_pvals = list(m_dict.keys())

        if len(p.columns) < 5:
            raise ValueError("Not enough columns in p for p-value adjustments.")

        # Assume that self.get_pval_from_zscores converts z-scores to p-values.
        pvals = self._get_pval_from_zscores(p.loc[:, cols_for_pvals])

        p_up = self._p_adjust_mat_per_label(pvals, p["programF"])

        p_down = self._p_adjust_mat_per_label(self._get_pval_from_zscores(-p.loc[:, cols_for_pvals]), p["programF"])

        # Ensure these are DataFrames.
        p_up = pd.DataFrame(p_up) if not isinstance(p_up, pd.DataFrame) else p_up
        p_down = pd.DataFrame(p_down) if not isinstance(p_down, pd.DataFrame) else p_down

        # 12. Combine p with additional columns.
        part1 = p[cols_for_pvals + ["programF", "genes"]]
        fisher_up = self._fisher_combine(p_up)

        fisher_down = self._fisher_combine(p_down)

        if not isinstance(fisher_up, pd.Series):
            fisher_up = pd.Series(fisher_up, index=p.index)
        else:
            fisher_up.index = p.index

        if not isinstance(fisher_down, pd.Series):
            fisher_down = pd.Series(fisher_down, index=p.index)
        else:
            fisher_down.index = p.index

        n_up = np.nansum(p_up < 0.1, axis=1)
        nf_up = np.nanmean(p_up < 0.1, axis=1)
        n_down = np.nansum(p_down < 0.1, axis=1)
        nf_down = np.nanmean(p_down < 0.1, axis=1)

        part2 = pd.DataFrame(
            {
                "p.up": fisher_up,
                "p.down": fisher_down,
                "n.up": n_up,
                "nf.up": nf_up,
                "n.down": n_down,
                "nf.down": nf_down,
                "program": p["program"],
                "up": p["up"],
            },
            index=p.index,
        )

        m_final = pd.concat([part1, part2], axis=1)

        m_final["N"] = m_final["n.up"]
        m_final.loc[~m_final["up"], "N"] = m_final.loc[~m_final["up"], "n.down"]
        m_final["Nf"] = m_final["nf.up"]
        m_final.loc[~m_final["up"], "Nf"] = m_final.loc[~m_final["up"], "nf.down"]
        m_final.loc[~m_final["up"], "p.up"] = 1
        m_final.loc[m_final["up"], "p.down"] = 1

        return m_final

    def _dlg_find_scoring(self, r1, R):
        """
        Mimics the R function DLG.find.scoring.

        Parameters:
        r1: dict
            A cell-type object (converted from an AnnData or similar) with keys such as:
            - "name": the cell type name
            - "genes": a list (or pandas Index) of gene names
            - "X": a pandas DataFrame of cell-level expression (cells × features) whose columns are named
                    so that subsetting by a set of feature names (e.g. rownames(WS)) is possible.
            - "tpm": a pandas DataFrame of TPM values (genes × samples) with gene names as index.
            - "metadata": a DataFrame of cell-level metadata.
            - "samples": a pandas Series or list of sample names.
            - "extra_scores": a dict that will be updated with new computed scores.
        R: dict
            A results dictionary containing keys such as "gene.pval", "cca", "MCP.cell.types", etc.

        Returns:
        Updated r1 with computed scores, zscores, residuals, and signatures.
        """
        # 1) Get gene.pval for this cell type.
        # print(R["gene.pval"])
        cell_type_val = np.unique(r1.obs[self.celltype_key])[0]
        gene_pval = R["gene.pval"].get(cell_type_val)
        if gene_pval is None:
            print("No MCPs identified.")
            r1 = self._dlg_initialize(r1, R)
            return r1

        # 3) Get the sorted unique gene names from gene.pval.
        g = sorted(gene_pval["genes"].unique())

        WS = R["ws"][cell_type_val]

        result = r1.obsm["X_pca"].dot(WS)

        # Create column names for the resulting DataFrame.
        n_cols = result.shape[1]
        col_names = [f"MCP{i+1}" for i in range(n_cols)]

        # Create a DataFrame with the same index as r1.obs.
        r1.obsm["cca_zero"] = pd.DataFrame(result, index=r1.obs.index, columns=col_names)

        # Convert r1.X (a numpy array) into a DataFrame with cells as rows and genes as columns.
        expr_df = pd.DataFrame(r1.X, index=r1.obs_names, columns=r1.var_names)

        # Transpose so that rows correspond to genes.
        gene_expr = expr_df.T
        # Select only the genes in g.

        expr_sub = gene_expr.loc[g, :]

        zscores_matrix = self._center_large_matrix(expr_sub, sd_flag=True)

        # For example, if you want to store the full zscores matrix in r1.uns:
        r1.obsm["zscores"] = zscores_matrix.T

        # 7) Define a helper function that, for each column name in cca0, computes iterative NNLS scores.
        def f(col_name):
            # y is the column of cca0 corresponding to col_name.
            y = r1.obsm["cca_zero"].loc[:, col_name]

            # b: boolean mask where gene_pval["program"] equals the current column name.
            b = gene_pval["program"] == col_name

            if not b.any():
                # If no entries, return the vector y and gene.pval = None.
                return {"gene_pval": None, "scores": y}

            gene_pval_subset = gene_pval[b].copy()

            # X: take the rows of r1["zscores"] corresponding to the genes in gene_pval_subset, then transpose.
            # (R: X <- t(r1@zscores[gene.pval$genes, ]))
            zscores_df = pd.DataFrame(r1.obsm["zscores"]).T
            # Now index the DataFrame by the genes of interest, then transpose.
            X = zscores_df.loc[gene_pval_subset["genes"]]

            # For columns of X that correspond to genes that are NOT up (i.e. gene_pval_subset["up"] is False),
            # multiply the values by -1.
            genes_not_up = gene_pval_subset.loc[~gene_pval_subset["up"], "genes"]
            # Create a boolean mask for X’s columns.

            X = X.T

            mask = X.columns.isin(genes_not_up)

            X.loc[:, X.columns[mask]] *= -1
            # Call the iterative NNLS function; assume it returns a dictionary with key "coef"
            # X = X.T
            gene_pval_iter = self._dlg_iterative_nnls(X, y, gene_pval_subset)
            coef_series = pd.Series(gene_pval_iter["coef"].values, index=X.columns)
            scores = X.dot(coef_series)

            return {"gene_pval": gene_pval_iter, "scores": scores}

        # 8) Apply the function f to each column name in the cca0 matrix.
        # In R: m <- lapply(colnames(r1@extra.scores$cca0), f)

        m = {col: f(col) for col in r1.obsm["cca_zero"].columns}

        scores_list = [np.array(item["scores"]) for item in m.values() if item["scores"] is not None]

        if scores_list:
            # Stack rows into a 2D array then transpose.
            nnl0 = np.vstack(scores_list).T

            r1.obsm["nnl0"] = pd.DataFrame(nnl0, index=r1.obs_names, columns=r1.obsm["cca_zero"].columns)
        else:
            r1.obsm["nnl0"] = None

        cca0_transposed = r1.obsm["cca_zero"].T
        if self.conf is None:
            r1.obsm["cca"] = cca0_transposed.T
        else:
            conf_m = r1.obs[self.conf]
            r1.obsm["cca"] = self._get_residuals(conf_m.values, cca0_transposed.values).T

        nnl0_transposed = r1.obsm["nnl0"].T if r1.obsm["nnl0"] is not None else None
        if nnl0_transposed is not None:
            if self.conf is None:
                r1.obsm["scores"] = nnl0_transposed.T
            else:
                conf_m = r1.obs[self.conf]
                r1.obsm["scores"] = self._get_residuals(conf_m.values, nnl0_transposed.values).T
        else:
            r1.obsm["scores"] = None

        df_scores = pd.DataFrame(r1.obsm["scores"], index=r1.obs.index)
        r1.uns["scoresAv"] = df_scores.groupby(r1.obs[self.sample_id]).median()

        # 12) Combine gene.pval results from each element of m.
        gene_pval_list = []
        for item in m.values():
            if item["gene_pval"] is not None:
                gene_pval_list.append(item["gene_pval"])
        if gene_pval_list:
            r1.uns["gene_pval"] = pd.concat(gene_pval_list, axis=0)
        else:
            r1.uns["gene_pval"] = None

        # 13) Create copies m1 and m2 of r1.gene_pval.
        m1 = r1.uns["gene_pval"].copy() if r1.uns["gene_pval"] is not None else pd.DataFrame()
        m2 = r1.uns["gene_pval"].copy() if r1.uns["gene_pval"] is not None else pd.DataFrame()

        # 14) Compute the number of cells for each MCP program from R["MCP.cell.types"].
        idx = {mcp: len(cells) for mcp, cells in R["samples_cells"].items()}

        # 15) For m1, add a new column "n.cells" from the mapping in idx.
        if not m1.empty:
            m1["n.cells"] = m1["program"].map(idx)
            lb = np.ceil(m1["n.cells"] / 2)
            # Filter m1 based on conditions.

            m1 = m1[
                (m1["coef"] > 0)
                | ((m1["n.up"] >= lb) & (m1["p.up"] < 1e-3))
                | ((m1["n.down"] >= lb) & (m1["p.down"] < 1e-3))
            ]
            # Filter m2.
            m2 = m2[(m2["Nf"] == 1) & ((m2["p.up"] < 0.05) | (m2["p.down"] < 0.05))]
        else:
            lb = None

        # 16) Split the gene lists by programF to form signatures.
        sig1_dict = {}
        sig2_dict = {}
        if not m1.empty:
            for prog, grp in m1.groupby("programF"):
                sig1_dict[prog] = grp["genes"].tolist()
        if not m2.empty:
            for prog, grp in m2.groupby("programF"):
                sig2_dict[prog] = grp["genes"].tolist()

        for mcp, genes in sig1_dict.items():
            print(f"{mcp}: {len(genes)}")

        r1.uns["sig"] = {"sig1": sig1_dict, "sig2": sig2_dict}

        return r1

    def _center_large_matrix(self, m, sd_flag, v=None):
        """
        Centers (and optionally scales) each row of a DataFrame.

        Parameters:
        m : pandas.DataFrame
            The input matrix with rows to center (and scale).
        sd_flag : bool
            If True, divide each row by its standard deviation.
        v : pandas.Series or None, optional
            If provided, these values are used as the row means; otherwise,
            they are computed from m.

        Returns:
        pandas.DataFrame with centered (and scaled) rows.
        """
        if v is None:
            v = m.mean(axis=1, skipna=True)
        if not sd_flag:
            return m.sub(v, axis=0)
        else:
            row_std = m.std(axis=1, skipna=True)
            return m.sub(v, axis=0).div(row_std, axis=0)

    def _get_pval_from_zscores(self, z):
        """
        Convert z-scores to one-sided p-values, mimicking the R function get.pval.from.zscores.

        For each z:
        - Compute p = 10^(-abs(z))
        - If z is negative, set p = 1 - p

        Parameters:
        -----------
        z : array-like (e.g., numpy array or pandas Series)
            The z-scores.

        Returns:
        --------
        p : numpy.ndarray
            The computed p-values.
        """
        # Convert z to a numpy array (if not already)
        z = np.asarray(z)
        # Compute 10^(-abs(z))
        p = 10 ** (-np.abs(z))
        # For negative z values, set p = 1 - p
        negative_mask = (z < 0) & ~np.isnan(z)
        p[negative_mask] = 1 - p[negative_mask]
        return p

    def _p_adjust_multiple_tests(self, p_vals):
        #    """
        #    Adjust a 1D array of p-values using the Benjamini-Hochberg FDR procedure.
        #    Parameters:
        #        p_vals : array-like
        #            An array of p-values.
        #    Returns:
        #        np.ndarray of adjusted p-values.
        #    """
        p_vals = np.asarray(p_vals)
        # multipletests returns: rejected, pvals_corrected, alphacS, alphacBonf
        _, pvals_corrected, _, _ = multipletests(p_vals, method="fdr_bh")
        return pvals_corrected

    def _p_adjust_mat(self, p_df):
        """
        Adjust a DataFrame or Series of p-values column-wise.

        Parameters:
            p_df : pandas.DataFrame or pandas.Series
                Input p-values.

        Returns:
            Adjusted p-values in the same structure (DataFrame or Series).
        """
        # If the input is a Series, adjust it directly.
        if isinstance(p_df, pd.Series):
            return pd.Series(self._p_adjust_multiple_tests(p_df.values), index=p_df.index)
        # If the input is a DataFrame, apply adjustment to each column.
        elif isinstance(p_df, pd.DataFrame):
            return p_df.apply(lambda col: pd.Series(self._p_adjust_multiple_tests(col.values), index=col.index), axis=0)
        else:
            raise ValueError("Input p must be a pandas Series or DataFrame.")

    def _p_adjust_mat_per_label(self, p, v):
        """
        Mimics the R function p.adjust.mat.per.label.

        Given a DataFrame (or numpy array) p of p-values and an array-like v (with one label per row of p),
        create a new DataFrame p1 (with the same index and columns as p) in which the p-values have been
        adjusted separately for each unique label in v.

        Parameters:
            p : pandas.DataFrame or numpy.ndarray
                A DataFrame (or 2D numpy array) of p-values.
            v : array-like
                An array-like object with one value per row of p.

        Returns:
            p1 : pandas.DataFrame
                A DataFrame of the same shape as p containing the adjusted p-values.
        """
        # If p is not a DataFrame, convert it.
        if not isinstance(p, pd.DataFrame):
            p = pd.DataFrame(p)

        # Create a new DataFrame with the same index and columns as p, filled with NaN.
        p1 = pd.DataFrame(np.nan, index=p.index, columns=p.columns)

        # Get all unique labels from v.
        unique_labels = pd.unique(v)

        # For each unique label, create a boolean mask and adjust the p-values for that group.
        for label in unique_labels:
            mask = np.array(v) == label
            # If there is only one column, adjust the vector.
            if p1.shape[1] < 2:
                # self._p_adjust is assumed to adjust a 1D array of p-values.
                p1.loc[mask] = self._p_adjust(p.loc[mask].values)
            else:
                # Otherwise, adjust the DataFrame columnwise.
                p1.loc[mask] = self._p_adjust_mat(p.loc[mask])

        return p1

    def _get_fisher_p_value(self, p_values):
        """
        Combine a vector of p-values using Fisher's method.

        Parameters:
            p_values (array-like): A list or array of p-values.

        Returns:
            float: The combined p-value.
        """
        # Convert to a numpy array and remove NaNs.
        p_values = np.array(p_values, dtype=float)
        p_values = p_values[~np.isnan(p_values)]
        if len(p_values) == 0:
            return np.nan
        # Compute Fisher's test statistic.
        chi_stat = -2 * np.sum(np.log(p_values))
        df = 2 * len(p_values)
        # Combined p-value.
        p_combined = 1 - stats.chi2.cdf(chi_stat, df)
        return p_combined

    def _fisher_combine(self, p):
        """
        Mimics the R function fisher.combine.

        Applies Fisher's method row-wise to combine p-values from a DataFrame.

        Parameters:
            p (pandas.DataFrame): A DataFrame where each row contains a set of p-values to be combined.

        Returns:
            pandas.Series: A Series containing the combined p-value for each row.
        """
        return p.apply(lambda row: self._get_fisher_p_value(row), axis=1)

    def _dlg_iterative_nnls(self, X, y, gene_pval):
        """
        Python version of DLG.iterative.nnls from R.

        This function applies iterative nonnegative least squares (NNLS) fitting based on the
        'f_rank' column in 'gene_pval', mimicking the R code you provided. It updates 'y'
        (the residuals) and 'y_fit' (the cumulative fitted values) at each iteration, and
        stops early if a correlation threshold is met. Finally, if columns remain with
        f_rank < last_n1, it does one more NNLS step on those columns.

        Args:
            X (pd.DataFrame): Shape (n_samples, n_features). Rows are samples, columns are genes/features.
            y (np.ndarray): A 1D array of shape (n_samples,) representing the response variable.
            gene_pval (pd.DataFrame): Must contain:
                - "Nf" (float) = f_rank values
                - "coef" (float) = updated in each iteration

        Returns:
            pd.DataFrame: Updated 'gene_pval' with final "coef" values.
        """
        # Set the random seed for reproducibility.
        np.random.default_rng(1234)

        # f.rank corresponds to gene_pval$Nf in R.
        f_rank = gene_pval["Nf"].values
        y1 = y.copy()  # original response
        y_fit = np.zeros_like(y)  # initialize fitted values to zero
        v = {}  # to store NNLS results per iteration

        # Initialize the "coef" column to 0.
        gene_pval["coef"] = 0

        # Get the unique values in f_rank in descending order and keep only those >= 1/3.
        unique_vals = np.unique(f_rank[~np.isnan(f_rank)])
        idx_vals = sorted(unique_vals, reverse=True)
        idx_vals = [val for val in idx_vals if val >= (1 / 3)]

        last_n1 = None
        for n1 in idx_vals:
            # Boolean mask for rows where f_rank equals n1.
            b1 = f_rank == n1
            if np.nansum(b1) < 5:
                continue  # skip if fewer than 5 entries

            last_n1 = n1

            # Subset X's columns corresponding to True in b1.
            # (Assumes X's columns correspond to the genes represented in gene_pval.)
            X1 = X.iloc[:, b1]

            main_str = "N" + str(n1)
            # Run NNLS on X1 and y.
            sol, rnorm = nnls(X1, y)
            fitted = X1.dot(sol)
            residuals = y - fitted

            # Store the NNLS result.
            v[main_str] = {"x": sol, "fitted": fitted, "residuals": residuals}

            # Update y to be the residuals.
            y = residuals
            # Accumulate the fitted values.
            y_fit = y_fit + fitted
            # Update gene_pval "coef" for rows where f_rank equals n1.
            gene_pval.loc[gene_pval["Nf"] == n1, "coef"] = sol

            # If sufficient variability exists and the correlation between y1 and y_fit is > 0.95, return gene_pval.
            if len(np.unique(y_fit)) > 10 and np.corrcoef(y1, y_fit)[0, 1] > 0.95:
                # Optionally, you could plot y_fit vs y1 here.
                # Optionally compute a linear regression line for y_fit vs y1.
                slope, intercept = np.polyfit(y_fit, y1, 1)
                x_vals = np.linspace(y_fit.min(), y_fit.max(), 100)
                y_vals = slope * x_vals + intercept

                plt.figure(figsize=(6, 6))
                plt.scatter(y_fit, y1, color="black", s=5)  # small black points
                plt.plot(x_vals, y_vals, color="red")  # regression line in red
                plt.title(f"NNLS fitting - {n1}")
                plt.xlabel("y_fit")
                plt.ylabel("y1")
                plt.show()
                return gene_pval

        # After iterating over the high-rank columns, check if there are fewer than 5 entries for f_rank < n1.

        if last_n1 is None:
            return gene_pval

        # Otherwise, process the remaining columns where f_rank < n1.
        X1 = X.iloc[:, f_rank < last_n1]
        main_str = "Ns" + str(last_n1)
        sol, rnorm = nnls(X1, y)
        fitted = X1.dot(sol)
        residuals = y - fitted
        v[main_str] = {"x": sol, "fitted": fitted, "residuals": residuals}
        y = residuals
        y_fit = y_fit + fitted
        gene_pval.loc[gene_pval["Nf"] < last_n1, "coef"] = sol

        # Optionally compute a linear regression line for y_fit vs y1.
        slope, intercept = np.polyfit(y_fit, y1, 1)
        x_vals = np.linspace(y_fit.min(), y_fit.max(), 100)
        y_vals = slope * x_vals + intercept

        plt.figure(figsize=(6, 6))
        plt.scatter(y_fit, y1, color="black", s=5)  # small black points
        plt.plot(x_vals, y_vals, color="red")  # regression line in red
        plt.title(f"NNLS fitting - {n1}")
        plt.xlabel("y_fit")
        plt.ylabel("y1")
        plt.show()

        return gene_pval

    def _dlg_hlm_pval(self, r1, r2, formula="y ~ (1 | samples) + x + cellQ"):
        """
        Mimics the R function DLG.hlm.pval.

        Parameters:
        r1, r2 : cell‐type objects (for example, AnnData objects converted
                to dictionaries via cell_type_2_list) for two cell types.
        formula : str, optional
                The mixed‐effects model formula. (Default: "y ~ (1 | samples) + x + cellQ")

        Returns:
        m_df : pandas.DataFrame
            A DataFrame with row index equal to the unique covariate names (derived
            from the column names of r1["scores"]) and two columns containing the p‐values
            computed via formula_HLM for r1 and r2 respectively.

        In the R code:
        1. An index is built from c("samples","scores","tme.OE",…) where covariate names are
            extracted from the formula.
        2. r1 and r2 are converted via cell.type.2.list.
        3. A helper function f(x) is defined that computes a vector c(p1, p2) for each covariate x.
        4. Then the p-values (the 2nd and 4th elements) are extracted from each vector.
        """
        # --- Step 1. Build an index vector.
        # For example, we want to include "samples", "scores", "tme_OE" plus any covariate names.
        # Here we simply set:
        idx = ["samples", "scores", "tme_OE"]
        # (Optionally, you might extract additional covariates from the formula string.)

        # --- Step 2. Convert r1 and r2 into a dictionary (list) with only the desired slots.
        r1 = self._cell_type_2_list(r1, idx=idx)
        r2 = self._cell_type_2_list(r2, idx=idx)

        # --- Step 3. Define a helper function f(x) that computes the model p-values.
        # For a given covariate x, it calls formula_HLM on r1 and r2.
        def f(x):
            # Here, we assume that r1["scores"] and r1["tme_OE"] are pandas DataFrames
            # and that column x exists in each.

            p1 = self._formula_HLM(
                r0=r1, y=r1["obsm"]["scores"].loc[:, x], x=r1["uns"]["tme_OE"].loc[:, x], formula=formula
            )
            p2 = self._formula_HLM(
                r0=r2, y=r2["obsm"]["scores"].loc[:, x], x=r2["uns"]["tme_OE"].loc[:, x], formula=formula
            )
            # We assume formula_HLM returns a Series with indices ["Estimate", "P"].
            # Concatenate the results so that we have: [r1_Est, r1_P, r2_Est, r2_P]
            return [p1["Estimate"], p1["P"], p2["Estimate"], p2["P"]]

        # --- Step 4. Build the unique list of covariate names.
        # In the R code, idx is built as the unique first part from splitting column names of r1$scores by "."

        covariate_keys = list(dict.fromkeys(col.split(".")[0] for col in r1["obsm"]["scores"].columns))

        covariate_keys = list(covariate_keys)
        # --- Step 5. For each covariate, apply f.
        # This will produce an array with one row per covariate and 4 columns.
        m_array = np.array([f(key) for key in covariate_keys])

        # --- Step 6. Select the 2nd and 4th columns (i.e. the p-values from r1 and r2).
        # (Remember: Python indexing is 0-based, so these are columns 1 and 3.)
        m_result = m_array[:, [1, 3]]

        # --- Step 7. Build a DataFrame from the result.
        m_df = pd.DataFrame(m_result, index=covariate_keys, columns=["r1_p", "r2_p"])
        return m_df

    def _sig2MCP(self, R_sig, k=5):
        """
        Mimics the R function sig2MCP but also includes the cell type in the key names.

        In R, the function flattens R.sig, keeps only keys containing "up" or "down",
        then for each MCP (from 1 to k) filters for keys starting with "MCPx." and removes that prefix.
        It then checks that the resulting keys (split by a dot) have at least two unique first parts.
        If not, that MCP is removed.

        In this Python version, we assume R_sig is a dictionary whose keys are cell type labels (e.g. "A", "B", "C")
        and whose values are themselves dictionaries with keys like "MCP1.down" and "MCP1.up". The output will be a
        dictionary with keys "MCP1", "MCP2", ... where each value is a dictionary with keys like "A.down", "A.up", etc.

        Parameters:
            R_sig: dict
                Dictionary of gene signature dictionaries keyed by cell type.
            k: int, default 5
                Number of MCPs to process.

        Returns:
            dict: A dictionary with keys "MCP1", ..., "MCPk". For each MCP,
                if there are at least two unique cell type labels, a dictionary is returned
                mapping "cellType.suffix" (e.g. "A.down", "B.up") to the corresponding gene list;
                otherwise, the value is None.
        """
        # Flatten while preserving the cell type.
        flat = {}
        for cell, subdict in R_sig.items():
            if subdict is None:
                continue
            for key, val in subdict.items():
                # We only want keys that contain "up" or "down"
                if "up" in key or "down" in key:
                    # Expect key to be like "MCP1.down" or "MCP1.up"
                    if key.startswith("MCP"):
                        parts = key.split(".", 1)
                        if len(parts) != 2:
                            continue
                        mcp_key, suffix = parts  # e.g. mcp_key = "MCP1", suffix = "down"
                        # Create a new key that includes the cell type: e.g., "A.down"
                        new_key = f"{cell}.{suffix}"
                        if mcp_key not in flat:
                            flat[mcp_key] = {}
                        flat[mcp_key][new_key] = val
        # Now build the output dictionary for each MCP from 1 to k.
        MCPs = {}
        for i in range(1, k + 1):
            mcp_key = f"MCP{i}"
            if mcp_key in flat:
                # Get the cell type prefixes from the keys (e.g., "A" from "A.down")
                cell_types = {key.split(".")[0] for key in flat[mcp_key].keys()}
                if len(cell_types) < 2:
                    print(f"Removing {mcp_key}")
                    MCPs[mcp_key] = None
                else:
                    MCPs[mcp_key] = flat[mcp_key]
            else:
                MCPs[mcp_key] = None
        return MCPs

    def DIALOGUE_pheno(self, R, pheno="clin.status", cca_flag=False, rA=None, frm=None, selected_samples=None):
        """
        Mimics the R function DIALOGUE.pheno.

        Parameters:
        R : dict
            Dictionary with keys such as "k", "frm", "covar", "cell.types", "cca.scores", "metadata", etc.
        pheno : str, default "clin.status"
            The name of the phenotype column.
        cca_flag : bool, default False
            If True, update R["scores"] by combining cca.scores and metadata.
        rA : object
            Provided for metadata addition if needed.
        frm : str or None
            A formula string; if None, derived from R["frm"] by removing occurrences of "+tme.qc" and the phenotype.
        selected_samples : list or None
            If provided, only these samples will be kept.

        Returns:
        Z_df : pd.DataFrame
            A DataFrame whose rows correspond to each cell type (plus an "All" row) and whose columns are the
            subset of columns from the first R["scores"] DataFrame that contain "MCP".
        """
        import numpy as np
        import pandas as pd

        # 1. Get k from R["k"]["DIALOGUE"]
        k = R["k"]["DIALOGUE"]

        # 2. If no formula is provided, derive it from R["frm"]
        if frm is None:
            frm = R["frm"].replace("+tme.qc", "").replace("+ tme.qc", "")
            frm = frm.replace("+" + pheno, "").replace("+ " + pheno, "")

        # 3. Define helper function f(X) that processes a DataFrame X.
        def f(X):
            # a) Get covariates excluding "tme.qc"
            covar = [c for c in R["covar"] if c != "tme.qc"]
            # b) Build a dictionary of covariate columns
            r1 = {var: X[var] for var in covar if var in X.columns}
            # c) Add the phenotype column
            r1["pheno"] = X[pheno]
            # d) If pheno is not boolean, convert it by comparing to its smallest unique value.
            if not pd.api.types.is_bool_dtype(r1["pheno"]):
                unique_vals = sorted(r1["pheno"].dropna().unique())
                if unique_vals:
                    r1["pheno"] = r1["pheno"] == unique_vals[0]
                else:
                    r1["pheno"] = r1["pheno"].astype(bool)
            # e) Add scores: use the first k columns (as a NumPy array)
            r1["scores"] = X.iloc[:, :k].to_numpy()
            # f) Also add the samples column from X
            r1["samples"] = X["samples"]
            # g) If any cells have missing phenotype, print a message and subset r1.
            if r1["pheno"].isna().any():
                n_missing = r1["pheno"].isna().sum()
                print(f"Identified {n_missing} cells with no phenotype.")
                mask = r1["pheno"].notna()
                for key, val in r1.items():
                    if isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
                        r1[key] = val.loc[mask]
                    elif isinstance(val, np.ndarray):
                        r1[key] = val[mask.to_numpy()]
            # h) If selected_samples is provided, subset r1 accordingly.
            if selected_samples is not None:
                mask = r1["samples"].isin(selected_samples)
                for key, val in r1.items():
                    if isinstance(val, pd.Series) or isinstance(val, pd.DataFrame):
                        r1[key] = val.loc[mask]
                    elif isinstance(val, np.ndarray):
                        r1[key] = val[mask.to_numpy()]
            # i) Apply the formula-based HLM. Assume self.apply_formula_HLM is defined.
            result = self.apply_formula_HLM(r=r1, Y=r1["scores"], X=r1["pheno"], margin=2, formula=frm)
            # Extract the first column of the result.
            if isinstance(result, pd.DataFrame):
                z = result.iloc[:, 0]
            else:
                z = result[:, 0]
            return z

        # 4. If cca_flag is True, update R["scores"] by combining cca.scores and metadata.
        if cca_flag:
            if R.get("metadata") is None:
                R = self.DLG_add_metadata(R, rA=rA)
            # For each cell type, concatenate the cca.scores and metadata DataFrames.
            R["scores"] = {
                ct: pd.concat([R["residual_scores"][ct], R["metadata"][ct]], axis=1) for ct in R["cell.types"]
            }

        # 5. Apply f() to each cell type's scores.
        Z_list = [f(R["scores"][ct]) for ct in R["cell.types"]]
        Z = np.array(Z_list)

        # 6. Combine all scores from all cell types (row-binding) into one DataFrame X.
        X = pd.concat([R["scores"][ct] for ct in R["cell.types"]], axis=0)

        # 7. Append "cell.type" to the covariate list.
        R["covar"].append("cell.type")

        # 8. Determine the column names for Z: select columns from the first scores DataFrame that contain "MCP".
        first_scores = R["scores"][R["cell.types"][0]]
        cols = [col for col in first_scores.columns if "MCP" in col]

        # 9. Compute f(X) on the combined DataFrame X.
        z_all = f(X)

        # 10. Append z_all as an extra row to Z.
        Z = np.vstack([Z, z_all])

        # 11. Build row names: one per cell type, plus "All".
        row_names = R["cell.types"] + ["All"]

        # 12. Build the final DataFrame.
        Z_df = pd.DataFrame(Z, index=row_names, columns=cols)

        return Z_df

    # def multilevel_modeling(
    #     self,
    #     ct_subs: dict,
    #     mcp_scores: dict,
    #     ws_dict: dict,
    #     confounder: str | None,
    #     formula: str = None,
    # ):
    #     """Runs the multilevel modeling step to match genes to MCPs and generate p-values for MCPs.

    #     Args:
    #         ct_subs: The DIALOGUE cell type objects.
    #         mcp_scores: The determined MCP scores from the PMD step.
    #         confounder: Any modeling confounders.
    #         formula: The hierarchical modeling formula. Defaults to y ~ x + n_counts.

    #     Returns:
    #         A Pandas DataFrame containing:
    #         - for each mcp: HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2
    #         - merged HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2 of all mcps

    #     Examples:
    #         >>> import pertpy as pt
    #         >>> import scanpy as sc
    #         >>> adata = pt.dt.dialogue_example()
    #         >>> sc.pp.pca(adata)
    #         >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
    #             n_counts_key = "nCount_RNA", n_mpcs = 3)
    #         >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
    #         >>> all_results, new_mcps = dl.multilevel_modeling(ct_subs=ct_subs, mcp_scores=mcps, ws_dict=ws, \
    #             confounder="gender")
    #     """
    #     # TODO the returns of the function better

    #     # all possible pairs of cell types without pairing same cell type
    #     cell_types = list(ct_subs.keys())
    #     pairs = list(itertools.combinations(cell_types, 2))

    #     if not formula:
    #         formula = f"y ~ x + {self.n_counts_key}"

    #     # Hierarchical modeling expects DataFrames
    #     mcp_cell_types = {f"MCP{i}": cell_types for i in range(self.n_mcps)}
    #     mcp_scores_df = {
    #         ct: pd.DataFrame(v, index=ct_subs[ct].obs.index, columns=list(mcp_cell_types.keys()))
    #         for ct, v in mcp_scores.items()
    #     }

    #     # run HLM for each pair
    #     all_results: dict[str, dict[Any, dict[str, tuple[DataFrame, dict[str, Any]]]]] = {}
    #     mlm_progress = Progress(
    #         SpinnerColumn(),
    #         TextColumn("[progress.description]{task.description}"),
    #         BarColumn(),
    #         TaskProgressColumn(),
    #     )
    #     mixed_model_progress = Progress(
    #         SpinnerColumn(),
    #         TextColumn("[progress.description]{task.description}"),
    #         BarColumn(),
    #         TaskProgressColumn(),
    #         transient=True,
    #     )
    #     group = Group(mlm_progress, mixed_model_progress)
    #     live = Live(group)
    #     with live:
    #         mlm_task = mlm_progress.add_task("[bold blue]Running multilevel modeling", total=len(pairs))

    #         for pair in pairs:
    #             cell_type_1 = pair[0]
    #             cell_type_2 = pair[1]
    #             mlm_progress.update(mlm_task, description=f"[bold blue]{cell_type_1} vs {cell_type_2}")

    #             ct_data_1 = ct_subs[cell_type_1]
    #             ct_data_2 = ct_subs[cell_type_2]

    #             # equivalent to dialogue2.pair
    #             mcps = []
    #             for mcp, cell_type_list in mcp_cell_types.items():
    #                 if cell_type_1 in cell_type_list and cell_type_2 in cell_type_list:
    #                     mcps.append(mcp)

    #             if len(mcps) == 0:
    #                 logger.warning(f"No shared MCPs between {cell_type_1} and {cell_type_2}.")
    #                 continue

    #             logger.info(f"{len(mcps)} MCPs identified for {cell_type_1} and {cell_type_2}.")

    #             new_mcp_scores: dict[Any, list[Any]]
    #             cca_sig, new_mcp_scores = self._calculate_cca_sig(
    #                 ct_subs, mcp_scores=mcp_scores, ws_dict=ws_dict, n_counts_key=self.n_counts_key
    #             )

    #             sig_1 = cca_sig[cell_type_1]  # TODO: only need the up and down genes from this here per MCP
    #             sig_2 = cca_sig[cell_type_2]
    #             # only use samples which have a minimum number of cells (default 2) in both cell types
    #             sample_ids = list(
    #                 set(self._get_abundant_elements_from_series(ct_data_1.obs[self.sample_id]))
    #                 & set(self._get_abundant_elements_from_series(ct_data_2.obs[self.sample_id]))
    #             )

    #             # subset cell types to valid samples (set.cell.types)
    #             ct_data_1 = ct_data_1[ct_data_1.obs[self.sample_id].isin(sample_ids)]
    #             ct_data_2 = ct_data_2[ct_data_2.obs[self.sample_id].isin(sample_ids)]

    #             # TODO: shouldn't need this aligning step for cells. corresponds to @scores / y
    #             #     scores_1 = cca_scores[cell_type_1].loc[ct_data_1.obs.index]
    #             #     scores_2 = cca_scores[cell_type_2].loc[ct_data_2.obs.index]

    #             # indexes into the average sample expression per gene with the sample id per cell. corresponds to @tme / x
    #             # TODO: Why is the sample_id type check failing?
    #             tme_1 = self._get_pseudobulks(ct_data_2, groupby=self.sample_id, strategy="mean").loc[
    #                 :, ct_data_1.obs[self.sample_id]
    #             ]  # unclear why we do this
    #             tme_1.columns = ct_data_1.obs.index
    #             tme_2 = self._get_pseudobulks(ct_data_1, groupby=self.sample_id, strategy="mean").loc[
    #                 :, ct_data_2.obs[self.sample_id]
    #             ]
    #             tme_2.columns = ct_data_2.obs.index

    #             merged_results = {}

    #             mm_task = mixed_model_progress.add_task("[bold blue]Determining mixed effects", total=len(mcps))
    #             for mcp in mcps:
    #                 mixed_model_progress.update(mm_task, description=f"[bold blue]Determining mixed effects for {mcp}")

    #                 # TODO Check whether the genes in result{sig_genes_1] are different and if so note that somewhere and explain why
    #                 result = {}
    #                 result["HLM_result_1"], result["sig_genes_1"] = self._apply_HLM_per_MCP_for_one_pair(
    #                     mcp_name=mcp,
    #                     scores_df=mcp_scores_df[cell_type_2],
    #                     ct_data=ct_data_2,
    #                     tme=tme_2,
    #                     sig=sig_1,
    #                     n_counts=self.n_counts_key,
    #                     formula=formula,
    #                     confounder=confounder,
    #                 )
    #                 result["HLM_result_2"], result["sig_genes_2"] = self._apply_HLM_per_MCP_for_one_pair(
    #                     mcp_name=mcp,
    #                     scores_df=mcp_scores_df[cell_type_1],
    #                     ct_data=ct_data_1,
    #                     tme=tme_1,
    #                     sig=sig_2,
    #                     n_counts=self.n_counts_key,
    #                     formula=formula,
    #                        confounder=confounder,
    #                    )
    #                    merged_results[mcp] = result

    #                    mixed_model_progress.update(mm_task, advance=1)
    #                mixed_model_progress.update(mm_task, visible=False)
    #                mlm_progress.update(mlm_task, advance=1)

    # merge results - TODO, but probably don't need
    #     merged_results['HLM_result_1'] = pd.concat([merged_result[mcp]['HLM_result_1'] for mcp in mcps])
    #     merged_results['HLM_result_2'] = pd.concat([merged_result[mcp]['HLM_result_2'] for mcp in mcps])
    #     merged_results['sig_genes_1'] = [**merged_result[mcp]['sig_genes_1'] for mcp in mcps]
    #     merged_results['sig_genes_2'] = [**merged_result[mcp]['sig_genes_2'] for mcp in mcps]

    #                all_results[f"{cell_type_1}_vs_{cell_type_2}"] = merged_results

    #        return all_results, new_mcp_scores

    def test_association(
        self,
        adata: AnnData,
        condition_label: str,
        residual_scores: dict[str, pd.DataFrame],
        conditions_compare: tuple[str, str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Tests the association between MCPs and a binary response variable (e.g. treatment response)
        using MCP values stored in a dictionary of residual scores. The MCP values are first aggregated
        by sample (using the mean) and then compared between two conditions via t-tests.

        Note: Benjamini–Hochberg correction is applied across cell types (not across MCPs).

        Args:
            adata (AnnData): AnnData object containing condition labels in obs.
            condition_label (str): Column name in adata.obs with condition labels (must be categorical).
            residual_scores (dict[str, pd.DataFrame]): Dictionary where each key is a cell type and each value
                is a DataFrame of MCP values. The DataFrame is not yet aggregated by sample, so each row corresponds
                to a cell and the index contains the sample ID.
            conditions_compare (tuple[str, str] | None, optional): Tuple of length 2 specifying the two conditions to compare.
                If None, the function uses the categories in adata.obs[condition_label] (which must be exactly 2).

        Returns:
            dict[str, pd.DataFrame]: A dictionary with the following keys:
                - "pvals": DataFrame of raw p-values (cell types × MCPs).
                - "tstats": DataFrame of t-statistics (cell types × MCPs).
                - "pvals_adj": DataFrame of adjusted p-values via Benjamini–Hochberg (cell types × MCPs).
        """
        from scipy import stats
        from statsmodels.stats.multitest import multipletests

        sample_label = self.sample_id
        n_mcps = self.n_mcps

        # Generate MCP column names as "MCP1", "MCP2", ..., "MCP{n_mcps}"
        mcp_cols = [f"MCP{n}" for n in range(1, n_mcps + 1)]

        # If conditions to compare are not provided, derive them from adata.obs[condition_label]
        if conditions_compare is None:
            conditions_compare = list(adata.obs[condition_label].cat.categories)  # type: ignore
            if len(conditions_compare) != 2:
                raise ValueError(
                    "Please specify two conditions to compare or supply an object with exactly 2 conditions."
                )

        # Use the keys of the residual_scores dictionary as cell types.
        celltypes = list(residual_scores.keys())
        pvals = pd.DataFrame(1, index=celltypes, columns=mcp_cols)
        tstats = pd.DataFrame(1, index=celltypes, columns=mcp_cols)
        pvals_adj = pd.DataFrame(1, index=celltypes, columns=mcp_cols)

        # Compute a sample-level response as the mode of the condition label.
        response = adata.obs.groupby(sample_label)[condition_label].agg(pd.Series.mode)

        print(response)

        # Loop over each cell type.
        for celltype in celltypes:
            # Get the MCP values DataFrame for this cell type.
            df = residual_scores[celltype]
            # Aggregate by sample (using the mean), similar to: mns = df.groupby(sample_label)[mcpnum].mean()
            mns = df.join(adata.obs[[sample_label]]).groupby(sample_label).mean()
            # Concatenate with the sample-level response.
            mns = pd.concat([mns, response], axis=1)

            for mcp in mcp_cols:
                if mcp not in mns.columns:
                    continue
                group1 = mns[mns[condition_label] == conditions_compare[0]][mcp]
                group2 = mns[mns[condition_label] == conditions_compare[1]][mcp]
                res = stats.ttest_ind(group1, group2, nan_policy="omit")
                pvals.loc[celltype, mcp] = res.pvalue
                tstats.loc[celltype, mcp] = res.statistic

        # Adjust p-values for each MCP (across cell types) using Benjamini–Hochberg.
        for mcp in mcp_cols:
            pvals_adj[mcp] = multipletests(pvals[mcp], method="fdr_bh")[1]

        return {"pvals": pvals, "tstats": tstats, "pvals_adj": pvals_adj}

    def get_mlm_mcp_genes(
        self,
        celltype: str,
        results: dict,
        MCP: str = "mcp_0",
        threshold: float = 0.70,
        focal_celltypes: list[str] | None = None,
    ):
        """Extracts MCP genes from the MCP multilevel modeling object for the cell type of interest.

        Args:
            celltype: Cell type of interest.
            results: dl.MultilevelModeling result object.
            MCP: MCP key of the result object.
            threshold: Number between [0,1]. The fraction of cell types compared against which must have the associated MCP gene.
            focal_celltypes: None (compare against all cell types) or a list of other cell types which you want to compare against.

        Returns:
            Dict with keys 'up_genes' and 'down_genes' and values of lists of genes

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> all_results, new_mcps = dl.multilevel_modeling(ct_subs=ct_subs, mcp_scores=mcps, ws_dict=ws, \
                confounder="gender")
            >>> mcp_genes = dl.get_mlm_mcp_genes(celltype='Macrophages', results=all_results)
        """
        # Convert "mcp_x" to "MCPx" format
        # REMOVE THIS BLOCK ONCE MLM OUTPUT MATCHES STANDARD
        if MCP.startswith("mcp_"):
            MCP = MCP.replace("mcp_", "MCP")
            MCP = "MCP" + str(int(MCP[3:]) - 1)

        # Extract all comparison keys from the results object
        comparisons = list(results.keys())

        filtered_keys = [key for key in comparisons if celltype in key]

        # If focal_celltypes are specified, further filter keys
        if focal_celltypes is not None:
            if celltype in focal_celltypes:
                focal_celltypes = [item for item in focal_celltypes if item != celltype]
            filtered_keys = [key for key in filtered_keys if any(foci in key for foci in focal_celltypes)]

        mcp_dict = {}
        for key in filtered_keys:
            if key.startswith(celltype):
                mcp_dict[key.split("_vs_")[1]] = results[key][MCP]["sig_genes_1"]
            else:
                mcp_dict[key.split("_vs_")[0]] = results[key][MCP]["sig_genes_2"]

        genes_dict_up = {}  # type: ignore
        genes_dict_down = {}  # type: ignore
        for celltype2 in mcp_dict.keys():
            for gene in mcp_dict[celltype2][MCP + ".up"]:
                if gene in genes_dict_up:
                    genes_dict_up[gene] += 1
                else:
                    genes_dict_up[gene] = 1
            for gene in mcp_dict[celltype2][MCP + ".down"]:
                if gene in genes_dict_down:
                    genes_dict_down[gene] += 1
                else:
                    genes_dict_down[gene] = 1

        up_genes_df = pd.DataFrame.from_dict(genes_dict_up, orient="index")
        down_genes_df = pd.DataFrame.from_dict(genes_dict_down, orient="index")

        min_cell_types = np.floor(len(filtered_keys) * threshold)

        final_output = {}
        final_output["up_genes"] = list(np.unique(up_genes_df[up_genes_df[0] >= min_cell_types].index.values.tolist()))
        final_output["down_genes"] = list(
            np.unique(down_genes_df[down_genes_df[0] >= min_cell_types].index.values.tolist())
        )

        return final_output

    def _get_extrema_MCP_genes_single(self, ct_subs: dict, mcp: str = "mcp_0", fraction: float = 0.1):
        """Identifies extreme cells based on their MCP score.

        Takes a dictionary of subpopulations AnnData objects as output from DIALOGUE,
        identifies the extreme cells based on their MCP score for the input mcp,
        calculates rank_gene_groups with default parameters between the high-extreme and low-extreme cells
        and returns a dictionary containing the resulting ct_subs objects with the extreme cells labeled.

        Args:
            ct_subs: Dialogue output ct_subs dictionary
            mcp: The name of the marker gene expression column.
            fraction: Fraction of extreme cells to consider for gene ranking.
                      Should be between 0 and 1.

        Returns:
            Dictionary where keys are subpopulation names and values are Anndata
            objects containing the results of gene ranking analysis.

        Examples:
            >>> ct_subs = {
            ...     "subpop1": anndata_obj1,
            ...     "subpop2": anndata_obj2,
            ...     # ... more subpopulations ...
            ... }
            >>> genes_results = _get_extrema_MCP_genes_single(ct_subs, mcp="mcp_4", fraction=0.2)
        """
        genes = {}
        for ct in ct_subs.keys():
            mini = ct_subs[ct]
            mini.obs["extrema"] = pd.qcut(
                mini.obs[mcp],
                [0, 0 + fraction, 1 - fraction, 1.0],
                labels=["low " + mcp + " " + ct, "no", "high" + mcp + " " + ct],
            )
            sc.tl.rank_genes_groups(
                mini, "extrema", groups=["high" + mcp + " " + ct], reference="low " + mcp + " " + ct
            )
            genes[ct] = mini  # .uns['rank_genes_groups']

        return genes

    def get_extrema_MCP_genes(self, ct_subs: dict, fraction: float = 0.1):
        """Identifies cells with extreme MCP scores.

        Takes as input a dictionary of subpopulations AnnData objects (DIALOGUE output),
        For each MCP it identifies cells with extreme MCP scores, then calls rank_genes_groups to
        identify genes which are differentially expressed between high-scoring and low-scoring cells.

        Args:
            ct_subs: Dialogue output ct_subs dictionary
            fraction: Fraction of extreme cells to consider for gene ranking.
                      Should be between 0 and 1.

        Returns:
            Nested dictionary where keys of the first level are MCPs (of the form "mcp_0" etc)
            and the second level keys are cell types. The values are dataframes containing the
            results of the rank_genes_groups analysis.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> extrema_mcp_genes = dl.get_extrema_MCP_genes(ct_subs)
        """
        rank_dfs: dict[str, dict[Any, Any]] = {}
        ct_sub = next(iter(ct_subs.values()))
        mcps = [col for col in ct_sub.obs.columns if col.startswith("mcp_")]

        for mcp in mcps:
            rank_dfs[mcp] = {}
            ct_ranked = self._get_extrema_MCP_genes_single(ct_subs, mcp=mcp, fraction=fraction)
            for celltype in ct_ranked.keys():
                rank_dfs[mcp][celltype] = sc.get.rank_genes_groups_df(ct_ranked[celltype], group=None)

        return rank_dfs

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_split_violins(
        self,
        adata: AnnData,
        split_key: str,
        celltype_key: str,
        *,
        split_which: tuple[str, str] = None,
        mcp: str = "mcp_0",
        return_fig: bool = False,
    ) -> Figure | None:
        """Plots split violin plots for a given MCP and split variable.

        Any cells with a value for split_key not in split_which are removed from the plot.

        Args:
            adata: Annotated data object.
            split_key: Variable in adata.obs used to split the data.
            celltype_key: Key for cell type annotations.
            split_which: Which values of split_key to plot. Required if more than 2 values in split_key.
            mcp: Key for MCP data.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> dl.plot_split_violins(adata, split_key='gender', celltype_key='cell.subtypes')

        Preview:
            .. image:: /_static/docstring_previews/dialogue_violin.png
        """
        df = sc.get.obs_df(adata, [celltype_key, mcp, split_key])
        if split_which is None:
            split_which = df[split_key].unique()
        df = df[df[split_key].isin(split_which)]
        df[split_key] = df[split_key].cat.remove_unused_categories()

        ax = sns.violinplot(data=df, x=celltype_key, y=mcp, hue=split_key, split=True)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_pairplot(
        self,
        adata: AnnData,
        celltype_key: str,
        color: str,
        sample_id: str,
        *,
        mcp: str = "mcp_0",
        return_fig: bool = False,
    ) -> Figure | None:
        """Generate a pairplot visualization for multi-cell perturbation (MCP) data.

        Computes the mean of a specified MCP feature (mcp) for each combination of sample and cell type,
        then creates a pairplot to visualize the relationships between these mean MCP values.

        Args:
            adata: Annotated data object.
            celltype_key: Key in `adata.obs` containing cell type annotations.
            color: Key in `adata.obs` for color annotations. This parameter is used as the hue
            sample_id: Key in `adata.obs` for the sample annotations.
            mcp: Key in `adata.obs` for MCP feature values.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> dl.plot_pairplot(adata, celltype_key="cell.subtypes", color="gender", sample_id="clinical.status")

        Preview:
            .. image:: /_static/docstring_previews/dialogue_pairplot.png
        """
        mean_mcps = adata.obs.groupby([sample_id, celltype_key])[mcp].mean()
        mean_mcps = mean_mcps.reset_index()
        mcp_pivot = pd.pivot(mean_mcps[[sample_id, celltype_key, mcp]], index=sample_id, columns=celltype_key)[mcp]

        aggstats = adata.obs.groupby([sample_id])[color].describe()
        aggstats = aggstats.loc[list(mcp_pivot.index), :]
        aggstats[color] = aggstats["top"]
        mcp_pivot = pd.concat([mcp_pivot, aggstats[color]], axis=1)
        sns.pairplot(mcp_pivot, hue=color, corner=True)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None
