from __future__ import annotations

import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as ssm
from anndata import AnnData
from lamin_utils import logger
from pandas import DataFrame
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from scipy import stats
from scipy.optimize import nnls
from seaborn import PairGrid
from sklearn.linear_model import LinearRegression
from sparsecca import lp_pmd, multicca_permute, multicca_pmd
from statsmodels.sandbox.stats.multicomp import multipletests

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Dialogue:
    """Python implementation of DIALOGUE."""

    def __init__(
        self,
        sample_id: str,
        celltype_key: str,
        n_counts_key: str,
        n_mpcs: int,
        feature_space_key: str = "X_pca",
        n_components: int = 50,
    ):
        """Constructor for Dialogue.

        Args:
            sample_id: The sample ID key in AnnData.obs which is used for pseudobulk determination.
            celltype_key: The key in AnnData.obs which contains the cell type column.
            n_counts_key: The key of the number of counts in Anndata.obs . Also commonly the size factor.
            n_mpcs: Number of PMD components which corresponds to the number of determined MCPs.
            feature_space_key: The key in adata.obsm for the feature space (e.g., "X_pca", "X_umap").
            n_components: The number of components of the feature space to use, e.g. PCA components.
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
        self.feature_space_key = feature_space_key
        self.n_components = n_components

    def _get_pseudobulks(
        self, adata: AnnData, groupby: str, strategy: Literal["median", "mean"] = "median"
    ) -> pd.DataFrame:
        """Return cell-averaged data by groupby.

        Copied from `https://github.com/schillerlab/sc-toolbox/blob/397e80dc5e8fb8017b75f6c3fa634a1e1213d484/sc_toolbox/tools/__init__.py#L458`

        Args:
            adata: Annotated data matrix.
            groupby: The key to groupby for pseudobulks
            strategy: The pseudobulking strategy. One of "median" or "mean"

        Returns:
            A Pandas DataFrame of pseudobulk counts
        """
        # TODO: Replace with decoupler's implementation
        pseudobulk = {"Genes": adata.var_names.values}

        for category in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == category
            if strategy == "median":
                pseudobulk[category] = adata[temp].X.median(axis=0)
            elif strategy == "mean":
                pseudobulk[category] = adata[temp].X.mean(axis=0)

        pseudobulk = pd.DataFrame(pseudobulk).set_index("Genes")

        return pseudobulk

    def _pseudobulk_feature_space(
        self,
        adata: AnnData,
        groupby: str,
    ) -> pd.DataFrame:
        """Return Cell-averaged components from a passed feature space.

        TODO: consider merging with `get_pseudobulks`
        TODO: DIALOGUE recommends running PCA on each cell type separately before running PMD - this should be implemented as an option here.

        Args:
            adata: Annotated data matrix.
            groupby: The key to groupby for pseudobulks.

        Returns:
            A pseudobulk DataFrame of the averaged components.
        """
        aggr = {}
        for category in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == category
            aggr[category] = adata[temp].obsm[self.feature_space_key][:, : self.n_components].mean(axis=0)
        aggr = pd.DataFrame(aggr)
        return aggr

    def _scale_data(self, pseudobulks: pd.DataFrame, normalize: bool = True) -> np.ndarray:
        """Row-wise mean center and scale by the standard deviation.

        Args:
            pseudobulks: The pseudobulk PCA components.
            normalize: Whether to mimic DIALOGUE behavior or not.

        Returns:
            The scaled count matrix.
        """
        # TODO: the `scale` function we implemented to match the R `scale` fn should already contain this functionality.
        # DIALOGUE doesn't scale the data before passing to multicca, unlike what is recommended by sparsecca.
        # However, performing this scaling _does_ increase overall correlation of the end result
        if normalize:
            return pseudobulks.to_numpy()
        else:
            return ((pseudobulks - pseudobulks.mean()) / pseudobulks.std()).to_numpy()

    def _concat_adata_mcp_scores(
        self, adata: AnnData, ct_subs: dict[str, AnnData], mcp_scores: dict[str, np.ndarray], celltype_key: str
    ) -> AnnData:
        """Concatenates the AnnData object with the mcp scores.

        Args:
            adata: The AnnData object to append mcp scores to.
            ct_subs: cell type objects.
            mcp_scores: The MCP scores dictionary.
            celltype_key: Key of the cell type column in obs.

        Returns:
            AnnData object with concatenated MCP scores in obs.
        """

        def __concat_obs(adata: AnnData, mcp_df: pd.DataFrame) -> AnnData:
            mcp_df.columns = [f"mcp_{col}" for col in mcp_df.columns]
            new_obs = pd.concat([adata.obs, mcp_df.set_index(adata.obs.index)], axis=1)
            adata.obs = new_obs

            return adata

        ad_mcp = {
            ct: __concat_obs(ct_subs[ct], pd.DataFrame(mcp_scores[ct])) for ct in adata.obs[celltype_key].cat.categories
        }

        adata = ad.concat(ad_mcp.values())

        return adata

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

    def _get_cor_zscores(self, estimate: pd.Series, p_val: pd.Series) -> pd.DataFrame:
        """Given estimate and p_values calculate zscore.

        Args:
            estimate: Hierarchical modeling estimate results.
            p_val: p-values of the Hierarchical modeling.

        Returns:
            A DataFrame containing the zscores indexed by the estimates.
        """
        p_val.replace(0, min(p for p in p_val if p is not None and p > 0))

        # check for all (negative) estimate values if >0 then divide p_value by 2 at same index else substract the p_value/2 from 1
        # pos_est and neg_est differ in calculation for values as negative estimation is used in neg_est
        pos_est = pd.DataFrame(
            [p_val.iloc[i] / 2 if estimate.iloc[i] > 0 else 1 - (p_val.iloc[i] / 2) for i in range(len(estimate))],
            columns=["pos_est"],
        )  # significant in pos_est will be positive
        neg_est = pd.DataFrame(
            [p_val.iloc[i] / 2 if -estimate.iloc[i] > 0 else 1 - (p_val.iloc[i] / 2) for i in range(len(estimate))],
            columns=["neg_est"],
        )  # significant in neg_est will be negative
        onesided_p_vals = pd.concat([pos_est, neg_est], axis=1)

        # calculate zscores
        z_scores = pd.DataFrame(
            [
                np.log10(row["neg_est"]) if row["pos_est"] > 0.5 else -np.log10(row["pos_est"])
                for _, row in onesided_p_vals.iterrows()
            ],
            columns=["z_score"],
        )
        z_scores = z_scores.set_index(estimate.index)

        return z_scores

    def _formula_hlm(
        self,
        y: pd.DataFrame,
        x_labels: pd.DataFrame,
        x_tme: pd.DataFrame,
        formula: str,
        sample_obs: str,
        return_all: bool = False,
    ):
        """Applies a mixed linear model using the specified formula (MCP scores used for the dependent var) and returns the coefficient and p-value.

        TODO: reduce runtime? Maybe we can use an approximation or something that isn't statsmodels.

        Args:
            y: Dataframe containing the MCP score for an individual gene
            x_labels: Dataframe with other .obs values needed for fitting the formula, specifically `n_counts_key`.
            x_tme: Transcript mean expression of `x`, with a respectively labeled column.
            formula: The mixedlm formula.
            sample_obs: Sample identifier in the obs dataframe, such as a confounder (treated as random effect)
            return_all: Whether to return model summary (estimate and p-value) or alternatively a list of the coefficient/p-value for x only

        Returns:
            The determined coefficients and p-values.
        """
        formula_data = pd.concat(
            [y, x_tme, x_labels], axis=1, join="inner"
        )  # TODO: sometimes these have mismatched sizes and they shouldn't... currently fixed with join=inner

        mdf = smf.mixedlm(formula, formula_data, groups=formula_data[sample_obs]).fit()

        if return_all:
            return pd.DataFrame({"estimate": mdf.params, "p_val": mdf.pvalues})

        return [mdf.params["x"], mdf.pvalues["x"]]

    def _mixed_effects(
        self,
        scores: pd.DataFrame,
        x_labels: pd.DataFrame,
        tme: pd.DataFrame,
        genes_in_mcp: list[str],
        formula: str,
        confounder: str,
    ) -> pd.DataFrame:
        """Determines z-scores, estimates and p-values with adjusted p-values and booleans marking if a gene is up or downregulated.

        Args:
            scores: A single MCP's scores with genes names in the index.
            x_labels: Dataframe that must contain a column named 'x' containing average expression values by sample
            tme: Transcript mean expression of `x`.
            genes_in_mcp: Up and down genes for this MCP.
            formula: Formula for hierachical modeling.
            confounder: Any model confounders.

        Returns:
            DataFrame with z-scores, estimates, p-values, adjusted p-values and booleans
            marking whether a gene is up (True) or downregulated (False).
        """
        scores.columns = ["y"]

        hlm_result = pd.DataFrame(columns=["estimate", "p_val"])

        for gene, expr_data in tme.loc[genes_in_mcp].iterrows():
            tme_gene = pd.DataFrame(data={"x": expr_data})
            hlm_result.loc[gene] = self._formula_hlm(
                y=scores, x_labels=x_labels, x_tme=tme_gene, formula=formula, sample_obs=confounder
            )

        hlm_result["z_score"] = self._get_cor_zscores(hlm_result["estimate"], hlm_result["p_val"])["z_score"]

        mt_results = ssm.multipletests(
            hlm_result["p_val"], alpha=0.05, method="fdr_bh", is_sorted=False, returnsorted=False
        )
        hlm_result["p_adjust"] = mt_results[1].tolist()

        return hlm_result

    def _get_top_cor(
        self, df: pd.DataFrame, mcp_name: str, max_length: int = 100, min_threshold: int = 0, index: str = "z_score"
    ) -> dict[str, Any]:
        """Determines the significant up- and downregulated genes by the passed index column.

        TODO: This function will eventually get merged with a second version from Faye. Talk to Yuge about it.

        Args:
            df: Dataframe with rowsindex being the gene names and column with name <index> containing floats.
            mcp_name: Name of mcp which was used for calculation of column value.
            max_length: Value needed to later decide at what index the threshold value should be extracted from column.
            min_threshold: Minimal threshold to select final scores by if it is smaller than calculated threshold.
            index: Column index to use eto calculate the significant genes.

        Returns:
            According to the values in a df column (default: zscore) the significant up and downregulated gene names
        """
        min_threshold = -abs(min_threshold)
        zscores_neg = df.loc[:, index]
        zscores_pos = pd.Series(data=[-score for score in zscores_neg], index=df.index)

        top_genes = {}
        for pair in [(zscores_pos, ".up"), (zscores_neg, ".down")]:
            zscores = pair[0]
            suffix = pair[1]
            ordered = zscores.argsort()
            # TODO: There is an off by one error here (1 too many)
            threshold = min(zscores[ordered[min(max_length, len(zscores) - 1)]], min_threshold)
            # extract all gene names where value in column is <= threshold -> get significant genes
            genes = zscores.index
            top_genes[mcp_name + suffix] = sorted(genes[i] for i in range(len(zscores)) if zscores[i] <= threshold)

        return top_genes

    def _apply_HLM_per_MCP_for_one_pair(
        self,
        mcp_name: str,
        scores_df: pd.DataFrame,
        ct_data: AnnData,
        tme: pd.DataFrame,
        sig: dict,
        n_counts: str,
        formula: str,
        confounder: str | None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Applies hierarchical modeling for a single MCP.

        TODO: separate the sig calculation so that this whole function is more tractable

        Args:
            mcp_name: The name of the MCP to model.
            scores_df: The MCP scores for a cell type. Number of MCPs x number of features.
            ct_data: The AnnData object containing the metadata and labels in obs.
            tme: Transcript mean expression in `x`.
            sig: DataFrame containing a series of up and downregulated MCPs.
            n_counts: The key of the gene counts.
            formula: The formula of the hierarchical modeling.
            confounder: Any modeling confounders.

        Returns:
            The HLM results together with significant up/downregulated genes per MCP
        """
        HLM_result = self._mixed_effects(
            scores=scores_df[[mcp_name]],
            x_labels=ct_data.obs[[n_counts, confounder]] if confounder else ct_data.obs[[n_counts]],
            tme=tme,
            genes_in_mcp=list(sig[mcp_name]["up"]) + list(sig[mcp_name]["down"]),
            formula=formula,
            confounder=confounder,
        )

        HLM_result["up"] = [gene in sig[mcp_name]["up"] for gene in HLM_result.index]

        # extract significant up and downregulated genes
        sig_genes = self._get_top_cor(df=HLM_result, mcp_name=mcp_name, max_length=100, min_threshold=1)

        HLM_result["program"] = mcp_name

        return HLM_result, sig_genes

    def _get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mimics DIALOGUE.get.residuals."""
        resid = []
        for y_sub in y:
            lr = LinearRegression().fit(X.reshape(-1, 1), y_sub)
            pred = lr.predict(X.reshape(-1, 1))
            resid.append(y_sub - pred)

        return np.array(resid)

    def _iterative_nnls(self, A_orig: np.ndarray, y_orig: np.ndarray, feature_ranks: list[int], n_iter: int = 1000):
        """Solves non-negative least-squares separately for different feature categories.

        Mimics DLG.iterative.nnls.
        Variables are notated according to:

            `argmin|Ax - y|`

        Returns:
            Returns the aggregated coefficients from nnls.
        """
        # TODO: Consider moving this internally to cca_sig
        y = y_orig.copy()

        sig_ranks = sorted(set(feature_ranks), reverse=True)
        sig_ranks = [rank for rank in sig_ranks if rank >= 1 / 3]  # code coverage only with n_mcps > 3
        masks = [feature_ranks == r for r in sig_ranks if sum(feature_ranks == r) >= 5]  # type: ignore

        # TODO: The few type ignores are dangerous and should be checked! They could be bugs.
        insig_mask = feature_ranks < sig_ranks[-1]  # type: ignore # TODO: rename variable after better understanding
        if sum(insig_mask) >= 5:  # such as genes with 0 rank, or those below 1/3
            masks.append(insig_mask)
            sig_ranks.append("insig")  # type: ignore

        x_final = np.zeros(A_orig.shape[0])
        Ax = np.zeros(A_orig.shape[1])
        for _, mask in zip(sig_ranks, masks, strict=False):
            A = A_orig[mask].T
            coef_nnls, _ = nnls(A, y, maxiter=n_iter)
            y = y - A @ coef_nnls  # residuals
            Ax += A @ coef_nnls
            x_final[mask] = coef_nnls

        return x_final

    def _corr2_coeff(self, A, B):
        # Rowwise mean of input arrays & subtract from input arrays themselves
        A_mA = A - A.mean(1)[:, None]
        B_mB = B - B.mean(1)[:, None]

        # Sum of squares across rows
        ssA = (A_mA**2).sum(1)
        ssB = (B_mB**2).sum(1)

        # Finally get corr coeff
        return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))

    # TODO: needs check for correctness and variable renaming
    # TODO: Confirm that this doesn't return duplicate gene names.
    def _get_top_elements(self, m: pd.DataFrame, max_length: int, min_threshold: float):
        """Get top elements.

        Args:
            m: Any DataFrame of Gene name as index with variable columns.
            max_length: Maximum number of correlated elements.
            min_threshold: p-value threshold  # TODO confirm

        Returns:
            Indices of the top elements
        """
        m_pos = -m
        m_neg = m
        df = pd.concat(
            [m_pos, m_neg], axis=1
        )  # check if the colnames has any significance, they are just split .up and .down in dialogue
        top_l = []
        for i in range(df.shape[1]):  # TODO: names are not very descriptive -> improve
            mi = df.iloc[:, i].dropna()
            ci = mi.iloc[np.argsort(mi)[min(max_length, len(mi)) - 1]]
            min_threshold = min_threshold if min_threshold is None else min(ci, min_threshold)
            b = df.iloc[:, i].fillna(False) <= min_threshold
            top_l.append(df.index[b])  # the index is actually rownames which are genes**
        return top_l  # returns indices as different appended lists

    def _calculate_cca_sig(
        self,
        ct_subs: dict,
        mcp_scores: dict,
        ws_dict: dict,
        n_counts_key: str,
        max_genes: int = 200,
    ) -> tuple[dict[Any, dict[str, Any]], dict[Any, list[Any]]]:
        # TODO this whole function should be standalone
        # It will contain the calculation of up/down + calculation (new final mcp scores)
        # Ensure that it'll still fit/work with the hierarchical multilevel_modeling
        """Determine the up and down genes per MCP."""
        # TODO: something is slightly slow here
        cca_sig_results: dict[Any, dict[str, Any]] = {}
        new_mcp_scores: dict[Any, list[Any]] = {}
        for ct in ct_subs:
            ct_adata = ct_subs[ct]
            conf_m = ct_adata.obs[n_counts_key].values

            if not isinstance(ct_adata.X, np.ndarray):
                ct_adata.X = ct_adata.X.toarray()
            R_cca_gene_cor1_x = self._corr2_coeff(
                ct_adata.X.T, mcp_scores[ct].T
            )  # TODO: there are some nans here, also in R

            # get genes that are most positively and negatively correlated across all MCPS
            top_ele = self._get_top_elements(  # TODO: consider renaming for clarify
                pd.DataFrame(R_cca_gene_cor1_x, index=ct_adata.var.index), max_length=max_genes, min_threshold=0.05
            )
            top_cor_genes_flattened = [x for lst in top_ele for x in lst]
            top_cor_genes_flattened = sorted(set(top_cor_genes_flattened))  # aka the mysterious g1 in R

            # MAJOR TODO: I've only used normal correlation instead of partial correlation as we wait on the implementation
            from scipy.stats import spearmanr

            def _pcor_mat(v1, v2, v3, method="spearman"):
                """MAJOR TODO: I've only used normal correlation instead of partial correlation as we wait on the implementation."""
                correlations = []  # R
                pvals = []  # P
                for x2 in v2:
                    c = []
                    p = []
                    for x1 in v1:
                        corr, pval = spearmanr(x1, x2)
                        c.append(corr)
                        p.append(pval)
                    correlations.append(c)
                    pvals.append(p)

                # TODO needs reshape
                #     mt_results = ssm.multipletests(
                #         pvals, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)
                #     pvals_adjusted = mt_results[1].tolist()

                return np.array(correlations), np.array(pvals)  # pvals_adjusted

            C1, P1 = _pcor_mat(ct_adata[:, top_cor_genes_flattened].X.toarray().T, mcp_scores[ct].T, conf_m)
            C1[(0.05 / ct_adata.shape[1]) < P1] = 0  # why?

            cca_sig_unformatted = self._get_top_elements(  # 3 up, 3 dn, for each mcp
                pd.DataFrame(C1.T, index=top_cor_genes_flattened), max_length=max_genes, min_threshold=0.05
            )

            # TODO: probably format the up and down within get_top_elements
            cca_sig: dict[str, Any] = defaultdict(dict)
            for i in range(int(len(cca_sig_unformatted) / 2)):
                cca_sig[f"MCP{i}"]["up"] = cca_sig_unformatted[i * 2]
                cca_sig[f"MCP{i}"]["down"] = cca_sig_unformatted[i * 2 + 1]

            cca_sig = dict(cca_sig)
            cca_sig_results[ct] = cca_sig

            # This is basically DIALOGUE 3 now
            pre_r_scores = {
                ct: ct_subs[ct].obsm[self.feature_space_key][:, : self.n_components] @ ws_dict[ct]
                for i, ct in enumerate(ct_subs.keys())
                # TODO This is a recalculation and not a new calculation
            }

            scores = []
            for i, mcp in enumerate(cca_sig.keys()):
                # TODO: I'm suspicious about how this code holds up given duplicate var_names - should test
                pre_r_score = pre_r_scores[ct][:, i]  # noqa: F841
                # TODO we should fix _get_top_elements to return the directionality
                # TODO this should be casted somewhere else
                sig_mcp_genes = list(cca_sig[mcp]["up"]) + list(cca_sig[mcp]["down"])
                # deuniqued_gene_indices = [top_cor_genes_flattened.index(g) for g in sig_mcp_genes]
                X = ct_adata[:, sig_mcp_genes].X.toarray()
                zscore = (X - np.mean(X, axis=0)) / np.std(
                    X, axis=0, ddof=1
                )  # adjusted ddof to match R even though it's wrong
                zscores = zscore.T

                # zscores = zscore.T[deuniqued_gene_indices]  # t(r1@zscores[gene.pval$genes,])  should really check this line, so troll

                new_b = [gene in cca_sig[mcp]["up"] for gene in sig_mcp_genes]
                zscores[new_b] = -zscores[new_b]
                # coef = self._iterative_nnls(zscores, pre_r_score, pvals_mcp['Nf'].values)
                # scores = zscores.T @ coef
                # TODO: The line below has to be new_mcp_scores.append(scores) after we implemented the NF value caluation
                scores.append(zscores)
            new_mcp_scores[ct] = scores

        return cca_sig_results, new_mcp_scores

    def _load(
        self,
        adata: AnnData,
        ct_order: list[str],
        agg_feature: bool = True,
        normalize: bool = True,
    ) -> tuple[list, dict]:
        """Separates cell into AnnDatas by celltype_key and creates the multifactor PMD input.

        Mimics DIALOGUE's `make.cell.types` and the pre-processing that occurs in DIALOGUE1.

        Args:
            adata: AnnData object generate celltype objects for
            ct_order: The order of cell types
            agg_feature: Whether to aggregate pseudobulks with some embeddings or not.
            normalize: Whether to mimic DIALOGUE behavior or not.

        Returns:
            A celltype_label:array dictionary.
        """
        ct_subs = {ct: adata[adata.obs[self.celltype_key] == ct].copy() for ct in ct_order}
        fn = self._pseudobulk_feature_space if agg_feature else self._get_pseudobulks
        ct_aggr = {ct: fn(ad, self.sample_id) for ct, ad in ct_subs.items()}  # type: ignore

        # TODO: implement check (as in https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L114-L119)
        # that there are at least 5 share samples here

        # TODO: https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L121-L131
        ct_preprocess = {ct: self._scale_data(ad, normalize=normalize).T for ct, ad in ct_aggr.items()}

        mcca_in = [ct_preprocess[ct] for ct in ct_order]

        return mcca_in, ct_subs

    def calculate_multifactor_PMD(
        self,
        adata: AnnData,
        penalties: list[int] | None = None,
        ct_order: list[str] | None = None,
        agg_feature: bool = True,
        solver: Literal["lp", "bs"] = "bs",
        normalize: bool = True,
    ) -> tuple[AnnData, dict[str, np.ndarray], dict[Any, Any], dict[Any, Any]]:
        """Runs multifactor PMD.

        Currently mimics DIALOGUE1.

        Args:
            adata: AnnData object to calculate PMD for.
            penalties: PMD penalties.
            ct_order: The order of cell types.
            agg_feature: Whether to calculate cell-averaged principal components.
            solver: Which solver to use for PMD. Must be one of "lp" (linear programming) or "bs" (binary search).
                    For differences between these to please refer to https://github.com/theislab/sparsecca/blob/main/examples/linear_programming_multicca.ipynb
            normalize: Whether to mimic DIALOGUE as close as possible

        Returns:
            MCP scores  # TODO this requires more detail

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(
            ...     sample_id="clinical.status", celltype_key="cell.subtypes", n_counts_key="nCount_RNA", n_mpcs=3
            ... )
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
        """
        # IMPORTANT NOTE: the order in which matrices are passed to multicca matters.
        # As such, it is important here that to obtain the same result as in R, we pass the matrices in the same order.
        if ct_order is not None:
            cell_types = ct_order
        else:
            ct_order = cell_types = adata.obs[self.celltype_key].astype("category").cat.categories

        mcca_in, ct_subs = self._load(adata, ct_order=cell_types, agg_feature=agg_feature, normalize=normalize)

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

        if solver == "bs":
            ws, _ = multicca_pmd(mcca_in, penalties, K=self.n_mcps, standardize=True, niter=100, mimic_R=normalize)
        elif solver == "lp":
            ws, _ = lp_pmd(mcca_in, penalties, K=self.n_mcps, standardize=True, mimic_R=normalize)
        else:
            raise ValueError('Please select a valid solver. Must be one of "lp" or "bs".')
        ws_dict = {ct: ws[i] for i, ct in enumerate(ct_order)}

        pre_r_scores = {
            ct: ct_subs[ct].obsm[self.feature_space_key][:, : self.n_components] @ ws[i]
            for i, ct in enumerate(cell_types)
        }

        # TODO: output format needs some cleanup, even though each MCP score is matched to one cell, it's not at all
        # matched at this point in the function and requires references to internals that shouldn't need exposing (ct_subs)

        mcp_scores = {
            ct: self._get_residuals(ct_subs[ct].obs[self.n_counts_key].values, pre_r_scores[ct].T).T
            for i, ct in enumerate(cell_types)
        }

        adata = self._concat_adata_mcp_scores(
            adata, ct_subs=ct_subs, mcp_scores=mcp_scores, celltype_key=self.celltype_key
        )

        return adata, mcp_scores, ws_dict, ct_subs

    def multilevel_modeling(
        self,
        ct_subs: dict,
        mcp_scores: dict,
        ws_dict: dict,
        confounder: str | None,
        formula: str = None,
    ) -> pd.DataFrame:
        """Runs the multilevel modeling step to match genes to MCPs and generate p-values for MCPs.

        Args:
            ct_subs: The DIALOGUE cell type objects.
            mcp_scores: The determined MCP scores from the PMD step.
            ws_dict: WS dictionary.
            confounder: Any modeling confounders.
            formula: The hierarchical modeling formula. Defaults to y ~ x + n_counts.

        Returns:
            - for each mcp: HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2
            - merged HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2 of all mcps

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
        """
        # TODO the returns of the function better

        # all possible pairs of cell types without pairing same cell type
        cell_types = list(ct_subs.keys())
        pairs = list(itertools.combinations(cell_types, 2))

        if not formula:
            formula = f"y ~ x + {self.n_counts_key}"

        # Hierarchical modeling expects DataFrames
        mcp_cell_types = {f"MCP{i}": cell_types for i in range(self.n_mcps)}
        mcp_scores_df = {
            ct: pd.DataFrame(v, index=ct_subs[ct].obs.index, columns=list(mcp_cell_types.keys()))
            for ct, v in mcp_scores.items()
        }

        # run HLM for each pair
        all_results: dict[str, dict[Any, dict[str, tuple[DataFrame, dict[str, Any]]]]] = {}
        mlm_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )
        mixed_model_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=True,
        )
        group = Group(mlm_progress, mixed_model_progress)
        live = Live(group)
        with live:
            mlm_task = mlm_progress.add_task("[bold blue]Running multilevel modeling", total=len(pairs))

            for pair in pairs:
                cell_type_1 = pair[0]
                cell_type_2 = pair[1]
                mlm_progress.update(mlm_task, description=f"[bold blue]{cell_type_1} vs {cell_type_2}")

                ct_data_1 = ct_subs[cell_type_1]
                ct_data_2 = ct_subs[cell_type_2]

                # equivalent to dialogue2.pair
                mcps = []
                for mcp, cell_type_list in mcp_cell_types.items():
                    if cell_type_1 in cell_type_list and cell_type_2 in cell_type_list:
                        mcps.append(mcp)

                if len(mcps) == 0:
                    logger.warning(f"No shared MCPs between {cell_type_1} and {cell_type_2}.")
                    continue

                logger.info(f"{len(mcps)} MCPs identified for {cell_type_1} and {cell_type_2}.")

                new_mcp_scores: dict[Any, list[Any]]
                cca_sig, new_mcp_scores = self._calculate_cca_sig(
                    ct_subs, mcp_scores=mcp_scores, ws_dict=ws_dict, n_counts_key=self.n_counts_key
                )

                sig_1 = cca_sig[cell_type_1]  # TODO: only need the up and down genes from this here per MCP
                sig_2 = cca_sig[cell_type_2]
                # only use samples which have a minimum number of cells (default 2) in both cell types
                sample_ids = list(
                    set(self._get_abundant_elements_from_series(ct_data_1.obs[self.sample_id]))
                    & set(self._get_abundant_elements_from_series(ct_data_2.obs[self.sample_id]))
                )

                # subset cell types to valid samples (set.cell.types)
                ct_data_1 = ct_data_1[ct_data_1.obs[self.sample_id].isin(sample_ids)]
                ct_data_2 = ct_data_2[ct_data_2.obs[self.sample_id].isin(sample_ids)]

                # TODO: shouldn't need this aligning step for cells. corresponds to @scores / y
                #     scores_1 = cca_scores[cell_type_1].loc[ct_data_1.obs.index]
                #     scores_2 = cca_scores[cell_type_2].loc[ct_data_2.obs.index]

                # indexes into the average sample expression per gene with the sample id per cell. corresponds to @tme / x
                # TODO: Why is the sample_id type check failing?
                tme_1 = self._get_pseudobulks(ct_data_2, groupby=self.sample_id, strategy="mean").loc[
                    :, ct_data_1.obs[self.sample_id]
                ]  # unclear why we do this
                tme_1.columns = ct_data_1.obs.index
                tme_2 = self._get_pseudobulks(ct_data_1, groupby=self.sample_id, strategy="mean").loc[
                    :, ct_data_2.obs[self.sample_id]
                ]
                tme_2.columns = ct_data_2.obs.index

                merged_results = {}

                mm_task = mixed_model_progress.add_task("[bold blue]Determining mixed effects", total=len(mcps))
                for mcp in mcps:
                    mixed_model_progress.update(mm_task, description=f"[bold blue]Determining mixed effects for {mcp}")

                    # TODO Check whether the genes in result{sig_genes_1] are different and if so note that somewhere and explain why
                    result = {}
                    result["HLM_result_1"], result["sig_genes_1"] = self._apply_HLM_per_MCP_for_one_pair(
                        mcp_name=mcp,
                        scores_df=mcp_scores_df[cell_type_2],
                        ct_data=ct_data_2,
                        tme=tme_2,
                        sig=sig_1,
                        n_counts=self.n_counts_key,
                        formula=formula,
                        confounder=confounder,
                    )
                    result["HLM_result_2"], result["sig_genes_2"] = self._apply_HLM_per_MCP_for_one_pair(
                        mcp_name=mcp,
                        scores_df=mcp_scores_df[cell_type_1],
                        ct_data=ct_data_1,
                        tme=tme_1,
                        sig=sig_2,
                        n_counts=self.n_counts_key,
                        formula=formula,
                        confounder=confounder,
                    )
                    merged_results[mcp] = result

                    mixed_model_progress.update(mm_task, advance=1)
                mixed_model_progress.update(mm_task, visible=False)
                mlm_progress.update(mlm_task, advance=1)

                # merge results - TODO, but probably don't need
                #     merged_results['HLM_result_1'] = pd.concat([merged_result[mcp]['HLM_result_1'] for mcp in mcps])
                #     merged_results['HLM_result_2'] = pd.concat([merged_result[mcp]['HLM_result_2'] for mcp in mcps])
                #     merged_results['sig_genes_1'] = [**merged_result[mcp]['sig_genes_1'] for mcp in mcps]
                #     merged_results['sig_genes_2'] = [**merged_result[mcp]['sig_genes_2'] for mcp in mcps]

                all_results[f"{cell_type_1}_vs_{cell_type_2}"] = merged_results

        return all_results, new_mcp_scores

    def test_association(
        self,
        adata: AnnData,
        condition_label: str,
        conditions_compare: tuple[str, str] = None,
    ):
        """Tests association between MCPs and a binary response variable (e.g. response to treatment).

        Note: benjamini-hochberg corrects for the number of cell types, NOT the number of MCPs

        Args:
            adata: AnnData object with MCPs in obs
            condition_label: Column name in adata.obs with condition labels. Must be categorical.
            conditions_compare: Tuple of length 2 with the two conditions to compare, must be in adata.obs[condition_label]

        Returns:
            Dict of data frames with pvals, tstats, and pvals_adj for each MCP

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> stats = dl.test_association(adata, condition_label="pathology")
        """
        celltype_label = self.celltype_key
        sample_label = self.sample_id
        n_mcps = self.n_mcps

        if conditions_compare is None:
            conditions_compare = list(adata.obs[condition_label].cat.categories)  # type: ignore
            if len(conditions_compare) != 2:
                raise ValueError("Please specify conditions to compare or supply an object with only 2 conditions")

        pvals = pd.DataFrame(1, adata.obs[celltype_label].unique(), ["mcp_" + str(n) for n in range(n_mcps)])
        tstats = pd.DataFrame(1, adata.obs[celltype_label].unique(), ["mcp_" + str(n) for n in range(n_mcps)])
        pvals_adj = pd.DataFrame(1, adata.obs[celltype_label].unique(), ["mcp_" + str(n) for n in range(n_mcps)])

        response = adata.obs.groupby(sample_label)[condition_label].agg(pd.Series.mode)
        for celltype in adata.obs[celltype_label].unique():
            df = adata.obs[adata.obs[celltype_label] == celltype]

            for mcpnum in ["mcp_" + str(n) for n in range(n_mcps)]:
                mns = df.groupby(sample_label)[mcpnum].mean()
                mns = pd.concat([mns, response], axis=1)
                res = stats.ttest_ind(
                    mns[mns[condition_label] == conditions_compare[0]][mcpnum],
                    mns[mns[condition_label] == conditions_compare[1]][mcpnum],
                )
                pvals.loc[celltype, mcpnum] = res[1]
                tstats.loc[celltype, mcpnum] = res[0]

        for mcpnum in ["mcp_" + str(n) for n in range(n_mcps)]:
            pvals_adj[mcpnum] = multipletests(pvals[mcpnum], method="fdr_bh")[1]

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
        for celltype2 in mcp_dict:
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
        for ct in ct_subs:
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
            for celltype in ct_ranked:
                rank_dfs[mcp][celltype] = sc.get.rank_genes_groups_df(ct_ranked[celltype], group=None)

        return rank_dfs

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_split_violins(  # pragma: no cover # noqa: D417
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
    def plot_pairplot(  # pragma: no cover # noqa: D417
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
