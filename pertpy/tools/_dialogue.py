from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as ssm
from anndata import AnnData
from pandas import DataFrame
from rich import print
from rich.console import Group
from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from sparsecca import lp_pmd, multicca_permute, multicca_pmd


class Dialogue:
    """Python implementation of DIALOGUE"""

    def _get_pseudobulks(
        self, adata: AnnData, groupby: str, strategy: Literal["median", "mean"] = "median"
    ) -> pd.DataFrame:
        """Return cell-averaged data by groupby.

        Copied from `https://github.com/schillerlab/sc-toolbox/blob/397e80dc5e8fb8017b75f6c3fa634a1e1213d484/sc_toolbox/tools/__init__.py#L458`

        # TODO: Replace with decoupler's implementation

        Args:
            groupby: The key to groupby for pseudobulks
            strategy: The pseudobulking strategy. One of "median" or "mean"

        Returns:
            A Pandas DataFrame of pseudobulk counts
        """
        pseudobulk = {"Genes": adata.var_names.values}

        for category in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == category
            if strategy == "median":
                pseudobulk[category] = adata[temp].X.median(axis=0).A1
            elif strategy == "mean":
                pseudobulk[category] = adata[temp].X.mean(axis=0).A1

        pseudobulk = pd.DataFrame(pseudobulk).set_index("Genes")

        return pseudobulk

    def _pseudobulk_pca(self, adata: AnnData, groupby: str, n_components: int = 50) -> pd.DataFrame:
        """Return cell-averaged PCA components.

        TODO: consider merging with `get_pseudobulks`
        TODO: DIALOGUE recommends running PCA on each cell type separately before running PMD - this should be implemented as an option here.

        Args:
            groupby: The key to groupby for pseudobulks
            n_components: The number of PCA components

        Returns:
            A pseudobulk of PCA components.
        """
        aggr = {}

        for category in adata.obs.loc[:, groupby].cat.categories:
            temp = adata.obs.loc[:, groupby] == category
            aggr[category] = adata[temp].obsm["X_pca"][:, :n_components].mean(axis=0)

        aggr = pd.DataFrame(aggr)

        return aggr

    def _scale_data(self, pseudobulks: pd.DataFrame, mimic_dialogue: bool = True) -> np.ndarray:
        """Row-wise mean center and scale by the standard deviation.

        TODO: the `scale` function we implemented to match the R `scale` fn should already contain this functionality.

        Args:
            pseudobulks: The pseudobulk PCA components.
            mimic_dialogue: Whether to mimic DIALOGUE behavior or not.

        Returns:
            The scaled count matrix.
        """
        # DIALOGUE doesn't scale the data before passing to multicca, unlike what is recommended by sparsecca.
        # However, performing this scaling _does_ increase overall correlation of the end result
        # WHEN SAMPLE ORDER AND DIALOGUE2+3 PROCESSING IS IGNORED.
        if mimic_dialogue:
            return pseudobulks.to_numpy()
        else:
            return ((pseudobulks - pseudobulks.mean()) / pseudobulks.std()).to_numpy()

    def _concat_adata_mcp_scores(
        self, adata: AnnData, ct_subs: dict[str, AnnData], mcp_scores: dict[str, np.ndarray], celltype_key: str
    ) -> AnnData:
        """Concatenates the AnnData object with the mcp scores.

        Args:
            adata: The AnnData object to append mcp scores to.
            mcp_scores: The MCP scores dictionary.
            celltype_key: Key of the cell type column in obs.

        Returns:
            AnnData object with concatenated MCP scores in obs.
        """

        def __concat_obs(adata: AnnData, mcp_df: pd.DataFrame) -> AnnData:
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
        p_val.replace(0, min([p for p in p_val if p is not None and p > 0]))

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
        """Applies a mixed linear model using the specified formula (MCP scores used for the dependent var) and returns the coefficient and p-value

        TODO: reduce runtime? Maybe we can use an approximation or something that isn't statsmodels.

        Args:
            y: Dataframe containing the MCP score for an individual gene
            x_labels: Dataframe that must contain a column named 'x' containing average expression values by sample
            x_tme: Transcript mean expression of `x`.
            formula: The mixedlm formula.
            sample_obs: Sample identifier in the obs dataframe, such as a confounder (treated as random effect)
            return_all: Whether to return model summary (estimate and p-value) or alternatively a list of the coefficient/p-value for x only

        Returns:
            The determined coefficients and p-values.
        """
        formula_data = pd.concat([y, x_tme, x_labels], axis=1)

        mdf = smf.mixedlm(formula, formula_data, groups=x_labels[sample_obs]).fit()

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
            index: Column index to use eto calculate the significant genes. Defaults to `z_score`.

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
            top_genes[mcp_name + suffix] = sorted([genes[i] for i in range(len(zscores)) if zscores[i] <= threshold])

        return top_genes

    def _apply_HLM_per_MCP_for_one_pair(  # noqa: N802
        self,
        mcp_name: str,
        scores_df: dict,
        ct_data: AnnData,
        tme: pd.DataFrame,
        sig: dict,
        n_counts: str,
        formula: str,
        confounder: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Applies hierarchical modeling for a single MCP.

        TODO: separate the sig calculation so that this whole function is more tractable

        Args:
            mcp_name: The name of the MCP to model.
            scores: The MCP scores for a cell type. Number of MCPs x number of features.
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
            x_labels=ct_data.obs[[n_counts, confounder]],
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
        """Solves non-negative least squares separately for different feature categories.

        Mimics DLG.iterative.nnls.
        Variables are notated according to:

            `argmin|Ax - y|`

        Args:
            A_orig:
            y_orig:
            feature_ranks:
            n_iter: Passed to scipy.optimize.nnls. Defaults to 1000.

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
        for _, mask in zip(sig_ranks, masks):
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

    def _get_top_elements(self, m: pd.DataFrame, max_length: int, min_threshold: float):
        """

        TODO: needs check for correctness and variable renaming
        TODO: Confirm that this doesn't return duplicate gene names

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
        for ct in ct_subs.keys():
            ct_adata = ct_subs[ct]
            conf_m = ct_adata.obs[n_counts_key].values  # defining this for the millionth time

            R_cca_gene_cor1_x = self._corr2_coeff(
                ct_adata.X.toarray().T, mcp_scores[ct].T
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
                """
                MAJOR TODO: I've only used normal correlation instead of partial correlation as we wait on the implementation
                """
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
            C1[P1 > (0.05 / ct_adata.shape[1])] = 0  # why?

            cca_sig_unformatted = self._get_top_elements(  # 3 up, 3 dn, for each mcp
                pd.DataFrame(C1.T, index=top_cor_genes_flattened), max_length=max_genes, min_threshold=0.05
            )

            # TODO: probably format the up and down within get_top_elements
            cca_sig: dict[str, Any] = defaultdict(dict)
            for i in range(0, int(len(cca_sig_unformatted) / 2)):
                cca_sig[f"MCP{i + 1}"]["up"] = cca_sig_unformatted[i * 2]
                cca_sig[f"MCP{i + 1}"]["down"] = cca_sig_unformatted[i * 2 + 1]

            cca_sig = dict(cca_sig)
            cca_sig_results[ct] = cca_sig

            # This is basically DIALOGUE 3 now
            pre_r_scores = {
                ct: ct_subs[ct].obsm["X_pca"][:, :50] @ ws_dict[ct]
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

    def load(
        self,
        adata: AnnData,
        sample_id: str,
        celltype_key: str,
        ct_order: list[str],
        agg_pca: bool = True,
        mimic_dialogue: bool = True,
    ) -> tuple[list, dict]:
        """Separates cell into AnnDatas by celltype_key and creates the multifactor PMD input.

        Mimics DIALOGUE's `make.cell.types` and the pre-processing that occurs in DIALOGUE1.

        Args:
            adata: AnnData object generate celltype objects for
            sample_id: The key to aggregate the pseudobulks for.
            celltype_key: The key of the cell type column.
            ct_order: The order of cell types
            agg_pca: Whether to aggregate pseudobulks with PCA or not (default: True).
            mimic_dialogue: Whether to mimic DIALOGUE behavior or not (default: True).

        Returns:
            A celltype_label:array dictionary.
        """
        ct_subs = {ct: adata[adata.obs[celltype_key] == ct].copy() for ct in ct_order}
        fn = self._pseudobulk_pca if agg_pca else self._get_pseudobulks
        ct_aggr = {ct: fn(ad, sample_id) for ct, ad in ct_subs.items()}  # type: ignore

        # TODO: implement check (as in https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L114-L119)
        # that there are at least 5 share samples here

        # TODO: https://github.com/livnatje/DIALOGUE/blob/55da9be0a9bf2fcd360d9e11f63e30d041ec4318/R/DIALOGUE.main.R#L121-L131
        ct_preprocess = {ct: self._scale_data(ad, mimic_dialogue=mimic_dialogue).T for ct, ad in ct_aggr.items()}

        mcca_in = [ct_preprocess[ct] for ct in ct_order]

        return mcca_in, ct_subs

    def calculate_multifactor_PMD(  # noqa: N802
        self,
        adata: AnnData,
        sample_id: str,
        celltype_key: str,
        n_counts_key: str,
        n_mcps: int = 3,
        penalties: list[int] = None,
        ct_order: list[str] = None,
        agg_pca: bool = True,
        solver: Literal["lp", "bs"] = "bs",
        normalize: bool = True,
    ) -> tuple[AnnData, dict[str, np.ndarray], dict[Any, Any], dict[Any, Any]]:
        """Runs multifactor PMD.

        Currently mimics DIALOGUE1.

        Args:
            adata: AnnData object to calculate PMD for.
            sample_id: Key to use for pseudobulk determination.
            celltype_key: Cell type column key.
            n_counts_key: Key of the number of counts in obs. Also commonly the size factor.
            n_mcps: Number of PMD components which corresponds to the number of determined MCPs.
            penalties: PMD penalties.
            ct_order: The order of cell types.
            agg_pca: Whether to calculate cell-averaged PCA components.
            solver: Which solver to use for PMD. Must be one of "lp" (linear programming) or "bs" (binary search).
                    For differences between these to please refer to https://github.com/theislab/sparsecca/blob/main/examples/linear_programming_multicca.ipynb
            normalize: Whether to mimic DIALOGUE as close as possible

        Returns:
            MCP scores  # TODO this requires more detail
        """
        # IMPORTANT NOTE: the order in which matrices are passed to multicca matters. As such,
        # it is important here that to obtain the same result as in R, we pass the matrices in
        # in the same order.
        if ct_order is not None:
            cell_types = ct_order
        else:
            ct_order = cell_types = adata.obs[celltype_key].astype("category").cat.categories

        mcca_in, ct_subs = self.load(
            adata, sample_id, celltype_key, ct_order=cell_types, agg_pca=agg_pca, mimic_dialogue=normalize
        )

        n_samples = mcca_in[0].shape[1]
        if penalties is None:
            penalties = multicca_permute(
                mcca_in, penalties=np.sqrt(n_samples) / 2, nperms=10, niter=50, standardize=True
            )["bestpenalties"]
        else:
            penalties = penalties

        if solver == "bs":
            ws, _ = multicca_pmd(mcca_in, penalties, K=n_mcps, standardize=True, niter=100, mimic_R=normalize)
        elif solver == "lp":
            ws, _ = lp_pmd(mcca_in, penalties, K=n_mcps, standardize=True, mimic_R=normalize)
        else:
            raise ValueError('Please select a valid solver. Must be one of "lp" or "bs".')
        ws_dict = {ct: ws[i] for i, ct in enumerate(ct_order)}

        pre_r_scores = {
            ct: ct_subs[ct].obsm["X_pca"][:, :50] @ ws[i] for i, ct in enumerate(cell_types)  # TODO change from 50
        }

        # TODO: output format needs some cleanup, even though each MCP score is matched to one cell, it's not at all
        # matched at this point in the function and requires references to internals that shouldn't need exposing (ct_subs)

        mcp_scores = {
            ct: self._get_residuals(ct_subs[ct].obs[n_counts_key].values, pre_r_scores[ct].T).T
            for i, ct in enumerate(cell_types)
        }

        adata = self._concat_adata_mcp_scores(adata, ct_subs=ct_subs, mcp_scores=mcp_scores, celltype_key=celltype_key)

        return adata, mcp_scores, ws_dict, ct_subs

    def multilevel_modeling(
        self,
        ct_subs: dict,
        mcp_scores: dict,
        ws_dict: dict,
        n_counts_key: str,
        n_mcps: int,
        sample_id: str,
        confounder: str,
        formula: str = None,
    ):
        """Runs the multilevel modeling step to match genes to MCPs and generate p-values for MCPs.

        Args:
            ct_subs: The DIALOGUE cell type objects.
            mcp_scores: The determined MCP scores from the PMD step.
            n_counts_key: The key of the number of counts.
            n_mcps: The number of MCPs.
            sample_id: The sample column number.
            confounder: Any modeling confounders.
            formula: The hierarchical modeling formula. Defaults to y ~ x + n_counts.

        Returns:
            A Pandas DataFrame containing:
            - for each mcp: HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2
            - merged HLM_result_1, HLM_result_2, sig_genes_1, sig_genes_2 of all mcps
            TODO: Describe both returns
        """
        cell_types = list(ct_subs.keys())

        # all possible pairs of cell types with out pairing same cell type
        pairs = list(itertools.combinations(cell_types, 2))

        if not formula:
            formula = f"y ~ x + {n_counts_key}"

        # Hierarchical modeling expects DataFrames
        mcp_cell_types = {f"MCP{i + 1}": cell_types for i in range(n_mcps)}
        mcp_scores_df = {  # noqa: F841
            ct: pd.DataFrame(v, index=ct_subs[ct].obs.index, columns=mcp_cell_types.keys())
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
                    print(f"[bold red]No shared MCPs between {cell_type_1} and {cell_type_2}.")
                    continue

                print(f"[bold blue]{len(mcps)} MCPs identified for {cell_type_1} and {cell_type_2}.")

                new_mcp_scores: dict[Any, list[Any]]
                cca_sig, new_mcp_scores = self._calculate_cca_sig(
                    ct_subs, mcp_scores=mcp_scores, ws_dict=ws_dict, n_counts_key=n_counts_key
                )

                sig_1 = cca_sig[  # noqa: F841
                    cell_type_1
                ]  # TODO: only need the up and down genes from this here per MCP
                sig_2 = cca_sig[cell_type_2]  # noqa: F841
                # only use samples which have a minimum number of cells (default 2) in both cell types
                sample_ids = list(
                    set(self._get_abundant_elements_from_series(ct_data_1.obs[sample_id]))
                    & set(self._get_abundant_elements_from_series(ct_data_2.obs[sample_id]))
                )

                # subset cell types to valid samples (set.cell.types)
                ct_data_1 = ct_data_1[ct_data_1.obs[sample_id].isin(sample_ids)]
                ct_data_2 = ct_data_2[ct_data_2.obs[sample_id].isin(sample_ids)]

                # TODO: shouldn't need this aligning step for cells. corresponds to @scores / y
                #     scores_1 = cca_scores[cell_type_1].loc[ct_data_1.obs.index]
                #     scores_2 = cca_scores[cell_type_2].loc[ct_data_2.obs.index]

                # indexes into the average sample expression per gene with the sample id per cell. corresponds to @tme / x
                tme_1 = self._get_pseudobulks(ct_data_2, sample_id, strategy="mean").loc[
                    :, ct_data_1.obs[sample_id]
                ]  # unclear why we do this
                tme_1.columns = ct_data_1.obs.index
                tme_2 = self._get_pseudobulks(ct_data_1, sample_id, strategy="mean").loc[:, ct_data_2.obs[sample_id]]
                tme_2.columns = ct_data_2.obs.index

                merged_results = {}

                mm_task = mixed_model_progress.add_task("[bold blue]Determining mixed effects", total=len(mcps))
                for mcp in mcps:
                    mixed_model_progress.update(mm_task, description=f"[bold blue]Determining mixed effects for {mcp}")

                    # TODO Check that the genes in result{sig_genes_1] are different and if so note that somewhere and explain why
                    result = {}
                    result["HLM_result_1"], result["sig_genes_1"] = self._apply_HLM_per_MCP_for_one_pair(
                        mcp_name=mcp,
                        scores_df=mcp_scores_df[cell_type_2],
                        ct_data=ct_data_2,
                        tme=tme_2,
                        sig=sig_1,
                        n_counts=n_counts_key,
                        formula=formula,
                        confounder=confounder,
                    )
                    result["HLM_result_2"], result["sig_genes_2"] = self._apply_HLM_per_MCP_for_one_pair(
                        mcp_name=mcp,
                        scores_df=mcp_scores_df[cell_type_1],
                        ct_data=ct_data_1,
                        tme=tme_1,
                        sig=sig_2,
                        n_counts=n_counts_key,
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
