from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import decoupler as dc
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from statsmodels.stats.multitest import fdrcorrection

if TYPE_CHECKING:
    from anndata import AnnData


class DifferentialGeneExpression:
    """Support for differential gene expression for scverse."""

    def get_pseudobulk(
        self,
        adata: AnnData,
        sample_col: str,
        groups_col: str,
        obs: pd.DataFrame = None,
        layer: str = None,
        use_raw: bool = False,
        mode: str = "sum",
        min_cells=10,
        min_counts: int = 1000,
        dtype: npt.DTypeLike = np.float32,
        skip_checks: bool = False,
    ) -> AnnData:
        """Summarizes expression profiles across cells per sample and group.

        Generates summarized expression profiles across cells per sample (e.g. sample id) and group (e.g. cell type) based on the metadata found in .obs.
        To ensure a minimum quality control, this function removes genes that are not expressed enough across cells (min_prop) or samples (min_smpls),
        and samples with not enough cells (min_cells) or gene counts (min_counts).

        By default this function expects raw integer counts as input and sums them per sample and group (mode='sum'), but other modes are available.

        This function produces some quality control metrics to assess if is necessary to filter some samples.
        The number of cells that belong to each sample is stored in `.obs['psbulk_n_cells']`,
        the total sum of counts per sample in .obs['psbulk_counts'], and the proportion of cells that express a given gene in `.layers[‘psbulk_props’]`.

        Wraps decoupler's `get_pseudobulk` function.
        See: https://decoupler-py.readthedocs.io/en/latest/generated/decoupler.get_pseudobulk.html#decoupler.get_pseudobulk
        for more details.

        Args:
            adata: Input AnnData object.
            sample_col: Column of obs where to extract the samples names.
            groups_col: Column of obs where to extract the groups names.
            obs: If provided, metadata DataFrame.
            layer: If provided, which layer to use.
            use_raw: Use raw attribute of the AnnData object if present.
            mode: How to perform the pseudobulk.
                  Available options are 'sum', 'mean' or 'median'. Also accepts callback functions to perform custom aggregations.
                  Additionally, it is also possible to provide a dictionary of different callback functions, each one stored in a different resulting `.layer`.
                  In this case, the result of the first callback function of the dictionary is stored in .X by default.
            min_cells: Filter to remove samples by a minimum number of cells in a sample-group pair.
            min_counts: Filter to remove samples by a minimum number of summed counts in a sample-group pair.
            dtype: Type of float used.
            skip_checks: Whether to skip input checks.
                         Set to True when working with positive and negative data, or when counts are not integers.

        Returns:
            Returns new AnnData object with unormalized pseudobulk profiles per sample and group.
        """
        pseudobulk_adata = dc.get_pseudobulk(
            adata,
            sample_col=sample_col,
            groups_col=groups_col,
            obs=obs,
            layer=layer,
            use_raw=use_raw,
            mode=mode,
            min_counts=min_counts,
            dtype=dtype,
            min_cells=min_cells,
            skip_checks=skip_checks,
        )

        return pseudobulk_adata

    def filter_by_expr(
        self,
        adata: AnnData,
        obs: pd.DataFrame = None,
        group: str | None = None,
        lib_size: int | float | None = None,
        min_count: int = 10,
        min_total_count: int = 15,
        large_n: int = 10,
        min_prop: float = 0.7,
    ) -> AnnData:
        """Filter AnnData by which genes have sufficiently large counts to be retained in a statistical analysis.

        Wraps decoupler's `filter_by_expr` function.
        See https://decoupler-py.readthedocs.io/en/latest/generated/decoupler.filter_by_expr.html#decoupler.filter_by_expr
        for more details.

        Args:
            adata: AnnData obtained after running `get_pseudobulk`.
            obs: Metadata dataframe, only needed if `adata` is not an `AnnData`.
            group: Name of the `.obs` column to group by. If None, assumes all samples belong to one group.
            lib_size: Library size. Defaults to the sum of reads per sample if None.
            min_count: Minimum count required per gene for at least some samples.
            min_total_count: Minimum total count required per gene across all samples.
            large_n: Number of samples per group considered to be "large".
            min_prop: Minimum proportion of samples in the smallest group that express the gene.

        Returns:
            AnnData with only the genes that are to be kept.
        """
        genes = dc.filter_by_expr(
            adata=adata,
            obs=obs,
            group=group,
            lib_size=lib_size,
            min_count=min_count,
            min_total_count=min_total_count,
            large_n=large_n,
            min_prop=min_prop,
        )
        filtered_adata = adata[:, genes].copy()

        return filtered_adata

    def filter_by_prop(self, adata: AnnData, min_prop: float = 0.2, min_samples: int = 2) -> AnnData:
        """Determine which genes are expressed in a sufficient proportion of cells across samples.

        This function selects genes that are sufficiently expressed across cells in each sample and that this condition
        is met across a minimum number of samples.

        Args:
            adata: AnnData obtained after running `get_pseudobulk`. It requieres `.layer['psbulk_props']`.
            min_prop: Minimum proportion of cells that express a gene in a sample.
            min_samples: Minimum number of samples with bigger or equal proportion of cells with expression than `min_prop`.

        Returns:
            AnnData with only the genes that are to be kept.
        """
        genes = dc.filter_by_prop(adata=adata, min_prop=min_prop, min_smpls=min_samples)
        filtered_adata = adata[:, genes].copy()

        return filtered_adata

    def calculate_correlation(
        self,
        de_res_1: pd.DataFrame,
        de_res_2: pd.DataFrame,
        method: Literal["spearman", "pearson", "kendall-tau"] = "spearman",
    ) -> pd.DataFrame:
        """Calculate the Spearman correlation coefficient for 'pvals_adj' and 'logfoldchanges' columns.

        Args:
            de_res_1: A DataFrame with DE result columns.
            de_res_2: Another DataFrame with the same DE result columns.
            method: The correlation method to apply. One of `spearman`, `pearson`, `kendall-tau`.
                    Defaults to `spearman`.

        Returns:
            A DataFrame with the Spearman correlation coefficients for 'pvals_adj' and 'logfoldchanges'.
        """
        columns_of_interest = ["pvals_adj", "logfoldchanges"]
        correlation_data = {}
        for col in columns_of_interest:
            match method:
                case "spearman":
                    correlation, _ = spearmanr(de_res_1[col], de_res_2[col])
                case "pearson":
                    correlation, _ = pearsonr(de_res_1[col], de_res_2[col])
                case "kendall-tau":
                    correlation, _ = kendalltau(de_res_1[col], de_res_2[col])
                case _:
                    raise ValueError("Unknown correlation method.")
            correlation_data[col] = correlation

        return pd.DataFrame([correlation_data], columns=columns_of_interest)

    def calculate_jaccard_index(self, de_res_1: pd.DataFrame, de_res_2: pd.DataFrame, threshold: float = 0.05) -> float:
        """Calculate the Jaccard index for sets of significantly expressed genes/features based on a p-value threshold.

        Args:
            de_res_1: A DataFrame with DE result columns, including 'pvals'.
            de_res_2: Another DataFrame with the same DE result columns.
            threshold: A threshold for determining significant expression (default is 0.05).

        Returns:
            The Jaccard index.
        """
        significant_set_1 = set(de_res_1[de_res_1["pvals"] <= threshold].index)
        significant_set_2 = set(de_res_2[de_res_2["pvals"] <= threshold].index)

        intersection = significant_set_1.intersection(significant_set_2)
        union = significant_set_1.union(significant_set_2)

        return len(intersection) / len(union) if union else 0

    def calculate_cohens_d(self, de_res_1: pd.DataFrame, de_res_2: pd.DataFrame) -> pd.Series:
        """Calculate Cohen's D for the logfoldchanges.

        Args:
            de_res_1: A DataFrame with DE result columns, including 'logfoldchanges'.
            de_res_2: Another DataFrame with the same DE result columns.

        Returns:
            A pandas Series containing Cohen's D for each gene/feature.
        """
        means_1 = de_res_1["logfoldchanges"].mean()
        means_2 = de_res_2["logfoldchanges"].mean()
        sd_1 = de_res_1["logfoldchanges"].std()
        sd_2 = de_res_2["logfoldchanges"].std()

        pooled_sd = np.sqrt((sd_1**2 + sd_2**2) / 2)
        cohens_d = (means_1 - means_2) / pooled_sd

        return cohens_d

    def de_res_to_anndata(
        self,
        adata: AnnData,
        de_res: pd.DataFrame,
        *,
        groupby: str,
        gene_id_col: str = "gene_symbols",
        score_col: str = "scores",
        pval_col: str = "pvals",
        pval_adj_col: str | None = "pvals_adj",
        lfc_col: str = "logfoldchanges",
        key_added: str = "rank_genes_groups",
    ) -> None:
        """Add tabular differential expression result to AnnData as if it was produced by `scanpy.tl.rank_genes_groups`.

        Args:
            adata:
                Annotated data matrix
            de_res:
                Tablular de result
            groupby:
                Column in `de_res` that indicates the group. This column must also exist in `adata.obs`.
            gene_id_col:
                Column in `de_res` that holds the gene identifiers
            score_col:
                Column in `de_res` that holds the score (results will be ordered by score).
            pval_col:
                Column in `de_res` that holds the unadjusted pvalue
            pval_adj_col:
                Column in `de_res` that holds the adjusted pvalue.
                If not specified, the unadjusted pvalues will be FDR-adjusted.
            lfc_col:
                Column in `de_res` that holds the log fold change
            key_added:
                Key under which the results will be stored in `adata.uns`
        """
        if groupby not in adata.obs.columns or groupby not in de_res.columns:
            raise ValueError("groupby column must exist in both adata and de_res.")
        res_dict = {
            "params": {
                "groupby": groupby,
                "reference": "rest",
                "method": "other",
                "use_raw": True,
                "layer": None,
                "corr_method": "other",
            },
            "names": [],
            "scores": [],
            "pvals": [],
            "pvals_adj": [],
            "logfoldchanges": [],
        }
        df_groupby = de_res.groupby(groupby)
        for _, tmp_df in df_groupby:
            tmp_df = tmp_df.sort_values(score_col, ascending=False)
            res_dict["names"].append(tmp_df[gene_id_col].values)  # type: ignore
            res_dict["scores"].append(tmp_df[score_col].values)  # type: ignore
            res_dict["pvals"].append(tmp_df[pval_col].values)  # type: ignore
            if pval_adj_col is not None:
                res_dict["pvals_adj"].append(tmp_df[pval_adj_col].values)  # type: ignore
            else:
                res_dict["pvals_adj"].append(fdrcorrection(tmp_df[pval_col].values)[1])  # type: ignore
            res_dict["logfoldchanges"].append(tmp_df[lfc_col].values)  # type: ignore

        for key in ["names", "scores", "pvals", "pvals_adj", "logfoldchanges"]:
            res_dict[key] = pd.DataFrame(
                np.vstack(res_dict[key]).T,
                columns=list(df_groupby.groups.keys()),
            ).to_records(index=False, column_dtypes="O")
        adata.uns[key_added] = res_dict

    def de_analysis(
        self,
        adata: AnnData,
        groupby: str,
        method: Literal["t-test", "wilcoxon", "pydeseq2", "deseq2", "edger"],
        *formula: str | None,
        contrast: str | None,
        inplace: bool = True,
        key_added: str | None,
    ) -> pd.DataFrame:
        """Perform differential expression analysis.

        Args:
            adata: single-cell or pseudobulk AnnData object
            groupby: Column in adata.obs that contains the factor to test, e.g. `treatment`.
                     For simple statistical tests (t-test, wilcoxon), it is sufficient to specify groupby.
                     Linear models require to specify a formula.
                     In that case, the `groupby` column is used to compute the contrast.
            method: Which method to use to perform the DE test.
            formula: model specification for linear models. E.g. `~ treatment + sex + age`.
                     MUST contain the factor specified in `groupby`.
            contrast: See e.g. https://www.statsmodels.org/devel/contrasts.html for more information.
            inplace: if True, save the result in `adata.varm[key_added]`
            key_added: Key under which the result is saved in `adata.varm` if inplace is True.
                       If set to None this defaults to `de_{method}_{groupby}`.
        Returns:
            Depending on the method a Pandas DataFrame containing at least:
            * gene_id
            * log2 fold change
            * mean expression
            * unadjusted p-value
            * adjusted p-value
        """
        raise NotImplementedError
