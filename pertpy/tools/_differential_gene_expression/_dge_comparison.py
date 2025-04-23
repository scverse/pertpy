import numpy as np
import pandas as pd
from anndata import AnnData


class DGEEVAL:
    def compare(
        self,
        adata: AnnData | None = None,
        de_key1: str = None,
        de_key2: str = None,
        de_df1: pd.DataFrame | None = None,
        de_df2: pd.DataFrame | None = None,
        shared_top: int = 100,
    ) -> dict[str, float]:
        """Compare two differential expression analyses.

        Compare two sets of DE results and evaluate the similarity by the overlap of top DEG and
        the correlation of their scores and adjusted p-values.

        Args:
            adata: AnnData object containing DE results in `uns`. Required if `de_key1` and `de_key2` are used.
            de_key1: Key for DE results in `adata.uns`, e.g., output of `tl.rank_genes_groups`.
            de_key2: Another key for DE results in `adata.uns`, e.g., output of `tl.rank_genes_groups`.
            de_df1: DataFrame containing DE results, e.g. output from pertpy differential gene expression interface.
            de_df2: DataFrame containing DE results, e.g. output from pertpy differential gene expression interface.
            shared_top: The number of top DEG to compute the proportion of their intersection.

        """
        if (de_key1 or de_key2) and (de_df1 is not None or de_df2 is not None):
            raise ValueError(
                "Please provide either both `de_key1` and `de_key2` with `adata`, or `de_df1` and `de_df2`, but not both."
            )

        if de_df1 is None and de_df2 is None:  # use keys
            if not de_key1 or not de_key2:
                raise ValueError("Both `de_key1` and `de_key2` must be provided together if using `adata`.")

        elif de_df1 is None or de_df2 is None:
            raise ValueError("Both `de_df1` and `de_df2` must be provided together if using DataFrames.")

        if de_key1:
            if not adata:
                raise ValueError("`adata` should be provided with `de_key1` and `de_key2`. ")
            assert all(k in adata.uns for k in [de_key1, de_key2]), (
                "Provided `de_key1` and `de_key2` must exist in `adata.uns`."
            )
            vars = adata.var_names

        if de_df1 is not None:
            for df in (de_df1, de_df2):
                if not {"variable", "log_fc", "adj_p_value"}.issubset(df.columns):
                    raise ValueError("Each DataFrame must contain columns: 'variable', 'log_fc', and 'adj_p_value'.")

            assert set(de_df1["variable"]) == set(de_df2["variable"]), "Variables in both dataframes must match."
            vars = de_df1["variable"].sort_values()

        shared_top = min(shared_top, len(vars))
        vars_ranks = np.arange(1, len(vars) + 1)
        results = pd.DataFrame(index=vars)
        top_names = []

        if de_key1 and de_key2:
            for i, k in enumerate([de_key1, de_key2]):
                label = adata.uns[k]["names"].dtype.names[0]
                srt_idx = np.argsort(adata.uns[k]["names"][label])
                results[f"scores_{i}"] = adata.uns[k]["scores"][label][srt_idx]
                results[f"pvals_adj_{i}"] = adata.uns[k]["pvals_adj"][label][srt_idx]
                results[f"ranks_{i}"] = vars_ranks[srt_idx]
                top_names.append(adata.uns[k]["names"][label][:shared_top])
        else:
            for i, df in enumerate([de_df1, de_df2]):
                srt_idx = np.argsort(df["variable"])
                results[f"scores_{i}"] = df["log_fc"].values[srt_idx]
                results[f"pvals_adj_{i}"] = df["adj_p_value"].values[srt_idx]
                results[f"ranks_{i}"] = vars_ranks[srt_idx]
                top_names.append(df["variable"][:shared_top])

        metrics = {}
        metrics["shared_top_genes"] = len(set(top_names[0]).intersection(top_names[1])) / shared_top
        metrics["scores_corr"] = results["scores_0"].corr(results["scores_1"], method="pearson")
        metrics["pvals_adj_corr"] = results["pvals_adj_0"].corr(results["pvals_adj_1"], method="pearson")
        metrics["scores_ranks_corr"] = results["ranks_0"].corr(results["ranks_1"], method="spearman")

        return metrics
