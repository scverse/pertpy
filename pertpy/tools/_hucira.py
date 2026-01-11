import os
import re
import warnings
from pathlib import Path
from typing import Literal

import blitzgsea
import gseapy as gp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from bokeh.palettes import all_palettes
from pycirclize import Circos
from tqdm.auto import tqdm


def _vprint(msg, verbose):
    if verbose:
        print(msg)


class Hucira:
    def load_cytokine_info(self, save_dir="", force_download=False):
        """Download and load Cytokine information sheet: includes information about sender and receptor genes (for cell-cell communication plot).

        Parameters
        ----------
        save_dir : str
            Directory where the file will be saved.
        force_download : bool
            Allows user to force a fresh download

        Returns:
        -------
        cytokine_info : pandas.DataFrame
        """
        url = (
            "https://raw.githubusercontent.com/theislab/huCIRA/"
            "main/src/hucira/data/"
            "20250125_cytokine_info_with_functional_classification_LV.xlsx"
        )

        if save_dir == "":
            save_dir = Path.cwd()
        save_dir.mkdir(parents=True, exist_ok=True)
        local_path = save_dir / "cytokine_info.xlsx"

        if force_download or not Path.exists(local_path):
            print("Downloading Cytokine Information sheet...")
            cytokine_info = pd.read_excel(url, sheet_name="all_cytokines", engine="openpyxl")
            cytokine_info.to_excel(local_path, sheet_name="all_cytokines")
        else:
            print(f"Loading from: {local_path}")
            cytokine_info = pd.read_excel(local_path)

        return cytokine_info

    def _get_genesets(
        self,
        adata: AnnData,
        df: pd.DataFrame,
        celltype_signature: str,
        direction: Literal["upregulated", "downregulated", "both"] | None = None,
        threshold_pval: float | None = None,
        threshold_lfc: float | None = None,
    ) -> tuple[dict[str, list[str]], pd.DataFrame]:
        """Get shared gene sets between query adata and the Human Cytokine Dictionary, CIP signatures, or custom gene signatures of a chosen cell type.

        Parameters
        ----------
        - adata: AnnData object with gene expression data.
        - df: Either hcd, CIP signature, or a custom dataframe containing columns ["gene", "query_program", "celltype"].
        - celltype_signature: celltype naming convention needs to match df.celltype
        - direction: Relevant for hcd, but not for CIP or custom gene program
        - threshold_pval: Relevant for hcd, but not for CIP or custom gene program
        - threshold_lfc: Relevant for hcd, but not for CIP or custom gene program

        Returns:
        -------
        - gene_set_dict: dictionary with cytokine/CIP as key and associated genes as values
        - gene_set_df: df containing information on gene overlap between query data and gene program for chosen cell type
        """
        required_for_hcd = ["log_fc", "adj_p_value", "cytokine"]
        required_for_CIP = ["gene", "CIP", "celltype"]

        # Construct signature gene set if input is human cytokine dictionary
        if set(required_for_hcd).issubset(df.columns):
            print(f"Computing gene sets of Human Cytokine Dictionary for {celltype_signature}.")
            select = (df.adj_p_value <= threshold_pval) & (df.celltype == celltype_signature)
            if direction == "upregulated":
                select = select & (df.log_fc >= threshold_lfc)
            elif direction == "downregulated":
                select = select & (df.log_fc <= threshold_lfc)
            elif direction == "both":
                select = select & (df.log_fc.abs() >= threshold_lfc)
            else:
                raise ValueError(f"Invalid direction: {direction}.")
            df = df.loc[select]

            gene_set_dict = {}
            gene_set_df = pd.DataFrame()
            for cytokine_i, cytokine in enumerate(df.cytokine.unique()):
                gene_set = df.loc[df.cytokine == cytokine].gene.values
                gene_set_shared = np.intersect1d(gene_set, adata.var_names)
                gene_set_df.loc[cytokine_i, "cytokine"] = cytokine
                gene_set_df.loc[cytokine_i, "num_genes_signature"] = len(gene_set)
                gene_set_df.loc[cytokine_i, "num_shared_genes_signature"] = len(gene_set_shared)
                gene_set_df.loc[cytokine_i, "frac_shared_genes_signature"] = len(gene_set_shared) / len(gene_set)
                gene_set_dict[cytokine] = gene_set_shared

        # Construct signature gene set if input is CIP signatures
        elif set(required_for_CIP).issubset(df.columns):
            print(f"Computing gene sets of Cytokine-induced gene programs for {celltype_signature}.")
            select = df.celltype == celltype_signature
            df = df.loc[select]
            gene_set_dict = {}
            gene_set_df = pd.DataFrame()
            for CIP_i, CIP in enumerate(df.CIP.unique()):
                gene_set = df.loc[df.CIP == CIP].gene.values
                gene_set_shared = np.intersect1d(gene_set, adata.var_names)
                gene_set_df.loc[CIP_i, "CIP"] = CIP
                gene_set_df.loc[CIP_i, "num_genes_signature"] = len(gene_set)
                gene_set_df.loc[CIP_i, "num_shared_genes_signature"] = len(gene_set_shared)
                gene_set_df.loc[CIP_i, "frac_shared_genes_signature"] = len(gene_set_shared) / len(gene_set)
                gene_set_dict[CIP] = gene_set_shared

        # Construct signature gene set for custom gene programs
        elif "query_program" in df.columns:
            print(f"Computing gene sets of user-defined gene programs for {celltype_signature}.")
            select = df.celltype == celltype_signature
            df = df.loc[select]
            gene_set_dict = {}
            gene_set_df = pd.DataFrame()
            for query_program_i, query_program in enumerate(df.query_program.unique()):
                gene_set = df.loc[df.query_program == query_program].gene.values
                gene_set_shared = np.intersect1d(gene_set, adata.var_names)
                gene_set_df.loc[query_program_i, "query_program"] = query_program
                gene_set_df.loc[query_program_i, "num_genes_signature"] = len(gene_set)
                gene_set_df.loc[query_program_i, "num_shared_genes_signature"] = len(gene_set_shared)
                gene_set_df.loc[query_program_i, "frac_shared_genes_signature"] = len(gene_set_shared) / len(gene_set)
                gene_set_dict[query_program] = gene_set_shared

        else:
            raise ValueError(
                "invalid input for df parameter. You can use either the Human Cytokine Dictionary with load_human_cytokine_dict(), or our CIP signatures with load_CIP_signatures(). If you want to compute enrichment of custom gene sets, df must have columns: ['gene', 'query_program', 'celltype']."
            )
            return
        return gene_set_dict, gene_set_df

    def _compute_mu_and_sigma(self, adata: AnnData, contrast_column: str, condition: str) -> pd.DataFrame:
        group = adata[adata.obs[contrast_column] == condition]
        num_cells = group.shape[0]
        X = group.X.toarray() if hasattr(group.X, "toarray") else group.X
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0, ddof=1)
        return {"mu": mu, "sigma": sigma, "num_cells": num_cells}

    def _compute_s2n(
        self,
        adata: AnnData,
        contrast_column: str,
        condition_1: str,
        condition_2: str,
        precomputed_stats: dict | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the signal-to-noise ratio (S2N) for each gene between two conditions in an AnnData object.

        Parameters
        ----------
        - adata: AnnData object with gene expression data.
        - contrast_column: Key in `adata.obs` indicating the condition labels (e.g. "disease_state").
        - condition_1: Name of the first condition (e.g., "flare").
        - condition_2: Name of the second condition (e.g., "healthy").

        Returns:
        -------
        - s2n_scores: pandas Series of S2N values indexed by gene names.
        """
        if precomputed_stats is None:
            # Select cells for each condition
            group1 = adata[adata.obs[contrast_column] == condition_1]
            group2 = adata[adata.obs[contrast_column] == condition_2]

            # number of cells per condition
            num_cells_1 = group1.shape[0]
            num_cells_2 = group2.shape[0]

            # Get expression matrices
            X1 = group1.X.toarray() if hasattr(group1.X, "toarray") else group1.X
            X2 = group2.X.toarray() if hasattr(group2.X, "toarray") else group2.X

            # Compute mean and std per gene
            mu1 = np.mean(X1, axis=0)
            mu2 = np.mean(X2, axis=0)
            sigma1 = np.std(X1, axis=0, ddof=1)
            sigma2 = np.std(X2, axis=0, ddof=1)

        else:
            _vprint("Using precomputed stats", True)
            num_cells_1 = precomputed_stats[condition_1]["num_cells"]
            num_cells_2 = precomputed_stats[condition_2]["num_cells"]
            mu1 = precomputed_stats[condition_1]["mu"]
            mu2 = precomputed_stats[condition_2]["mu"]
            sigma1 = precomputed_stats[condition_1]["sigma"]
            sigma2 = precomputed_stats[condition_2]["sigma"]

        # Compute S2N
        s2n = (mu1 - mu2) / (sigma1 + sigma2 + 1e-8)  # epsilon to avoid division by zero

        num_cells = pd.DataFrame(
            index=[f"{condition_1}_vs_{condition_2}"],
            columns=["num_cells_1", "num_cells_2"],
            data=[[num_cells_1, num_cells_2]],
        )
        stats = pd.DataFrame(s2n, index=adata.var_names, columns=[f"{condition_1}_vs_{condition_2}"])

        return stats, num_cells

    def _compute_ranking_statistic(
        self, adata: AnnData, contrast_column: str, contrasts_combo: list[tuple[str, str]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        rnk_stats, num_cells = [], []
        precomputed_stats = {}

        conditions = []
        for condition in contrasts_combo:
            conditions.extend([condition[0], condition[1]])
        conditions = np.unique(conditions)

        for condition in conditions:
            precomputed_stats[condition] = self._compute_mu_and_sigma(
                adata, contrast_column=contrast_column, condition=condition
            )

        for condition in contrasts_combo:
            _rnk_stats, _num_cells = self._compute_s2n(
                adata,
                contrast_column=contrast_column,
                condition_1=condition[0],
                condition_2=condition[1],
                precomputed_stats=precomputed_stats,
            )
            rnk_stats.append(_rnk_stats)
            num_cells.append(_num_cells)
        return pd.concat(rnk_stats, axis=1), pd.concat(num_cells, axis=0)

    def run_one_enrichment_test(
        self,
        adata: AnnData,
        df: pd.DataFrame,
        celltype_combo: tuple[str, str] = ("B cell", "B_cell"),
        celltype_column: str = "cell_type",
        contrasts_combo: tuple[str, str] | list[tuple[str, str]] = None,
        contrast_column: str = "disease_state",
        direction: Literal["upregulated", "downregulated", "both"] = "upregulated",
        # Filtering parameters for gene set construction
        threshold_lfc: float = 1.0,
        threshold_expression: float = 0.0,
        threshold_pval: float = 0.01,
        # GSEA parameters
        min_size: int = 10,
        max_size: int = 1000,
        permutation_num: int = 1000,
        weight: float = 1.0,
        seed: int = 2025,
        verbose: bool = False,
        threads: int = 6,
    ) -> pd.DataFrame:
        """Computes cytokine enrichment activity in one celltype using GSEA scoring.

        1. "Looks up" query cell type in human cytokine dictionary and retrieves associated up-/downregulated genes per cytokine as reference.
        2. Creates ranking of query data genes contrasting condition1 vs condition2. A continuum from genes most associated with condition1 (top) to genes most associated with condition2 (bottom)
        3. Computes enrichment of each cytokine by matching their associated gene set in the ranked list.

        Parameters
        ----------
        - adata
            The query adata object.
        - df
            Human Cytokine Dictionary
        - celltype_combo
            A tuple with the celltype name of query adata in first position and respective celltype name of df in second position. Simulates "lookup of query in dictionary".
        - celltype_column
            Column name of adata.obs object that stores the cell types.
        - contrasts_combo
            Tuple that stores two biological conditions that are compared to each other in enrichment. E.g., which cytokines are enriched in healthy samples vs disease samples? Can be a list of tuples, function automatically loops through them.
        - contrast_column
            Column name of adata.obs object that stores the biological condition of samples.
        - direction
            "upregulated", "downregulated", or "both" are valid input. Up-/downregulation w.r.t condition1 (condition1 is the first of the two elements in each contrasts tuple.
        - threshold_pval
            Constructs the gene set: Filters for genes in human df with an adj. p-val lower than threshold_pval.
        - threshold_lfc
            Constructs the gene set: Filters for genes in human df that are up/downregulated with a lfc higher than threshold_lfc.
        - threshold_expression
            Filters out genes with mean gene expression across all cells lower than threshold_expression.

        Returns:
        -------
        - results
            A DataFrame with all computed enrichment scores and statistical parameters. Not filtered by significance or robustness yet.
        """
        print(type(contrasts_combo))
        if not isinstance(contrasts_combo, list):
            assert isinstance(contrasts_combo, tuple)
            contrasts_combo = [contrasts_combo]

        celltype_adata = celltype_combo[0]
        celltype_signature = celltype_combo[1]

        # allows potential loop of celltype combos to continue
        if celltype_adata not in adata.obs[celltype_column].unique():
            print(
                f"'{celltype_adata}' is not present in celltype_column ({celltype_column}) of query adata. Skipping enrichment test of this celltype.\n"
            )
            return None

        # filter for cell type
        _vprint("Filter for cell type:", verbose)
        adata = adata[adata.obs[celltype_column] == celltype_adata]
        _vprint("Filter for cell type: done.", verbose)

        # filter based on gene expression
        _vprint("Filter for gene expression:", verbose)
        adata = adata[:, adata.X.mean(axis=0) >= threshold_expression]
        _vprint("Filter for gene expression: done.", verbose)

        # get genesets
        _vprint("Get gene sets:", verbose)
        gene_set_dict, gene_set_df = self._get_genesets(
            adata=adata,
            df=df,
            celltype_signature=celltype_signature,
            direction=direction,
            threshold_pval=threshold_pval,
            threshold_lfc=threshold_lfc,
        )

        gene_set_dict = {
            key: gene_set for key, gene_set in gene_set_dict.items() if min_size < len(gene_set) < max_size
        }

        _vprint("Get gene sets: done.", verbose)

        # compute ranking stat
        _vprint("Get ranking stats:", verbose)
        rnk_stats, num_cells_per_condition = self._compute_ranking_statistic(
            adata, contrast_column=contrast_column, contrasts_combo=contrasts_combo
        )
        _vprint("Get ranking stats: done.", verbose)
        results = []

        for contrast_name in rnk_stats.columns:
            print(contrast_name)
            # format stat so that it can be processed by blitzgsea. E.g., needs col "0": genenames, and "1": scores
            rnk = (
                rnk_stats.loc[:, contrast_name]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values(ascending=False)
                .to_frame()
                .reset_index()
                .rename(columns={"index": "0", contrast_name: "1"})
            )

            # run enrichment (changed gp.prerank() to blitzgsea. The output result is in slightly diff format.
            """
            gp_res = gp.prerank(
                rnk=rnk,
                gene_sets=gene_set_dict,
                min_size=min_size,
                max_size=max_size,
                permutation_num=permutation_num,
                weight=weight,
                outdir=None,
                seed=seed,
                verbose=verbose,
                threads=threads,
            )
            """

            _res = blitzgsea.gsea(rnk, gene_set_dict, permutations=permutation_num)

            _res.loc[:, "Term"] = _res.index
            _res.loc[:, "contrast"] = contrast_name
            _res.loc[:, "num_cells_1"] = num_cells_per_condition.loc[contrast_name, "num_cells_1"]
            _res.loc[:, "num_cells_2"] = num_cells_per_condition.loc[contrast_name, "num_cells_2"]
            _res.loc[:, "percent_duplicate_ranking_stats"] = (rnk.duplicated(keep="first").sum() / rnk.shape[0]) * 100
            results.append(_res)
            _vprint(f"{contrast_name}: done.", verbose)

        # combine results and save hyperparams
        results = pd.concat(results, axis=0, ignore_index=True)
        results.loc[:, "celltype_adata"] = celltype_adata
        results.loc[:, "celltype_signature"] = celltype_signature
        results.loc[:, "celltype_combo"] = f"{celltype_adata} ({celltype_signature})"
        results.loc[:, "direction"] = direction
        results.loc[:, "threshold_pval"] = threshold_pval
        results.loc[:, "threshold_lfc"] = threshold_lfc
        results.loc[:, "threshold_expression"] = threshold_expression
        results.loc[:, "min_size"] = min_size
        results.loc[:, "max_size"] = max_size
        results.loc[:, "permutation_num"] = permutation_num
        results.loc[:, "weight"] = weight
        results.loc[:, "seed"] = seed
        results.loc[:, "threads"] = threads

        required_for_hcd = ["log_fc", "adj_p_value", "cytokine"]
        if set(required_for_hcd).issubset(df.columns):
            results.rename({"Term": "cytokine"}, inplace=True, axis=1)
            results = pd.merge(results, gene_set_df, on="cytokine")
        elif "CIP" in df.columns:
            results.rename({"Term": "CIP"}, inplace=True, axis=1)
            results = pd.merge(results, gene_set_df, on="CIP")
            results.direction = "upregulated"
        elif "query_program" in df.columns:
            results.rename({"Term": "query_program"}, inplace=True, axis=1)
            results = pd.merge(results, gene_set_df, on="query_program")
            results.direction = "custom input"

        return results

    def _check_robustness_fractions(
        self,
        df_pivot: pd.DataFrame,
        threshold_qval: float = 0.1,  # adjusted p value
        threshold_valid: float = 0.1,  # fraction of results required to even consider this condition. I.e. if the test only ran for one set of thresholds, then it is not very robust.
        threshold_below_alpha: float = 0.75,  # fraction of results that need to be significant
    ):
        n_total = np.prod(df_pivot.shape)
        n_valid = n_total - df_pivot.isna().sum().sum()
        n_below_alpha = (
            (df_pivot < threshold_qval).sum().sum()
        )  # number of results below pval threshold, i.e., number of significant results
        frac_valid_results = n_valid / n_total
        frac_pval_below_alpha = n_below_alpha / n_valid  # fraction of significant results relative to valid results
        is_robust = (frac_pval_below_alpha > threshold_below_alpha) & (frac_valid_results > threshold_valid)
        return frac_valid_results, frac_pval_below_alpha, is_robust

    def check_robustness(
        self,
        all_results: pd.DataFrame,
        threshold_qval: float = 0.1,
        threshold_valid: float = 0.1,
        threshold_below_alpha: float = 0.9,
    ):
        """Filters for robust and significant results (<threshold_qval/alpha) out of original enrichments (run_enrichment_test() output).

        Returns only the enrichments that are stable across many different tests and that are statistically significant.


        Parameters
        ----------
        - results
            The DataFrame output from run_enrichment_test().
        - threshold_qval
            Threshold that checks significance of results (leniently). Result is considered significant if its q-val is below this threshold.
        - threshold_valid
            The fraction of results required to even consider this condition. I.e. if the test only ran for one set of thresholds, then it is not very robust.
        - threshold_below_alpha
            The fraction of results that need to be significant


        Returns:
        -------
        - robust_results
            DataFrame with robust and significant enrichments (includes min and max of nes)

        """
        all_thresholds_expression = all_results.threshold_expression.sort_values(ascending=False).unique()
        all_thresholds_lfc = sorted(all_results.threshold_lfc.unique())

        df = pd.DataFrame(index=all_thresholds_expression, columns=all_thresholds_lfc)
        df.index.rename("threshold_expression", inplace=True)
        df.columns.rename("threshold_lfc", inplace=True)

        robust_results = []

        # Get gene_program name of your enrichment analysis.
        if "cytokine" in all_results.columns:
            gene_program = "cytokine"
        elif "CIP" in all_results.columns:
            gene_program = "CIP"
        elif "query_program" in all_results.columns:
            gene_program = "query_program"
        else:
            raise ValueError("Missing column that is defining gene programs in 'all_results'.")
            return

        for contrast in tqdm(all_results.contrast.unique()):
            for celltype_combo in all_results.celltype_combo.unique():
                results_ct = all_results.loc[
                    (all_results.celltype_combo == celltype_combo) & (all_results.contrast == contrast)
                ]
                for program in results_ct[gene_program].unique():
                    results_ct_cy = results_ct.loc[results_ct[gene_program] == program]
                    df_pivot = results_ct_cy.pivot(index="threshold_expression", columns="threshold_lfc", values="fdr")
                    with warnings.catch_warnings():
                        warnings.simplefilter(action="ignore", category=FutureWarning)
                        df_combined = pd.concat([df, df_pivot])
                    df_merged = df_combined.combine_first(df_pivot)
                    df_merged = df_merged.loc[~df_merged.index.duplicated()]
                    df_pivot = df_merged.loc[all_thresholds_expression, all_thresholds_lfc].astype(float)
                    frac_valid_results, frac_pval_below_alpha, is_robust = self._check_robustness_fractions(
                        df_pivot,
                        threshold_qval=threshold_qval,
                        threshold_valid=threshold_valid,
                        threshold_below_alpha=threshold_below_alpha,
                    )

                    if is_robust:
                        robust_results.append(
                            (
                                celltype_combo,
                                contrast,
                                program,
                                frac_valid_results,
                                frac_pval_below_alpha,
                                is_robust,
                                results_ct_cy.nes.min(),
                                results_ct_cy.nes.max(),
                                threshold_qval,
                                threshold_below_alpha,
                            )
                        )

        robust_results = pd.DataFrame(robust_results).rename(
            {
                0: "celltype_combo",
                1: "contrast",
                2: gene_program,
                3: "frac_valid",
                4: "frac_significant",
                5: "is_robust",
                6: "NES_min",
                7: "NES_max",
                8: "qval_threshold",
                9: "threshold_frac_below_alpha",
            },
            axis=1,
        )
        return robust_results

    def get_robust_significant_results(
        self,
        results: pd.DataFrame,
        alphas: list[float] | None = None,
        threshold_valid: float = 0.1,
        threshold_below_alpha: float = 0.9,
    ):
        """Function Wrapper: Filters for robust and signifcant results across several alpha/q-val from original enrichments (run_enrichment_test() output).

        Returns only the enrichments that are statistically significant (q-val), and stable across many different tests (per contrast).
        Calls check_robustness for different qval thresholds to explore more stringent significance thresholds. Use for visualization of results (e.g. in a heatmap). If using thresholds [0.1, 0.05, 0.01] for significant testing, returns significance notations as well (*, **, ***)

        Parameters
        ----------
        - results
            The DataFrame output from run_enrichment_test().
        - alphas
            List of thresholds (q-val) to check significance of results. Result is considered significant if its q-val is below this threshold.
        - threshold_valid
            The fraction of results required to even consider this condition. I.e. if the test only ran for one set of thresholds, then it is not very robust.
        - threshold_below_alpha
            The fraction of results that need to be significant

        Returns:
        -------
        - robust_results_dict
            Dictionary mapping contrasts to lists of the enrichment score results (pivot_df), their significance annotations (annot_df), and significance thresholds (robust_sub).
            robust_results_dict = {contrast1: [pivot_df1, annot_df1, robust_sub1],
                                   contrast2: [pivot_df2, annot_df2, robust_sub2]}
        """
        # default significant values (matching significance stars)
        if alphas is None:
            alphas = [0.1, 0.05, 0.01]

        # Get gene_program name of your enrichment analysis.
        if "cytokine" in results.columns:
            gene_program = "cytokine"
        elif "CIP" in results.columns:
            gene_program = "CIP"
        elif "query_program" in results.columns:
            gene_program = "query_program"
        else:
            raise ValueError("Missing column that is defining gene programs in 'results'.")
            return

        results_robust = [
            self.check_robustness(
                results,
                threshold_qval=alpha,
                threshold_valid=threshold_valid,
                threshold_below_alpha=threshold_below_alpha,
            )
            for alpha in alphas
        ]

        results_robust = pd.concat(results_robust)

        # if none of the results in the df pass the filter, exit out and don't return anything.
        if results_robust.empty:
            print("No robust results to process. Exiting function.")
            return

        results_robust = (
            results_robust.groupby(["contrast", "celltype_combo", gene_program])["qval_threshold"]
            .min()
            .to_frame()
            .reset_index()
        )

        results_mean = (
            results.assign(nes=pd.to_numeric(results.nes, errors="coerce"))  # ensure numeric
            .fillna({"nes": 0})  # only fill NES
            .groupby(["contrast", "celltype_combo", gene_program])["nes"]
            .mean()
            .to_frame()
            .reset_index()
        )

        # Create separate robust results dict for every contrast pair.
        robust_results_dict = {}
        for contrast in results.contrast.unique():
            subset = results_mean[results_mean.contrast == contrast]
            pivot_df = subset.pivot(index=gene_program, columns="celltype_combo", values="nes")

            # create empty annotation df
            annot_df = pivot_df.copy().astype(object)
            annot_df[:] = ""

            # fill annotations based on results_robust
            robust_sub = results_robust[results_robust.contrast == contrast]
            for program in annot_df.index:
                for celltype in annot_df.columns:
                    qval = robust_sub.loc[
                        (robust_sub[gene_program] == program) & (robust_sub.celltype_combo == celltype),
                        "qval_threshold",
                    ]
                    if len(qval) != 0:
                        qval = qval.values[0]
                        if qval == 0.1:
                            annot_df.loc[program, celltype] = "*"
                        elif qval == 0.05:
                            annot_df.loc[program, celltype] = "**"
                        elif qval == 0.01:
                            annot_df.loc[program, celltype] = "***"

            robust_results_dict[contrast] = [pivot_df, annot_df, robust_sub]

        return robust_results_dict

    def _get_senders(
        self,
        adata: AnnData,
        cytokine_info: pd.DataFrame,
        cytokine: str = "IL-32-beta",
        show: bool = False,
        column_cell_type: str = "cell_type",
    ) -> pd.DataFrame:
        genes = np.unique(re.split(", ", cytokine_info.loc[cytokine_info.name == cytokine, "gene"].values[0]))
        mask = np.isin(genes, adata.var_names)

        if not mask.any():
            print(f"None of the cytokine producing genes ({genes}) were found in dataset for cytokine {cytokine}.")
            return None
        if not mask.all():
            print(
                f"The following cytokine producing genes were not found in the dataset and are excluded: {genes[~mask]}"
            )
            genes = genes[mask]
        adata = adata[:, genes]

        # Ranks gene(s) of query sender cytokine across immune cell types.
        adata_out = sc.tl.rank_genes_groups(
            adata,
            groupby=column_cell_type,
            copy=True,
            use_raw=False,
            method="wilcoxon",
        )
        result = adata_out.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        results_mean, results_frac = [], []
        rank_genes_df = []
        for g in groups:
            df = pd.DataFrame(
                {
                    "gene": result["names"][g],
                    "logfoldchanges": result["logfoldchanges"][g],
                    "pvals": result["pvals"][g],
                    "pvals_adj": result["pvals_adj"][g],
                    column_cell_type: g,
                }
            )
            rank_genes_df.append(df)
        rank_genes_df = pd.concat(rank_genes_df, axis=0)
        rank_genes_df.set_index(column_cell_type, inplace=True)
        grouped = rank_genes_df.groupby(column_cell_type)

        # Chooses minimum rank_genes_group() statistical parameters (considers limiting gene, if there are multiple per cytokine)
        grouped_rank_genes_df_all = []
        for celltype in grouped.groups:
            grouped_celltype_df = grouped.get_group(celltype)

            # get gene with smallest log_fold_change (representing limiting gene), and retrieve stat. parameters
            limiting_gene_idx = np.argmin(grouped_celltype_df["logfoldchanges"].values)
            limiting_gene_vals = grouped_celltype_df.iloc[limiting_gene_idx][["logfoldchanges", "pvals", "pvals_adj"]]
            gene_concat = ", ".join(grouped_celltype_df["gene"])
            grouped_rank_genes_df = limiting_gene_vals.to_frame().T
            grouped_rank_genes_df["gene"] = gene_concat
            grouped_rank_genes_df.index = [celltype]
            grouped_rank_genes_df_all.append(grouped_rank_genes_df)

        grouped_rank_genes_df_all = pd.concat(grouped_rank_genes_df_all, axis=0)
        grouped_rank_genes_df_all = grouped_rank_genes_df_all.rename(
            columns={"logfoldchanges": "min_logfoldchanges", "pvals": "min_pvals", "pvals_adj": "min_pvals_adj"}
        )

        # Minimum of mean gene expression of sender cytokine genes:
        X_df = adata[:, genes].to_df()
        frac_df = X_df > 0
        X_df.loc[:, column_cell_type] = adata.obs.loc[:, column_cell_type].values
        frac_df.loc[:, column_cell_type] = adata.obs.loc[:, column_cell_type].values

        # take minimum average gene expression across all genes required for this sender
        results_mean = (
            X_df.groupby(column_cell_type, observed=False).mean().min(axis=1).to_frame().rename({0: "mean_X"}, axis=1)
        )
        # take minimum expression fraction across all genes required for this sender
        results_frac = (
            frac_df.groupby(column_cell_type, observed=False)
            .mean()
            .min(axis=1)
            .to_frame()
            .rename({0: "frac_X"}, axis=1)
        )

        # Final df with information about active sender cytokines.
        results = pd.concat([grouped_rank_genes_df_all, results_mean, results_frac], axis=1)
        results["mean_X>0"] = results["mean_X"].where(results["mean_X"] > 0, None)
        results.loc[:, "cytokine"] = cytokine
        return results

    def _get_receivers(
        self, adata: AnnData, cytokine_info: pd.DataFrame, cytokine: str, column_cell_type: str = "cell_type"
    ) -> pd.DataFrame | None:
        # get receptor genes for this cytokine
        _receptor_genes = cytokine_info.loc[cytokine_info.name == cytokine, "receptor gene"]
        if _receptor_genes.isna().all():
            print(f"No receptor gene found in cytokine_info for cytokine: {cytokine}")
            return None
        assert len(_receptor_genes) == 1, _receptor_genes
        _receptor_genes = _receptor_genes.values[0]
        # there can be multiple receptors
        candidates = re.split("; ", _receptor_genes)
        results_mean, results_frac = [], []
        # each receptor may require the expression of multiple genes
        for candidate in candidates:
            # print(candidate)
            genes = np.array(re.split(", ", candidate))
            mask = np.isin(genes, adata.var_names)
            if not mask.any():
                print(f"None of the cytokine receptor genes ({genes}) were found in dataset for cytokine {cytokine}.")
                continue
            if not mask.all():
                print(
                    f"The following cytokine receptor genes were not found in the dataset and are excluded: {genes[~mask]}"
                )
                genes = genes[mask]
            X_df = adata[:, genes].to_df()
            frac_df = X_df > 0
            X_df.loc[:, column_cell_type] = adata.obs.loc[:, column_cell_type].values
            frac_df.loc[:, column_cell_type] = adata.obs.loc[:, column_cell_type].values
            # take minimum average gene expression across all genes required for this receptor
            results_mean.append(X_df.groupby(column_cell_type, observed=False).mean().min(axis=1).to_frame())
            # take minimum expression fraction across all genes required for this receptor
            results_frac.append(frac_df.groupby(column_cell_type, observed=False).mean().min(axis=1).to_frame())
        if len(results_mean) == 0:
            return None

        results_mean = pd.concat(results_mean, axis=1).max(axis=1).to_frame().rename({0: "mean_X"}, axis=1)
        results_frac = pd.concat(results_frac, axis=1).max(axis=1).to_frame().rename({0: "frac_X"}, axis=1)
        results = pd.concat([results_mean, results_frac], axis=1)
        results.loc[:, "cytokine"] = cytokine
        return results

    def get_one_senders_and_receivers(
        self,
        adata: AnnData,
        cytokine_info: pd.DataFrame,
        cytokine: str,
        celltype_colname: str = "cell_type",
        sender_pvalue_threshold: float = 0.1,
        receiver_mean_X_threshold: float = 0,
        sender_lfc_threshold: float = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates cytokine producer and receiver statistics (senders and receivers of cell-cell communication) for one cytokine.

        Best for exploration purposes of a singular cytokine.

        Parameters
        ----------
        adata : AnnData
            Query adata object of analysis
        cytokine_info : pd.DataFrame
            External file containing info about receptor genes of each cytokine in format
            pd.DataFrame({"name": cytokine, "receptor gene": [gene1, gene2]})
        cytokine : str
            A cytokine, which ideally should be present in robust_results
            (the outcome of the robust enrichment analysis)
        celltype_colname : str, default "cell_type"
            Column name of where cell types are stored in adata

        Returns:
        -------
        df_senders : pd.DataFrame
            Cytokine signal senders per cell type
        df_receivers : pd.DataFrame
            Cytokine signal receivers per cell type
        """
        df_senders = self._get_senders(
            adata=adata, cytokine_info=cytokine_info, cytokine=cytokine, column_cell_type=celltype_colname
        )
        df_receivers = self._get_receivers(
            adata=adata, cytokine_info=cytokine_info, cytokine=cytokine, column_cell_type=celltype_colname
        )
        if df_senders is not None:
            df_senders = df_senders.loc[
                (df_senders.min_pvals < sender_pvalue_threshold)
                & (df_senders.min_logfoldchanges > sender_lfc_threshold)
            ]
        if df_receivers is not None:
            df_receivers = df_receivers.loc[df_receivers.mean_X > receiver_mean_X_threshold]

        return df_senders, df_receivers

    def get_all_senders_and_receivers(
        self,
        adata: AnnData,
        cytokine_info: pd.DataFrame,
        cytokine_list: list = None,
        celltype_colname: str = "cell_type",
        sender_pvalue_threshold: float = 0.1,
        receiver_mean_X_threshold: float = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generates cytokine producer and receiver statistics (senders and receivers of cell-cell communication) for a list of cytokines.

        Best for visualization purposes (for plot_communication function).

        Parameters
        ----------
        adata : AnnData
            Query adata object of analysis
        cytokine_info : pd.DataFrame
            External file containing info about receptor genes of each cytokine in format
            pd.DataFrame({"name": cytokine, "receptor gene": [gene1, gene2]})
        cytokine_list : list, optional
            List of cytokines, which ideally should be present in robust_results
            (the outcome of the robust enrichment analysis). Default is None.
        celltype_colname : str, default "cell_type"
            Column name of where cell types are stored in adata

        Returns:
        -------
        df_src : pd.DataFrame
            All cytokine signal senders
        df_tgt : pd.DataFrame
            All cytokine signal receivers
        """
        senders, receivers = [], []
        for cytokine in cytokine_list:
            df_senders, df_receivers = self.get_one_senders_and_receivers(
                adata=adata,
                cytokine_info=cytokine_info,
                cytokine=cytokine,
                celltype_colname=celltype_colname,
                sender_pvalue_threshold=0.1,
                receiver_mean_X_threshold=0,
            )

            if cytokine == "IL-32-beta":
                # no known receptor genes - create non-informative df_receivers manually.
                all_celltypes = sorted(adata.obs[celltype_colname].unique())
                df_receivers = pd.DataFrame.from_dict(
                    dict(zip(all_celltypes, np.ones([len(all_celltypes), 2]) * np.inf, strict=True)),
                    orient="index",
                ).rename({0: "mean_X", 1: "frac_X"}, axis=1)
                df_receivers.loc[:, "cytokine"] = cytokine

            if df_senders is not None and df_receivers is not None:
                df_senders = df_senders.assign(celltype=df_senders.index)
                df_receivers = df_receivers.assign(celltype=df_receivers.index)

                senders.append(df_senders)
                receivers.append(df_receivers)

        df_src = pd.concat(senders)
        df_tgt = pd.concat(receivers)

        return df_src, df_tgt

    ######## PLOTTING: #########

    def _format_cytokine_names(self, x):
        if isinstance(x, (list, np.ndarray, pd.Index)):
            return [self._format_cytokine_names(_x) for _x in x]
        text = x.get_text() if hasattr(x, "get_text") else x
        text = text.replace("beta", r"$\beta$")
        text = text.replace("alpha", r"$\alpha$")
        text = text.replace("gamma", r"$\gamma$")
        text = text.replace("lambda", r"$\lambda$")
        text = text.replace("omega", r"$\omega$")
        return text

    def plot_significant_results(
        self,
        results_pivot: pd.DataFrame,
        df_annot: pd.DataFrame,
        robust_results_dict: dict[str, pd.DataFrame] | None = None,
        selected_celltypes: list[str] | None = None,
        selected_cytokines: list[str] | None = None,
        fontsize: float = 6.0,
        save_fig: bool = False,
        fig_path: str = "",
        fig_width: float = 10.0,
        fig_height: float = 12.0,
    ):
        """Optional heatmap plotting aid: Plots either the robust results from a dict of contrasts or individually per contrast.

        Parameters
        ----------
        - robust_results_dict:
            robust enrichment score dictionary from get_significant_results(). If this argument is present it has precedence over results_pivot and df_annot.
        - results_pivot:
            pandas DataFrame of robust enrichment for results from one contrast
        - df_annot:
            pandas DataFrame of robust enrichment significance annotations for results from one contrast
        - selected_celltypes:
            Can choose to only visualize selected celltypes out of available from robust results. Must be in robust results, otherwise error.
        - selected_cytokines:
            Can choose to only visualize selected celltypes out of available from robust results. Must be in robust results, otherwise error.

        Returns:
        -------
        - Nothing. Plotting function only

        """
        # Case 1: robust_results_dict is provided. This precedes the other arguments. Plots all contrasts together.
        if robust_results_dict is not None and len(robust_results_dict) > 0:
            n = len(robust_results_dict)
            fig, axes = plt.subplots(1, n, squeeze=False)

            for i, (contrast, (_pivot, _annot, _)) in enumerate(robust_results_dict.items()):
                ax = axes[0, i]
                pivot = _pivot
                annot = _annot

                # Apply filtering if requested
                if selected_celltypes:
                    pivot = pivot.T.loc[selected_celltypes].T
                    annot = annot.T.loc[selected_celltypes].T
                if selected_cytokines:
                    pivot = pivot.loc[selected_cytokines]
                    annot = annot.loc[selected_cytokines]

                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                sns.heatmap(
                    pivot,
                    square=True,
                    annot=annot,
                    cmap="RdBu_r",
                    center=0,
                    annot_kws={"fontsize": fontsize, "family": "sans-serif"},
                    fmt="",
                    linewidths=0.5,
                    linecolor="white",
                    cbar=True,
                    cbar_kws={"shrink": 0.5, "fraction": 0.04, "pad": 0.02},
                    ax=ax,
                )

                ax.set_title(contrast, fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_facecolor("lightgray")
                ax.tick_params(axis="both", which="both", length=0)

                # Axis labels
                ax.set_xticks(0.5 + np.arange(pivot.shape[1]))
                ax.set_xticklabels(pivot.columns, fontsize=fontsize, rotation=90, ha="center")
                ax.set_yticks(0.5 + np.arange(pivot.shape[0]))
                ax.set_yticklabels(self._format_cytokine_names(pivot.index), fontsize=fontsize, rotation=0, ha="right")

            if save_fig:
                fig_file = Path(fig_path) / "all_contrasts_significant_results.svg"
                # Ensure the directory exists
                fig_file.parent.mkdir(parents=True, exist_ok=True)

                plt.savefig(
                    fig_file,
                    bbox_inches="tight",
                    pad_inches=0,
                    dpi=500,
                )
            plt.tight_layout()
            plt.show()
            return

        # Case 2: single robust_result is provided, only the one chosen contrast comparison is plotted.
        if isinstance(results_pivot, pd.DataFrame) and isinstance(df_annot, pd.DataFrame):
            if selected_celltypes:
                results_pivot = results_pivot.T.loc[selected_celltypes].T
                df_annot = df_annot.T.loc[selected_celltypes].T
            if selected_cytokines:
                results_pivot = results_pivot.loc[selected_cytokines]
                df_annot = df_annot.loc[selected_cytokines]

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                results_pivot,
                square=True,
                annot=df_annot,
                cmap="RdBu_r",
                center=0,
                annot_kws={"fontsize": fontsize, "family": "sans-serif"},
                fmt="",
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                cbar_kws={"shrink": 0.5, "fraction": 0.04, "pad": 0.02},
                ax=ax,
            )
            ax.set_title("Contrast1_vs_Contrast2", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_facecolor("lightgray")
            ax.tick_params(axis="both", which="both", length=0)

            # Axis labels
            ax.set_xticks(0.5 + np.arange(results_pivot.shape[1]))
            ax.set_xticklabels(results_pivot.columns, fontsize=fontsize, rotation=90, ha="center")
            ax.set_yticks(0.5 + np.arange(results_pivot.shape[0]))
            ax.set_yticklabels(
                self._format_cytokine_names(results_pivot.index), fontsize=fontsize, rotation=0, ha="right"
            )

            plt.show()

            if save_fig:
                fig_file = Path(fig_path) / "significant_results.svg"
                # Ensure the directory exists
                fig_file.parent.mkdir(parents=True, exist_ok=True)

                plt.savefig(fig_file, bbox_inches="tight", pad_inches=0, dpi=500)
            return

        print("Nothing was plotted. Check input data!")
        return

    def plot_communication(
        self,
        df_src: pd.DataFrame,
        df_tgt: pd.DataFrame,
        frac_expressing_cells_sender: float | None = 0.05,
        frac_expressing_cells_receiver: float | None = 0.05,
        mean_cytokine_gene_expression_sender: float | None = None,
        mean_cytokine_gene_expression_receiver: float | None = None,
        df_enrichment: pd.DataFrame | None = None,
        all_celltypes: list | None = None,
        cytokine2color: dict | None = None,
        celltype2color: dict | None = None,
        figsize: tuple[float, float] = (5, 5),
        show_legend: bool = True,
        save_path: str | None = None,
        lw: float = 1.0,
        fontsize: int = 6,
        loc: str = "upper left",
        bbox_to_anchor: tuple[float, float] = (1, 1),
    ):
        """Generates a Circos plot to visualize cell-cell communication based on cytokine producers and receivers.

        The function filters the input dataframes based on thresholds for fraction of expressing cells
        and mean cytokine gene expression, then creates a circular layout with cell type partitions
        and draws directed links representing cytokine communication between producers and receivers.

        Parameters
        ----------
        df_src : pd.DataFrame
            DataFrame containing producer cell type and cytokine expression statistics,
            typically from `_get_expression_stats`. Must have 'celltype', 'cytokine',
            'mean_cytokine_gene_expression', and 'frac_expressing_cells' columns.
        df_tgt : pd.DataFrame
            DataFrame containing receiver cell type and cytokine expression statistics,
            typically from `_get_expression_stats`. Must have 'celltype', 'cytokine',
            'mean_cytokine_gene_expression', and 'frac_expressing_cells' columns.
        frac_expressing_cells_sender : float | None, default 0.05
            Minimum fraction of cells expressing a cytokine gene for a producer cell type.
            If None, no filtering is applied.
        frac_expressing_cells_receiver : float | None, default 0.05
            Minimum fraction of cells expressing a cytokine gene for a receiver cell type.
            If None, no filtering is applied.
        mean_cytokine_gene_expression_sender : float | None, default None
            Minimum mean expression of a cytokine gene for a producer cell type. If None, no filtering is applied.
        mean_cytokine_gene_expression_receiver : float | None, default None
            Minimum mean expression of a cytokine gene for a receiver cell type. If None, no filtering is applied.
        df_enrichment : pd.DataFrame | None, optional
            Optional dataframe with enrichment information. Default is None.
        all_celltypes : list | None, optional
            List of all cell types. If None, inferred from df_src and df_tgt.
        cytokine2color : dict | None, optional
            Optional mapping from cytokine names to colors.
        celltype2color : dict | None, optional
            Optional mapping from cell type names to colors.
        figsize : tuple[float, float], default (5, 5)
            Figure size for the plot.
        show_legend : bool, default True
            Whether to show the legend.
        save_path : str | None, optional
            Path to save the figure. If None, figure is not saved.
        lw : float, default 1.0
            Line width for links.
        fontsize : int, default 6
            Font size for labels.
        loc : str, default "upper left"
            Legend location.
        bbox_to_anchor : tuple[float, float], default (1, 1)
            Bounding box anchor for the legend.

        """
        if frac_expressing_cells_sender is not None:
            df_src = df_src.loc[df_src.frac_X > frac_expressing_cells_sender]
        if frac_expressing_cells_receiver is not None:
            df_tgt = df_tgt.loc[df_tgt.frac_X > frac_expressing_cells_receiver]
        if mean_cytokine_gene_expression_sender is not None:
            df_src = df_src.loc[df_src.mean_X > mean_cytokine_gene_expression_sender]
        if frac_expressing_cells_receiver is not None:
            df_tgt = df_tgt.loc[df_tgt.mean_X > mean_cytokine_gene_expression_receiver]

        if all_celltypes is None:
            all_celltypes = sorted(np.union1d(df_src.celltype.unique(), df_tgt.celltype.unique()))
        # celltype_colors = all_palettes["Set3"][len(all_celltypes)]
        if celltype2color is None:
            n = len(all_celltypes)

            # Get first 20 colors from Category20
            palette_20 = all_palettes["Category20"][20]
            # Get 20 colors from Category20b
            palette_20b = all_palettes["Category20b"][20]

            # Combine palettes
            combined_palette = palette_20 + palette_20b

            if n > 40:
                raise ValueError(f"Too many cell types ({n}) for available palettes (max 40).")

            # Assign colors to cell types
            celltype_colors = combined_palette[:n]
            celltype2color = dict(zip(all_celltypes, celltype_colors, strict=True))

        all_cytokines = np.union1d(df_src.cytokine.unique(), df_tgt.cytokine.unique())
        cytokine2idx = {cytokine: k for k, cytokine in enumerate(all_cytokines)}
        # cytokine_colors = all_palettes["Category20"][len(all_cytokines)]
        # cytokine2color = dict(zip(all_cytokines, cytokine_colors, strict=True))

        unique_cytokines = df_src.cytokine.unique()
        if df_enrichment is not None:
            significant_cytokines = df_enrichment.cytokine.unique()
            unique_cytokines = np.intersect1d(unique_cytokines, significant_cytokines)

        if cytokine2color is None:
            cytokine_colors = all_palettes["Colorblind"][max(3, len(unique_cytokines))]
            cytokine_colors = cytokine_colors[: len(unique_cytokines)]  # in case there are less than 3 unique cytokines
            # cytokine_colors = all_palettes["Set3"][max(3, len(unique_cytokines))]
            cytokine2color = dict(zip(unique_cytokines, cytokine_colors, strict=True))

        # draw outer circle / cell type partitions
        sectors = dict(zip(all_celltypes, (2 * len(all_cytokines) + 3) * np.ones(len(all_celltypes)), strict=True))

        circos = Circos(sectors, space=3)
        for sector in circos.sectors:
            start, stop = sector.deg_lim
            center = (start + stop) / 2
            track = sector.add_track((92, 100))

            if 160 >= center >= 20:
                ha = "left"
            elif 340 >= center >= 200:
                ha = "right"
            else:
                ha = "center"

            va = "bottom" if center < 90 or center > 270 else "top"

            track.axis(facecolor=celltype2color[sector.name])
            # track.text(shorten_cell_type_names(sector.name), color="black", size=6, r=110, rotation="horizontal", adjust_rotation=False, family="sans-serif", ha=ha)
            track.text(
                sector.name,
                color="black",
                size=fontsize,
                r=110,
                rotation="horizontal",
                adjust_rotation=False,
                family="sans-serif",
                ha=ha,
                va=va,
            )

        # draw links
        legend_cytokine2color = {}
        for _row_idx, row in df_src.iterrows():
            src_celltype = row.celltype
            cytokine_idx = cytokine2idx[row.cytokine]
            tgt_celltypes = df_tgt.loc[df_tgt.cytokine == row.cytokine].celltype.unique()

            for tgt_celltype in tgt_celltypes:
                is_enriched = True  # default --> plot if enriched or whenever no enrichment info is provided

                if df_enrichment is not None:
                    df_enrichment.loc[:, "celltype"] = df_enrichment.celltype_combo.apply(lambda x: x.split(" (")[0])
                    select = (df_enrichment.celltype == tgt_celltype) & (df_enrichment.cytokine == row.cytokine)
                    is_enriched = df_enrichment.loc[select].shape[0] > 0

                if is_enriched:
                    linestyle = None
                    _score = df_tgt.loc[
                        (df_tgt.cytokine == row.cytokine) & (df_tgt.celltype == tgt_celltype), "mean_X"
                    ].values
                    assert len(_score) == 1
                    if not np.isfinite(_score[0]):
                        linestyle = "--"

                    circos.link_line(
                        (src_celltype, 1 + cytokine_idx),  # src node
                        (tgt_celltype, 2 + len(all_cytokines) + cytokine_idx),  # tgt node
                        direction=1,
                        color=cytokine2color[row.cytokine],
                        # color=celltype2color[src_celltype],
                        lw=lw,
                        arrow_height=8.0,
                        arrow_width=8.0,
                        linestyle=linestyle,
                    )
                    if row.cytokine not in legend_cytokine2color:
                        legend_cytokine2color[row.cytokine] = cytokine2color[row.cytokine]

        circos.plotfig(figsize=figsize)
        plt.gca()

        legend_handles = []
        legend_labels = []
        for cytokine, color in legend_cytokine2color.items():
            legend_handles.append(mlines.Line2D([], [], color=color, lw=1.5))
            legend_labels.append(cytokine)
        if show_legend:
            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                title="Cytokines",
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                prop={"family": "sans-serif", "size": 6},
                title_fontsize=6,
            )
        plt.tight_layout()
        if save_path:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=400,
            )
        plt.show()

        return legend_handles, legend_labels
