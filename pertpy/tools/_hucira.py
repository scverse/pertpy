from typing import Literal

import blitzgsea
import gseapy as gp
import numpy as np
import pandas as pd
from anndata import AnnData


def _vprint(msg, verbose):
    if verbose:
        print(msg)


class hucira:
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
            _res = blitzgsea.gsea(rnk, gene_set_dict)

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
