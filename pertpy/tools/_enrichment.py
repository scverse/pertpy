from collections import ChainMap
from collections.abc import Sequence
from typing import Any, Literal

import blitzgsea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes
from scanpy.plotting import DotPlot
from scanpy.tools._score_genes import _sparse_nanmean
from scipy.sparse import issparse
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy.metadata import Drug


def _prepare_targets(
    targets: dict[str, list[str]] | dict[str, dict[str, list[str]]] = None,
    nested: bool = False,
    categories: str | Sequence[str] = None,
) -> ChainMap | dict:
    if categories is not None:
        categories = [categories] if isinstance(categories, str) else list(categories)

    if targets is None:
        pt_drug = Drug()
        pt_drug.chembl.set()
        targets = pt_drug.chembl.dictionary
        nested = True
    else:
        targets = targets.copy()
    if categories is not None:
        targets = {k: targets[k] for k in categories}  # type: ignore
    if nested:
        targets = dict(ChainMap(*[targets[cat] for cat in targets]))  # type: ignore

    return targets


def _mean(X, names, axis):
    """Helper function to compute a mean of X across an axis, respecting names and possible nans."""
    if issparse(X):
        obs_avg = pd.Series(
            np.array(_sparse_nanmean(X, axis=axis)).flatten(),
            index=names,
        )
    else:
        obs_avg = pd.Series(np.nanmean(X, axis=axis), index=names)
    return obs_avg


class Enrichment:
    def score(
        self,
        adata: AnnData,
        layer: str = None,
        targets: dict[str, list[str]] | dict[str, dict[str, list[str]]] = None,
        nested: bool = False,
        categories: Sequence[str] = None,
        method: Literal["mean", "seurat"] = "mean",
        n_bins: int = 25,
        ctrl_size: int = 50,
        key_added: str = "pertpy_enrichment",
    ) -> None:
        """Obtain per-cell scoring of gene groups of interest.

        Inspired by drug2cell score: https://github.com/Teichlab/drug2cell.
        Ensure that the gene nomenclature in your target sets is compatible with your
        `.var_names`. The ChEMBL drug targets use HGNC.

        Args:
            adata: An AnnData object. It is recommended to use log-normalised data.
            targets: Gene groups to evaluate, which can be targets of known drugs, GO terms, pathway memberships, etc.
                     Accepts two forms:
                     - A dictionary with group names as keys and corresponding gene lists as entries.
                     - A dictionary of dictionaries with group categories as keys. Use `nested=True` in this case.
                     If not provided, ChEMBL-derived drug target sets are used.
            nested: Indicates if `targets` is a dictionary of dictionaries with group categories as keys.
            categories: To subset the gene groups to specific categories, especially when `targets=None` or `nested=True`.
                        For ChEMBL drug targets, these are ATC level 1/level 2 category codes.
            method: Method for scoring gene groups. `"mean"` calculates the mean over all genes,
                    while `"seurat"` uses a background profile subtraction approach.
            layer: Specifies which `.layers` of AnnData to use for expression values.
            n_bins: The number of expression bins for the `'seurat'` method.
            ctrl_size: The number of genes to randomly sample from each expression bin for the `"seurat"` method.
            key_added: Prefix key that adds the results to `uns`.
                       Note that the actual values are `key_added_score`, `key_added_variables`, `key_added_genes`, `key_added_all_genes`.

        Returns:
            An AnnData object with scores.
        """
        mtx = adata.layers[layer] if layer is not None else adata.X

        targets = _prepare_targets(targets=targets, nested=nested, categories=categories)  # type: ignore
        full_targets = targets.copy()

        for drug in targets:
            targets[drug] = np.isin(adata.var_names, targets[drug])

        # Scoring is done via matrix multiplication of the original cell by gene matrix by a new gene by drug matrix
        # with the entries in the new matrix being the weights of each gene for that group (such as drug)
        # The mean across targets is constant -> prepare weights for that
        weights = pd.DataFrame(targets, index=adata.var_names)
        weights = weights.loc[:, weights.sum() > 0]
        weights = weights / weights.sum()
        scores = mtx.dot(weights) if issparse(mtx) else np.dot(mtx, weights)

        if method == "seurat":
            obs_avg = _mean(mtx, names=adata.var_names, axis=0)
            n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
            obs_cut = obs_avg.rank(method="min") // n_items
            obs_cut = obs_cut.values

            control_groups = {}
            for cut in np.unique(obs_cut):
                mask = obs_cut == cut
                r_genes = np.nonzero(mask)[0]
                rng = np.random.default_rng()
                rng.shuffle(r_genes)
                mask[r_genes[ctrl_size:]] = False
                control_groups[cut] = mask
            control_gene_weights = pd.DataFrame(control_groups, index=adata.var_names)
            control_gene_weights = control_gene_weights / control_gene_weights.sum()

            control_profiles = mtx.dot(control_gene_weights) if issparse(mtx) else np.dot(mtx, control_gene_weights)
            drug_bins = {}
            for drug in weights.columns:
                bins = np.unique(obs_cut[targets[drug]])
                drug_bins[drug] = np.isin(control_gene_weights.columns, bins)
            drug_weights = pd.DataFrame(drug_bins, index=control_gene_weights.columns)
            drug_weights = drug_weights / drug_weights.sum()
            seurat = np.dot(control_profiles, drug_weights)
            scores = scores - seurat

        adata.uns[f"{key_added}_score"] = scores
        adata.uns[f"{key_added}_variables"] = weights.columns

        adata.uns[f"{key_added}_genes"] = {"var": pd.DataFrame(columns=["genes"]).astype(object)}
        adata.uns[f"{key_added}_all_genes"] = {"var": pd.DataFrame(columns=["all_genes"]).astype(object)}

        for drug in weights.columns:
            adata.uns[f"{key_added}_genes"]["var"].loc[drug, "genes"] = "|".join(adata.var_names[targets[drug]])
            adata.uns[f"{key_added}_all_genes"]["var"].loc[drug, "all_genes"] = "|".join(full_targets[drug])

    def hypergeometric(
        self,
        adata: AnnData,
        targets: dict[str, list[str] | dict[str, list[str]]] | None = None,
        nested: bool = False,
        categories: str | list[str] | None = None,
        pvals_adj_thresh: float = 0.05,
        direction: str = "both",
        corr_method: Literal["benjamini-hochberg", "bonferroni"] = "benjamini-hochberg",
    ):
        """Perform a hypergeometric test to assess the overrepresentation of gene group members.

        Args:
            adata: With marker genes computed via `sc.tl.rank_genes_groups()` in the original expression space.
            targets: The gene groups to evaluate. Can be targets of known drugs, GO terms, pathway memberships, anything you can assign genes to.
                     If `None`, will use `d2c.score()` output if present, and if not present load the ChEMBL-derived drug target sets distributed with the package.
                     Accepts two forms:
                     - A dictionary with the names of the groups as keys, and the entries being the corresponding gene lists.
                     - A dictionary of dictionaries defined like above, with names of gene group categories as keys.
                     If passing one of those, specify `nested=True`.
            nested: Whether `targets` is a dictionary of dictionaries with group categories as keys.
            categories: If `targets=None` or `nested=True`, this argument can be used to subset the gene groups to one or more categories (keys of the original dictionary).
                        In case of the ChEMBL drug targets, these are ATC level 1/level 2 category codes.
            pvals_adj_thresh: The `pvals_adj` cutoff to use on the `sc.tl.rank_genes_groups()` output to identify markers.
            direction: Whether to seek out up/down-regulated genes for the groups, based on the values from `scores`.
                       Can be `up`, `down`, or `both` (for no selection).
            corr_method: Which FDR correction to apply to the p-values of the hypergeometric test.
                         Can be `benjamini-hochberg` or `bonferroni`.

        Returns:
            Dictionary with clusters for which the original object markers were computed as the keys,
            and data frames of test results sorted on q-value as the items.
        """
        universe = set(adata.var_names)
        targets = _prepare_targets(targets=targets, nested=nested, categories=categories)  # type: ignore
        for group in targets:
            targets[group] = set(targets[group]).intersection(universe)  # type: ignore
        # We remove empty keys since we don't need them
        targets = {k: v for k, v in targets.items() if v}

        overrepresentation = {}
        for cluster in adata.uns["rank_genes_groups"]["names"].dtype.names:
            results = pd.DataFrame(
                1,
                index=list(targets.keys()),
                columns=[
                    "intersection",
                    "gene_group",
                    "markers",
                    "universe",
                    "pvals",
                    "pvals_adj",
                ],
            )
            mask = adata.uns["rank_genes_groups"]["pvals_adj"][cluster] < pvals_adj_thresh
            if direction == "up":
                mask = mask & (adata.uns["rank_genes_groups"]["scores"][cluster] > 0)
            elif direction == "down":
                mask = mask & (adata.uns["rank_genes_groups"]["scores"][cluster] < 0)
            markers = set(adata.uns["rank_genes_groups"]["names"][cluster][mask])
            results["markers"] = len(markers)
            results["universe"] = len(universe)
            results["pvals"] = results["pvals"].astype(float)

            for ind in results.index:
                gene_group = targets[ind]
                common = gene_group.intersection(markers)  # type: ignore
                results.loc[ind, "intersection"] = len(common)
                results.loc[ind, "gene_group"] = len(gene_group)
                # need to subtract 1 from the intersection length
                # https://alexlenail.medium.com/understanding-and-implementing-the-hypergeometric-test-in-python-a7db688a7458
                pval = hypergeom.sf(len(common) - 1, len(universe), len(markers), len(gene_group))
                results.loc[ind, "pvals"] = pval
            # Just in case any NaNs popped up somehow, fill them to 1 so FDR works
            results = results.fillna(1)
            if corr_method == "benjamini-hochberg":
                results["pvals_adj"] = multipletests(results["pvals"], method="fdr_bh")[1]
            elif corr_method == "bonferroni":
                results["pvals_adj"] = np.minimum(results["pvals"] * results.shape[0], 1.0)
            overrepresentation[cluster] = results.sort_values("pvals_adj")

        return overrepresentation

    def gsea(
        self,
        adata: "AnnData",
        targets: dict[str, list[str] | dict[str, list[str]]] | None = None,
        nested: bool = False,
        categories: str | list[str] | None = None,
        absolute: bool = False,
        key_added: str = "pertpy_enrichment_gsea",
    ) -> dict[str, pd.DataFrame] | tuple[dict[str, pd.DataFrame], dict[str, dict]]:  # pragma: no cover
        """Perform gene set enrichment analysis on the marker gene scores using blitzgsea.

        Args:
            adata: AnnData object with marker genes computed via `sc.tl.rank_genes_groups()`
                   in the original expression space.
            targets: The gene groups to evaluate, either as a dictionary with names of the
                     groups as keys and gene lists as values, or a dictionary of dictionaries
                     with names of gene group categories as keys.
                     case it uses `d2c.score()` output or loads ChEMBL-derived drug target sets.
            nested: Indicates if `targets` is a dictionary of dictionaries with group
                    categories as keys.
            categories: Used to subset the gene groups to one or more categories,
                        applicable if `targets=None` or `nested=True`.
            absolute: If True, passes the absolute values of scores to GSEA, improving
                      statistical power.
            key_added: Prefix key that adds the results to `uns`.

        Returns:
            A dictionary with clusters as keys and data frames of test results sorted on
            q-value as the items.
        """
        targets = _prepare_targets(targets=targets, nested=nested, categories=categories)  # type: ignore
        enrichment = {}
        plot_gsea_args: dict[str, Any] = {"targets": targets, "scores": {}}
        for cluster in adata.uns["rank_genes_groups"]["names"].dtype.names:
            df = pd.DataFrame(
                {
                    "0": adata.uns["rank_genes_groups"]["names"][cluster],
                    "1": adata.uns["rank_genes_groups"]["scores"][cluster],
                }
            )
            if absolute:
                df["1"] = np.absolute(df["1"])
                df = df.sort_values("1", ascending=False)
            enrichment[cluster] = blitzgsea.gsea(df, targets)
            plot_gsea_args["scores"][cluster] = df

        adata.uns[key_added] = plot_gsea_args

        return enrichment

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_dotplot(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        *,
        targets: dict[str, dict[str, list[str]]] = None,
        source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
        category_name: str = "interaction_type",
        categories: Sequence[str] = None,
        groupby: str = None,
        key: str = "pertpy_enrichment",
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> DotPlot | None:
        """Plots a dotplot by groupby and categories.

        Wraps scanpy's dotplot but formats it nicely by categories.

        Args:
            adata: An AnnData object with enrichment results stored in `.uns["pertpy_enrichment_score"]`.
            targets: Gene groups to evaluate, which can be targets of known drugs, GO terms, pathway memberships, etc.
                     Accepts a dictionary of dictionaries with group categories as keys.
                     If not provided, ChEMBL-derived or dgbidb drug target sets are used, given by `source`.
            source: Source of drug target sets when `targets=None`, `chembl`, `dgidb` or `pharmgkb`.
            categories: To subset the gene groups to specific categories, especially when `targets=None`.
                            For ChEMBL drug targets, these are ATC level 1/level 2 category codes.
            category_name: The name of category used to generate a nested drug target set when `targets=None` and `source=dgidb|pharmgkb`.
            groupby: dotplot groupby such as clusters or cell types.
            key: Prefix key of enrichment results in `uns`.
            {common_plot_args}
            kwargs: Passed to scanpy dotplot.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> pt_enrichment = pt.tl.Enrichment()
            >>> adata = sc.datasets.pbmc3k_processed()
            >>> pt_enrichment.score(adata)
            >>> sc.tl.rank_genes_groups(adata, method="wilcoxon", groupby="louvain")
            >>> pt_enrichment.plot_dotplot(adata, categories=["B01", "B02", "B03"], groupby="louvain")

        Preview:
            .. image:: /_static/docstring_previews/enrichment_dotplot.png
        """
        if categories is not None:
            categories = [categories] if isinstance(categories, str) else list(categories)

        if targets is None:
            pt_drug = Drug()
            if source == "chembl":
                pt_drug.chembl.set()
                targets = pt_drug.chembl.dictionary
            elif source == "dgidb":
                pt_drug.dgidb.set()
                interaction = pt_drug.dgidb.data
                if category_name not in interaction.columns:
                    raise ValueError("The category name is not available in dgidb drug target data.")
                interaction[category_name] = interaction[category_name].fillna("Unknown/Other")
                targets = (
                    interaction.groupby(category_name)
                    .apply(lambda x: x.groupby("drug_claim_name")["gene_claim_name"].apply(list).to_dict())
                    .to_dict()
                )
            else:
                pt_drug.pharmgkb.set()
                interaction = pt_drug.pharmgkb.data
                if category_name not in interaction.columns:
                    raise ValueError("The category name is not available in pharmgkb drug target data.")
                interaction[category_name] = interaction[category_name].fillna("Unknown/Other")
                targets = (
                    interaction.groupby(category_name)
                    .apply(lambda x: x.groupby("Compound|Disease")["Gene"].apply(list).to_dict())
                    .to_dict()
                )
        else:
            targets = targets.copy()
        if categories is not None:
            targets = {k: targets[k] for k in categories}  # type: ignore

        for group in targets:
            targets[group] = list(targets[group].keys())  # type: ignore

        var_names: list[str] = []
        var_group_positions: list[tuple[int, int]] = []
        var_group_labels: list[str] = []
        start = 0

        enrichment_score_adata = AnnData(adata.uns[f"{key}_score"], obs=adata.obs)
        enrichment_score_adata.var_names = adata.uns[f"{key}_variables"]

        for group in targets:
            targets[group] = list(  # type: ignore
                enrichment_score_adata.var_names[np.isin(enrichment_score_adata.var_names, targets[group])]
            )
            if len(targets[group]) == 0:
                continue
            var_names = var_names + targets[group]  # type: ignore
            var_group_positions = var_group_positions + [(start, len(var_names) - 1)]
            var_group_labels = var_group_labels + [group]
            start = len(var_names)

        plot_args = {
            "var_names": var_names,
            "var_group_positions": var_group_positions,
            "var_group_labels": var_group_labels,
        }

        fig = sc.pl.dotplot(
            enrichment_score_adata,
            groupby=groupby,
            swap_axes=True,
            ax=ax,
            show=False,
            **plot_args,
            **kwargs,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    def plot_gsea(
        self,
        adata: AnnData,
        enrichment: dict[str, pd.DataFrame],
        *,
        n: int = 10,
        key: str = "pertpy_enrichment_gsea",
        interactive_plot: bool = False,
    ) -> None:
        """Generates a blitzgsea top_table plot.

        This function is designed to visualize the results from a Gene Set Enrichment Analysis (GSEA).
        It uses the output from the `gsea()` method, which provides the enrichment data,
        and displays the top results using blitzgsea's `top_table()` plot.

        Args:
            adata: AnnData object to plot.
            enrichment: Cluster names as keys, blitzgsea's ``gsea()`` output as values.
            n: How many top scores to show for each group.
            key: GSEA results key in `uns`.
            interactive_plot: Whether to plot interactively or not.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> pt_enrichment = pt.tl.Enrichment()
            >>> adata = sc.datasets.pbmc3k_processed()
            >>> pt_enrichment.score(adata)
            >>> sc.tl.rank_genes_groups(adata, method="wilcoxon", groupby="louvain")
            >>> enrichment = pt_enrichment.gsea(adata)
            >>> pt_enrichment.plot_gsea(adata, enrichment, interactive_plot=True)

        Preview:
            .. image:: /_static/docstring_previews/enrichment_gsea.png
        """
        for cluster in enrichment:
            fig = blitzgsea.plot.top_table(
                adata.uns[key]["scores"][cluster],
                adata.uns[key]["targets"],
                enrichment[cluster],
                n=n,
                interactive_plot=interactive_plot,
            )
            fig.suptitle(cluster)
            fig.show()
