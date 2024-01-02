from collections import ChainMap
from collections.abc import Sequence
from typing import Literal, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scanpy.plotting import DotPlot
from scanpy.tools._score_genes import _sparse_nanmean
from scipy.sparse import issparse

from pertpy.metadata import Drug


def _prepare_targets(
    targets: dict[str, list[str]] | dict[str, dict[str, list[str]]] = None,
    nested: bool = False,
    categories: str | Sequence[str] = None,
) -> Union[ChainMap, dict]:
    if categories is not None:
        if isinstance(categories, str):
            categories = [categories]
        else:
            categories = list(categories)

    if targets is None:
        pt_drug = Drug()
        targets = pt_drug._chembl_json
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
            nested: Indicates if `targets` is a dictionary of dictionaries with group categories as keys. Defaults to False.
            categories: To subset the gene groups to specific categories, especially when `targets=None` or `nested=True`.
                        For ChEMBL drug targets, these are ATC level 1/level 2 category codes.
            method: Method for scoring gene groups. `"mean"` calculates the mean over all genes, while `"seurat"` uses a background profile subtraction approach.
                    Defaults to 'mean'.
            layer: Specifies which `.layers` of AnnData to use for expression values. Defaults to `.X` if None.
            n_bins: The number of expression bins for the `'seurat'` method.
            ctrl_size: The number of genes to randomly sample from each expression bin for the `"seurat"` method.

        Returns:
            An AnnData object with scores.
        """
        if layer is not None:
            mtx = adata.layers[layer]
        else:
            mtx = adata.X

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
        if issparse(mtx):
            scores = mtx.dot(weights)
        else:
            scores = np.dot(mtx, weights)

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

            if issparse(mtx):
                control_profiles = mtx.dot(control_gene_weights)
            else:
                control_profiles = np.dot(mtx, control_gene_weights)
            drug_bins = {}
            for drug in weights.columns:
                bins = np.unique(obs_cut[targets[drug]])
                drug_bins[drug] = np.isin(control_gene_weights.columns, bins)
            drug_weights = pd.DataFrame(drug_bins, index=control_gene_weights.columns)
            drug_weights = drug_weights / drug_weights.sum()
            seurat = np.dot(control_profiles, drug_weights)
            scores = scores - seurat

        adata.uns["pertpy_enrichment_score"] = scores
        adata.uns["pertpy_enrichment_variables"] = weights.columns

        adata.uns["pertpy_enrichment_genes"] = {"var": pd.DataFrame(columns=["genes"]).astype(object)}
        adata.uns["pertpy_enrichment_all_genes"] = {"var": pd.DataFrame(columns=["all_genes"]).astype(object)}

        for drug in weights.columns:
            adata.uns["pertpy_enrichment_genes"]["var"].loc[drug, "genes"] = "|".join(adata.var_names[targets[drug]])
            adata.uns["pertpy_enrichment_all_genes"]["var"].loc[drug, "all_genes"] = "|".join(full_targets[drug])

    def plot_dotplot(
        self,
        adata: AnnData,
        targets: dict[str, list[str]] | dict[str, dict[str, list[str]]] = None,
        categories: Sequence[str] = None,
        groupby: str = None,
        **kwargs,
    ) -> Union[DotPlot, dict, None]:
        """Plots a dotplot by groupby and categories.

        Wraps scanpy's dotplot but formats it nicely by categories.

        Args:
            adata: An AnnData object with enrichment results stored in `.uns["pertpy_enrichment_score"]`.
            targets: Gene groups to evaluate, which can be targets of known drugs, GO terms, pathway memberships, etc.
                     Accepts two forms:
                     - A dictionary with group names as keys and corresponding gene lists as entries.
                     - A dictionary of dictionaries with group categories as keys. Use `nested=True` in this case.
                     If not provided, ChEMBL-derived drug target sets are used.
            categories: To subset the gene groups to specific categories, especially when `targets=None` or `nested=True`.
                            For ChEMBL drug targets, these are ATC level 1/level 2 category codes.
            groupby: dotplot groupby such as clusters or cell types.
            kwargs: Passed to scanpy dotplot.

        Returns:
            If `return_fig` is `True`, returns a :class:`~scanpy.pl.DotPlot` object,
            else if `show` is false, return axes dict.

        Examples:
            >>> import pertpy as pt
            >>> pt_enrichment = pt.tl.Enrichment()
            >>> pt_enrichment.plot_dotplot(adata, categories=["B01","B02","B03"], groupby="leiden")
        """
        if categories is not None:
            if isinstance(categories, str):
                categories = [categories]
            else:
                categories = list(categories)

        if targets is None:
            pt_drug = Drug()
            targets = pt_drug._chembl_json
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

        enrichment_score_adata = AnnData(adata.uns["pertpy_enrichment_score"], obs=adata.obs)
        enrichment_score_adata.var_names = adata.uns["pertpy_enrichment_variables"]

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

        return sc.pl.dotplot(enrichment_score_adata, groupby=groupby, swap_axes=True, **plot_args, **kwargs)
