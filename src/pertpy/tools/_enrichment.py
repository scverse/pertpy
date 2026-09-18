from collections import ChainMap
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import blitzgsea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from matplotlib.axes import Axes
from scanpy.plotting import DotPlot
from scanpy.tools._score_genes import _sparse_nanmean
from scipy.stats import hypergeom
from scverse_misc import Deprecation, deprecated_arg
from statsmodels.stats.multitest import multipletests

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._types import CSBase, cast_frame, cast_matrix
from pertpy.metadata import Drug


def _prepare_targets(
    targets: Mapping[str, list[str] | dict[str, list[str]]] | None = None,
    nested: bool = False,
    categories: str | Sequence[str] | None = None,
) -> ChainMap | dict:
    if categories is not None:
        categories = [categories] if isinstance(categories, str) else list(categories)

    groups: dict[str, Any]
    if targets is None:
        pt_drug = Drug()
        pt_drug.chembl.set()
        groups = pt_drug.chembl.dictionary
        nested = True
    else:
        groups = dict(targets)
    if categories is not None:
        groups = {k: groups[k] for k in categories}
    if nested:
        groups = dict(ChainMap(*[groups[cat] for cat in groups]))

    return groups


def _mean(X, names, axis):
    """Helper function to compute a mean of X across an axis, respecting names and possible nans."""
    if isinstance(X, CSBase):
        obs_avg = pd.Series(
            np.array(_sparse_nanmean(X, axis=axis)).flatten(),
            index=names,
        )
    else:
        obs_avg = pd.Series(np.nanmean(X, axis=axis), index=names)
    return obs_avg


def _get_signature_matrix(adata: AnnData, layer: str | None) -> np.ndarray | CSBase:
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer {layer!r} does not exist in the .layers attribute.")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X
    return cast_matrix(matrix)


def _get_signature_genes(adata: AnnData, gene_symbols_key: str | None) -> np.ndarray:
    raw_genes: pd.Index[Any] | pd.Series[Any]
    if gene_symbols_key is None:
        raw_genes = adata.var_names
    else:
        var = cast_frame(adata.var)
        if gene_symbols_key not in var:
            raise ValueError(f"Column {gene_symbols_key!r} does not exist in the .var attribute.")
        raw_genes = var[gene_symbols_key]

    if pd.isna(raw_genes).any():
        raise ValueError("Gene identifiers used for signature matching must not be missing.")
    genes = pd.Index(raw_genes.astype(str))
    if genes.has_duplicates:
        duplicates = genes[genes.duplicated()].unique().tolist()
        raise ValueError(f"Gene identifiers used for signature matching must be unique; found {duplicates!r}.")
    return genes.to_numpy()


def _prepare_query_signature(
    query_signature: Mapping[str, float] | pd.Series | None,
    up_genes: Sequence[str] | None,
    down_genes: Sequence[str] | None,
) -> pd.Series:
    if query_signature is not None and (up_genes is not None or down_genes is not None):
        raise ValueError("Pass either `query_signature` or `up_genes`/`down_genes`, not both.")

    if query_signature is not None:
        if isinstance(query_signature, pd.Series):
            signature = query_signature.astype(float).copy()
        else:
            signature = pd.Series(dict(query_signature), dtype=float)
    else:
        up = [up_genes] if isinstance(up_genes, str) else ([] if up_genes is None else list(up_genes))
        down = [down_genes] if isinstance(down_genes, str) else ([] if down_genes is None else list(down_genes))
        up = [str(gene) for gene in up]
        down = [str(gene) for gene in down]
        if len(up) != len(set(up)):
            raise ValueError("`up_genes` must not contain duplicate genes.")
        if len(down) != len(set(down)):
            raise ValueError("`down_genes` must not contain duplicate genes.")

        values: dict[str, float] = {}
        for gene in up:
            values[gene] = 1.0
        for gene in down:
            if gene in values:
                raise ValueError(f"Gene {gene!r} occurs in both `up_genes` and `down_genes`.")
            values[gene] = -1.0
        signature = pd.Series(values, dtype=float)

    if pd.isna(signature.index).any():
        raise ValueError("Query gene identifiers must not be missing.")
    signature.index = signature.index.astype(str)
    if signature.index.has_duplicates:
        duplicates = signature.index[signature.index.duplicated()].unique().tolist()
        raise ValueError(f"Query gene identifiers must be unique; found {duplicates!r}.")
    if not np.isfinite(signature.to_numpy(dtype=float)).all():
        raise ValueError("Query signature values must be finite.")
    signature = signature[signature != 0]
    if signature.empty:
        raise ValueError("The query signature must contain at least one non-zero gene.")
    return signature


def _weighted_enrichment_score(values: np.ndarray, hits: np.ndarray) -> float:
    n_hits = int(hits.sum())
    n_misses = len(hits) - n_hits
    if n_hits == 0 or n_misses == 0:
        raise ValueError("Weighted enrichment requires at least one hit and one non-hit gene.")

    order = np.argsort(values, kind="mergesort")[::-1]
    ranked_values = values[order]
    ranked_hits = hits[order]
    ranked_weights = np.abs(ranked_values)
    hit_weights = ranked_weights * ranked_hits
    max_hit_weight = hit_weights.max()
    if max_hit_weight == 0:
        return 0.0

    hit_weights = hit_weights / max_hit_weight
    hit_weight_sum = hit_weights.sum()
    hit_step = hit_weights / hit_weight_sum
    miss_step = (~ranked_hits) / n_misses
    tie_starts = np.r_[0, np.flatnonzero(ranked_values[1:] != ranked_values[:-1]) + 1]
    running: np.ndarray = np.cumsum(np.add.reduceat(hit_step - miss_step, tie_starts))
    running[-1] = 0.0
    max_score = float(running.max())
    min_score = float(running.min())
    return max_score if abs(max_score) >= abs(min_score) else min_score


def _cmap_connectivity(values: np.ndarray, up_mask: np.ndarray, down_mask: np.ndarray) -> float:
    es_up = _weighted_enrichment_score(values, up_mask) if up_mask.any() else float("nan")
    es_down = _weighted_enrichment_score(values, down_mask) if down_mask.any() else float("nan")

    if up_mask.any() and down_mask.any():
        if np.sign(es_up) == np.sign(es_down):
            return 0.0
        return float((es_up - es_down) / 2.0)
    if up_mask.any():
        return es_up
    return float(-es_down)


class Enrichment:
    def score(
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        targets: dict[str, list[str]] | dict[str, dict[str, list[str]]] | None = None,
        nested: bool = False,
        categories: Sequence[str] | None = None,
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
        mtx = cast_matrix(adata.layers[layer] if layer is not None else adata.X)

        target_groups: ChainMap | dict = _prepare_targets(targets=targets, nested=nested, categories=categories)
        full_targets = target_groups.copy()

        for drug in target_groups:
            target_groups[drug] = np.isin(adata.var_names, target_groups[drug])

        # Scoring is done via matrix multiplication of the original cell by gene matrix by a new gene by drug matrix
        # with the entries in the new matrix being the weights of each gene for that group (such as drug)
        # The mean across targets is constant -> prepare weights for that
        weights = pd.DataFrame(target_groups, index=adata.var_names)
        weights = weights.loc[:, weights.sum() > 0]
        weights = weights / weights.sum()
        scores = mtx.dot(weights) if isinstance(mtx, CSBase) else np.dot(mtx, weights)

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

            control_profiles = (
                mtx.dot(control_gene_weights) if isinstance(mtx, CSBase) else np.dot(mtx, control_gene_weights)
            )
            drug_bins = {}
            for drug in weights.columns:
                bins = np.unique(obs_cut[target_groups[drug]])
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
            adata.uns[f"{key_added}_genes"]["var"].loc[drug, "genes"] = "|".join(adata.var_names[target_groups[drug]])
            adata.uns[f"{key_added}_all_genes"]["var"].loc[drug, "all_genes"] = "|".join(full_targets[drug])

    def signature_reversal(
        self,
        adata: AnnData,
        query_signature: Mapping[str, float] | pd.Series | None = None,
        *,
        up_genes: Sequence[str] | None = None,
        down_genes: Sequence[str] | None = None,
        layer: str | None = None,
        gene_symbols_key: str | None = None,
        min_genes: int = 1,
        key_added: str = "signature_reversal",
    ) -> None:
        """Score perturbations by how strongly they reverse a disease/query signature.

        This computes a raw CMap-style weighted connectivity score (WTCS) on a perturbation-level AnnData object, where observations are perturbations and variables are genes.
        Values must be finite, signed perturbation effects relative to an appropriate matched control, such as z-scores, log-fold changes, or control-subtracted expression.
        Raw or pseudobulk mean expression is not a perturbation signature.

        Positive query genes are up-regulated in the query state and negative genes are down-regulated.
        Only the sign of values in `query_signature` is used.
        Connectivity ranges from -1 (opposing) to 1 (similar), and the stored reversal score is its negative so that higher values indicate stronger opposition.
        This method does not compute CMap's normalized connectivity score, tau, p-values, or false-discovery rates.
        Genes with equal perturbation values are treated as a single rank group so their input order cannot affect the score.

        A high reversal score is a hypothesis for follow-up, not evidence of efficacy or safety.
        In particular, suppressing a compensatory or protective transcriptional response can also produce a high reversal score.

        Args:
            adata: Perturbation-level AnnData with perturbations as observations, genes as variables, and finite signed perturbation effects relative to matched controls in `.X` or `layer`.
            query_signature: Signed query signature.
                Positive values indicate query-up genes and negative values query-down genes; magnitudes are ignored.
            up_genes: Query-up genes. Used when `query_signature` is not provided.
            down_genes: Query-down genes. Used when `query_signature` is not provided.
            layer: Layer containing perturbation signatures. Defaults to `.X`.
            gene_symbols_key: Optional `.var` column used to match query gene names instead of `.var_names`.
                Gene identifiers used for matching must be unique and non-missing.
            min_genes: Minimum total number of query genes that must be present in `adata`.
                CMap recommends query sets containing roughly 10 to 200 genes; very small matches should be treated as exploratory.
            key_added: Prefix used to store results in `.obs` and `.uns`.

        Returns:
            Updates `adata` with `{key_added}_score`, `{key_added}_connectivity` and `{key_added}_rank` in `.obs`, and query metadata in `.uns[key_added]`.

        Examples:
            >>> import numpy as np
            >>> import pertpy as pt
            >>> from anndata import AnnData
            >>> effect_adata = AnnData(np.array([[-2.0, 1.0, 0.5]]))
            >>> effect_adata.var_names = ["IL6", "CCR7", "other"]
            >>> enr = pt.tl.Enrichment()
            >>> enr.signature_reversal(effect_adata, up_genes=["IL6"], down_genes=["CCR7"])
        """
        if min_genes < 1:
            raise ValueError("`min_genes` must be at least 1.")

        matrix = _get_signature_matrix(adata, layer)
        genes = _get_signature_genes(adata, gene_symbols_key)
        query = _prepare_query_signature(query_signature, up_genes, down_genes)
        aligned_query = query.reindex(genes).to_numpy(dtype=float)
        present = np.isfinite(aligned_query)
        n_present = int(present.sum())
        if n_present < min_genes:
            raise ValueError(f"Only {n_present} query genes are present in `adata`; at least {min_genes} are required.")

        up_mask = aligned_query > 0
        down_mask = aligned_query < 0
        if (query > 0).any() and not up_mask.any():
            raise ValueError("None of the query-up genes are present in `adata`.")
        if (query < 0).any() and not down_mask.any():
            raise ValueError("None of the query-down genes are present in `adata`.")

        connectivity_scores = np.empty(adata.n_obs, dtype=float)
        for idx in range(adata.n_obs):
            values = np.asarray(to_dense(matrix[idx]), dtype=float).reshape(-1)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Perturbation signature {adata.obs_names[idx]!r} contains non-finite values; "
                    "all signatures must be finite and use the same gene universe."
                )
            connectivity_scores[idx] = _cmap_connectivity(values, up_mask, down_mask)

        score_key = f"{key_added}_score"
        connectivity_key = f"{key_added}_connectivity"
        rank_key = f"{key_added}_rank"
        reversal_scores = -connectivity_scores
        adata.obs[score_key] = reversal_scores
        adata.obs[connectivity_key] = connectivity_scores
        adata.obs[rank_key] = (
            pd.Series(reversal_scores, index=adata.obs_names)
            .rank(ascending=False, method="min", na_option="keep")
            .astype("Int64")
        )
        adata.uns[key_added] = {
            "score_key": score_key,
            "connectivity_key": connectivity_key,
            "rank_key": rank_key,
            "method": "cmap_wtcs",
            "layer": layer,
            "gene_symbols_key": gene_symbols_key,
            "min_genes": min_genes,
            "query_signature": query.to_dict(),
            "up_genes": query.index[query > 0].tolist(),
            "down_genes": query.index[query < 0].tolist(),
            "matched_genes": genes[present].tolist(),
            "n_query_genes": int(len(query)),
            "n_matched_genes": n_present,
            "n_matched_up_genes": int(up_mask.sum()),
            "n_matched_down_genes": int(down_mask.sum()),
        }

    @deprecated_arg(
        "pvals_adj_thresh",
        Deprecation("1.0.6", "Use `padj_threshold`."),
    )
    def hypergeometric(
        self,
        adata: AnnData,
        *,
        targets: dict[str, list[str] | dict[str, list[str]]] | None = None,
        nested: bool = False,
        categories: str | list[str] | None = None,
        padj_threshold: float = 0.05,
        pvals_adj_thresh: float | None = None,
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
            padj_threshold: The `pvals_adj` cutoff to use on the `sc.tl.rank_genes_groups()` output to identify markers.
            direction: Whether to seek out up/down-regulated genes for the groups, based on the values from `scores`.
                       Can be `up`, `down`, or `both` (for no selection).
            corr_method: Which FDR correction to apply to the p-values of the hypergeometric test.
                         Can be `benjamini-hochberg` or `bonferroni`.
            pvals_adj_thresh: Deprecated and will be removed in a future release. Use `padj_threshold`.

        Returns:
            Dictionary with clusters for which the original object markers were computed as the keys,
            and data frames of test results sorted on q-value as the items.
        """
        if pvals_adj_thresh is not None:
            padj_threshold = pvals_adj_thresh

        universe = set(adata.var_names)
        prepared: ChainMap | dict = _prepare_targets(targets=targets, nested=nested, categories=categories)
        for group in prepared:
            prepared[group] = set(prepared[group]).intersection(universe)
        # We remove empty keys since we don't need them
        target_groups = {k: v for k, v in prepared.items() if v}

        overrepresentation = {}
        for cluster in adata.uns["rank_genes_groups"]["names"].dtype.names:
            results = pd.DataFrame(
                1,
                index=list(target_groups.keys()),
                columns=[
                    "intersection",
                    "gene_group",
                    "markers",
                    "universe",
                    "pvals",
                    "pvals_adj",
                ],
            )
            mask = adata.uns["rank_genes_groups"]["pvals_adj"][cluster] < padj_threshold
            if direction == "up":
                mask = mask & (adata.uns["rank_genes_groups"]["scores"][cluster] > 0)
            elif direction == "down":
                mask = mask & (adata.uns["rank_genes_groups"]["scores"][cluster] < 0)
            markers = set(adata.uns["rank_genes_groups"]["names"][cluster][mask])
            results["markers"] = len(markers)
            results["universe"] = len(universe)
            results["pvals"] = results["pvals"].astype(float)

            for ind in results.index:
                gene_group = target_groups[ind]
                common = gene_group.intersection(markers)
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
        *,
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
        targets: dict[str, dict[str, list[str]]] | None = None,
        source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
        category_name: str = "interaction_type",
        categories: Sequence[str] | None = None,
        groupby: str | None = None,
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
        nested_targets: dict[str, Any] = dict(targets)
        if categories is not None:
            nested_targets = {k: nested_targets[k] for k in categories}

        group_genes: dict[str, list[str]] = {group: list(genes) for group, genes in nested_targets.items()}

        var_names: list[str] = []
        var_group_positions: list[tuple[int, int]] = []
        var_group_labels: list[str] = []
        start = 0

        enrichment_score_adata = AnnData(adata.uns[f"{key}_score"], obs=cast_frame(adata.obs))
        enrichment_score_adata.var_names = adata.uns[f"{key}_variables"]

        for group in group_genes:
            group_genes[group] = list(
                enrichment_score_adata.var_names[np.isin(enrichment_score_adata.var_names, group_genes[group])]
            )
            if len(group_genes[group]) == 0:
                continue
            var_names = var_names + group_genes[group]
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
            groupby=groupby,  # type: ignore[arg-type]
            swap_axes=True,
            ax=ax,  # type: ignore[arg-type]
            show=False,
            **plot_args,  # type: ignore[arg-type]
            **kwargs,
        )

        if return_fig:
            return fig  # type: ignore[return-value]
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
