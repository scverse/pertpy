from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as pl
from matplotlib.axes import Axes
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_bar,
    geom_density,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from scanpy import get
from scanpy._settings import settings
from scanpy._utils import _check_use_raw, sanitize_anndata
from scanpy.plotting import _utils

import pertpy as pt


class MixscapePlot:
    """Plotting functions for Mixscape."""

    @staticmethod
    def barplot(  # pragma: no cover
        adata: AnnData,
        control: str = "NT",
        mixscape_class_global="mixscape_class_global",
        axis_text_x_size: int = 8,
        axis_text_y_size: int = 6,
        axis_title_size: int = 8,
        strip_text_size: int = 6,
        panel_spacing_x: float = 0.3,
        panel_spacing_y: float = 0.3,
        legend_title_size: int = 8,
        legend_text_size: int = 8,
        show: bool | None = None,
        save: bool | str | None = None,
    ):
        """Barplot to visualize perturbation scores calculated from RunMixscape function.

        Args:
            adata: The annotated data object.
            control: Control category from the `pert_key` column. Default is 'NT'.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            show: Show the plot, do not return axis.
            save: If True or a str, save the figure. A string is appended to the default filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.

        Returns:
            If show is False, return ggplot object used to draw the plot.
        """
        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        count = pd.crosstab(index=adata.obs[mixscape_class_global], columns=adata.obs[control])
        all_cells_percentage = pd.melt(count / count.sum(), ignore_index=False).reset_index()
        KO_cells_percentage = all_cells_percentage[all_cells_percentage[mixscape_class_global] == "KO"]
        KO_cells_percentage = KO_cells_percentage.sort_values("value", ascending=False)

        new_levels = KO_cells_percentage[control]
        all_cells_percentage[control] = pd.Categorical(
            all_cells_percentage[control], categories=new_levels, ordered=False
        )
        all_cells_percentage[mixscape_class_global] = pd.Categorical(
            all_cells_percentage[mixscape_class_global], categories=["NT", "NP", "KO"], ordered=False
        )
        all_cells_percentage["gene"] = all_cells_percentage[control].str.rsplit("g", expand=True)[0]
        all_cells_percentage["guide_number"] = all_cells_percentage[control].str.rsplit("g", expand=True)[1]
        all_cells_percentage["guide_number"] = "g" + all_cells_percentage["guide_number"]
        NP_KO_cells = all_cells_percentage[all_cells_percentage["gene"] != "NT"]

        p1 = (
            ggplot(NP_KO_cells, aes(x="guide_number", y="value", fill="mixscape_class_global"))
            + scale_fill_manual(values=["#7d7d7d", "#c9c9c9", "#ff7256"])
            + geom_bar(stat="identity")
            + theme_classic()
            + xlab("sgRNA")
            + ylab("% of cells")
        )

        p1 = (
            p1
            + theme(
                axis_text_x=element_text(size=axis_text_x_size, hjust=2),
                axis_text_y=element_text(size=axis_text_y_size),
                axis_title=element_text(size=axis_title_size),
                strip_text=element_text(size=strip_text_size, face="bold"),
                panel_spacing_x=panel_spacing_x,
                panel_spacing_y=panel_spacing_y,
            )
            + facet_wrap("gene", ncol=5, scales="free")
            + labs(fill="mixscape class")
            + theme(legend_title=element_text(size=legend_title_size), legend_text=element_text(size=legend_text_size))
        )

        _utils.savefig_or_show("mixscape_barplot", show=show, save=save)
        if not show:
            return p1

    @staticmethod
    def heatmap(  # pragma: no cover
        adata: AnnData,
        labels: str,
        target_gene: str,
        layer: str | None = None,
        control: str | None = "NT",
        method: str | None = "wilcoxon",
        subsample_number: int | None = 900,
        vmin: float | None = -2,
        vmax: float | None = 2,
        show: bool | None = None,
        save: bool | str | None = None,
        **kwds,
    ):
        """Heatmap plot using mixscape results. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            target_gene: Target gene name to visualize heatmap for.
            layer: Key from `adata.layers` whose value will be used to perform tests on.
            control: Control category from the `pert_key` column. Default is 'NT'.
            method: The default method is 'wilcoxon', see `method` parameter in `scanpy.tl.rank_genes_groups` for more options.
            subsample_number: Subsample to this number of observations.
            vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
            vmax: The value representing the upper limit of the color scale. Values larger than vmax are plotted with the same color as vmax.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            ax: A matplotlib axes object. Only works if plotting a single component.
            **kwds: Additional arguments to `scanpy.pl.rank_genes_groups_heatmap`.
        """
        if "mixscape_class" not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        adata_subset = adata[(adata.obs[labels] == target_gene) | (adata.obs[labels] == control)].copy()
        sc.tl.rank_genes_groups(adata_subset, layer=layer, groupby=labels, method=method)
        sc.pp.scale(adata_subset, max_value=vmax)
        sc.pp.subsample(adata_subset, n_obs=subsample_number)
        return sc.pl.rank_genes_groups_heatmap(
            adata_subset,
            groupby="mixscape_class",
            vmin=vmin,
            vmax=vmax,
            n_genes=20,
            groups=["NT"],
            show=show,
            save=save,
            **kwds,
        )

    @staticmethod
    def perturbscore(  # pragma: no cover
        adata: AnnData,
        labels: str,
        target_gene: str,
        mixscape_class="mixscape_class",
        color="orange",
        split_by: str = None,
        before_mixscape=False,
        perturbation_type: str = "KO",
    ):
        """Density plots to visualize perturbation scores calculated by the `pt.tl.mixscape` function. Requires `pt.tl.mixscape` to be run first.

        https://satijalab.org/seurat/reference/plotperturbscore

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            target_gene: Target gene name to visualize perturbation scores for.
            mixscape_class: The column of `.obs` with mixscape classifications.
            color: Specify color of target gene class or knockout cell class. For control non-targeting and non-perturbed cells, colors are set to different shades of grey.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                the perturbation signature for every replicate separately.
            before_mixscape: Option to split densities based on mixscape classification (default) or original target gene classification. Default is set to NULL and plots cells by original class ID.
            perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.

        Returns:
            The ggplot object used for drawn.
        """
        if "mixscape" not in adata.uns:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        perturbation_score = None
        for key in adata.uns["mixscape"][target_gene].keys():
            perturbation_score_temp = adata.uns["mixscape"][target_gene][key]
            perturbation_score_temp["name"] = key
            if perturbation_score is None:
                perturbation_score = copy.deepcopy(perturbation_score_temp)
            else:
                perturbation_score = pd.concat([perturbation_score, perturbation_score_temp])
        perturbation_score["mix"] = adata.obs[mixscape_class][perturbation_score.index]
        gd = list(set(perturbation_score[labels]).difference({target_gene}))[0]
        # If before_mixscape is True, split densities based on original target gene classification
        if before_mixscape is True:
            cols = {gd: "#7d7d7d", target_gene: color}
            p = ggplot(perturbation_score, aes(x="pvec", color="gene_target")) + geom_density() + theme_classic()
            p_copy = copy.deepcopy(p)
            p_copy._build()
            top_r = max(p_copy.layers[0].data["density"])
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            perturbation_score.loc[perturbation_score["gene_target"] == gd, "y_jitter"] = np.random.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score["gene_target"] == gd)
            )
            perturbation_score.loc[perturbation_score["gene_target"] == target_gene, "y_jitter"] = np.random.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["gene_target"] == target_gene)
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                perturbation_score["split"] = adata.obs[split_by][perturbation_score.index]
                p2 = (
                    p
                    + scale_color_manual(values=cols, drop=False)
                    + geom_density(size=1.5)
                    + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                    + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                    + ylab("Cell density")
                    + xlab("Perturbation score")
                    + theme(
                        legend_key_size=1,
                        legend_text=element_text(colour="black", size=14),
                        legend_title=element_blank(),
                        plot_title=element_text(size=16, face="bold"),
                    )
                    + facet_wrap("split")
                )
            else:
                p2 = (
                    p
                    + scale_color_manual(values=cols, drop=False)
                    + geom_density(size=1.5)
                    + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                    + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                    + ylab("Cell density")
                    + xlab("Perturbation score")
                    + theme(
                        legend_key_size=1,
                        legend_text=element_text(colour="black", size=14),
                        legend_title=element_blank(),
                        plot_title=element_text(size=16, face="bold"),
                    )
                )
        # If before_mixscape is False, split densities based on mixscape classifications
        else:
            cols = {gd: "#7d7d7d", f"{target_gene} NP": "#c9c9c9", f"{target_gene} {perturbation_type}": color}
            p = ggplot(perturbation_score, aes(x="pvec", color="mix")) + geom_density() + theme_classic()
            p_copy = copy.deepcopy(p)
            p_copy._build()
            top_r = max(p_copy.layers[0].data["density"])
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            gd2 = list(
                set(perturbation_score["mix"]).difference([f"{target_gene} NP", f"{target_gene} {perturbation_type}"])
            )[0]
            perturbation_score.loc[perturbation_score["mix"] == gd2, "y_jitter"] = np.random.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score["mix"] == gd2)
            )
            perturbation_score.loc[
                perturbation_score["mix"] == f"{target_gene} {perturbation_type}", "y_jitter"
            ] = np.random.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} {perturbation_type}")
            )
            perturbation_score.loc[perturbation_score["mix"] == f"{target_gene} NP", "y_jitter"] = np.random.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} NP")
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                perturbation_score["split"] = adata.obs[split_by][perturbation_score.index]
                p2 = (
                    ggplot(perturbation_score, aes(x="pvec", color="mix"))
                    + scale_color_manual(values=cols, drop=False)
                    + geom_density(size=1.5)
                    + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                    + theme_classic()
                    + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                    + ylab("Cell density")
                    + xlab("Perturbation score")
                    + theme(
                        legend_key_size=1,
                        legend_text=element_text(colour="black", size=14),
                        legend_title=element_blank(),
                        plot_title=element_text(size=16, face="bold"),
                    )
                    + facet_wrap("split")
                )
            else:
                p2 = (
                    p
                    + scale_color_manual(values=cols, drop=False)
                    + geom_density(size=1.5)
                    + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                    + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                    + ylab("Cell density")
                    + xlab("Perturbation score")
                    + theme(
                        legend_key_size=1,
                        legend_text=element_text(colour="black", size=14),
                        legend_title=element_blank(),
                        plot_title=element_text(size=16, face="bold"),
                    )
                )
        return p2

    @staticmethod
    def violin(  # noqa: C901  pragma: no cover
        adata: AnnData,
        target_gene_idents: str | list[str],
        keys: str | Sequence[str] = "mixscape_class_p_ko",
        groupby: str | None = "mixscape_class",
        log: bool = False,
        use_raw: bool | None = None,
        stripplot: bool = True,
        hue: str | None = None,
        jitter: float | bool = True,
        size: int = 1,
        layer: str | None = None,
        scale: Literal["area", "count", "width"] = "width",
        order: Sequence[str] | None = None,
        multi_panel: bool | None = None,
        xlabel: str = "",
        ylabel: str | Sequence[str] | None = None,
        rotation: float | None = None,
        show: bool | None = None,
        save: bool | str | None = None,
        ax: Axes | None = None,
        **kwds,
    ):
        """Violin plot using mixscape results. Requires `pt.tl.mixscape` to be run first.

        Args:
            adata: The annotated data object.
            target_gene: Target gene name to plot.
            keys: Keys for accessing variables of `.var_names` or fields of `.obs`. Default is 'mixscape_class_p_ko'.
            groupby: The key of the observation grouping to consider. Default is 'mixscape_class'.
            log: Plot on logarithmic axis.
            use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
            stripplot: Add a stripplot on top of the violin plot.
            order: Order in which to show the categories.
            xlabel: Label of the x axis. Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
            ylabel: Label of the y axis. If `None` and `groupby` is `None`, defaults to `'value'`. If `None` and `groubpy` is not `None`, defaults to `keys`.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            ax: A matplotlib axes object. Only works if plotting a single component.
            **kwds: Additional arguments to `seaborn.violinplot`.

        Returns:
            A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.
        """
        if isinstance(target_gene_idents, str):
            mixscape_class_mask = adata.obs[groupby] == target_gene_idents
        elif isinstance(target_gene_idents, list):
            mixscape_class_mask = np.full_like(adata.obs[groupby], False, dtype=bool)
            for ident in target_gene_idents:
                mixscape_class_mask |= adata.obs[groupby] == ident
        adata = adata[mixscape_class_mask]

        import seaborn as sns  # Slow import, only import if called

        sanitize_anndata(adata)
        use_raw = _check_use_raw(adata, use_raw)
        if isinstance(keys, str):
            keys = [keys]
        keys = list(OrderedDict.fromkeys(keys))  # remove duplicates, preserving the order

        if isinstance(ylabel, (str, type(None))):
            ylabel = [ylabel] * (1 if groupby is None else len(keys))
        if groupby is None:
            if len(ylabel) != 1:
                raise ValueError(f"Expected number of y-labels to be `1`, found `{len(ylabel)}`.")
        elif len(ylabel) != len(keys):
            raise ValueError(f"Expected number of y-labels to be `{len(keys)}`, " f"found `{len(ylabel)}`.")

        if groupby is not None:
            if hue is not None:
                obs_df = get.obs_df(adata, keys=[groupby] + keys + [hue], layer=layer, use_raw=use_raw)
            else:
                obs_df = get.obs_df(adata, keys=[groupby] + keys, layer=layer, use_raw=use_raw)

        else:
            obs_df = get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw)
        if groupby is None:
            obs_tidy = pd.melt(obs_df, value_vars=keys)
            x = "variable"
            ys = ["value"]
        else:
            obs_tidy = obs_df
            x = groupby
            ys = keys

        if multi_panel and groupby is None and len(ys) == 1:
            # This is a quick and dirty way for adapting scales across several
            # keys if groupby is None.
            y = ys[0]

            g = sns.catplot(
                y=y,
                data=obs_tidy,
                kind="violin",
                scale=scale,
                col=x,
                col_order=keys,
                sharey=False,
                order=keys,
                cut=0,
                inner=None,
                **kwds,
            )

            if stripplot:
                grouped_df = obs_tidy.groupby(x)
                for ax_id, key in zip(range(g.axes.shape[1]), keys):
                    sns.stripplot(
                        y=y,
                        data=grouped_df.get_group(key),
                        jitter=jitter,
                        size=size,
                        color="black",
                        ax=g.axes[0, ax_id],
                    )
            if log:
                g.set(yscale="log")
            g.set_titles(col_template="{col_name}").set_xlabels("")
            if rotation is not None:
                for ax in g.axes[0]:
                    ax.tick_params(axis="x", labelrotation=rotation)
        else:
            # set by default the violin plot cut=0 to limit the extend
            # of the violin plot (see stacked_violin code) for more info.
            kwds.setdefault("cut", 0)
            kwds.setdefault("inner")

            if ax is None:
                axs, _, _, _ = _utils.setup_axes(
                    ax=ax,
                    panels=["x"] if groupby is None else keys,
                    show_ticks=True,
                    right_margin=0.3,
                )
            else:
                axs = [ax]
            for ax, y, ylab in zip(axs, ys, ylabel):  # noqa: F402
                ax = sns.violinplot(
                    x=x,
                    y=y,
                    data=obs_tidy,
                    order=order,
                    orient="vertical",
                    scale=scale,
                    ax=ax,
                    hue=hue,
                    **kwds,
                )
                # Get the handles and labels.
                handles, labels = ax.get_legend_handles_labels()
                if stripplot:
                    ax = sns.stripplot(
                        x=x,
                        y=y,
                        data=obs_tidy,
                        order=order,
                        jitter=jitter,
                        color="black",
                        size=size,
                        ax=ax,
                        hue=hue,
                        dodge=True,
                    )
                if xlabel == "" and groupby is not None and rotation is None:
                    xlabel = groupby.replace("_", " ")
                ax.set_xlabel(xlabel)
                if ylab is not None:
                    ax.set_ylabel(ylab)

                if log:
                    ax.set_yscale("log")
                if rotation is not None:
                    ax.tick_params(axis="x", labelrotation=rotation)

        show = settings.autoshow if show is None else show
        if hue is not None and stripplot is True:
            pl.legend(handles, labels)
        _utils.savefig_or_show("mixscape_violin", show=show, save=save)

        if not show:
            if multi_panel and groupby is None and len(ys) == 1:
                return g
            elif len(axs) == 1:
                return axs[0]
            else:
                return axs

    @staticmethod
    def lda(  # pragma: no cover
        adata: AnnData,
        labels: str,
        mixscape_class_global="mixscape_class_global",
        layer: str | None = None,
        control: str | None = "NT",
        n_comps: int | None = 10,
        min_de_genes: int | None = 5,
        logfc_threshold: float | None = 0.25,
        split_by: str | None = None,
        pval_cutoff: float | None = 5e-2,
        perturbation_type: str | None = "KO",
        show: bool | None = None,
        save: bool | str | None = None,
        **kwds,
    ):
        """Visualizing perturbation responses with Linear Discriminant Analysis. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            layer: Key from `adata.layers` whose value will be used to perform tests on.
            control: Control category from the `pert_key` column. Default is 'NT'.
            n_comps: Number of principal components to use. Defaults to 10.
            min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
            logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells (default: 0.25).
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
            pval_cutoff: P-value cut-off for selection of significantly DE genes.
            perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwds: Additional arguments to `scanpy.pl.umap`.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        if split_by is None:
            split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = split_obs.unique()
            split_masks = [split_obs == category for category in categories]

        mixscape_identifier = pt.tl.Mixscape()
        # determine gene sets across all splits/groups through differential gene expression
        perturbation_markers = mixscape_identifier._get_perturbation_markers(
            adata=adata,
            split_masks=split_masks,
            categories=categories,
            labels=labels,
            control=control,
            layer=layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
        )
        adata_subset = adata[
            (adata.obs[mixscape_class_global] == perturbation_type) | (adata.obs[mixscape_class_global] == control)
        ].copy()
        projected_pcs: dict[str, np.ndarray] = {}
        # performs PCA on each mixscape class separately and projects each subspace onto all cells in the data.
        for _, (key, value) in enumerate(perturbation_markers.items()):
            if len(value) == 0:
                continue
            else:
                gene_subset = adata_subset[
                    (adata_subset.obs[labels] == key[1]) | (adata_subset.obs[labels] == control)
                ].copy()
                sc.pp.scale(gene_subset)
                sc.tl.pca(gene_subset, n_comps=n_comps)
                sc.pp.neighbors(gene_subset)
                # projects each subspace onto all cells in the data.
                sc.tl.ingest(adata=adata_subset, adata_ref=gene_subset, embedding_method="pca")
                projected_pcs[key[1]] = adata_subset.obsm["X_pca"]
        # concatenate all pcs into a single matrix.
        for index, (_, value) in enumerate(projected_pcs.items()):
            if index == 0:
                projected_pcs_array = value
            else:
                projected_pcs_array = np.concatenate((projected_pcs_array, value), axis=1)

        clf = LinearDiscriminantAnalysis(n_components=len(projected_pcs))
        clf.fit(projected_pcs_array, adata_subset.obs["gene_target"])
        cell_embeddings = clf.transform(projected_pcs_array)
        adata_subset.obsm["cell_embeddings"] = cell_embeddings
        sc.pp.neighbors(adata_subset, use_rep="cell_embeddings")
        sc.tl.umap(adata_subset, n_components=11)
        sc.pl.umap(adata_subset, color="mixscape_class", show=show, save=save, **kwds)
