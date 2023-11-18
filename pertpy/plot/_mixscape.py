from __future__ import annotations

import copy
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as pl
from scanpy import get
from scanpy._settings import settings
from scanpy._utils import _check_use_raw, sanitize_anndata
from scanpy.plotting import _utils

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from matplotlib.axes import Axes


class MixscapePlot:
    """Plotting functions for Mixscape."""

    @staticmethod
    def barplot(  # pragma: no cover
        adata: AnnData,
        guide_rna_column: str,
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
            guide_rna_column: The column of `.obs` with guide RNA labels. The target gene labels.
                              The format must be <gene_target>g<#>. For example, 'STAT2g1' and 'ATF2g1'.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            show: Show the plot, do not return axis.
            save: If True or a str, save the figure. A string is appended to the default filename.
                  Infer the filetype if ending on {'.pdf', '.png', '.svg'}.

        Returns:
            If show is False, return ggplot object used to draw the plot.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> mixscape_identifier = pt.tl.Mixscape()
            >>> mixscape_identifier.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
            >>> mixscape_identifier.mixscape(adata = mdata['rna'], control = 'NT', labels='gene_target', layer='X_pert')
            >>> pt.pl.ms.barplot(mdata['rna'], guide_rna_column='NT')
        """
        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        count = pd.crosstab(index=adata.obs[mixscape_class_global], columns=adata.obs[guide_rna_column])
        all_cells_percentage = pd.melt(count / count.sum(), ignore_index=False).reset_index()
        KO_cells_percentage = all_cells_percentage[all_cells_percentage[mixscape_class_global] == "KO"]
        KO_cells_percentage = KO_cells_percentage.sort_values("value", ascending=False)

        new_levels = KO_cells_percentage[guide_rna_column]
        all_cells_percentage[guide_rna_column] = pd.Categorical(
            all_cells_percentage[guide_rna_column], categories=new_levels, ordered=False
        )
        all_cells_percentage[mixscape_class_global] = pd.Categorical(
            all_cells_percentage[mixscape_class_global], categories=["NT", "NP", "KO"], ordered=False
        )
        all_cells_percentage["gene"] = all_cells_percentage[guide_rna_column].str.rsplit("g", expand=True)[0]
        all_cells_percentage["guide_number"] = all_cells_percentage[guide_rna_column].str.rsplit("g", expand=True)[1]
        all_cells_percentage["guide_number"] = "g" + all_cells_percentage["guide_number"]
        NP_KO_cells = all_cells_percentage[all_cells_percentage["gene"] != "NT"]
        _utils.savefig_or_show("mixscape_barplot", show=show, save=save)
        if not show:
            color_mapping = {"KO": "gray", "NP": "lightgray", "NT": "salmon"}
            unique_genes = NP_KO_cells["gene"].unique()
            fig, axs = pl.subplots(int(len(unique_genes)/5),5, figsize=(25, 25), sharey=True)
            for i, gene in enumerate(unique_genes):
                ax = axs[int(i/5),i%5]
                grouped_df = NP_KO_cells[NP_KO_cells["gene"]==gene].groupby(["guide_number", "mixscape_class_global"])["value"].sum().unstack()
                grouped_df.plot(kind="bar", stacked=True, color=[color_mapping[col] for col in grouped_df.columns], ax=ax, width=0.8,legend=False)
                ax.set_title(gene, bbox=dict(facecolor='white', edgecolor='black', pad=1),fontsize=axis_title_size)
                ax.set(xlabel="sgRNA", ylabel="% of cells")
                sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='right',fontsize=axis_text_x_size) 
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=axis_text_y_size)
            fig.subplots_adjust(right=0.8)
            #Increase the space between subplots
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            ax.legend(title="mixscape_class_global",loc='center right', bbox_to_anchor=(2.2, 3.5),frameon=True, fontsize=legend_text_size, title_fontsize=legend_title_size)
            return fig

    @staticmethod
    def heatmap(  # pragma: no cover
        adata: AnnData,
        labels: str,
        target_gene: str,
        control: str,
        layer: str | None = None,
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
            control: Control category from the `pert_key` column.
            layer: Key from `adata.layers` whose value will be used to perform tests on.
            method: The default method is 'wilcoxon', see `method` parameter in `scanpy.tl.rank_genes_groups` for more options.
            subsample_number: Subsample to this number of observations.
            vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
            vmax: The value representing the upper limit of the color scale. Values larger than vmax are plotted with the same color as vmax.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            ax: A matplotlib axes object. Only works if plotting a single component.
            **kwds: Additional arguments to `scanpy.pl.rank_genes_groups_heatmap`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> mixscape_identifier = pt.tl.Mixscape()
            >>> mixscape_identifier.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
            >>> mixscape_identifier.mixscape(adata = mdata['rna'], control = 'NT', labels='gene_target', layer='X_pert')
            >>> pt.pl.ms.heatmap(adata = mdata['rna'], labels='gene_target', target_gene='IFNGR2', layer='X_pert', control='NT')
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

        Examples:
            Visualizing the perturbation scores for the cells in a dataset:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> mixscape_identifier = pt.tl.Mixscape()
            >>> mixscape_identifier.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
            >>> mixscape_identifier.mixscape(adata = mdata['rna'], control = 'NT', labels='gene_target', layer='X_pert')
            >>> pt.pl.ms.perturbscore(adata = mdata['rna'], labels='gene_target', target_gene='IFNGR2', color = 'orange')
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
            p = sns.kdeplot(data=perturbation_score, x='pvec', hue=labels, fill=False, common_norm=False)
            top_r = max(p.get_lines()[i].get_data()[1].max() for i in range(len(p.get_lines())))
            pl.close()
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            rng = np.random.default_rng()
            perturbation_score.loc[perturbation_score[labels] == gd, "y_jitter"] = rng.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score[labels] == gd)
            )
            perturbation_score.loc[perturbation_score[labels] == target_gene, "y_jitter"] = rng.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score[labels] == target_gene)
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                sns.set(style="whitegrid")
                sns.FacetGrid(data=perturbation_score, col=split_by, hue=split_by, palette=cols, height=5, sharey=False)
                g.map(sns.kdeplot, 'pvec', fill=True, common_norm=False)
                g.map(sns.scatterplot, 'pvec', 'y_jitter', s=10, alpha=0.5)
                g.set_axis_labels("Perturbation score", "Cell density")
                g.add_legend(title=split_by, fontsize=14, title_fontsize=16)
                g.despine(left=True)

            # If split_by is not provided, create a single plot
            else:
                sns.set(style="whitegrid")
                sns.kdeplot(data=perturbation_score, x='pvec', hue='gene_target', fill=True, common_norm=False, palette=cols)
                sns.scatterplot(data=perturbation_score, x='pvec', y='y_jitter', hue='gene_target', palette=cols, s=10, alpha=0.5)
                pl.xlabel('Perturbation score', fontsize=16)
                pl.ylabel('Cell density', fontsize=16)
                pl.title('Density Plot using Seaborn and Matplotlib', fontsize=18)
                pl.legend(title='gene_target', title_fontsize=14, fontsize=12)
                sns.despine()
              
                
        # If before_mixscape is False, split densities based on mixscape classifications
        else:
            cols = {gd: "#7d7d7d", f"{target_gene} NP": "#c9c9c9", f"{target_gene} {perturbation_type}": color}
            p = sns.kdeplot(data=perturbation_score, x='pvec', hue=labels, fill=False, common_norm=False)
            top_r = max(p.get_lines()[i].get_data()[1].max() for i in range(len(p.get_lines())))
            pl.close()
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            rng = np.random.default_rng()
            gd2 = list(
                set(perturbation_score["mix"]).difference([f"{target_gene} NP", f"{target_gene} {perturbation_type}"])
            )[0]
            perturbation_score.loc[perturbation_score["mix"] == gd2, "y_jitter"] = rng.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score["mix"] == gd2)
            )
            perturbation_score.loc[
                perturbation_score["mix"] == f"{target_gene} {perturbation_type}", "y_jitter"
            ] = rng.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} {perturbation_type}")
            )
            perturbation_score.loc[perturbation_score["mix"] == f"{target_gene} NP", "y_jitter"] = rng.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} NP")
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                sns.set(style="whitegrid")
                g = sns.FacetGrid(data=perturbation_score, col=split_by, hue='mix', palette=cols, height=5, sharey=False)
                g.map(sns.kdeplot, 'pvec', fill=True, common_norm=False, alpha=0.7)
                g.map(sns.scatterplot, 'pvec', 'y_jitter', s=10, alpha=0.5)
                g.set_axis_labels("Perturbation score", "Cell density")
                g.add_legend(title='mix', fontsize=14, title_fontsize=16)
                g.despine(left=True)
                

            # If split_by is not provided, create a single plot
            else:
                sns.set(style="whitegrid")
                sns.kdeplot(data=perturbation_score, x='pvec', hue='mix', fill=True, common_norm=False, palette=cols, alpha=0.7)
                sns.scatterplot(data=perturbation_score, x='pvec', y='y_jitter', hue='mix', palette=cols, s=10, alpha=0.5)
                pl.xlabel('Perturbation score', fontsize=16)
                pl.ylabel('Cell density', fontsize=16)
                pl.title('Density Plot using Seaborn and Matplotlib', fontsize=18)
                pl.legend(title='mix', title_fontsize=14, fontsize=12)
                sns.despine()
                
        return pl.gcf()

    @staticmethod
    def violin(  # pragma: no cover
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

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> mixscape_identifier = pt.tl.Mixscape()
            >>> mixscape_identifier.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
            >>> mixscape_identifier.mixscape(adata = mdata['rna'], control = 'NT', labels='gene_target', layer='X_pert')
            >>> pt.pl.ms.violin(adata = mdata['rna'], target_gene_idents=['NT', 'IFNGR2 NP', 'IFNGR2 KO'], groupby='mixscape_class')
        """
        if isinstance(target_gene_idents, str):
            mixscape_class_mask = adata.obs[groupby] == target_gene_idents
        elif isinstance(target_gene_idents, list):
            mixscape_class_mask = np.full_like(adata.obs[groupby], False, dtype=bool)
            for ident in target_gene_idents:
                mixscape_class_mask |= adata.obs[groupby] == ident
        adata = adata[mixscape_class_mask]



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
        control: str,
        mixscape_class="mixscape_class",
        mixscape_class_global="mixscape_class_global",
        perturbation_type: str | None = "KO",
        lda_key: str | None = "mixscape_lda",
        n_components: int | None = None,
        show: bool | None = None,
        save: bool | str | None = None,
        **kwds,
    ):
        """Visualizing perturbation responses with Linear Discriminant Analysis. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            control: Control category from the `pert_key` column.
            labels: The column of `.obs` with target gene labels.
            mixscape_class: The column of `.obs` with the mixscape classification result.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Defaults to 'KO'.
            lda_key: If not speficied, lda looks .uns["mixscape_lda"] for the LDA results.
            n_components: The number of dimensions of the embedding.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwds: Additional arguments to `scanpy.pl.umap`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> mixscape_identifier = pt.tl.Mixscape()
            >>> mixscape_identifier.perturbation_signature(mdata['rna'], 'perturbation', 'NT', 'replicate')
            >>> mixscape_identifier.mixscape(adata = mdata['rna'], control = 'NT', labels='gene_target', layer='X_pert')
            >>> mixscape_identifier.lda(adata=mdata['rna'], control='NT', labels='gene_target', layer='X_pert')
            >>> pt.pl.ms.lda(adata=mdata['rna'], control='NT')
        """
        if mixscape_class not in adata.obs:
            raise ValueError(f'Did not find .obs["{mixscape_class!r}"]. Please run `pt.tl.mixscape` first.')
        if lda_key not in adata.uns:
            raise ValueError(f'Did not find .uns["{lda_key!r}"]. Run `pt.tl.neighbors` first.')

        adata_subset = adata[
            (adata.obs[mixscape_class_global] == perturbation_type) | (adata.obs[mixscape_class_global] == control)
        ].copy()
        adata_subset.obsm[lda_key] = adata_subset.uns[lda_key]
        if n_components is None:
            n_components = adata_subset.uns[lda_key].shape[1]
        sc.pp.neighbors(adata_subset, use_rep=lda_key)
        sc.tl.umap(adata_subset, n_components=n_components)
        sc.pl.umap(adata_subset, color=mixscape_class, show=show, save=save, **kwds)
