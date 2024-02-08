from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

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
        legend_title_size: int = 18,
        legend_text_size: int = 18,
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
            >>> ms = pt.tl.Mixscape()
            >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", "replicate")
            >>> ms.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> ms.plot_barplot(mdata["rna"], guide_rna_column="NT")
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Mixscape

        ms = Mixscape()
        return ms.plot_barplot(
            adata=adata,
            guide_rna_column=guide_rna_column,
            mixscape_class_global=mixscape_class_global,
            axis_text_x_size=axis_text_x_size,
            axis_text_y_size=axis_text_y_size,
            axis_title_size=axis_title_size,
            legend_title_size=legend_title_size,
            legend_text_size=legend_text_size,
            show=show,
            save=save,
        )

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
        **kwargs,
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
            **kwds: Additional arguments to `scanpy.pl.rank_genes_groups_heatmap`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms = pt.tl.Mixscape()
            >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", "replicate")
            >>> ms.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> ms.plot_heatmap(
            ...     adata=mdata["rna"], labels="gene_target", target_gene="IFNGR2", layer="X_pert", control="NT"
            ... )
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Mixscape

        ms = Mixscape()
        return ms.plot_heatmap(
            adata=adata,
            labels=labels,
            target_gene=target_gene,
            control=control,
            layer=layer,
            method=method,
            subsample_number=subsample_number,
            vmin=vmin,
            vmax=vmax,
            show=show,
            save=save,
            **kwargs,
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
            >>> mixscape_identifier.perturbation_signature(mdata["rna"], "perturbation", "NT", "replicate")
            >>> mixscape_identifier.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> mixscape_identifier.perturbscore(
            ...     adata=mdata["rna"], labels="gene_target", target_gene="IFNGR2", color="orange"
            ... )
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Mixscape

        ms = Mixscape()
        return ms.plot_perturbscore(
            adata=adata,
            labels=labels,
            target_gene=target_gene,
            mixscape_class=mixscape_class,
            color=color,
            split_by=split_by,
            before_mixscape=before_mixscape,
            perturbation_type=perturbation_type,
        )

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
        **kwargs,
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
            **kwargs: Additional arguments to `seaborn.violinplot`.

        Returns:
            A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms = pt.tl.Mixscape()
            >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", "replicate")
            >>> ms.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> ms.plot_violin(
            ...     adata=mdata["rna"], target_gene_idents=["NT", "IFNGR2 NP", "IFNGR2 KO"], groupby="mixscape_class"
            ... )
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Mixscape

        ms = Mixscape()
        return ms.plot_violin(
            adata=adata,
            target_gene_idents=target_gene_idents,
            keys=keys,
            groupby=groupby,
            log=log,
            use_raw=use_raw,
            stripplot=stripplot,
            hue=hue,
            jitter=jitter,
            size=size,
            layer=layer,
            scale=scale,
            order=order,
            multi_panel=multi_panel,
            xlabel=xlabel,
            ylabel=ylabel,
            rotation=rotation,
            show=show,
            save=save,
            ax=ax,
            **kwargs,
        )

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
        **kwargs,
    ):
        """Visualizing perturbation responses with Linear Discriminant Analysis. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            control: Control category from the `pert_key` column.
            labels: The column of `.obs` with target gene labels.
            mixscape_class: The column of `.obs` with the mixscape classification result.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            perturbation_type: Specify type of CRISPR perturbation expected for labeling mixscape classifications.
                               Defaults to 'KO'.
            lda_key: If not speficied, lda looks .uns["mixscape_lda"] for the LDA results.
            n_components: The number of dimensions of the embedding.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename.
                  Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwargs: Additional arguments to `scanpy.pl.umap`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms = pt.tl.Mixscape()
            >>> ms.perturbation_signature(mdata["rna"], "perturbation", "NT", "replicate")
            >>> ms.mixscape(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> ms.lda(adata=mdata["rna"], control="NT", labels="gene_target", layer="X_pert")
            >>> ms.plot_lda(adata=mdata["rna"], control="NT")
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Mixscape

        ms = Mixscape()

        return ms.plot_lda(
            adata=adata,
            control=control,
            mixscape_class=mixscape_class,
            mixscape_class_global=mixscape_class_global,
            perturbation_type=perturbation_type,
            lda_key=lda_key,
            n_components=n_components,
            show=show,
            save=save,
            **kwargs,
        )
