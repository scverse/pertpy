import warnings
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from mudata import MuData

from pertpy.tools._coda._base_coda import CompositionalModel2

sns.set_style("ticks")


class CodaPlot:
    @staticmethod
    def __stackbar(  # pragma: no cover
        y: np.ndarray,
        type_names: list[str],
        title: str,
        level_names: list[str],
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        cmap: ListedColormap | None = cm.tab20,
        show_legend: bool | None = True,
    ) -> plt.Axes:
        """Plots a stacked barplot for one (discrete) covariate.

        Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

        Args:
            y: The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows,
               one for each group, containing the count mean of each cell type
            type_names: The names of all cell types
            title: Plot title, usually the covariate's name
            level_names: Names of the covariate's levels
            figsize: Figure size. Defaults to None.
            dpi: Dpi setting. Defaults to 100.
            cmap: The color map for the barplot. Defaults to cm.tab20.
            show_legend: If True, adds a legend. Defaults to True.

        Returns:
            A :class:`~matplotlib.axes.Axes` object
        """
        n_bars, n_types = y.shape

        figsize = rcParams["figure.figsize"] if figsize is None else figsize

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        r = np.array(range(n_bars))
        sample_sums = np.sum(y, axis=1)

        barwidth = 0.85
        cum_bars = np.zeros(n_bars)

        for n in range(n_types):
            bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_bars)], sample_sums, strict=False)]
            plt.bar(
                r,
                bars,
                bottom=cum_bars,
                color=cmap(n % cmap.N),
                width=barwidth,
                label=type_names[n],
                linewidth=0,
            )
            cum_bars += bars

        ax.set_title(title)
        if show_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
        ax.set_xticks(r)
        ax.set_xticklabels(level_names, rotation=45, ha="right")
        ax.set_ylabel("Proportion")

        return ax

    @staticmethod
    def stacked_barplot(  # pragma: no cover
        data: AnnData | MuData,
        feature_name: str,
        modality_key: str = "coda",
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        cmap: ListedColormap | None = cm.tab20,
        show_legend: bool | None = True,
        level_order: list[str] = None,
    ) -> plt.Axes:
        """Plots a stacked barplot for all levels of a covariate or all samples (if feature_name=="samples").

        Args:
            data: AnnData object or MuData object.
            feature_name: The name of the covariate to plot. If feature_name=="samples", one bar for every sample will be plotted
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            figsize: Figure size. Defaults to None.
            dpi: Dpi setting. Defaults to 100.
            cmap: The matplotlib color map for the barplot. Defaults to cm.tab20.
            show_legend: If True, adds a legend. Defaults to True.
            level_order: Custom ordering of bars on the x-axis. Defaults to None.

        Returns:
            A :class:`~matplotlib.axes.Axes` object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_stacked_barplot(mdata, feature_name="samples")
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Sccoda

        coda = Sccoda()
        return coda.plot_stacked_barplot(
            data=data,
            feature_name=feature_name,
            modality_key=modality_key,
            figsize=figsize,
            dpi=dpi,
            palette=cmap,
            show_legend=show_legend,
            level_order=level_order,
        )

    @staticmethod
    def effects_barplot(  # pragma: no cover
        data: AnnData | MuData,
        modality_key: str = "coda",
        covariates: str | list | None = None,
        parameter: Literal["log2-fold change", "Final Parameter", "Expected Sample"] = "log2-fold change",
        plot_facets: bool = True,
        plot_zero_covariate: bool = True,
        plot_zero_cell_type: bool = False,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        cmap: str | ListedColormap | None = cm.tab20,
        level_order: list[str] = None,
        args_barplot: dict | None = None,
    ) -> plt.Axes | sns.axisgrid.FacetGrid | None:
        """Barplot visualization for effects.

        The effect results for each covariate are shown as a group of barplots, with intra--group separation by cell types.
        The covariates groups can either be ordered along the x-axis of a single plot (plot_facets=False) or as plot facets (plot_facets=True).

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            covariates: The name of the covariates in data.obs to plot. Defaults to None.
            parameter: The parameter in effect summary to plot. Defaults to "log2-fold change".
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets.
                         Defaults to True.
            plot_zero_covariate: If True, plot covariate that have all zero effects. If False, do not plot.
                                 Defaults to True.
            plot_zero_cell_type: If True, plot cell type that have zero effect. If False, do not plot.
                                 Defaults to False.
            figsize: Figure size. Defaults to None.
            dpi: Figure size. Defaults to 100.
            cmap: The seaborn color map for the barplot. Defaults to cm.tab20.
            level_order: Custom ordering of bars on the x-axis. Defaults to None.
            args_barplot: Arguments passed to sns.barplot. Defaults to None.

        Returns:
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`)
            or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> sccoda.plot_effects_barplot(mdata)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Sccoda

        coda = Sccoda()
        return coda.plot_effects_barplot(
            data=data,
            modality_key=modality_key,
            covariates=covariates,
            parameter=parameter,
            plot_facets=plot_facets,
            plot_zero_covariate=plot_zero_covariate,
            plot_zero_cell_type=plot_zero_cell_type,
            figsize=figsize,
            dpi=dpi,
            palette=cmap,
            level_order=level_order,
            args_barplot=args_barplot,
        )

    @staticmethod
    def boxplots(  # pragma: no cover
        data: AnnData | MuData,
        feature_name: str,
        modality_key: str = "coda",
        y_scale: Literal["relative", "log", "log10", "count"] = "relative",
        plot_facets: bool = False,
        add_dots: bool = False,
        cell_types: list | None = None,
        args_boxplot: dict | None = None,
        args_swarmplot: dict | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        cmap: str | None = "Blues",
        show_legend: bool | None = True,
        level_order: list[str] = None,
    ) -> plt.Axes | sns.axisgrid.FacetGrid | None:
        """Grouped boxplot visualization. The cell counts for each cell type are shown as a group of boxplots,
            with intra--group separation by a covariate from data.obs.

        Args:
            data: AnnData object or MuData object
            feature_name: The name of the feature in data.obs to plot
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            y_scale: Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count),
                     "log10" - log10(count), "count" - absolute abundance (cell counts).
                     Defaults to "relative".
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets. Defaults to False.
            add_dots: If True, overlay a scatterplot with one dot for each data point. Defaults to False.
            model: When draw_effects, specify a tasCODA model
            cell_types: Subset of cell types that should be plotted. Defaults to None.
            args_boxplot: Arguments passed to sns.boxplot. Defaults to {}.
            args_swarmplot: Arguments passed to sns.swarmplot. Defaults to {}.
            figsize: Figure size. Defaults to None.
            dpi: Dpi setting. Defaults to 100.
            cmap: The seaborn color map for the barplot. Defaults to "Blues".
            show_legend: If True, adds a legend. Defaults to True.
            level_order: Custom ordering of bars on the x-axis. Defaults to None.

        Returns:
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`)
            or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_boxplots(mdata, feature_name="condition", add_dots=True)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Sccoda

        coda = Sccoda()
        return coda.plot_boxplots(
            data=data,
            feature_name=feature_name,
            modality_key=modality_key,
            y_scale=y_scale,
            plot_facets=plot_facets,
            add_dots=add_dots,
            cell_types=cell_types,
            args_boxplot=args_boxplot,
            args_swarmplot=args_swarmplot,
            figsize=figsize,
            dpi=dpi,
            palette=cmap,
            show_legend=show_legend,
            level_order=level_order,
        )

    @staticmethod
    def rel_abundance_dispersion_plot(  # pragma: no cover
        data: AnnData | MuData,
        modality_key: str = "coda",
        abundant_threshold: float | None = 0.9,
        default_color: str | None = "Grey",
        abundant_color: str | None = "Red",
        label_cell_types: bool = True,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        ax: Axes = None,
    ) -> plt.Axes:
        """Plots total variance of relative abundance versus minimum relative abundance of all cell types for determination of a reference cell type.

        If the count of the cell type is larger than 0 in more than abundant_threshold percent of all samples, the cell type will be marked in a different color.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
                          Defaults to "coda".
            abundant_threshold: Presence threshold for abundant cell types. Defaults to 0.9.
            default_color: Bar color for all non-minimal cell types. Defaults to "Grey".
            abundant_color: Bar color for cell types with abundant percentage larger than abundant_threshold.
                            Defaults to "Red".
            label_cell_types: Label dots with cell type names. Defaults to True.
            figsize: Figure size. Defaults to None.
            dpi: Dpi setting. Defaults to 100.
            ax: A matplotlib axes object. Only works if plotting a single component. Defaults to None.

        Returns:
            A :class:`~matplotlib.axes.Axes` object

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> sccoda.plot_rel_abundance_dispersion_plot(mdata)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Sccoda

        coda = Sccoda()
        return coda.plot_rel_abundance_dispersion_plot(
            data=data,
            modality_key=modality_key,
            abundant_threshold=abundant_threshold,
            default_color=default_color,
            abundant_color=abundant_color,
            label_cell_types=label_cell_types,
            figsize=figsize,
            dpi=dpi,
            ax=ax,
        )

    @staticmethod
    def draw_tree(  # pragma: no cover
        data: AnnData | MuData,
        modality_key: str = "coda",
        tree: str = "tree",  # Also type ete3.Tree. Omitted due to import errors
        tight_text: bool | None = False,
        show_scale: bool | None = False,
        show: bool | None = True,
        save: str | None = None,
        units: Literal["px", "mm", "in"] | None = "px",
        figsize: tuple[float, float] | None = (None, None),
        dpi: int | None = 90,
    ):
        """Plot a tree using input ete3 tree object.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
                          Defaults to "coda".
            tree: A ete3 tree object or a str to indicate the tree stored in `.uns`.
                  Defaults to "tree".
            tight_text: When False, boundaries of the text are approximated according to general font metrics,
                        producing slightly worse aligned text faces but improving
                        the performance of tree visualization in scenes with a lot of text faces.
                        Default to False.
            show_scale: Include the scale legend in the tree image or not.
                        Defaults to False.
            show: If True, plot the tree inline. If false, return tree and tree_style objects.
                  Defaults to True.
            file_name: Path to the output image file. Valid extensions are .SVG, .PDF, .PNG.
                       Output image can be saved whether show is True or not.
                       Defaults to None.
            units: Unit of image sizes. “px”: pixels, “mm”: millimeters, “in”: inches.
                   Defaults to "px".
            h: Height of the image in units. Defaults to None.
            w: Width of the image in units. Defaults to None.
            dpi: Dots per inches. Defaults to 90.

        Returns:
            Depending on `show`, returns :class:`ete3.TreeNode` and :class:`ete3.TreeStyle` (`show = False`) or plot the tree inline (`show = False`)

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
            >>> mdata = tasccoda.prepare(
            >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
            >>> )
            >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
            >>> tasccoda.plot_draw_tree(mdata, tree="lineage")

        Preview: #TODO: Add preview
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Tasccoda

        coda = Tasccoda()
        return coda.plot_draw_tree(
            data=data,
            modality_key=modality_key,
            tree=tree,
            tight_text=tight_text,
            show_scale=show_scale,
            show=show,
            save=save,
            units=units,
            figsize=figsize,
            dpi=dpi,
        )

    @staticmethod
    def draw_effects(  # pragma: no cover
        data: AnnData | MuData,
        covariate: str,
        modality_key: str = "coda",
        tree: str = "tree",  # Also type ete3.Tree. Omitted due to import errors
        show_legend: bool | None = None,
        show_leaf_effects: bool | None = False,
        tight_text: bool | None = False,
        show_scale: bool | None = False,
        show: bool | None = True,
        save: str | None = None,
        units: Literal["px", "mm", "in"] | None = "in",
        figsize: tuple[float, float] | None = (None, None),
        dpi: int | None = 90,
    ):
        """Plot a tree with colored circles on the nodes indicating significant effects with bar plots which indicate leave-level significant effects.

        Args:
            data: AnnData object or MuData object.
            covariate: The covariate, whose effects should be plotted.
            modality_key: If data is a MuData object, specify which modality to use.
                          Defaults to "coda".
            tree: A ete3 tree object or a str to indicate the tree stored in `.uns`.
                  Defaults to "tree".
            show_legend: If show legend of nodes significant effects or not.
                         Defaults to False if show_leaf_effects is True.
            show_leaf_effects: If True, plot bar plots which indicate leave-level significant effects.
                               Defaults to False.
            tight_text: When False, boundaries of the text are approximated according to general font metrics,
                        producing slightly worse aligned text faces but improving the performance of tree visualization in scenes with a lot of text faces.
                        Defaults to False.
            show_scale: Include the scale legend in the tree image or not. Defaults to False.
            show: If True, plot the tree inline. If false, return tree and tree_style objects. Defaults to True.
            file_name: Path to the output image file. valid extensions are .SVG, .PDF, .PNG. Output image can be saved whether show is True or not.
                       Defaults to None.
            units: Unit of image sizes. “px”: pixels, “mm”: millimeters, “in”: inches. Default is "in". Defaults to "in".
            h: Height of the image in units. Defaults to None.
            w: Width of the image in units. Defaults to None.
            dpi: Dots per inches. Defaults to 90.

        Returns:
            Depending on `show`, returns :class:`ete3.TreeNode` and :class:`ete3.TreeStyle` (`show = False`)
            or  plot the tree inline (`show = False`)

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.tasccoda_example()
            >>> tasccoda = pt.tl.Tasccoda()
            >>> mdata = tasccoda.load(
            >>>     adata, type="sample_level",
            >>>     levels_agg=["Major_l1", "Major_l2", "Major_l3", "Major_l4", "Cluster"],
            >>>     key_added="lineage", add_level_name=True
            >>> )
            >>> mdata = tasccoda.prepare(
            >>>     mdata, formula="Health", reference_cell_type="automatic", tree_key="lineage", pen_args={"phi": 0}
            >>> )
            >>> tasccoda.run_nuts(mdata, num_samples=1000, num_warmup=100, rng_key=42)
            >>> tasccoda.plot_draw_effects(mdata, covariate="Health[T.Inflamed]", tree="lineage")
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Tasccoda

        coda = Tasccoda()
        return coda.plot_draw_effects(
            data=data,
            modality_key=modality_key,
            covariate=covariate,
            tree=tree,
            show_legend=show_legend,
            show_leaf_effects=show_leaf_effects,
            tight_text=tight_text,
            show_scale=show_scale,
            show=show,
            save=save,
            units=units,
            figsize=figsize,
            dpi=dpi,
        )

    @staticmethod
    def effects_umap(  # pragma: no cover
        data: MuData,
        effect_name: str | list | None,
        cluster_key: str,
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
        show: bool = None,
        ax: Axes = None,
        **kwargs,
    ):
        """Plot a UMAP visualization colored by effect strength.

        Effect results in .varm of aggregated sample-level AnnData (default is data['coda']) are assigned to cell-level AnnData
        (default is data['rna']) depending on the cluster they were assigned to.

        Args:
            data: AnnData object or MuData object.
            effect_name: The name of the effect results in .varm of aggregated sample-level AnnData to plot
            cluster_key: The cluster information in .obs of cell-level AnnData (default is data['rna']).
                         To assign cell types' effects to original cells.
            modality_key_1: Key to the cell-level AnnData in the MuData object. Defaults to "rna".
            modality_key_2: Key to the aggregated sample-level AnnData object in the MuData object.
                            Defaults to "coda".
            show: Whether to display the figure or return axis. Defaults to None.
            ax: A matplotlib axes object. Only works if plotting a single component.
                Defaults to None.
            **kwargs: All other keyword arguments are passed to `scanpy.plot.umap()`

        Returns:
            If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

        Examples:
            >>> import pertpy as pt
            >>> import schist
            >>> adata = pt.dt.haber_2017_regions()
            >>> schist.inference.nested_model(adata, samples=100, random_seed=5678)
            >>> tasccoda_model = pt.tl.Tasccoda()
            >>> tasccoda_data = tasccoda_model.load(adata, type="cell_level",
            >>>                 cell_type_identifier="nsbm_level_1",
            >>>                 sample_identifier="batch", covariate_obs=["condition"],
            >>>                 levels_orig=["nsbm_level_4", "nsbm_level_3", "nsbm_level_2", "nsbm_level_1"],
            >>>                 add_level_name=True)sccoda = pt.tl.Sccoda()
            >>> tasccoda_model.prepare(
            >>>     tasccoda_data,
            >>>     modality_key="coda",
            >>>     reference_cell_type="18",
            >>>     formula="condition",
            >>>     pen_args={"phi": 0, "lambda_1": 3.5},
            >>>     tree_key="tree"
            >>> )
            >>> tasccoda_model.run_nuts(
            ...     tasccoda_data, modality_key="coda", rng_key=1234, num_samples=10000, num_warmup=1000
            ... )
            >>> tasccoda_model.plot_effects_umap(tasccoda_data,
            >>>                         effect_name=["effect_df_condition[T.Salmonella]",
            >>>                                      "effect_df_condition[T.Hpoly.Day3]",
            >>>                                      "effect_df_condition[T.Hpoly.Day10]"],
            >>>                                       cluster_key="nsbm_level_1",
            >>>                         )
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object for plotting function directly.",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Tasccoda

        coda = Tasccoda()
        coda.plot_effects_umap(
            data=data,
            effect_name=effect_name,
            cluster_key=cluster_key,
            modality_key_1=modality_key_1,
            modality_key_2=modality_key_2,
            show=show,
            ax=ax,
            **kwargs,
        )
