import os
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from ete3 import CircleFace, NodeStyle, TextFace, Tree, TreeStyle, faces
from matplotlib import cm, rcParams
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from mudata import MuData
from statannotations.Annotator import Annotator

from pertpy.tools._base_coda import CompositionalModel2, collapse_singularities_2

sns.set_style("ticks")


class CodaPlot:
    @staticmethod
    def __stackbar(  # pragma: no cover
        y: np.ndarray,
        type_names: List[str],
        title: str,
        level_names: List[str],
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[ListedColormap] = cm.tab20,
        show_legend: Optional[bool] = True,
    ) -> plt.Axes:
        """Plots a stacked barplot for one (discrete) covariate

        Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)


        Args:
            y: The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows, one for each group, containing the count mean of each cell type
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
            bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_bars)], sample_sums)]
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
        data: Union[AnnData, MuData],
        feature_name: str,
        modality_key: str = "coda",
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[ListedColormap] = cm.tab20,
        show_legend: Optional[bool] = True,
        level_order: List[str] = None,
    ) -> plt.Axes:
        """Plots a stacked barplot for all levels of a covariate or all samples (if feature_name=="samples").

        Usage: plot_feature_stackbars(data, ["cov1", "cov2", "cov3"])


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
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data

        # cell type names
        type_names = data.var.index

        # option to plot one stacked barplot per sample
        if feature_name == "samples":
            if level_order:
                assert set(level_order) == set(data.obs.index), "level order is inconsistent with levels"
                data = data[level_order]
            ax = CodaPlot.__stackbar(
                data.X,
                type_names=data.var.index,
                title="samples",
                level_names=data.obs.index,
                figsize=figsize,
                dpi=dpi,
                cmap=cmap,
                show_legend=show_legend,
            )
        else:
            # Order levels
            if level_order:
                assert set(level_order) == set(data.obs[feature_name]), "level order is inconsistent with levels"
                levels = level_order
            elif hasattr(data.obs[feature_name], "cat"):
                levels = data.obs[feature_name].cat.categories.to_list()
            else:
                levels = pd.unique(data.obs[feature_name])
            n_levels = len(levels)
            feature_totals = np.zeros([n_levels, data.X.shape[1]])

            for level in range(n_levels):
                l_indices = np.where(data.obs[feature_name] == levels[level])
                feature_totals[level] = np.sum(data.X[l_indices], axis=0)

            ax = CodaPlot.__stackbar(
                feature_totals,
                type_names=type_names,
                title=feature_name,
                level_names=levels,
                figsize=figsize,
                dpi=dpi,
                cmap=cmap,
                show_legend=show_legend,
            )
        return ax

    @staticmethod
    def effects_barplot(  # noqa: C901 # pragma: no cover
        data: Union[AnnData, MuData],
        modality_key: str = "coda",
        covariates: Optional[Union[str, List]] = None,
        parameter: Literal["log2-fold change", "Final Parameter", "Expected Sample"] = "log2-fold change",
        plot_facets: bool = True,
        plot_zero_covariate: bool = True,
        plot_zero_cell_type: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[Union[str, ListedColormap]] = cm.tab20,
        level_order: List[str] = None,
        args_barplot: Optional[dict] = None,
    ) -> Optional[Union[plt.Axes, sns.axisgrid.FacetGrid]]:
        """Barplot visualization for effects.

        The effect results for each covariate are shown as a group of barplots, with intra--group separation by cell types.
        The covariates groups can either be ordered along the x-axis of a single plot (plot_facets=False) or as plot facets (plot_facets=True).

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            covariates: The name of the covariates in data.obs to plot. Defaults to None.
            parameter: The parameter in effect summary to plot. Defaults to "log2-fold change".
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets. Defaults to True.
            plot_zero_covariate: If True, plot covariate that have all zero effects. If False, do not plot. Defaults to True.
            plot_zero_cell_type: If True, plot cell type that have zero effect. If False, do not plot. Defaults to False.
            figsize: Figure size. Defaults to None.
            dpi: Figure size. Defaults to 100.
            cmap: The seaborn color map for the barplot. Defaults to cm.tab20.
            level_order: Custom ordering of bars on the x-axis. Defaults to None.
            args_barplot: Arguments passed to sns.barplot. Defaults to {}.

        Returns:
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`) or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object
        """
        if args_barplot is None:
            args_barplot = {}
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        # Get covariate names from adata, partition into those with nonzero effects for min. one cell type/no cell types
        covariate_names = data.uns["scCODA_params"]["covariate_names"]
        if covariates is not None:
            if isinstance(covariates, str):
                covariates = [covariates]
            partial_covariate_names = [
                covariate_name
                for covariate_name in covariate_names
                if any(covariate in covariate_name for covariate in covariates)
            ]
            covariate_names = partial_covariate_names
        covariate_names_non_zero = [
            covariate_name
            for covariate_name in covariate_names
            if data.varm[f"effect_df_{covariate_name}"][parameter].any()
        ]
        covariate_names_zero = list(set(covariate_names) - set(covariate_names_non_zero))
        if not plot_zero_covariate:
            covariate_names = covariate_names_non_zero

        # set up df for plotting
        plot_df = pd.concat(
            [data.varm[f"effect_df_{covariate_name}"][parameter] for covariate_name in covariate_names],
            axis=1,
        )
        plot_df.columns = covariate_names
        plot_df = pd.melt(plot_df, ignore_index=False, var_name="Covariate")

        plot_df = plot_df.reset_index()

        if len(covariate_names_zero) != 0:
            if plot_facets:
                if plot_zero_covariate and not plot_zero_cell_type:
                    plot_df = plot_df[plot_df["value"] != 0]
                    for covariate_name_zero in covariate_names_zero:
                        new_row = {
                            "Covariate": covariate_name_zero,
                            "Cell Type": "zero",
                            "value": 0,
                        }
                        plot_df = plot_df.append(new_row, ignore_index=True)
                    plot_df["covariate_"] = pd.Categorical(plot_df["Covariate"], covariate_names)
                    plot_df = plot_df.sort_values(["covariate_"])
        if not plot_zero_cell_type:
            cell_type_names_zero = [
                name
                for name in plot_df["Cell Type"].unique()
                if (plot_df[plot_df["Cell Type"] == name]["value"] == 0).all()
            ]
            plot_df = plot_df[~plot_df["Cell Type"].isin(cell_type_names_zero)]

        # If plot as facets, create a FacetGrid and map barplot to it.
        if plot_facets:
            if isinstance(cmap, ListedColormap):
                cmap = np.array([cmap(i % cmap.N) for i in range(len(plot_df["Cell Type"].unique()))])
            if figsize is not None:
                height = figsize[0]
                aspect = np.round(figsize[1] / figsize[0], 2)
            else:
                height = 3
                aspect = 2

            g = sns.FacetGrid(
                plot_df,
                col="Covariate",
                sharey=True,
                sharex=False,
                height=height,
                aspect=aspect,
            )

            g.map(
                sns.barplot,
                "Cell Type",
                "value",
                palette=cmap,
                order=level_order,
                **args_barplot,
            )
            g.set_xticklabels(rotation=90)
            g.set(ylabel=parameter)
            axes = g.axes.flatten()
            for i, ax in enumerate(axes):
                ax.set_title(covariate_names[i])
                if len(ax.get_xticklabels()) < 5:
                    ax.set_aspect(10 / len(ax.get_xticklabels()))
                    if len(ax.get_xticklabels()) == 1:
                        if ax.get_xticklabels()[0]._text == "zero":
                            ax.set_xticks([])
            return g

        # If not plot as facets, call barplot to plot cell types on the x-axis.
        else:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if len(covariate_names) == 1:
                if isinstance(cmap, ListedColormap):
                    cmap = np.array([cmap(i % cmap.N) for i in range(len(plot_df["Cell Type"].unique()))])
                sns.barplot(
                    data=plot_df,
                    x="Cell Type",
                    y="value",
                    palette=cmap,
                    ax=ax,
                )
                ax.set_title(covariate_names[0])
            else:
                if isinstance(cmap, ListedColormap):
                    cmap = np.array([cmap(i % cmap.N) for i in range(len(covariate_names))])
                sns.barplot(
                    data=plot_df,
                    x="Cell Type",
                    y="value",
                    hue="Covariate",
                    palette=cmap,
                    ax=ax,
                )
            cell_types = pd.unique(plot_df["Cell Type"])
            ax.set_xticklabels(cell_types, rotation=90)
            return ax

    @staticmethod
    def boxplots(  # noqa: C901 # pragma: no cover
        data: Union[AnnData, MuData],
        feature_name: str,
        modality_key: str = "coda",
        y_scale: Literal["relative", "log", "log10", "count"] = "relative",
        plot_facets: bool = False,
        add_dots: bool = False,
        draw_effects: bool = False,
        model: CompositionalModel2 = None,
        cell_types: Optional[list] = None,
        args_boxplot: Optional[dict] = None,
        args_swarmplot: Optional[dict] = None,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = 100,
        cmap: Optional[str] = "Blues",
        show_legend: Optional[bool] = True,
        level_order: List[str] = None,
    ) -> Optional[Union[plt.Axes, sns.axisgrid.FacetGrid]]:
        """Grouped boxplot visualization. The cell counts for each cell type are shown as a group of boxplots,
            with intra--group separation by a covariate from data.obs.

        Args:
            data: AnnData object or MuData object
            feature_name: The name of the feature in data.obs to plot
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            y_scale: Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count), "log10" - log10(count), "count" - absolute abundance (cell counts). Defaults to "relative".
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets. Defaults to False.
            add_dots: If True, overlay a scatterplot with one dot for each data point. Defaults to False.
            draw_effects: If True, draw horizontal bars for credible effects (You have to run inference on model before using this option!). Defaults to False.
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
            Depending on `plot_facets`, returns a :class:`~matplotlib.axes.Axes` (`plot_facets = False`) or :class:`~sns.axisgrid.FacetGrid` (`plot_facets = True`) object
        """
        if args_boxplot is None:
            args_boxplot = {}
        if args_swarmplot is None:
            args_swarmplot = {}
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        # y scale transformations
        if y_scale == "relative":
            sample_sums = np.sum(data.X, axis=1, keepdims=True)
            X = data.X / sample_sums
            value_name = "Proportion"
        # add pseudocount 0.5 if using log scale
        elif y_scale == "log":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log(count)"
        elif y_scale == "log10":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log10(count)"
        elif y_scale == "count":
            X = data.X
            value_name = "count"
        else:
            raise ValueError("Invalid y_scale transformation")

        count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).merge(
            data.obs[feature_name], left_index=True, right_index=True
        )
        plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)
        if cell_types is not None:
            plot_df = plot_df[plot_df["Cell type"].isin(cell_types)]

        # Get credible effects results from model
        if draw_effects:
            if model is not None:
                credible_effects_df = model.credible_effects(data, modality_key).to_frame().reset_index()
            else:
                print("Specify a tasCODA model to draw effects")
            credible_effects_df[feature_name] = credible_effects_df["Covariate"].str.removeprefix(f"{feature_name}[T.")
            credible_effects_df[feature_name] = credible_effects_df[feature_name].str.removesuffix("]")
            credible_effects_df = credible_effects_df[credible_effects_df["Final Parameter"]]

        # If plot as facets, create a FacetGrid and map boxplot to it.
        if plot_facets:
            if level_order is None:
                level_order = pd.unique(plot_df[feature_name])

            K = X.shape[1]

            if figsize is not None:
                height = figsize[0]
                aspect = np.round(figsize[1] / figsize[0], 2)
            else:
                height = 3
                aspect = 2

            g = sns.FacetGrid(
                plot_df,
                col="Cell type",
                sharey=False,
                col_wrap=int(np.floor(np.sqrt(K))),
                height=height,
                aspect=aspect,
            )
            g.map(
                sns.boxplot,
                feature_name,
                value_name,
                palette=cmap,
                order=level_order,
                **args_boxplot,
            )

            if add_dots:
                if "hue" in args_swarmplot:
                    hue = args_swarmplot.pop("hue")
                else:
                    hue = None

                if hue is None:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        color="black",
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")
                else:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        hue,
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")
            return g

        # If not plot as facets, call boxplot to plot cell types on the x-axis.
        else:
            if level_order:
                args_boxplot["hue_order"] = level_order
                args_swarmplot["hue_order"] = level_order

            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax = sns.boxplot(
                x="Cell type",
                y=value_name,
                hue=feature_name,
                data=plot_df,
                fliersize=1,
                palette=cmap,
                ax=ax,
                **args_boxplot,
            )

            if draw_effects:
                pairs = [
                    [(row["Cell Type"], row[feature_name]), (row["Cell Type"], "Control")]
                    for _, row in credible_effects_df.iterrows()
                ]
                annot = Annotator(ax, pairs, data=plot_df, x="Cell type", y=value_name, hue=feature_name)
                annot.configure(test=None, loc="outside", color="red", line_height=0, verbose=False)
                annot.set_custom_annotations([row[feature_name] for _, row in credible_effects_df.iterrows()])
                annot.annotate()

            if add_dots:
                sns.swarmplot(
                    x="Cell type",
                    y=value_name,
                    data=plot_df,
                    hue=feature_name,
                    ax=ax,
                    dodge=True,
                    color="black",
                    **args_swarmplot,
                )

            cell_types = pd.unique(plot_df["Cell type"])
            ax.set_xticklabels(cell_types, rotation=90)

            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                handout = []
                labelout = []
                for h, l in zip(handles, labels):
                    if l not in labelout:
                        labelout.append(l)
                        handout.append(h)
                ax.legend(
                    handout,
                    labelout,
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    ncol=1,
                    title=feature_name,
                )

            plt.tight_layout()
            return ax

    @staticmethod
    def rel_abundance_dispersion_plot(  # pragma: no cover
        data: Union[AnnData, MuData],
        modality_key: str = "coda",
        abundant_threshold: Optional[float] = 0.9,
        default_color: Optional[str] = "Grey",
        abundant_color: Optional[str] = "Red",
        label_cell_types: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = 100,
        ax: Axes = None,
    ) -> plt.Axes:
        """Plots total variance of relative abundance versus minimum relative abundance of all cell types for determination of a reference cell type.

        If the count of the cell type is larger than 0 in more than abundant_threshold percent of all samples, the cell type will be marked in a different color.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda". Defaults to "coda".
            abundant_threshold: Presence threshold for abundant cell types. Defaults to 0.9.
            default_color: Bar color for all non-minimal cell types. Defaults to "Grey".
            abundant_color: Bar color for cell types with abundant percentage larger than abundant_threshold. Defaults to "Red".
            label_cell_types: Label dots with cell type names. Defaults to True.
            figsize: Figure size. Defaults to None.
            dpi: Dpi setting. Defaults to 100.
            ax: A matplotlib axes object. Only works if plotting a single component. Defaults to None.

        Returns:
            A :class:`~matplotlib.axes.Axes` object
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        rel_abun = data.X / np.sum(data.X, axis=1, keepdims=True)

        percent_zero = np.sum(data.X == 0, axis=0) / data.X.shape[0]
        nonrare_ct = np.where(percent_zero < 1 - abundant_threshold)[0]

        # select reference
        cell_type_disp = np.var(rel_abun, axis=0) / np.mean(rel_abun, axis=0)

        is_abundant = [x in nonrare_ct for x in range(data.X.shape[1])]

        # Scatterplot
        plot_df = pd.DataFrame(
            {
                "Total dispersion": cell_type_disp,
                "Cell type": data.var.index,
                "Presence": 1 - percent_zero,
                "Is abundant": is_abundant,
            }
        )

        if len(np.unique(plot_df["Is abundant"])) > 1:
            palette = [default_color, abundant_color]
        elif np.unique(plot_df["Is abundant"]) == [False]:
            palette = [default_color]
        else:
            palette = [abundant_color]

        ax = sns.scatterplot(
            data=plot_df,
            x="Presence",
            y="Total dispersion",
            hue="Is abundant",
            palette=palette,
            ax=ax,
        )

        # Text labels for abundant cell types

        abundant_df = plot_df.loc[plot_df["Is abundant"], :]

        def label_point(x, y, val, ax):
            a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
            texts = [
                ax.text(
                    point["x"],
                    point["y"],
                    str(point["val"]),
                )
                for i, point in a.iterrows()
            ]
            adjust_text(texts)

        if label_cell_types:
            label_point(
                abundant_df["Presence"],
                abundant_df["Total dispersion"],
                abundant_df["Cell type"],
                plt.gca(),
            )

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, title="Is abundant")

        plt.tight_layout()
        return ax

    @staticmethod
    def draw_tree(  # pragma: no cover
        data: Union[AnnData, MuData],
        modality_key: str = "coda",
        tree: Union[Tree, str] = "tree",
        tight_text: Optional[bool] = False,
        show_scale: Optional[bool] = False,
        show: Optional[bool] = True,
        file_name: Optional[str] = None,
        units: Optional[Literal["px", "mm", "in"]] = "px",
        h: Optional[float] = None,
        w: Optional[float] = None,
        dpi: Optional[int] = 90,
    ):
        """Plot a tree using input ete3 tree object.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            tree: A ete3 tree object or a str to indicate the tree stored in `.uns`. Defaults to "tree".
            tight_text: When False, boundaries of the text are approximated according to general font metrics, producing slightly worse aligned text faces but improving the performance of tree visualization in scenes with a lot of text faces. Default to False.
            show_scale: Include the scale legend in the tree image or not. Default to False.
            show: If True, plot the tree inline. If false, return tree and tree_style objects. Defaults to True.
            file_name: Path to the output image file. Valid extensions are .SVG, .PDF, .PNG. Output image can be saved whether show is True or not. Defaults to None.
            units: Unit of image sizes. “px”: pixels, “mm”: millimeters, “in”: inches. Defaults to "px".
            h: Height of the image in units. Defaults to None.
            w: Width of the image in units. Defaults to None.
            dpi: Dots per inches. Defaults to 90.

        Returns:
            Depending on `show`, returns :class:`ete3.TreeNode` and :class:`ete3.TreeStyle` (`show = False`) or  plot the tree inline (`show = False`)
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        if isinstance(tree, str):
            tree = data.uns[tree]

        def my_layout(node):
            text_face = TextFace(node.name, tight_text=tight_text)
            faces.add_face_to_node(text_face, node, column=0, position="branch-right")

        tree_style = TreeStyle()
        tree_style.show_leaf_name = False
        tree_style.layout_fn = my_layout
        tree_style.show_scale = show_scale
        if file_name is not None:
            tree.render(file_name, tree_style=tree_style, units=units, w=w, h=h, dpi=dpi)
        if show:
            return tree.render("%%inline", tree_style=tree_style, units=units, w=w, h=h, dpi=dpi)
        else:
            return tree, tree_style

    @staticmethod
    def draw_effects(  # noqa: C901 # pragma: no cover
        data: Union[AnnData, MuData],
        covariate: str,
        modality_key: str = "coda",
        tree: Union[Tree, str] = "tree",
        show_legend: Optional[bool] = None,
        show_leaf_effects: Optional[bool] = False,
        tight_text: Optional[bool] = False,
        show_scale: Optional[bool] = False,
        show: Optional[bool] = True,
        file_name: Optional[str] = None,
        units: Optional[Literal["px", "mm", "in"]] = "in",
        h: Optional[float] = None,
        w: Optional[float] = None,
        dpi: Optional[int] = 90,
    ):
        """Plot a tree with colored circles on the nodes indicating significant effects with bar plots which indicate leave-level significant effects.

        Args:
            data: AnnData object or MuData object.
            covariate: The covariate, whose effects should be plotted.
            modality_key: If data is a MuData object, specify which modality to use. Defaults to "coda".
            tree: A ete3 tree object or a str to indicate the tree stored in `.uns`. Defaults to "tree".
            show_legend: If show legend of nodes significant effects or not. Default is False if show_leaf_effects is True.
            show_leaf_effects: If True, plot bar plots which indicate leave-level significant effects. Defaults to False.
            tight_text: When False, boundaries of the text are approximated according to general font metrics, producing slightly worse aligned text faces but improving the performance of tree visualization in scenes with a lot of text faces. Defaults to False.
            show_scale: Include the scale legend in the tree image or not. Defaults to False.
            show: If True, plot the tree inline. If false, return tree and tree_style objects. Defaults to True.
            file_name: Path to the output image file. valid extensions are .SVG, .PDF, .PNG. Output image can be saved whether show is True or not. Defaults to None.
            units: Unit of image sizes. “px”: pixels, “mm”: millimeters, “in”: inches. Default is "in". Defaults to "in".
            h: Height of the image in units. Defaults to None.
            w: Width of the image in units. Defaults to None.
            dpi: Dots per inches. Defaults to 90.

        Returns:
            Depending on `show`, returns :class:`ete3.TreeNode` and :class:`ete3.TreeStyle` (`show = False`) or  plot the tree inline (`show = False`)
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        if show_legend is None:
            show_legend = not show_leaf_effects
        elif show_legend:
            print("Tree leaves and leaf effect bars won't be aligned when legend is shown!")

        if isinstance(tree, str):
            tree = data.uns[tree]
        # Collapse tree singularities
        tree2 = collapse_singularities_2(tree)

        node_effs = (
            data.uns["scCODA_params"]["node_df"]
            .loc[
                (covariate + "_node",),
            ]
            .copy()
        )
        node_effs.index = node_effs.index.get_level_values("Node")

        covariates = data.uns["scCODA_params"]["covariate_names"]
        effect_dfs = [data.varm[f"effect_df_{cov}"] for cov in covariates]
        eff_df = pd.concat(effect_dfs)
        eff_df.index = pd.MultiIndex.from_product(
            (covariates, data.var.index.tolist()),
            names=["Covariate", "Cell Type"],
        )
        leaf_effs = eff_df.loc[
            (covariate,),
        ].copy()
        leaf_effs.index = leaf_effs.index.get_level_values("Cell Type")

        # Add effect values
        for n in tree2.traverse():
            nstyle = NodeStyle()
            nstyle["size"] = 0
            n.set_style(nstyle)
            if n.name in node_effs.index:
                e = node_effs.loc[n.name, "Final Parameter"]
                n.add_feature("node_effect", e)
            else:
                n.add_feature("node_effect", 0)
            if n.name in leaf_effs.index:
                e = leaf_effs.loc[n.name, "Effect"]
                n.add_feature("leaf_effect", e)
            else:
                n.add_feature("leaf_effect", 0)

        # Scale effect values to get nice node sizes
        eff_max = np.max([np.abs(n.node_effect) for n in tree2.traverse()])
        leaf_eff_max = np.max([np.abs(n.leaf_effect) for n in tree2.traverse()])

        def my_layout(node):
            text_face = TextFace(node.name, tight_text=tight_text)
            text_face.margin_left = 10
            faces.add_face_to_node(text_face, node, column=0, aligned=True)

            # if node.is_leaf():
            size = (np.abs(node.node_effect) * 10 / eff_max) if node.node_effect != 0 else 0
            if np.sign(node.node_effect) == 1:
                color = "blue"
            elif np.sign(node.node_effect) == -1:
                color = "red"
            else:
                color = "cyan"
            if size != 0:
                faces.add_face_to_node(CircleFace(radius=size, color=color), node, column=0)

        tree_style = TreeStyle()
        tree_style.show_leaf_name = False
        tree_style.layout_fn = my_layout
        tree_style.show_scale = show_scale
        tree_style.draw_guiding_lines = True
        tree_style.legend_position = 1

        if show_legend:
            tree_style.legend.add_face(TextFace("Effects"), column=0)
            tree_style.legend.add_face(TextFace("       "), column=1)
            for i in range(4, 0, -1):
                tree_style.legend.add_face(
                    CircleFace(
                        float(f"{np.abs(eff_max) * 10 * i / (eff_max * 4):.2f}"),
                        "red",
                    ),
                    column=0,
                )
                tree_style.legend.add_face(TextFace(f"{-eff_max * i / 4:.2f} "), column=0)
                tree_style.legend.add_face(
                    CircleFace(
                        float(f"{np.abs(eff_max) * 10 * i / (eff_max * 4):.2f}"),
                        "blue",
                    ),
                    column=1,
                )
                tree_style.legend.add_face(TextFace(f" {eff_max * i / 4:.2f}"), column=1)

        if show_leaf_effects:
            leaf_name = [node.name for node in tree2.traverse("postorder") if node.is_leaf()]
            leaf_effs = leaf_effs.loc[leaf_name].reset_index()
            palette = ["blue" if Effect > 0 else "red" for Effect in leaf_effs["Effect"].tolist()]

            dir_path = os.getcwd()
            dir_path = os.path.join(dir_path, "tree_effect.png")
            tree2.render(dir_path, tree_style=tree_style, units="in")
            _, ax = plt.subplots(1, 2, figsize=(10, 10))
            sns.barplot(data=leaf_effs, x="Effect", y="Cell Type", palette=palette, ax=ax[1])
            img = mpimg.imread(dir_path)
            ax[0].imshow(img)
            ax[0].get_xaxis().set_visible(False)
            ax[0].get_yaxis().set_visible(False)
            ax[0].set_frame_on(False)

            ax[1].get_yaxis().set_visible(False)
            ax[1].spines["left"].set_visible(False)
            ax[1].spines["right"].set_visible(False)
            ax[1].spines["top"].set_visible(False)
            plt.xlim(-leaf_eff_max, leaf_eff_max)
            plt.subplots_adjust(wspace=0)

            if file_name is not None:
                plt.savefig(file_name)

        if file_name is not None and not show_leaf_effects:
            tree2.render(file_name, tree_style=tree_style, units=units)
        if show:
            if not show_leaf_effects:
                return tree2.render("%%inline", tree_style=tree_style, units=units, w=w, h=h, dpi=dpi)
        else:
            if not show_leaf_effects:
                return tree2, tree_style

    @staticmethod
    def effects_umap(  # pragma: no cover
        data: MuData,
        effect_name: Optional[Union[str, list]],
        cluster_key: str,
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
        show: bool = None,
        ax: Axes = None,
        **kwargs,
    ):
        """Plot a UMAP visualization colored by effect strength. Effect results in .varm of aggregated sample-level AnnData (default is data['coda']) are assigned to cell-level AnnData (default is data['rna']) depending on the cluster they were assigned to.

        Args:
            data: AnnData object or MuData object.
            effect_name: The name of the effect results in .varm of aggregated sample-level AnnData (default is data['coda']) to plot
            cluster_key: The cluster information in .obs of cell-level AnnData (default is data['rna']). To assign cell types' effects to original cells.
            modality_key_1: Key to the cell-level AnnData in the MuData object. Defaults to "rna".
            modality_key_2: Key to the aggregated sample-level AnnData object in the MuData object. Defaults to "coda".
            show: Whether to display the figure or return axis. Defaults to None.
            ax: A matplotlib axes object. Only works if plotting a single component. Defaults to None.
            **kwargs: All other keyword arguments are passed to `scanpy.plot.umap()`

        Returns:
            If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
        """
        data_rna = data[modality_key_1]
        data_coda = data[modality_key_2]
        if isinstance(effect_name, str):
            effect_name = [effect_name]
        for _, effect in enumerate(effect_name):
            data_rna.obs[effect] = [data_coda.varm[effect].loc[f"{c}", "Effect"] for c in data_rna.obs[cluster_key]]
        if kwargs.get("vmin"):
            vmin = kwargs["vmin"]
            kwargs.pop("vmin")
        else:
            vmin = min([data_rna.obs[effect].min() for _, effect in enumerate(effect_name)])
        if kwargs.get("vmax"):
            vmax = kwargs["vmax"]
            kwargs.pop("vmax")
        else:
            vmax = max([data_rna.obs[effect].max() for _, effect in enumerate(effect_name)])
        return sc.pl.umap(data_rna, color=effect_name, vmax=vmax, vmin=vmin, ax=ax, show=show, **kwargs)
