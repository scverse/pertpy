import contextlib
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from itertools import zip_longest
from types import MappingProxyType

import adjustText
import anndata as ad
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from formulaic_contrasts import FormulaicContrasts
from lamin_utils import logger
from matplotlib.pyplot import Figure
from matplotlib.ticker import MaxNLocator

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy.tools import PseudobulkSpace
from pertpy.tools._differential_gene_expression._checks import check_is_numeric_matrix


class MethodBase(ABC):
    def __init__(self, adata, *, mask=None, layer=None, **kwargs):
        """Initialize the method.

        Args:
            adata: AnnData object, usually pseudobulked.
            mask: A column in `adata.var` that contains a boolean mask with selected features.
            layer: Layer to use in fit(). If None, use the X array.
            **kwargs: Keyword arguments specific to the method implementation.
        """
        self.adata = adata
        if mask is not None:
            self.adata = self.adata[:, self.adata.var[mask]]

        self.layer = layer
        check_is_numeric_matrix(self.data)

    @property
    def data(self):
        """Get the data matrix from anndata this object was initalized with (X or layer)."""
        if self.layer is None:
            return self.adata.X
        else:
            return self.adata.layers[self.layer]

    @classmethod
    @abstractmethod
    def compare_groups(
        cls,
        adata,
        column,
        baseline,
        groups_to_compare,
        *,
        paired_by=None,
        mask=None,
        layer=None,
        fit_kwargs=MappingProxyType({}),
        test_kwargs=MappingProxyType({}),
    ):
        """Compare between groups in a specified column.

        Args:
            adata: AnnData object.
            column: column in obs that contains the grouping information.
            baseline: baseline value (one category from variable).
            groups_to_compare: One or multiple categories from variable to compare against baseline.
            paired_by: Column from `obs` that contains information about paired sample (e.g. subject_id).
            mask: Subset anndata by a boolean mask stored in this column in `.obs` before making any tests.
            layer: Use this layer instead of `.X`.
            fit_kwargs: Additional fit options.
            test_kwargs: Additional test options.

        Returns:
            Pandas dataframe with results ordered by significance. If multiple comparisons were performed this is indicated in an additional column.

        Examples:
            >>> # Example with EdgeR
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata,
            ...     target_col="Patient",
            ...     groups_col="Cluster",
            ...     layer_key="counts",
            ...     mode="sum",
            ...     min_cells=10,
            ...     min_counts=1000,
            ... )
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> res_df = edgr.compare_groups(pdata, column="Efficacy", baseline="SD", groups_to_compare=["PR", "PD"])
        """
        ...

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_volcano(  # pragma: no cover # noqa: D417
        self,
        data: pd.DataFrame | ad.AnnData,
        *,
        log2fc_col: str = "log_fc",
        pvalue_col: str = "adj_p_value",
        symbol_col: str = "variable",
        pval_thresh: float = 0.05,
        log2fc_thresh: float = 0.75,
        to_label: int | list[str] = 5,
        s_curve: bool | None = False,
        colors: list[str] = None,
        varm_key: str | None = None,
        color_dict: dict[str, list[str]] | None = None,
        shape_dict: dict[str, list[str]] | None = None,
        size_col: str | None = None,
        fontsize: int = 10,
        top_right_frame: bool = False,
        figsize: tuple[int, int] = (5, 5),
        legend_pos: tuple[float, float] = (1.6, 1),
        point_sizes: tuple[int, int] = (15, 150),
        shapes: list[str] | None = None,
        shape_order: list[str] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        return_fig: bool = False,
        **kwargs: int,
    ) -> Figure | None:
        """Creates a volcano plot from a pandas DataFrame or Anndata.

        Args:
            data: DataFrame or Anndata to plot.
            log2fc_col: Column name of log2 Fold-Change values.
            pvalue_col: Column name of the p values.
            symbol_col: Column name of gene IDs.
            varm_key: Key in Anndata.varm slot to use for plotting if an Anndata object was passed.
            size_col: Column name to size points by.
            point_sizes: Lower and upper bounds of point sizes.
            pval_thresh: Threshold p value for significance.
            log2fc_thresh: Threshold for log2 fold change significance.
            to_label: Number of top genes or list of genes to label.
            s_curve: Whether to use a reciprocal threshold for up and down gene determination.
            color_dict: Dictionary for coloring dots by categories.
            shape_dict: Dictionary for shaping dots by categories.
            fontsize: Size of gene labels.
            colors: Colors for [non-DE, up, down] genes. Defaults to ['gray', '#D62728', '#1F77B4'].
            top_right_frame: Whether to show the top and right frame of the plot.
            figsize: Size of the figure.
            legend_pos: Position of the legend as determined by matplotlib.
            shapes: List of matplotlib marker ids.
            shape_order: Order of categories for shapes.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            {common_plot_args}
            **kwargs: Additional arguments for seaborn.scatterplot.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> # Example with EdgeR
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata,
            ...     target_col="Patient",
            ...     groups_col="Cluster",
            ...     layer_key="counts",
            ...     mode="sum",
            ...     min_cells=10,
            ...     min_counts=1000,
            ... )
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> edgr.fit()
            >>> res_df = edgr.test_contrasts(
            ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
            ... )
            >>> edgr.plot_volcano(res_df, log2fc_thresh=0)

        Preview:
            .. image:: /_static/docstring_previews/de_volcano.png
        """
        if colors is None:
            colors = ["gray", "#D62728", "#1F77B4"]

        def _pval_reciprocal(lfc: float) -> float:
            """Function for relating -log10(pvalue) and logfoldchange in a reciprocal.

            Used for plotting the S-curve
            """
            return pval_thresh / (lfc - log2fc_thresh)

        def _map_shape(symbol: str) -> str:
            if shape_dict is not None:
                for k in shape_dict:
                    if shape_dict[k] is not None and symbol in shape_dict[k]:
                        return k
            return "other"

        # TODO join the two mapping functions
        def _map_genes_categories(
            row: pd.Series,
            log2fc_col: str,
            nlog10_col: str,
            log2fc_thresh: float,
            pval_thresh: float = None,
            s_curve: bool = False,
        ) -> str:
            """Map genes to categorize based on log2fc and pvalue.

            These categories are used for coloring the dots.
            Used when no color_dict is passed, sets up/down/nonsignificant.
            """
            log2fc = row[log2fc_col]
            nlog10 = row[nlog10_col]

            if s_curve:
                # S-curve condition for Up or Down categorization
                reciprocal_thresh = _pval_reciprocal(abs(log2fc))
                if log2fc > log2fc_thresh and nlog10 > reciprocal_thresh:
                    return "Up"
                elif log2fc < -log2fc_thresh and nlog10 > reciprocal_thresh:
                    return "Down"
                else:
                    return "not DE"
            # Standard condition for Up or Down categorization
            elif log2fc > log2fc_thresh and nlog10 > pval_thresh:
                return "Up"
            elif log2fc < -log2fc_thresh and nlog10 > pval_thresh:
                return "Down"
            else:
                return "not DE"

        def _map_genes_categories_highlight(
            row: pd.Series,
            log2fc_col: str,
            nlog10_col: str,
            log2fc_thresh: float,
            pval_thresh: float = None,
            s_curve: bool = False,
            symbol_col: str = None,
        ) -> str:
            """Map genes to categorize based on log2fc and pvalue.

            These categories are used for coloring the dots.
            Used when color_dict is passed, sets DE / not DE for background and user supplied highlight genes.
            """
            log2fc = row[log2fc_col]
            nlog10 = row[nlog10_col]
            symbol = row[symbol_col]

            if color_dict is not None:
                for k in color_dict:
                    if symbol in color_dict[k]:
                        return k

            if s_curve:
                # Use S-curve condition for filtering DE
                if nlog10 > _pval_reciprocal(abs(log2fc)) and abs(log2fc) > log2fc_thresh:
                    return "DE"
                return "not DE"
            else:
                # Use standard condition for filtering DE
                if abs(log2fc) < log2fc_thresh or nlog10 < pval_thresh:
                    return "not DE"
                return "DE"

        if isinstance(data, ad.AnnData):
            if varm_key is None:
                raise ValueError("Please pass a .varm key to use for plotting")

            raise NotImplementedError("Anndata not implemented yet")  # TODO: Implement this
            df = data.varm[varm_key].copy()

        df = data.copy(deep=True)

        # clean and replace 0s as they would lead to -inf
        if df[[log2fc_col, pvalue_col]].isnull().values.any():
            print("NaNs encountered, dropping rows with NaNs")
            df = df.dropna(subset=[log2fc_col, pvalue_col])

        if df[pvalue_col].min() == 0:
            print("0s encountered for p value, replacing with 1e-323")
            df.loc[df[pvalue_col] == 0, pvalue_col] = 1e-323

        # convert p value threshold to nlog10
        pval_thresh = -np.log10(pval_thresh)
        # make nlog10 column
        df["nlog10"] = -np.log10(df[pvalue_col])
        y_max = df["nlog10"].max() + 1
        # make a column to pick top genes
        df["top_genes"] = df["nlog10"] * df[log2fc_col]

        # Label everything with assigned color / shape
        if shape_dict or color_dict:
            combined_labels = []
            if isinstance(shape_dict, dict):
                combined_labels.extend([item for sublist in shape_dict.values() for item in sublist])
            if isinstance(color_dict, dict):
                combined_labels.extend([item for sublist in color_dict.values() for item in sublist])
            label_df = df[df[symbol_col].isin(combined_labels)]

        # Label top n_gens
        elif isinstance(to_label, int):
            label_df = pd.concat(
                (
                    df.sort_values("top_genes")[-to_label:],
                    df.sort_values("top_genes")[0:to_label],
                )
            )

        # assume that a list of genes was passed to label
        else:
            label_df = df[df[symbol_col].isin(to_label)]

        # By default mode colors by up/down if no dict is passed

        if color_dict is None:
            df["color"] = df.apply(
                lambda row: _map_genes_categories(
                    row,
                    log2fc_col=log2fc_col,
                    nlog10_col="nlog10",
                    log2fc_thresh=log2fc_thresh,
                    pval_thresh=pval_thresh,
                    s_curve=s_curve,
                ),
                axis=1,
            )

            # order of colors
            hues = ["not DE", "Up", "Down"][: len(df.color.unique())]

        else:
            df["color"] = df.apply(
                lambda row: _map_genes_categories_highlight(
                    row,
                    log2fc_col=log2fc_col,
                    nlog10_col="nlog10",
                    log2fc_thresh=log2fc_thresh,
                    pval_thresh=pval_thresh,
                    symbol_col=symbol_col,
                    s_curve=s_curve,
                ),
                axis=1,
            )

            user_added_cats = [x for x in df.color.unique() if x not in ["DE", "not DE"]]
            hues = ["DE", "not DE"] + user_added_cats

            # order of colors
            hues = hues[: len(df.color.unique())]
            colors = [
                "dimgrey",
                "lightgrey",
                "tab:blue",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:brown",
                "tab:pink",
                "tab:olive",
                "tab:cyan",
            ]

        # coloring if dictionary passed, subtle background + highlight
        # map shapes if dictionary exists
        if shape_dict is not None:
            df["shape"] = df[symbol_col].map(_map_shape)
            user_added_cats = [x for x in df["shape"].unique() if x != "other"]
            shape_order = ["other"] + user_added_cats
            if shapes is None:
                shapes = ["o", "^", "s", "X", "*", "d"]
            shapes = shapes[: len(df["shape"].unique())]
            shape_col = "shape"
        else:
            shape_col = None

        # build palette
        colors = colors[: len(df.color.unique())]

        # We want plot highlighted genes on top + at bigger size, split dataframe
        df_highlight = None
        if shape_dict or color_dict:
            label_genes = label_df[symbol_col].unique()
            df_highlight = df[df[symbol_col].isin(label_genes)]
            df = df[~df[symbol_col].isin(label_genes)]

        plt.figure(figsize=figsize)
        # Plot non-highlighted genes
        ax = sns.scatterplot(
            data=df,
            x=log2fc_col,
            y="nlog10",
            hue="color",
            hue_order=hues,
            palette=colors,
            size=size_col,
            sizes=point_sizes,
            style=shape_col,
            style_order=shape_order,
            markers=shapes,
            **kwargs,
        )
        # Plot highlighted genes
        if df_highlight is not None:
            ax = sns.scatterplot(
                data=df_highlight,
                x=log2fc_col,
                y="nlog10",
                hue="color",
                hue_order=hues,
                palette=colors,
                size=size_col,
                sizes=point_sizes,
                style=shape_col,
                style_order=shape_order,
                markers=shapes,
                legend=False,
                edgecolor="black",
                linewidth=1,
                **kwargs,
            )

        # plot vertical and horizontal lines
        if s_curve:
            x = np.arange((log2fc_thresh + 0.000001), y_max, 0.01)
            y = _pval_reciprocal(x)
            ax.plot(x, y, zorder=1, c="k", lw=2, ls="--")
            ax.plot(-x, y, zorder=1, c="k", lw=2, ls="--")

        else:
            ax.axhline(pval_thresh, zorder=1, c="k", lw=2, ls="--")
            ax.axvline(log2fc_thresh, zorder=1, c="k", lw=2, ls="--")
            ax.axvline(log2fc_thresh * -1, zorder=1, c="k", lw=2, ls="--")
        plt.ylim(0, y_max)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # make labels
        texts = []
        for i in range(len(label_df)):
            txt = plt.text(
                x=label_df.iloc[i][log2fc_col],
                y=label_df.iloc[i].nlog10,
                s=label_df.iloc[i][symbol_col],
                fontsize=fontsize,
            )

            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="w")])
            texts.append(txt)

        adjustText.adjust_text(texts, arrowprops={"arrowstyle": "-", "color": "k", "zorder": 5})

        # make things pretty
        for axis in ["bottom", "left", "top", "right"]:
            ax.spines[axis].set_linewidth(2)

        if not top_right_frame:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax.tick_params(width=2)
        plt.xticks(size=11, fontsize=10)
        plt.yticks(size=11)

        # Set default axis titles
        if x_label is None:
            x_label = log2fc_col
        if y_label is None:
            y_label = f"-$log_{{10}}$ {pvalue_col}"

        plt.xlabel(x_label, size=15)
        plt.ylabel(y_label, size=15)

        plt.legend(loc=1, bbox_to_anchor=legend_pos, frameon=False)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_paired(  # pragma: no cover # noqa: D417
        self,
        adata: ad.AnnData,
        results_df: pd.DataFrame,
        groupby: str,
        pairedby: str,
        *,
        var_names: Sequence[str] = None,
        n_top_vars: int = 15,
        layer: str = None,
        pvalue_col: str = "adj_p_value",
        symbol_col: str = "variable",
        n_cols: int = 4,
        panel_size: tuple[int, int] = (5, 5),
        show_legend: bool = True,
        size: int = 10,
        y_label: str = "expression",
        pvalue_template=lambda x: f"p={x:.2e}",
        boxplot_properties=None,
        palette=None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Creates a pairwise expression plot from a Pandas DataFrame or Anndata.

        Visualizes a panel of paired scatterplots per variable.

        Args:
            adata: AnnData object, can be pseudobulked.
            results_df: DataFrame with results from a differential expression test.
            groupby: .obs column containing the grouping. Must contain exactly two different values.
            pairedby: .obs column containing the pairing (e.g. "patient_id"). If None, an independent t-test is performed.
            var_names: Variables to plot.
            n_top_vars: Number of top variables to plot.
            layer: Layer to use for plotting.
            pvalue_col: Column name of the p values.
            symbol_col: Column name of gene IDs.
            n_cols: Number of columns in the plot.
            panel_size: Size of each panel.
            show_legend: Whether to show the legend.
            size: Size of the points.
            y_label: Label for the y-axis.
            pvalue_template: Template for the p-value string displayed in the title of each panel.
            boxplot_properties: Additional properties for the boxplot, passed to seaborn.boxplot.
            palette: Color palette for the line- and stripplot.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> # Example with EdgeR
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata,
            ...     target_col="Patient",
            ...     groups_col="Cluster",
            ...     layer_key="counts",
            ...     mode="sum",
            ...     min_cells=10,
            ...     min_counts=1000,
            ... )
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> edgr.fit()
            >>> res_df = edgr.test_contrasts(
            ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
            ... )
            >>> edgr.plot_paired(pdata, results_df=res_df, n_top_vars=8, groupby="Treatment", pairedby="Efficacy")

        Preview:
            .. image:: /_static/docstring_previews/de_paired_expression.png
        """
        if boxplot_properties is None:
            boxplot_properties = {}
        groups = adata.obs[groupby].unique()
        if len(groups) != 2:
            raise ValueError("The number of groups in the group_by column must be exactly 2 to enable paired testing")

        if var_names is None:
            var_names = results_df.head(n_top_vars)[symbol_col].tolist()

        adata = adata[:, var_names]

        if any(adata.obs[[groupby, pairedby]].value_counts() > 1):
            logger.info("Performing pseudobulk for paired samples")
            ps = PseudobulkSpace()
            adata = ps.compute(
                adata, target_col=groupby, groups_col=pairedby, layer_key=layer, mode="sum", min_cells=1, min_counts=1
            )

        X = adata.layers[layer] if layer is not None else adata.X
        with contextlib.suppress(AttributeError):
            X = X.toarray()

        groupby_cols = [pairedby, groupby]
        df = adata.obs.loc[:, groupby_cols].join(pd.DataFrame(X, index=adata.obs_names, columns=var_names))

        # remove unpaired samples
        paired_samples = set(df[df[groupby] == groups[0]][pairedby]) & set(df[df[groupby] == groups[1]][pairedby])
        df = df[df[pairedby].isin(paired_samples)]
        removed_samples = adata.obs[pairedby].nunique() - len(df[pairedby].unique())
        if removed_samples > 0:
            logger.warning(f"{removed_samples} unpaired samples removed")

        pvalues = results_df.set_index(symbol_col).loc[var_names, pvalue_col].values
        df.reset_index(drop=False, inplace=True)

        # transform data for seaborn
        df_melt = df.melt(
            id_vars=groupby_cols,
            var_name="var",
            value_name="val",
        )

        n_panels = len(var_names)
        nrows = math.ceil(n_panels / n_cols)
        ncols = min(n_cols, n_panels)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(ncols * panel_size[0], nrows * panel_size[1]),
            tight_layout=True,
            squeeze=False,
        )
        axes = axes.flatten()
        for i, (var, ax) in enumerate(zip_longest(var_names, axes)):
            if var is not None:
                sns.boxplot(
                    x=groupby,
                    data=df_melt.loc[df_melt["var"] == var],
                    y="val",
                    ax=ax,
                    color="white",
                    fliersize=0,
                    **boxplot_properties,
                )
                if pairedby is not None:
                    sns.lineplot(
                        x=groupby,
                        data=df_melt.loc[df_melt["var"] == var],
                        y="val",
                        ax=ax,
                        hue=pairedby,
                        legend=False,
                        errorbar=None,
                        palette=palette,
                    )
                jitter = 0 if pairedby else True
                sns.stripplot(
                    x=groupby,
                    data=df_melt.loc[df_melt["var"] == var],
                    y="val",
                    ax=ax,
                    hue=pairedby,
                    jitter=jitter,
                    size=size,
                    linewidth=1,
                    palette=palette,
                )

                ax.set_xlabel("")
                ax.tick_params(
                    axis="x",
                    labelsize=15,
                )
                ax.legend().set_visible(False)
                ax.set_ylabel(y_label)
                ax.set_title(f"{var}\n{pvalue_template(pvalues[i])}")
            else:
                ax.set_visible(False)
        fig.tight_layout()

        if show_legend is True:
            axes[n_panels - 1].legend().set_visible(True)
            axes[n_panels - 1].legend(
                bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=adata.obs[pairedby].nunique()
            )

        plt.tight_layout()
        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_fold_change(  # pragma: no cover # noqa: D417
        self,
        results_df: pd.DataFrame,
        *,
        var_names: Sequence[str] = None,
        n_top_vars: int = 15,
        log2fc_col: str = "log_fc",
        symbol_col: str = "variable",
        y_label: str = "Log2 fold change",
        figsize: tuple[int, int] = (10, 5),
        return_fig: bool = False,
        **barplot_kwargs,
    ) -> Figure | None:
        """Plot a metric from the results as a bar chart, optionally with additional information about paired samples in a scatter plot.

        Args:
            results_df: DataFrame with results from DE analysis.
            var_names: Variables to plot. If None, the top n_top_vars variables based on the log2 fold change are plotted.
            n_top_vars: Number of top variables to plot. The top and bottom n_top_vars variables are plotted, respectively.
            log2fc_col: Column name of log2 Fold-Change values.
            symbol_col: Column name of gene IDs.
            y_label: Label for the y-axis.
            figsize: Size of the figure.
            {common_plot_args}
            **barplot_kwargs: Additional arguments for seaborn.barplot.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> # Example with EdgeR
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata,
            ...     target_col="Patient",
            ...     groups_col="Cluster",
            ...     layer_key="counts",
            ...     mode="sum",
            ...     min_cells=10,
            ...     min_counts=1000,
            ... )
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> edgr.fit()
            >>> res_df = edgr.test_contrasts(
            ...     edgr.contrast(column="Treatment", baseline="Chemo", group_to_compare="Anti-PD-L1+Chemo")
            ... )
            >>> edgr.plot_fold_change(res_df)

        Preview:
            .. image:: /_static/docstring_previews/de_fold_change.png
        """
        if var_names is None:
            var_names = results_df.sort_values(log2fc_col, ascending=False).head(n_top_vars)[symbol_col].tolist()
            var_names += results_df.sort_values(log2fc_col, ascending=True).head(n_top_vars)[symbol_col].tolist()
            assert len(var_names) == 2 * n_top_vars

        df = results_df[results_df[symbol_col].isin(var_names)]
        df.sort_values(log2fc_col, ascending=False, inplace=True)

        plt.figure(figsize=figsize)
        sns.barplot(
            x=symbol_col,
            y=log2fc_col,
            data=df,
            palette="RdBu",
            legend=False,
            **barplot_kwargs,
        )
        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel(y_label)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_multicomparison_fc(  # pragma: no cover # noqa: D417
        self,
        results_df: pd.DataFrame,
        *,
        n_top_vars=15,
        contrast_col: str = "contrast",
        log2fc_col: str = "log_fc",
        pvalue_col: str = "adj_p_value",
        symbol_col: str = "variable",
        marker_size: int = 100,
        figsize: tuple[int, int] = (10, 2),
        x_label: str = "Contrast",
        y_label: str = "Gene",
        return_fig: bool = False,
        **heatmap_kwargs,
    ) -> Figure | None:
        """Plot a matrix of log2 fold changes from the results.

        Args:
            results_df: DataFrame with results from DE analysis.
            n_top_vars: Number of top variables to plot per group.
            contrast_col: Column in results_df containing information about the contrast.
            log2fc_col: Column in results_df containing the log2 fold change.
            pvalue_col: Column in results_df containing the p-value. Can be used to switch between adjusted and unadjusted p-values.
            symbol_col: Column in results_df containing the gene symbol.
            marker_size: Size of the biggest marker for significant variables.
            figsize: Size of the figure.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            {common_plot_args}
            **heatmap_kwargs: Additional arguments for seaborn.heatmap.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> # Example with EdgeR
            >>> import pertpy as pt
            >>> adata = pt.dt.zhang_2021()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> pdata = ps.compute(
            ...     adata,
            ...     target_col="Patient",
            ...     groups_col="Cluster",
            ...     layer_key="counts",
            ...     mode="sum",
            ...     min_cells=10,
            ...     min_counts=1000,
            ... )
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> res_df = edgr.compare_groups(pdata, column="Efficacy", baseline="SD", groups_to_compare=["PR", "PD"])
            >>> edgr.plot_multicomparison_fc(res_df)

        Preview:
            .. image:: /_static/docstring_previews/de_multicomparison_fc.png
        """
        groups = results_df[contrast_col].unique().tolist()

        results_df["abs_log_fc"] = results_df[log2fc_col].abs()

        def _get_significance(p_val):
            if p_val < 0.001:
                return "< 0.001"
            elif p_val < 0.01:
                return "< 0.01"
            elif p_val < 0.1:
                return "< 0.1"
            else:
                return "n.s."

        results_df["significance"] = results_df[pvalue_col].apply(_get_significance)

        var_names = []
        for group in groups:
            var_names += (
                results_df[results_df[contrast_col] == group]
                .sort_values("abs_log_fc", ascending=False)
                .head(n_top_vars)[symbol_col]
                .tolist()
            )

        results_df = results_df[results_df[symbol_col].isin(var_names)]
        df = results_df.pivot(index=contrast_col, columns=symbol_col, values=log2fc_col)[var_names]

        plt.figure(figsize=figsize)
        sns.heatmap(df, **heatmap_kwargs, cmap="coolwarm", center=0, cbar_kws={"label": "Log2 fold change"})

        _size = {"< 0.001": marker_size, "< 0.01": math.floor(marker_size / 2), "< 0.1": math.floor(marker_size / 4)}
        x_locs, x_labels = plt.xticks()[0], [label.get_text() for label in plt.xticks()[1]]
        y_locs, y_labels = plt.yticks()[0], [label.get_text() for label in plt.yticks()[1]]

        for _i, row in results_df.iterrows():
            if row["significance"] != "n.s.":
                plt.scatter(
                    x=x_locs[x_labels.index(row[symbol_col])],
                    y=y_locs[y_labels.index(row[contrast_col])],
                    s=_size[row["significance"]],
                    marker="*",
                    c="white",
                )

        plt.scatter([], [], s=marker_size, marker="*", c="black", label="< 0.001")
        plt.scatter([], [], s=math.floor(marker_size / 2), marker="*", c="black", label="< 0.01")
        plt.scatter([], [], s=math.floor(marker_size / 4), marker="*", c="black", label="< 0.1")
        plt.legend(title="Significance", bbox_to_anchor=(1.2, -0.05))

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if return_fig:
            return plt.gcf()
        plt.show()
        return None


class LinearModelBase(MethodBase):
    def __init__(self, adata, design, *, mask=None, layer=None, **kwargs):
        """Initialize the method.

        Args:
            adata: AnnData object, usually pseudobulked.
            design: Model design. Can be either a design matrix, a formulaic formula.Formulaic formula in the format 'x + z' or '~x+z'.
            mask: A column in adata.var that contains a boolean mask with selected features.
            layer: Layer to use in fit(). If None, use the X array.
            **kwargs: Keyword arguments specific to the method implementation.
        """
        super().__init__(adata, mask=mask, layer=layer)
        self._check_counts()

        self.formulaic_contrasts = None
        if isinstance(design, str):
            self.formulaic_contrasts = FormulaicContrasts(adata.obs, design)
            self.design = self.formulaic_contrasts.design_matrix
        else:
            self.design = design

    @classmethod
    def compare_groups(
        cls,
        adata: ad.AnnData,
        column: str,
        baseline: str,
        groups_to_compare: str | Iterable[str],
        *,
        paired_by: str | None = None,
        mask: pd.Series | None = None,
        layer: str | None = None,
        fit_kwargs=MappingProxyType({}),
        test_kwargs=MappingProxyType({}),
    ):
        design = f"~{column}"
        if paired_by is not None:
            design += f"+{paired_by}"
        if isinstance(groups_to_compare, str):
            groups_to_compare = [groups_to_compare]
        model = cls(adata, design=design, mask=mask, layer=layer)

        model.fit(**fit_kwargs)

        de_res = model.test_contrasts(
            {
                group_to_compare: model.contrast(column=column, baseline=baseline, group_to_compare=group_to_compare)
                for group_to_compare in groups_to_compare
            },
            **test_kwargs,
        )

        return de_res

    @property
    def variables(self):
        """Get the names of the variables used in the model definition."""
        if self.formulaic_contrasts is None:
            raise ValueError(
                "Retrieving variables is only possible if the model was initialized using a formula."
            ) from None
        else:
            return self.formulaic_contrasts.variables

    @abstractmethod
    def _check_counts(self):
        """Check that counts are valid for the specific method.

        Raises:
            ValueError: if the data matrix does not comply with the expectations.
        """
        ...

    @abstractmethod
    def fit(self, **kwargs):
        """Fit the model.

        Args:
            **kwargs: Additional arguments for fitting the specific method.
        """
        ...

    @abstractmethod
    def _test_single_contrast(self, contrast, **kwargs): ...

    def test_contrasts(self, contrasts: np.ndarray | Mapping[str | None, np.ndarray], **kwargs):
        """Perform a comparison as specified in a contrast vector.

        Args:
            contrasts: Either a numeric contrast vector, or a dictionary of numeric contrast vectors.
            **kwargs: passed to the respective implementation.

        Returns:
            A dataframe with the results.
        """
        if not isinstance(contrasts, dict):
            contrasts = {None: contrasts}
        results = []
        for name, contrast in contrasts.items():
            results.append(self._test_single_contrast(contrast, **kwargs).assign(contrast=name))

        results_df = pd.concat(results)

        return results_df

    def test_reduced(self, modelB):
        """Test against a reduced model.

        Args:
            modelB: the reduced model against which to test.

        Example:
            >>> import pertpy as pt
            >>> modelA = Model().fit()
            >>> modelB = Model().fit()
            >>> modelA.test_reduced(modelB)
        """
        raise NotImplementedError

    def cond(self, **kwargs):
        """Get a contrast vector representing a specific condition.

        Args:
            **kwargs: column/value pairs.

        Returns:
            A contrast vector that aligns to the columns of the design matrix.
        """
        if self.formulaic_contrasts is None:
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a formulaic formula. Please manually provide a contrast vector."
            )
        return self.formulaic_contrasts.cond(**kwargs)

    def contrast(self, *args, **kwargs):  # noqa: D417
        """Build a simple contrast for pairwise comparisons.

        Args:
            column: column in adata.obs to test on.
            baseline: baseline category (denominator).
            group_to_compare: category to compare against baseline (nominator).

        Returns:
            Numeric contrast vector.
        """
        if self.formulaic_contrasts is None:
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a formulaic formula. Please manually provide a contrast vector."
            )
        return self.formulaic_contrasts.contrast(*args, **kwargs)
