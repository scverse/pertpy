import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType

import adjustText
import anndata as ad
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from pertpy.tools._differential_gene_expression._checks import check_is_numeric_matrix
from pertpy.tools._differential_gene_expression._formulaic import (
    AmbiguousAttributeError,
    Factor,
    get_factor_storage_and_materializer,
    resolve_ambiguous,
)


@dataclass
class Contrast:
    """Simple contrast for comparison between groups"""

    column: str
    baseline: str
    group_to_compare: str


ContrastType = Contrast | tuple[str, str, str]


class MethodBase(ABC):
    def __init__(self, adata, *, mask=None, layer=None, **kwargs):
        """
        Initialize the method.

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
            return self.adata.layer[self.layer]

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
        """
        Compare between groups in a specified column.

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
        """
        ...

    def plot_volcano(
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
        save: bool | str | None = None,
        shapes: list[str] | None = None,
        shape_order: list[str] | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        **kwargs: int,
    ) -> None:
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
            save: Saves the plot if True or to the path provided.
            shapes: List of matplotlib marker ids.
            shape_order: Order of categories for shapes.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            **kwargs: Additional arguments for seaborn.scatterplot.
        """
        if colors is None:
            colors = ["gray", "#D62728", "#1F77B4"]

        def _pval_reciprocal(lfc: float) -> float:
            """
            Function for relating -log10(pvalue) and logfoldchange in a reciprocal.

            Used for plotting the S-curve
            """
            return pval_thresh / (lfc - log2fc_thresh)

        def _map_shape(symbol: str) -> str:
            if shape_dict is not None:
                for k in shape_dict.keys():
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
            """
            Map genes to categorize based on log2fc and pvalue.

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
            else:
                # Standard condition for Up or Down categorization
                if log2fc > log2fc_thresh and nlog10 > pval_thresh:
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
            """
            Map genes to categorize based on log2fc and pvalue.

            These categories are used for coloring the dots.
            Used when color_dict is passed, sets DE / not DE for background and user supplied highlight genes.
            """
            log2fc = row[log2fc_col]
            nlog10 = row[nlog10_col]
            symbol = row[symbol_col]

            if color_dict is not None:
                for k in color_dict.keys():
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

            raise NotImplementedError("Anndata not implemented yet")
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

        # TODO replace with scanpy save style
        if save:
            files = os.listdir()
            for x in range(100):
                file_pref = "volcano_" + "%02d" % (x,)
                if len([x for x in files if x.startswith(file_pref)]) == 0:
                    plt.savefig(file_pref + ".png", dpi=300, bbox_inches="tight")
                    plt.savefig(file_pref + ".svg", bbox_inches="tight")
                    break
        elif isinstance(save, str):
            plt.savefig(save + ".png", dpi=300, bbox_inches="tight")
            plt.savefig(save + ".svg", bbox_inches="tight")

        plt.show()


class LinearModelBase(MethodBase):
    def __init__(self, adata, design, *, mask=None, layer=None, **kwargs):
        """
        Initialize the method.

        Args:
            adata: AnnData object, usually pseudobulked.
            design: Model design. Can be either a design matrix, a formulaic formula.Formulaic formula in the format 'x + z' or '~x+z'.
            mask: A column in adata.var that contains a boolean mask with selected features.
            layer: Layer to use in fit(). If None, use the X array.
            **kwargs: Keyword arguments specific to the method implementation.
        """
        super().__init__(adata, mask=mask, layer=layer)
        self._check_counts()

        self.factor_storage = None
        self.variable_to_factors = None

        if isinstance(design, str):
            self.factor_storage, self.variable_to_factors, materializer_class = get_factor_storage_and_materializer()
            self.design = materializer_class(adata.obs, record_factor_metadata=True).get_model_matrix(design)
        else:
            self.design = design

    @classmethod
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
        try:
            return self.design.model_spec.variables_by_source["data"]
        except AttributeError:
            raise ValueError(
                "Retrieving variables is only possible if the model was initialized using a formula."
            ) from None

    @abstractmethod
    def _check_counts(self):
        """
        Check that counts are valid for the specific method.

        Raises:
            ValueError: if the data matrix does not comply with the expectations.
        """
        ...

    @abstractmethod
    def fit(self, **kwargs):
        """
        Fit the model.

        Args:
            **kwargs: Additional arguments for fitting the specific method.
        """
        ...

    @abstractmethod
    def _test_single_contrast(self, contrast, **kwargs): ...

    def test_contrasts(self, contrasts, **kwargs):
        """
        Perform a comparison as specified in a contrast vector.

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
        """
        Test against a reduced model.

        Args:
            modelB: the reduced model against which to test.

        Example:
            modelA = Model().fit()
            modelB = Model().fit()
            modelA.test_reduced(modelB)
        """
        raise NotImplementedError

    def cond(self, **kwargs):
        """
        Get a contrast vector representing a specific condition.

        Args:
            **kwargs: column/value pairs.

        Returns:
            A contrast vector that aligns to the columns of the design matrix.
        """
        if self.factor_storage is None:
            raise RuntimeError(
                "Building contrasts with `cond` only works if you specified the model using a formulaic formula. Please manually provide a contrast vector."
            )
        cond_dict = kwargs
        if not set(cond_dict.keys()).issubset(self.variables):
            raise ValueError(
                "You specified a variable that is not part of the model. Available variables: "
                + ",".join(self.variables)
            )
        for var in self.variables:
            if var in cond_dict:
                self._check_category(var, cond_dict[var])
            else:
                cond_dict[var] = self._get_default_value(var)
        df = pd.DataFrame([kwargs])
        return self.design.model_spec.get_model_matrix(df).iloc[0]

    def _get_factor_metadata_for_variable(self, var):
        factors = self.variable_to_factors[var]
        return list(chain.from_iterable(self.factor_storage[f] for f in factors))

    def _get_default_value(self, var):
        factor_metadata = self._get_factor_metadata_for_variable(var)
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL:
            try:
                tmp_base = resolve_ambiguous(factor_metadata, "base")
            except AmbiguousAttributeError as e:
                raise ValueError(
                    f"Could not automatically resolve base category for variable {var}. Please specify it explicity in `model.cond`."
                ) from e
            return tmp_base if tmp_base is not None else "\0"
        else:
            return 0

    def _check_category(self, var, value):
        factor_metadata = self._get_factor_metadata_for_variable(var)
        tmp_categories = resolve_ambiguous(factor_metadata, "categories")
        if resolve_ambiguous(factor_metadata, "kind") == Factor.Kind.CATEGORICAL and value not in tmp_categories:
            raise ValueError(
                f"You specified a non-existant category for {var}. Possible categories: {', '.join(tmp_categories)}"
            )

    def contrast(self, column, baseline, group_to_compare):
        """
        Build a simple contrast for pairwise comparisons.

        Args:
            column: column in adata.obs to test on.
            baseline: baseline category (denominator).
            group_to_compare: category to compare against baseline (nominator).

        Returns:
            Numeric contrast vector.
        """
        return self.cond(**{column: group_to_compare}) - self.cond(**{column: baseline})
