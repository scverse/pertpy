import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from numpy import ndarray
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from scipy.sparse import issparse

from pertpy._doc import _doc_params, doc_common_plot_args

from ._base import LinearModelBase
from ._checks import check_is_integer_matrix

warnings.filterwarnings("always", message=".*(pval_thresh|pvalue_col|log2fc_thresh).*")


class PyDESeq2(LinearModelBase):
    """Differential expression test using a PyDESeq2."""

    def __init__(
        self, adata: AnnData, design: str | ndarray, *, mask: str | None = None, layer: str | None = None, **kwargs
    ):
        super().__init__(adata, design, mask=mask, layer=layer, **kwargs)
        # work around pydeseq2 issue with sparse matrices
        # see also https://github.com/owkin/PyDESeq2/issues/25
        if issparse(self.data):
            if self.layer is None:
                self.adata.X = self.adata.X.toarray()
            else:
                self.adata.layers[self.layer] = self.adata.layers[self.layer].toarray()

    def _check_counts(self):
        check_is_integer_matrix(self.data)

    def fit(self, **kwargs) -> pd.DataFrame:
        """Fit dds model using pydeseq2.

        Note: this creates its own AnnData object for downstream processing.

        Args:
            **kwargs: Keyword arguments specific to DeseqDataSet(), except for `n_cpus` which will use all available CPUs minus one if the argument is not passed.
        """
        try:
            usable_cpus = len(os.sched_getaffinity(0))  # type: ignore # os.sched_getaffinity is not available on Windows and macOS
        except AttributeError:
            usable_cpus = os.cpu_count() or 1

        inference = DefaultInference(n_cpus=kwargs.pop("n_cpus", usable_cpus))

        adata_for_dds = self.adata
        if self.layer is not None:
            # pydeseq2 always uses `adata.X` as the count matrix, so ensure that `X`
            # reflects the chosen layer without mutating the user-provided `.X`.
            adata_for_dds = AnnData(X=self.data, obs=self.adata.obs, var=self.adata.var)

        dds = DeseqDataSet(
            adata=adata_for_dds,
            design=self.design,  # initialize using design matrix, not formula
            refit_cooks=True,
            inference=inference,
            **kwargs,
        )

        dds.deseq2()
        self.dds = dds

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_disp_ests(
        self,
        *,
        ymin: float | None = None,
        cv: bool = False,
        gene_col: str = "black",
        fit_col: str = "red",
        final_col: str = "dodgerblue",
        legend: bool = True,
        xlabel: str | None = None,
        ylabel: str | None = None,
        log: str = "xy",
        s: float = 0.45,
        return_fig: bool = False,
        **kwargs,
    ) -> None:
        """Plot dispersion estimates.

        A helper function that visualizes per-gene dispersion estimates together
        with the fitted mean–dispersion relationship.

        This plot shows:
        - gene-wise dispersion estimates
        - fitted dispersion trend
        - final dispersion estimates used for testing

        Optionally, the square root of dispersion (coefficient of variation) can
        be shown instead of raw dispersion values.

        Args:
            ymin: Lower bound for plotted values. Points below this threshold
                are clipped and drawn at ymin using triangle markers.
            cv: If True, plot the square root of dispersion (coefficient of
                variation) instead of dispersion.
            gene_col: Color for gene-wise dispersion estimates.
            fit_col: Color for fitted dispersion trend.
            final_col: Color for final dispersion estimates used for testing.
            legend: Whether to draw a legend.
            xlabel: Label for the x-axis (defaults to "mean of normalized counts").
            ylabel: Label for the y-axis (defaults to dispersion or CV).
            log: Axis scaling. "x", "y", or "xy" for log scaling.
            s: Scaling factor for point sizes.
            return_fig: If True, returns the matplotlib figure instead of None.
            {common_plot_args}
            **kwargs: Additional arguments for TODO

        Returns:
            matplotlib.figure.Figure | None

        Examples:
            >>> model.plot_disp_ests()

        Preview:
            .. image:: /_static/docstring_previews/dispersion_estimates.png
        """
        if not hasattr(self, "dds"):
            raise ValueError("Model not fitted yet. Call .fit() first.")

        dds = self.dds

        genecol = gene_col
        fitcol = fit_col
        finalcol = final_col

        if xlabel is None:
            xlabel = "mean of normalized counts"
        if ylabel is None:
            ylabel = "coefficient of variation" if cv else "dispersion"

        px = np.asarray(dds.var["_normed_means"])
        sel = px > 0
        px = px[sel]

        py = np.asarray(dds.var["genewise_dispersions"])[sel]
        if cv:
            py = np.sqrt(py)

        if ymin is None:
            positive = py[(py > 0) & np.isfinite(py)]
            ymin = 1e-08 if positive.size == 0 else 10 ** np.floor(np.log10(np.min(positive)) - 0.1)

        py_plot = np.maximum(py, ymin)

        fig, ax = plt.subplots(dpi=300)

        below = py < ymin
        above = ~below

        if above.any():
            ax.scatter(
                px[above],
                py_plot[above],
                facecolor=genecol,
                edgecolors="none",
                s=s * 20,
                marker="o",
                **kwargs,
            )

        if below.any():
            ax.scatter(
                px[below],
                py_plot[below],
                c=genecol,
                s=s * 20,
                marker="v",
                **kwargs,
            )

        outliers = np.asarray(
            dds.var.get(
                "_outlier_genes",
                pd.Series(False, index=dds.var_names),
            )
        )[sel]

        final_disp = np.asarray(dds.var["dispersions"])[sel]
        final_y = np.sqrt(final_disp) if cv else final_disp

        ax.scatter(
            px,
            final_y,
            s=s * (20 + 20 * outliers.astype(int)),
            facecolor=np.where(outliers, "none", finalcol),
            edgecolors=np.where(outliers, finalcol, "none"),
        )

        fitted_disp = np.asarray(dds.var["fitted_dispersions"])[sel]
        fitted_y = np.sqrt(fitted_disp) if cv else fitted_disp

        ax.scatter(
            px,
            fitted_y,
            facecolor=fitcol,
            edgecolors="none",
            marker="o",
            s=s * 20,
        )

        if "x" in log:
            ax.set_xscale("log")
        if "y" in log:
            ax.set_yscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if legend:
            from matplotlib.lines import Line2D

            handles = [
                Line2D([0], [0], marker="o", linestyle="", color=genecol, label="gene-est"),
                Line2D([0], [0], marker="o", linestyle="", color=fitcol, label="fitted"),
                Line2D([0], [0], marker="o", linestyle="", color=finalcol, label="final"),
            ]
            ax.legend(handles=handles, loc="lower right", frameon=True)

        plt.tight_layout()

        if return_fig:
            return plt.gcf()

        plt.show()
        return None

    def _test_single_contrast(self, contrast, alpha=0.05, *, lfc_shrink=None, **kwargs) -> pd.DataFrame:
        """Conduct a specific test and returns a Pandas DataFrame.

        Args:
            contrast: list of three strings of the form `["variable", "tested level", "reference level"]`.
            alpha: p value threshold used for controlling fdr with independent hypothesis weighting
            lfc_shrink: If given, apply apeGLM LFC shrinkage to the named coefficient (must be a column of the fitted `DeseqStats.LFC` matrix).
                Opt-in because apeGLM shrinks individual coefficients rather than arbitrary contrasts, so the right coefficient depends on the design and cannot be inferred from a numerical contrast vector.
            **kwargs: extra arguments to pass to DeseqStats()
        """
        contrast = np.array(contrast)
        stat_res = DeseqStats(self.dds, contrast=contrast, alpha=alpha, **kwargs)
        # Calling `.summary()` is required to fill the `results_df` data frame
        stat_res.summary()
        if lfc_shrink is not None:
            stat_res.lfc_shrink(coeff=lfc_shrink)
        res_df = (
            pd.DataFrame(stat_res.results_df)
            .rename(columns={"pvalue": "p_value", "padj": "adj_p_value", "log2FoldChange": "log_fc"})
            .sort_values("p_value")
        )
        res_df.index.name = "variable"
        res_df = res_df.reset_index()
        return res_df
