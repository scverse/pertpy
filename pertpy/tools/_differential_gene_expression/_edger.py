from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.pyplot import Figure
from scipy.sparse import issparse

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._logger import logger

from ._base import LinearModelBase
from ._checks import check_is_integer_matrix


class EdgeR(LinearModelBase):
    """Differential expression test using EdgeR."""

    def _check_counts(self):
        check_is_integer_matrix(self.data)

    def fit(self, **kwargs):  # adata, design, mask, layer
        """Fit model using edgeR.

        Note: this creates its own AnnData object for downstream.

        Args:
            **kwargs: Keyword arguments specific to glmQLFit()
        """
        ro, edger = self._ensure_deps("ro", "edger")

        if not hasattr(self, "dge") or not hasattr(self, "design_r"):
            self._prepare_dge()

        logger.info("Fitting linear model")
        fit = edger.glmQLFit(self.dge, design=self.design_r, **kwargs)

        ro.globalenv["fit"] = fit
        self.fit = fit

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_bcv(  # pragma: no cover # noqa: D417
        self,
        *,
        xlabel: str | None = "Average log CPM",
        ylabel: str | None = "Biological coefficient of variation",
        marker: str = "o",
        point_size: float = 0.2,
        common_col: str = "red",
        trend_col: str = "blue",
        tagwise_col: str = "black",
        legend: bool = True,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Plot biological coefficient of variation (BCV) like edgeR::plotBCV.

        Args:
            xlabel: Label for the x-axis (default: "Average log CPM").
            ylabel: Label for the y-axis (default: "Biological coefficient of variation").
            marker: Marker style.
            point_size: Scaling factor for point sizes.
            common_col: Color for common dispersion line.
            trend_col: Color for trended dispersion line.
            tagwise_col: Color for tagwise dispersion points.
            legend: Whether to draw a legend.
            {common_plot_args}
            **kwargs: Additional arguments for ax.scatter and ax.axhline.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import decoupler as dc
            >>> adata = pt.dt.zhang_2021()
            >>> adata = adata[adata.obs["Origin"] == "t", :].copy()
            >>> adata.layers["counts"] = adata.X.copy()
            >>> pdata = dc.pp.pseudobulk(adata, sample_col="Patient", groups_col="Cluster", layer="counts", mode="sum")
            >>> dc.pp.filter_samples(pdata, inplace=True)
            >>> edgr = pt.tl.EdgeR(pdata, design="~Efficacy+Treatment")
            >>> edgr.plot_bcv()

        Preview:
            .. image:: /_static/docstring_previews/de_plot_bcv.png
        """
        if not hasattr(self, "dge"):
            self._prepare_dge()

        numpy2ri, get_conversion, localconverter = self._ensure_deps("numpy2ri", "get_conversion", "localconverter")

        with localconverter(get_conversion() + numpy2ri.converter):
            A = np.asarray(self.dge.rx2("AveLogCPM"))
            tagwise = np.asarray(self.dge.rx2("tagwise.dispersion"))
            common = float(self.dge.rx2("common.dispersion")[0])
            trended = np.asarray(self.dge.rx2("trended.dispersion"))

        fig, ax = plt.subplots(dpi=300)

        ax.scatter(A, np.sqrt(tagwise), c=tagwise_col, s=point_size * 20, marker=marker, linewidths=0, **kwargs)

        ax.axhline(np.sqrt(common), color=common_col, linewidth=2, **kwargs)

        order = np.argsort(A)

        ax.plot(A[order], np.sqrt(trended)[order], color=trend_col, linewidth=2, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if legend:
            handles = [
                Line2D([0], [0], marker=marker, linestyle="", color=tagwise_col, label="Tagwise"),
                Line2D([0], [0], linestyle="-", color=common_col, label="Common"),
                Line2D([0], [0], linestyle="-", color=trend_col, label="Trend"),
            ]
            ax.legend(handles=handles, loc="upper right", frameon=True)

        plt.tight_layout()

        if return_fig:
            return fig

        plt.show()
        return None

    def _prepare_dge(self) -> None:
        """Create DGEList, calculate normalization factors, and estimate dispersions."""
        numpy2ri, pandas2ri, get_conversion, localconverter, edger = self._ensure_deps(
            "numpy2ri", "pandas2ri", "get_conversion", "localconverter", "edger"
        )

        # Convert dataframe
        with localconverter(get_conversion() + numpy2ri.converter):
            expr = self.adata.X if self.layer is None else self.adata.layers[self.layer]
            expr = expr.T.toarray() if issparse(expr) else expr.T

        with localconverter(get_conversion() + pandas2ri.converter) as cv:
            expr_r = cv.py2rpy(pd.DataFrame(expr, index=self.adata.var_names, columns=self.adata.obs_names))
            samples_r = cv.py2rpy(self.adata.obs)

        dge = edger.DGEList(counts=expr_r, samples=samples_r)

        logger.info("Calculating NormFactors")
        dge = edger.calcNormFactors(dge)

        with localconverter(get_conversion() + numpy2ri.converter) as cv:
            design_r = cv.py2rpy(self.design.values)

        logger.info("Estimating Dispersions")
        dge = edger.estimateDisp(dge, design=design_r)

        self.dge = dge
        self.design_r = design_r

    def _test_single_contrast(self, contrast: Sequence[float], **kwargs) -> pd.DataFrame:  # noqa: D417
        """Conduct test for each contrast and return a data frame.

        Args:
            contrast: numpy array of integars indicating contrast i.e. [-1, 0, 1, 0, 0]
        """
        ## -- Check installations
        # For running in notebook
        # pandas2ri.activate()
        # rpy2.robjects.numpy2ri.activate()

        # ToDo:
        #  parse **kwargs to R function
        #  Fix mask for .fit()

        ro, numpy2ri, pandas2ri, get_conversion, localconverter = self._ensure_deps(
            "ro", "numpy2ri", "pandas2ri", "get_conversion", "localconverter"
        )

        # Convert vector to R, which drops a category like `self.design_matrix` to use the intercept for the left out.
        with localconverter(get_conversion() + numpy2ri.converter) as cv:
            contrast_vec_r = cv.py2rpy(np.asarray(contrast))
        ro.globalenv["contrast_vec"] = contrast_vec_r

        # Test contrast with R
        ro.r(
            """
            test = edgeR::glmQLFTest(fit, contrast=contrast_vec)
            de_res =  edgeR::topTags(test, n=Inf, adjust.method="BH", sort.by="PValue")$table
            """
        )

        # Retrieve the `de_res` object
        de_res = ro.globalenv["de_res"]

        # If already a Pandas DataFrame, return it directly
        if isinstance(de_res, pd.DataFrame):
            de_res.index.name = "variable"
            return de_res.reset_index().rename(columns={"PValue": "p_value", "logFC": "log_fc", "FDR": "adj_p_value"})

        # Convert to Pandas DataFrame if still an R object
        with localconverter(get_conversion() + pandas2ri.converter) as cv:
            de_res = cv.rpy2py(de_res)

        de_res.index.name = "variable"
        de_res = de_res.reset_index()

        return de_res.rename(columns={"PValue": "p_value", "logFC": "log_fc", "FDR": "adj_p_value"})

    def _ensure_deps(self, *names):
        """Lazy loader for rpy2 objects with per-instance caching.

        Example:
            ro, numpy2ri, edger = self._ensure_deps("ro", "numpy2ri", "edger")
        """
        if not hasattr(self, "_imports_cache"):
            try:
                from rpy2 import robjects as ro
                from rpy2.robjects import numpy2ri, pandas2ri
                from rpy2.robjects.conversion import get_conversion, localconverter
                from rpy2.robjects.packages import importr

            except ImportError:
                raise ImportError("edger requires rpy2 to be installed.") from None

            try:
                edger = importr("edgeR")
            except ImportError as e:
                raise ImportError(
                    "edgeR requires a valid R installation with the following packages:\nedgeR, BiocParallel, RhpcBLASctl"
                ) from e

            self._imports_cache = {
                "ro": ro,
                "numpy2ri": numpy2ri,
                "pandas2ri": pandas2ri,
                "get_conversion": get_conversion,
                "localconverter": localconverter,
                "edger": edger,
            }

        results = {}

        for name in names:
            if name in self._imports_cache:
                results[name] = self._imports_cache[name]
            else:
                raise KeyError(f"Unknown import request: '{name}'")

        return tuple(results[name] for name in names)
