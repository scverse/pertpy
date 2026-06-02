from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.pyplot import Figure
from scipy.sparse import issparse

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

        logger.info("Fitting linear model")
        fit = edger.glmQLFit(dge, design=design_r, **kwargs)

        ro.globalenv["fit"] = fit
        self.fit = fit

    def plot_bcv(
        self,
        *,
        xlab: str = "Average log CPM",
        ylab: str = "Biological coefficient of variation",
        pch: str = "o",
        cex: float = 0.2,
        col_common: str = "red",
        col_trend: str = "blue",
        col_tagwise: str = "black",
        legend: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plot biological coefficient of variation (BCV) like edgeR::plotBCV.

        Must be called after `.fit()`.

        Args:
            xlab:
                X-axis label.
            ylab:
                Y-axis label.
            pch:
                Marker style (matplotlib-compatible).
            cex:
                Point size scaling (similar to R cex).
            col_common:
                Color for common dispersion line.
            col_trend:
                Color for trended dispersion line.
            col_tagwise:
                Color for tagwise dispersion points.
            legend:
                Whether to show legend.
            return_fig:
                If True, return matplotlib figure instead of showing it.
        """
        if not hasattr(self, "dge"):
            raise ValueError("Model not fitted yet. Call `.fit()` first.")

        # dge = self.dge

        try:
            from rpy2 import robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import get_conversion, localconverter
            from rpy2.robjects.packages import importr

        except ImportError:
            raise ImportError("edger requires rpy2 to be installed.") from None

        with localconverter(get_conversion() + numpy2ri.converter):
            A = np.asarray(self.dge.rx2("AveLogCPM"))
            tagwise = np.asarray(self.dge.rx2("tagwise.dispersion")) if "tagwise.dispersion" in self.dge.names else None
            trended = np.asarray(self.dge.rx2("trended.dispersion")) if "trended.dispersion" in self.dge.names else None
            common = np.asarray(self.dge.rx2("common.dispersion"))[0] if "common.dispersion" in self.dge.names else None

        if common is None and trended is None and tagwise is None:
            raise ValueError("No dispersions found in DGEList. Did you run estimateDisp()?")

        # ensure correct shapes
        A = np.asarray(A)

        # -----------------------------
        # Figure setup
        # -----------------------------
        fig, ax = plt.subplots(dpi=300)

        labels = []

        # -----------------------------
        # Tagwise dispersion
        # -----------------------------
        if tagwise is not None:
            tagwise = np.asarray(tagwise)

            ax.scatter(
                A,
                np.sqrt(tagwise),
                c=col_tagwise,
                s=cex * 20,
                marker=pch,
                linewidths=0,
            )
            labels.append("Tagwise")

        # -----------------------------
        # Common dispersion
        # -----------------------------
        if common is not None:
            common_val = float(np.asarray(common).reshape(-1)[0])
            ax.axhline(
                np.sqrt(common_val),
                color=col_common,
                linewidth=2,
            )
            labels.append("Common")

        # -----------------------------
        # Trended dispersion
        # -----------------------------
        if trended is not None:
            trended = np.asarray(trended)
            order = np.argsort(A)

            ax.plot(
                A[order],
                np.sqrt(trended)[order],
                color=col_trend,
                linewidth=2,
            )
            labels.append("Trend")

        # -----------------------------
        # Axes styling
        # -----------------------------
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        # -----------------------------
        # Legend (R-style)
        # -----------------------------
        if legend:
            handles = []

            if tagwise is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=pch,
                        linestyle="",
                        color=col_tagwise,
                        label="Tagwise",
                    )
                )

            if common is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color=col_common,
                        label="Common",
                    )
                )

            if trended is not None:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color=col_trend,
                        label="Trend",
                    )
                )

            ax.legend(handles=handles, loc="upper right", frameon=True)

        plt.tight_layout()

        if return_fig:
            return fig

        plt.show()
        return None

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

        try:
            from rpy2 import robjects as ro
            from rpy2.robjects import numpy2ri, pandas2ri
            from rpy2.robjects.conversion import get_conversion, localconverter
            from rpy2.robjects.packages import importr

        except ImportError:
            raise ImportError("edger requires rpy2 to be installed.") from None

        try:
            importr("edgeR")
        except ImportError:
            raise ImportError(
                "edgeR requires a valid R installation with the following packages: edgeR, BiocParallel, RhpcBLASctl"
            ) from None

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
