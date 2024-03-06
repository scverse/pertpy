from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mudata import MuData


class MilopyPlot:
    """Plotting functions for Milopy."""

    @staticmethod
    def nhood_graph(
        mdata: MuData,
        alpha: float = 0.1,
        min_logFC: float = 0,
        min_size: int = 10,
        plot_edges: bool = False,
        title: str = "DA log-Fold Change",
        show: bool | None = None,
        save: bool | str | None = None,
        **kwargs,
    ) -> None:
        """Visualize DA results on abstracted graph (wrapper around sc.pl.embedding)

        Args:
            mdata: MuData object
            alpha: Significance threshold. (default: 0.1)
            min_logFC: Minimum absolute log-Fold Change to show results. If is 0, show all significant neighbourhoods. (default: 0)
            min_size: Minimum size of nodes in visualization. (default: 10)
            plot_edges: If edges for neighbourhood overlaps whould be plotted. Defaults to False.
            title: Plot title. Defaults to "DA log-Fold Change".
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename.
                  Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata,
            >>>            design='~label',
            >>>            model_contrasts='labelwithdraw_15d_Cocaine-labelwithdraw_48h_Cocaine')
            >>> milo.build_nhood_graph(mdata)
            >>> milo.plot_nhood_graph(mdata)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Milo

        milo = Milo()

        return milo.plot_nhood_graph(
            madata=mdata,
            alpha=alpha,
            min_logFC=min_logFC,
            min_size=min_size,
            plot_edges=plot_edges,
            title=title,
            show=show,
            save=save,
            **kwargs,
        )

    @staticmethod
    def nhood(
        mdata: MuData,
        ix: int,
        feature_key: str | None = "rna",
        basis="X_umap",
        show: bool | None = None,
        save: bool | str | None = None,
        **kwargs,
    ) -> None:
        """Visualize cells in a neighbourhood.

        Args:
            mdata: MuData object with feature_key slot, storing neighbourhood assignments in `mdata[feature_key].obsm['nhoods']`
            ix: index of neighbourhood to visualize
            basis: Embedding to use for visualization. Defaults to "X_umap".
            show: Show the plot, do not return axis.
            save: If True or a str, save the figure. A string is appended to the default filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
            **kwargs: Additional arguments to `scanpy.pl.embedding`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> sc.tl.umap(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> pt.pl.milo.nhood(mdata, ix=0)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Milo

        milo = Milo()

        milo.plot_nhood(mdata=mdata, ix=ix, feature_key=feature_key, basis=basis, show=show, save=save, **kwargs)

    @staticmethod
    def da_beeswarm(
        mdata: MuData,
        feature_key: str | None = "rna",
        anno_col: str = "nhood_annotation",
        alpha: float = 0.1,
        subset_nhoods: list[str] = None,
        palette: str | Sequence[str] | dict[str, str] | None = None,
    ):
        """Plot beeswarm plot of logFC against nhood labels

        Args:
            mdata: MuData object
            anno_col: Column in mdata['milo'].var to use as annotation. (default: 'nhood_annotation'.)
            alpha: Significance threshold. (default: 0.1)
            subset_nhoods: List of nhoods to plot. If None, plot all nhoods. (default: None)
            palette: Name of Seaborn color palette for violinplots.
                     Defaults to pre-defined category colors for violinplots.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.bhattacherjee()
            >>> milo = pt.tl.Milo()
            >>> mdata = milo.load(adata)
            >>> sc.pp.neighbors(mdata["rna"])
            >>> milo.make_nhoods(mdata["rna"])
            >>> mdata = milo.count_nhoods(mdata, sample_col="orig.ident")
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.annotate_nhoods(mdata, anno_col="cell_type")
            >>> milo.plot_da_beeswarm(mdata)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Milo

        milo = Milo()

        milo.plot_da_beeswarm(
            mdata=mdata,
            feature_key=feature_key,
            anno_col=anno_col,
            alpha=alpha,
            subset_nhoods=subset_nhoods,
            palette=palette,
        )

    @staticmethod
    def nhood_counts_by_cond(
        mdata: MuData,
        test_var: str,
        subset_nhoods: list = None,
        log_counts: bool = False,
    ):
        """Plot boxplot of cell numbers vs condition of interest

        Args:
            mdata: MuData object storing cell level and nhood level information
            test_var: Name of column in adata.obs storing condition of interest (y-axis for boxplot)
            subset_nhoods: List of obs_names for neighbourhoods to include in plot. If None, plot all nhoods. (default: None)
            log_counts: Whether to plot log1p of cell counts. (default: False)
        """
        warnings.warn(
            "This function is deprecated and will be removed in pertpy 0.8.0!"
            " Please use the corresponding 'pt.tl' object",
            FutureWarning,
            stacklevel=2,
        )

        from pertpy.tools import Milo

        milo = Milo()

        milo.plot_nhood_counts_by_cond(
            mdata=mdata, test_var=test_var, subset_nhoods=subset_nhoods, log_counts=log_counts
        )
