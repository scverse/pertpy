from __future__ import annotations

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
            >>> milo.da_nhoods(mdata, design="~label")
            >>> milo.build_nhood_graph(mdata)
            >>> pt.pl.milo.nhood_graph(mdata)
            # TODO: If necessary adjust after fixing StopIteration error, which is currently thrown
        """
        nhood_adata = mdata["milo"].T.copy()

        if "Nhood_size" not in nhood_adata.obs.columns:
            raise KeyError(
                'Cannot find "Nhood_size" column in adata.uns["nhood_adata"].obs -- \
                    please run milopy.utils.build_nhood_graph(adata)'
            )

        nhood_adata.obs["graph_color"] = nhood_adata.obs["logFC"]
        nhood_adata.obs.loc[nhood_adata.obs["SpatialFDR"] > alpha, "graph_color"] = np.nan
        nhood_adata.obs["abs_logFC"] = abs(nhood_adata.obs["logFC"])
        nhood_adata.obs.loc[nhood_adata.obs["abs_logFC"] < min_logFC, "graph_color"] = np.nan

        # Plotting order - extreme logFC on top
        nhood_adata.obs.loc[nhood_adata.obs["graph_color"].isna(), "abs_logFC"] = np.nan
        ordered = nhood_adata.obs.sort_values("abs_logFC", na_position="first").index
        nhood_adata = nhood_adata[ordered]

        vmax = np.max([nhood_adata.obs["graph_color"].max(), abs(nhood_adata.obs["graph_color"].min())])
        vmin = -vmax

        sc.pl.embedding(
            nhood_adata,
            "X_milo_graph",
            color="graph_color",
            cmap="RdBu_r",
            size=nhood_adata.obs["Nhood_size"] * min_size,
            edges=plot_edges,
            neighbors_key="nhood",
            sort_order=False,
            frameon=False,
            vmax=vmax,
            vmin=vmin,
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

        mdata[feature_key].obs["Nhood"] = mdata[feature_key].obsm["nhoods"][:, ix].toarray().ravel()
        sc.pl.embedding(
            mdata[feature_key], basis, color="Nhood", size=30, title="Nhood" + str(ix), show=show, save=save, **kwargs
        )

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
            anno_col: Column in adata.uns['nhood_adata'].obs to use as annotation. (default: 'nhood_annotation'.)
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
            >>> milo.annotate_nhoods(mdata, anno_col='cell_type')
            >>> pt.pl.milo.da_beeswarm(mdata)
        """
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run 'milopy.count_nhoods(adata)' first."
            ) from None

        if subset_nhoods is not None:
            nhood_adata = nhood_adata[subset_nhoods]

        try:
            nhood_adata.obs[anno_col]
        except KeyError:
            raise RuntimeError(
                f"Unable to find {anno_col} in mdata.uns['nhood_adata']. Run 'milopy.utils.annotate_nhoods(adata, anno_col)' first"
            ) from None

        try:
            nhood_adata.obs["logFC"]
        except KeyError:
            raise RuntimeError(
                "Unable to find 'logFC' in mdata.uns['nhood_adata'].obs. Run 'core.da_nhoods(adata)' first."
            ) from None

        sorted_annos = (
            nhood_adata.obs[[anno_col, "logFC"]].groupby(anno_col).median().sort_values("logFC", ascending=True).index
        )

        anno_df = nhood_adata.obs[[anno_col, "logFC", "SpatialFDR"]].copy()
        anno_df["is_signif"] = anno_df["SpatialFDR"] < alpha
        anno_df = anno_df[anno_df[anno_col] != "nan"]

        try:
            obs_col = nhood_adata.uns["annotation_obs"]
            if palette is None:
                palette = dict(
                    zip(mdata[feature_key].obs[obs_col].cat.categories, mdata[feature_key].uns[f"{obs_col}_colors"])
                )
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                size=190,
                inner=None,
                orient="h",
                palette=palette,
                linewidth=0,
                scale="width",
            )
        except BaseException:  # noqa: BLE001
            sns.violinplot(
                data=anno_df,
                y=anno_col,
                x="logFC",
                order=sorted_annos,
                size=190,
                inner=None,
                orient="h",
                linewidth=0,
                scale="width",
            )
        sns.stripplot(
            data=anno_df,
            y=anno_col,
            x="logFC",
            order=sorted_annos,
            size=2,
            hue="is_signif",
            palette=["grey", "black"],
            orient="h",
            alpha=0.5,
        )
        plt.legend(loc="upper left", title=f"< {int(alpha * 100)}% SpatialFDR", bbox_to_anchor=(1, 1), frameon=False)
        plt.axvline(x=0, ymin=0, ymax=1, color="black", linestyle="--")

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
        try:
            nhood_adata = mdata["milo"].T.copy()
        except KeyError:
            raise RuntimeError(
                "mdata should be a MuData object with two slots: feature_key and 'milo'. Run milopy.count_nhoods(mdata) first"
            ) from None

        if subset_nhoods is None:
            subset_nhoods = nhood_adata.obs_names

        pl_df = pd.DataFrame(nhood_adata[subset_nhoods].X.A, columns=nhood_adata.var_names).melt(
            var_name=nhood_adata.uns["sample_col"], value_name="n_cells"
        )
        pl_df = pd.merge(pl_df, nhood_adata.var)
        pl_df["log_n_cells"] = np.log1p(pl_df["n_cells"])
        if not log_counts:
            sns.boxplot(data=pl_df, x=test_var, y="n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="n_cells", color="black", s=3)
            plt.ylabel("# cells")
        else:
            sns.boxplot(data=pl_df, x=test_var, y="log_n_cells", color="lightblue")
            sns.stripplot(data=pl_df, x=test_var, y="log_n_cells", color="black", s=3)
            plt.ylabel("log(# cells + 1)")

        plt.xticks(rotation=90)
        plt.xlabel(test_var)
