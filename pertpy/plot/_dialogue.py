import os
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData


class DialoguePlot:
    @staticmethod
    def split_violins(
        adata: AnnData,
        split_key: str,
        celltype_key=str,
        split_which: (str, str) = None,
        mcp: str = "mcp_0",
    ) -> plt.Axes:
        """
        Usage: Plots split violin plots for a given MCP and split variable. Any cells
        with a value for split_key not in split_which are removed from the plot.

        Args:
            adata (AnnData): Annotated data object.
            split_key (str): Variable in adata.obs used to split the data.
            split_which (str,str): Which values of split_key to plot. Required if more than 2 values in split_key.
            celltype_key (str): Key for cell type annotations.
            mcp (str, optional): Key for MCP data. Defaults to "mcp_0".

        Returns:
            A :class:`~matplotlib.axes.Axes` object"""

        # Get a DataFrame with necessary columns
        df = sc.get.obs_df(adata, [celltype_key, mcp, split_key])
        if split_which is None:
            split_which = df[split_key].unique()
        df = df[df[split_key].isin(split_which)]
        df[split_key] = df[split_key].cat.remove_unused_categories()

        # Create split violin plot using Seaborn
        p2 = sns.violinplot(data=df, x=celltype_key, y=mcp, hue=split_key, split=True)

        # rotate labels for readability
        p2.set_xticklabels(p2.get_xticklabels(), rotation=90)

        # Return the matplotlib Axes object
        return p2

    def pairplot(adata: AnnData, celltype_key: str, color: str, sample_id: str, mcp: str = "mcp_0") -> plt.Axes:
        """Generate a pairplot visualization for multi-cell perturbation (MCP) data.
        This function computes the mean of a specified MCP feature (mcp) for each combination of
        sample and cell type, then creates a pairplot to visualize the relationships between these
        mean MCP values.

        Usage: pairplot(adata, celltype_key="Cluster", color="Efficacy", mcp="mcp_0", sample_key="Sample"))

        Args:
            adata: AnnData object
            celltype_key: the key in adata.obs for cell type annotations.
            color: the key in adata.obs for color annotations. This parameter is used as the hue
            sample_id: the key in adata.obs for the sample annotations.
            mcp: the key in adata.obs for MCP feature values. Default is "mcp_0".
        Returns:
            A :class:`~matplotlib.axes.Axes` object"""
        mean_mcps = adata.obs.groupby([sample_id, celltype_key])[mcp].mean()
        mean_mcps = mean_mcps.reset_index()
        mcp_pivot = pd.pivot(mean_mcps[[sample_id, celltype_key, mcp]], index=sample_id, columns=celltype_key)[mcp]
        # now for each sample I want to get the value of the color variable
        aggstats = adata.obs.groupby([sample_id])[color].describe()
        aggstats = aggstats.loc[list(mcp_pivot.index), :]
        aggstats[color] = aggstats["top"]
        mcp_pivot = pd.concat([mcp_pivot, aggstats[color]], axis=1)
        p = sns.pairplot(mcp_pivot, hue=color, corner=True)
        return p
