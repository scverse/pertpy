import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from seaborn import PairGrid


class DialoguePlot:
    @staticmethod
    def split_violins(
        adata: AnnData,
        split_key: str,
        celltype_key=str,
        split_which: tuple[str, str] = None,
        mcp: str = "mcp_0",
    ) -> plt.Axes:
        """Plots split violin plots for a given MCP and split variable.

        Any cells with a value for split_key not in split_which are removed from the plot.

        Args:
            adata: Annotated data object.
            split_key: Variable in adata.obs used to split the data.
            celltype_key: Key for cell type annotations.
            split_which: Which values of split_key to plot. Required if more than 2 values in split_key.
            mcp: Key for MCP data. Defaults to "mcp_0".

        Returns:
            A :class:`~matplotlib.axes.Axes` object

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> pt.pl.dl.split_violins(adata, split_key='clinical.status', celltype_key='cell.subtypes')
        """
        df = sc.get.obs_df(adata, [celltype_key, mcp, split_key])
        if split_which is None:
            split_which = df[split_key].unique()
        df = df[df[split_key].isin(split_which)]
        df[split_key] = df[split_key].cat.remove_unused_categories()

        ax = sns.violinplot(data=df, x=celltype_key, y=mcp, hue=split_key, split=True)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        return ax

    def pairplot(self, adata: AnnData, celltype_key: str, color: str, sample_id: str, mcp: str = "mcp_0") -> PairGrid:
        """Generate a pairplot visualization for multi-cell perturbation (MCP) data.

        Computes the mean of a specified MCP feature (mcp) for each combination of sample and cell type,
        then creates a pairplot to visualize the relationships between these mean MCP values.

        Args:
            adata: Annotated data object.
            celltype_key: Key in adata.obs containing cell type annotations.
            color: Key in adata.obs for color annotations. This parameter is used as the hue
            sample_id: Key in adata.obs for the sample annotations.
            mcp: Key in adata.obs for MCP feature values. Defaults to "mcp_0".

        Returns:
            Seaborn Pairgrid object.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> adata = pt.dt.dialogue_example()
            >>> sc.pp.pca(adata)
            >>> dl = pt.tl.Dialogue(sample_id = "clinical.status", celltype_key = "cell.subtypes", \
                n_counts_key = "nCount_RNA", n_mpcs = 3)
            >>> adata, mcps, ws, ct_subs = dl.calculate_multifactor_PMD(adata, normalize=True)
            >>> pt.pl.dl.pairplot(adata, celltype_key="cell.subtypes", color="gender", sample_id="clinical.status")
        """
        mean_mcps = adata.obs.groupby([sample_id, celltype_key])[mcp].mean()
        mean_mcps = mean_mcps.reset_index()
        mcp_pivot = pd.pivot(mean_mcps[[sample_id, celltype_key, mcp]], index=sample_id, columns=celltype_key)[mcp]

        aggstats = adata.obs.groupby([sample_id])[color].describe()
        aggstats = aggstats.loc[list(mcp_pivot.index), :]
        aggstats[color] = aggstats["top"]
        mcp_pivot = pd.concat([mcp_pivot, aggstats[color]], axis=1)
        ax = sns.pairplot(mcp_pivot, hue=color, corner=True)

        return ax
