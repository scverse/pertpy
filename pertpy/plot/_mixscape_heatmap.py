from __future__ import annotations

import scanpy as sc
from anndata import AnnData


def mixscape_heatmap(
    adata: AnnData,
    labels: str,
    target_gene: str,
    layer: str | None = None,
    control: str | None = "NT",
    method: str | None = "wilcoxon",
    subsample_number: int | None = 900,
    vmin: float | None = -2,
    vmax: float | None = 2,
    show: bool | None = None,
    save: bool | str | None = None,
    **kwds,
):
    """Heatmap plot using mixscape results. Requires `pt.tl.mixscape()` to be run first.

    Args:
        adata: The annotated data object.
        labels: The column of `.obs` with target gene labels.
        target_gene: Target gene name to visualize heatmap for.
        layer: Key from `adata.layers` whose value will be used to perform tests on.
        control: Control category from the `pert_key` column. Default is 'NT'.
        method: The default method is 'wilcoxon', see `method` parameter in `scanpy.tl.rank_genes_groups` for more options.
        subsample_number: Subsample to this number of observations.
        vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
        vmax: The value representing the upper limit of the color scale. Values larger than vmax are plotted with the same color as vmax.
        show: Show the plot, do not return axis.
        save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
        ax: A matplotlib axes object. Only works if plotting a single component.
        **kwds: Additional arguments to `scanpy.pl.rank_genes_groups_heatmap`.
    """
    adata_subset = adata[(adata.obs[labels] == target_gene) | (adata.obs[labels] == control)].copy()
    sc.tl.rank_genes_groups(adata_subset, layer=layer, groupby=labels, method=method)
    sc.pp.scale(adata_subset, max_value=vmax)
    sc.pp.subsample(adata_subset, n_obs=subsample_number)
    return sc.pl.rank_genes_groups_heatmap(
        adata_subset,
        groupby="mixscape_class",
        vmin=vmin,
        vmax=vmax,
        n_genes=20,
        groups=["NT"],
        show=show,
        save=save,
        **kwds,
    )
