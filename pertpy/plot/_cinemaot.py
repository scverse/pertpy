from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scanpy.plotting import _utils

from typing import Optional

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

class CinemaotPlot:
    """Plotting functions for CINEMA-OT. Only includes new functions beyond the scanpy.pl.embedding family."""

    @staticmethod
    def vis_matching(
        adata: AnnData,
        de: AnnData,
        pert_key: str,
        control: str,
        de_label: str,
        source_label: str,
        matching_rep: str = 'ot',
        resolution: float = 0.5,
        normalize: str = 'col',
        title: str = "CINEMA-OT matching matrix",
        min_val: float = 0.01,
        show: bool = True,
        save: Optional[str] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> None:
        """Visualize the CINEMA-OT matching matrix. 

        Args:
            adata: the original anndata after running cinemaot.causaleffect or cinemaot.causaleffect_weighted.
            de: The anndata output from Cinemaot.causaleffect() or Cinemaot.causaleffect_weighted().
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            de_label: the label for differential response. If none, use leiden cluster labels at resolution 1.0.
            source_label: the confounder / cell type label.
            matching_rep: the place that stores the matching matrix. default de.obsm['ot'].
            normalize: normalize the coarse-grained matching matrix by row / column.
            title: the title for the figure.
            min_val: The min value to truncate the matching matrix.
            show: Show the plot, do not return axis.
            save: If `True` or a `str`, save the figure. A string is appended to the default filename.
                Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
            **kwargs: Other parameters to input for seaborn.heatmap.

        """
        adata_ = adata[adata.obs[pert_key]==control]

        df = pd.DataFrame(de.obsm[matching_rep])
        if de_label is None:
            de_label = 'leiden'
            sc.pp.neighbors(de,use_rep='X_embedding')
            sc.tl.leiden(de,resolution=resolution)
        df['de_label'] = de.obs[de_label].astype(str).values
        df['de_label'] = 'Response ' + df['de_label']
        df = df.groupby('de_label').sum().T
        df['source_label'] = adata_.obs[source_label].astype(str).values
        df = df.groupby('source_label').sum()

        if normalize == 'col':
            df = df/df.sum(axis=0)
        else:
            df = (df.T/df.sum(axis=1)).T
        df = df.clip(lower=min_val) - min_val
        if normalize == 'col':
            df = df/df.sum(axis=0)
        else:
            df = (df.T/df.sum(axis=1)).T

        g = sns.heatmap(df,annot=True,ax=ax,**kwargs)
        plt.title(title)
        _utils.savefig_or_show("matching_heatmap", show=show, save=save)
        if not show:
            if ax is not None:
                return ax
            else:
                return g