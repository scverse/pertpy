from __future__ import annotations

from collections import OrderedDict
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib import pyplot as pl
from matplotlib.axes import Axes
from scanpy import get
from scanpy._settings import settings
from scanpy._utils import _check_use_raw, sanitize_anndata
from scanpy.plotting import _utils
from scanpy.plotting._utils import setup_axes


def mixscape_violin(
    adata: AnnData,
    target_gene_idents: str | list[str],
    keys: str | Sequence[str] = "mixscape_class_p_ko",
    groupby: str | None = "mixscape_class",
    log: bool = False,
    use_raw: bool | None = None,
    stripplot: bool = True,
    hue: str | None = None,
    jitter: float | bool = True,
    size: int = 1,
    layer: str | None = None,
    scale: Literal["area", "count", "width"] = "width",
    order: Sequence[str] | None = None,
    multi_panel: bool | None = None,
    xlabel: str = "",
    ylabel: str | Sequence[str] | None = None,
    rotation: float | None = None,
    show: bool | None = None,
    save: bool | str | None = None,
    ax: Axes | None = None,
    **kwds,
):
    """Violin plot using mixscape results. Need to run `pt.tl.mixscape` first.

    Args:
        adata: The annotated data object.
        target_gene: Target gene name to plot.
        keys: Keys for accessing variables of `.var_names` or fields of `.obs`. Default is 'mixscape_class_p_ko'.
        groupby: The key of the observation grouping to consider. Default is 'mixscape_class'.
        order: Order in which to show the categories.
        xlabel: Label of the x axis. Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
        ylabel: Label of the y axis. If `None` and `groupby` is `None`, defaults to `'value'`. If `None` and `groubpy` is not `None`, defaults to `keys`.
        stripplot: Add a stripplot on top of the violin plot.
        log: Plot on logarithmic axis.
        use_raw: Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
        show: Show the plot, do not return axis.
        save: If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
        ax: A matplotlib axes object. Only works if plotting a single component.
        **kwds: Additional arguments to `seaborn.violinplot`.

    Returns:
        A :class:`~matplotlib.axes.Axes` object if `ax` is `None` else `None`.
    """

    if isinstance(target_gene_idents, str):
        mixscape_class_mask = adata.obs[groupby] == target_gene_idents
    elif isinstance(target_gene_idents, list):
        mixscape_class_mask = np.full_like(adata.obs[groupby], False, dtype=bool)
        for ident in target_gene_idents:
            mixscape_class_mask |= adata.obs[groupby] == ident
    adata = adata[mixscape_class_mask]

    import seaborn as sns  # Slow import, only import if called

    sanitize_anndata(adata)
    use_raw = _check_use_raw(adata, use_raw)
    if isinstance(keys, str):
        keys = [keys]
    keys = list(OrderedDict.fromkeys(keys))  # remove duplicates, preserving the order

    if isinstance(ylabel, (str, type(None))):
        ylabel = [ylabel] * (1 if groupby is None else len(keys))
    if groupby is None:
        if len(ylabel) != 1:
            raise ValueError(f"Expected number of y-labels to be `1`, found `{len(ylabel)}`.")
    elif len(ylabel) != len(keys):
        raise ValueError(f"Expected number of y-labels to be `{len(keys)}`, " f"found `{len(ylabel)}`.")

    if groupby is not None:
        if hue is not None:
            obs_df = get.obs_df(adata, keys=[groupby] + keys + [hue], layer=layer, use_raw=use_raw)
        else:
            obs_df = get.obs_df(adata, keys=[groupby] + keys, layer=layer, use_raw=use_raw)

    else:
        obs_df = get.obs_df(adata, keys=keys, layer=layer, use_raw=use_raw)
    if groupby is None:
        obs_tidy = pd.melt(obs_df, value_vars=keys)
        x = "variable"
        ys = ["value"]
    else:
        obs_tidy = obs_df
        x = groupby
        ys = keys

    if multi_panel and groupby is None and len(ys) == 1:
        # This is a quick and dirty way for adapting scales across several
        # keys if groupby is None.
        y = ys[0]

        g = sns.catplot(
            y=y,
            data=obs_tidy,
            kind="violin",
            scale=scale,
            col=x,
            col_order=keys,
            sharey=False,
            order=keys,
            cut=0,
            inner=None,
            **kwds,
        )

        if stripplot:
            grouped_df = obs_tidy.groupby(x)
            for ax_id, key in zip(range(g.axes.shape[1]), keys):
                sns.stripplot(
                    y=y,
                    data=grouped_df.get_group(key),
                    jitter=jitter,
                    size=size,
                    color="black",
                    ax=g.axes[0, ax_id],
                )
        if log:
            g.set(yscale="log")
        g.set_titles(col_template="{col_name}").set_xlabels("")
        if rotation is not None:
            for ax in g.axes[0]:
                ax.tick_params(axis="x", labelrotation=rotation)
    else:
        # set by default the violin plot cut=0 to limit the extend
        # of the violin plot (see stacked_violin code) for more info.
        kwds.setdefault("cut", 0)
        kwds.setdefault("inner")

        if ax is None:
            axs, _, _, _ = setup_axes(
                ax=ax,
                panels=["x"] if groupby is None else keys,
                show_ticks=True,
                right_margin=0.3,
            )
        else:
            axs = [ax]
        for ax, y, ylab in zip(axs, ys, ylabel):
            ax = sns.violinplot(
                x=x,
                y=y,
                data=obs_tidy,
                order=order,
                orient="vertical",
                scale=scale,
                ax=ax,
                hue=hue,
                **kwds,
            )
            # Get the handles and labels.
            handles, labels = ax.get_legend_handles_labels()
            if stripplot:
                ax = sns.stripplot(
                    x=x,
                    y=y,
                    data=obs_tidy,
                    order=order,
                    jitter=jitter,
                    color="black",
                    size=size,
                    ax=ax,
                    hue=hue,
                    dodge=True,
                )
            if xlabel == "" and groupby is not None and rotation is None:
                xlabel = groupby.replace("_", " ")
            ax.set_xlabel(xlabel)
            if ylab is not None:
                ax.set_ylabel(ylab)

            if log:
                ax.set_yscale("log")
            if rotation is not None:
                ax.tick_params(axis="x", labelrotation=rotation)

    show = settings.autoshow if show is None else show
    if hue is not None and stripplot is True:
        pl.legend(handles, labels)
    _utils.savefig_or_show("mixscape_violin", show=show, save=save)

    if not show:
        if multi_panel and groupby is None and len(ys) == 1:
            return g
        elif len(axs) == 1:
            return axs[0]
        else:
            return axs
