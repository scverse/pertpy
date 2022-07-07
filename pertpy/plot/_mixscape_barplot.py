from __future__ import annotations

import pandas as pd
from anndata import AnnData
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_classic,
    xlab,
    ylab,
)
from scanpy.plotting import _utils


def mixscape_barplot(
    adata: AnnData,
    control: str = "NT",
    mixscape_class_global="mixscape_class_global",
    axis_text_x_size: int = 8,
    axis_text_y_size: int = 6,
    axis_title_size: int = 8,
    strip_text_size: int = 6,
    panel_spacing_x: float = 0.3,
    panel_spacing_y: float = 0.3,
    legend_title_size: int = 8,
    legend_text_size: int = 8,
    show: bool | None = None,
    save: bool | str | None = None,
):
    """Barplot to visualize perturbation scores calculated from RunMixscape function.

    Args:
        adata: The annotated data object.
        control: Control category from the `pert_key` column. Default is 'NT'.
        mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).        show: Show the plot, do not return axis.
        save: If True or a str, save the figure. A string is appended to the default filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.

    Returns:
        If show is False, return ggplot object used for drawn.
    """

    count = pd.crosstab(index=adata.obs[mixscape_class_global], columns=adata.obs[control])
    df = count / count.sum()
    df2 = pd.melt(df, ignore_index=False).reset_index()
    test = df2[df2[mixscape_class_global] == "KO"]
    test = test.sort_values("value", ascending=False)

    new_levels = test[control]
    df2[control] = pd.Categorical(df2[control], categories=new_levels, ordered=False)
    df2[mixscape_class_global] = pd.Categorical(
        df2[mixscape_class_global], categories=["NT", "NP", "KO"], ordered=False
    )
    df2["gene"] = df2[control].str.rsplit("g", expand=True)[0]
    df2["guide_number"] = df2[control].str.rsplit("g", expand=True)[1]
    df2["guide_number"] = "g" + df2["guide_number"]
    df3 = df2[df2["gene"] != "NT"]

    p1 = (
        ggplot(df3, aes(x="guide_number", y="value", fill="mixscape_class_global"))
        + scale_fill_manual(values=["#7d7d7d", "#c9c9c9", "#ff7256"])
        + geom_bar(stat="identity")
        + theme_classic()
        + xlab("sgRNA")
        + ylab("% of cells")
    )

    p1 = (
        p1
        + theme(
            axis_text_x=element_text(size=axis_text_x_size, hjust=2),
            axis_text_y=element_text(size=axis_text_y_size),
            axis_title=element_text(size=axis_title_size),
            strip_text=element_text(size=strip_text_size, face="bold"),
            panel_spacing_x=panel_spacing_x,
            panel_spacing_y=panel_spacing_y,
        )
        + facet_wrap("gene", ncol=5, scales="free")
        + labs(fill="mixscape class")
        + theme(legend_title=element_text(size=legend_title_size), legend_text=element_text(size=legend_text_size))
    )

    _utils.savefig_or_show("mixscape_barplot", show=show, save=save)
    if not show:
        return p1
