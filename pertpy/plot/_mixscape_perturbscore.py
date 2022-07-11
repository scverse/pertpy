from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from anndata import AnnData
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_density,
    geom_point,
    ggplot,
    scale_color_manual,
    theme,
    theme_classic,
    xlab,
    ylab,
)


def mixscape_perturbscore(
    adata: AnnData,
    labels: str,
    target_gene: str,
    mixscape_class="mixscape_class",
    color="orange",
    split_by: str = None,
    before_mixscape=False,
    perturbation_type: str = "KO",
):
    """Density plots to visualize perturbation scores calculated by the `pt.tl.mixscape` function.
    https://satijalab.org/seurat/reference/plotperturbscore

    Args:
        adata: The annotated data object.
        labels: The column of `.obs` with target gene labels.
        target_gene: Target gene name to visualize perturbation scores for.
        mixscape_class: The column of `.obs` with mixscape classifications.
        color: Specify color of target gene class or knockout cell class. For control non-targeting and non-perturbed cells, colors are set to different shades of grey.
        split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
            the perturbation signature for every replicate separately.
        before_mixscape: Option to split densities based on mixscape classification (default) or original target gene classification. Default is set to NULL and plots cells by original class ID.
        perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.
    
    Returns:
        The ggplot object used for drawn.
    """
    perturbation_score = None
    for key in adata.uns["mixscape"][target_gene].keys():
        perturbation_score_temp = adata.uns["mixscape"][target_gene][key]
        perturbation_score_temp["name"] = key
        if perturbation_score is None:
            perturbation_score = copy.deepcopy(perturbation_score_temp)
        else:
            perturbation_score = pd.concat([perturbation_score, perturbation_score_temp])
    perturbation_score["mix"] = adata.obs[mixscape_class][perturbation_score.index]
    gd = list(set(perturbation_score[labels]).difference({target_gene}))[0]
    # If before_mixscape is True, split densities based on original target gene classification
    if before_mixscape is True:
        cols = {gd: "#7d7d7d", target_gene: color}
        p = ggplot(perturbation_score, aes(x="pvec", color="gene_target")) + geom_density() + theme_classic()
        p_copy = copy.deepcopy(p)
        p_copy._build()
        top_r = max(p_copy.layers[0].data["density"])
        perturbation_score["y_jitter"] = perturbation_score["pvec"]
        perturbation_score.loc[perturbation_score["gene_target"] == gd, "y_jitter"] = np.random.uniform(
            low=0.001, high=top_r / 10, size=sum(perturbation_score["gene_target"] == gd)
        )
        perturbation_score.loc[perturbation_score["gene_target"] == target_gene, "y_jitter"] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(perturbation_score["gene_target"] == target_gene)
        )
        # If split_by is provided, split densities based on the split_by
        if split_by is not None:
            perturbation_score["split"] = adata.obs[split_by][perturbation_score.index]
            p2 = (
                p
                + scale_color_manual(values=cols, drop=False)
                + geom_density(size=1.5)
                + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                + ylab("Cell density")
                + xlab("Perturbation score")
                + theme(
                    legend_key_size=1,
                    legend_text=element_text(colour="black", size=14),
                    legend_title=element_blank(),
                    plot_title=element_text(size=16, face="bold"),
                )
                + facet_wrap("split")
            )
        else:
            p2 = (
                p
                + scale_color_manual(values=cols, drop=False)
                + geom_density(size=1.5)
                + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                + ylab("Cell density")
                + xlab("Perturbation score")
                + theme(
                    legend_key_size=1,
                    legend_text=element_text(colour="black", size=14),
                    legend_title=element_blank(),
                    plot_title=element_text(size=16, face="bold"),
                )
            )
    # If before_mixscape is False, split densities based on mixscape classifications
    else:
        cols = {gd: "#7d7d7d", f"{target_gene} NP": "#c9c9c9", f"{target_gene} {perturbation_type}": color}
        p = ggplot(perturbation_score, aes(x="pvec", color="mix")) + geom_density() + theme_classic()
        p_copy = copy.deepcopy(p)
        p_copy._build()
        top_r = max(p_copy.layers[0].data["density"])
        perturbation_score["y_jitter"] = perturbation_score["pvec"]
        gd2 = list(
            set(perturbation_score["mix"]).difference([f"{target_gene} NP", f"{target_gene} {perturbation_type}"])
        )[0]
        perturbation_score.loc[perturbation_score["mix"] == gd2, "y_jitter"] = np.random.uniform(
            low=0.001, high=top_r / 10, size=sum(perturbation_score["mix"] == gd2)
        )
        perturbation_score.loc[
            perturbation_score["mix"] == f"{target_gene} {perturbation_type}", "y_jitter"
        ] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} {perturbation_type}")
        )
        perturbation_score.loc[perturbation_score["mix"] == f"{target_gene} NP", "y_jitter"] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} NP")
        )
        # If split_by is provided, split densities based on the split_by
        if split_by is not None:
            perturbation_score["split"] = adata.obs[split_by][perturbation_score.index]
            p2 = (
                ggplot(perturbation_score, aes(x="pvec", color="mix"))
                + scale_color_manual(values=cols, drop=False)
                + geom_density(size=1.5)
                + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                + theme_classic()
                + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                + ylab("Cell density")
                + xlab("Perturbation score")
                + theme(
                    legend_key_size=1,
                    legend_text=element_text(colour="black", size=14),
                    legend_title=element_blank(),
                    plot_title=element_text(size=16, face="bold"),
                )
                + facet_wrap("split")
            )
        else:
            p2 = (
                p
                + scale_color_manual(values=cols, drop=False)
                + geom_density(size=1.5)
                + geom_point(aes(x="pvec", y="y_jitter"), size=0.1)
                + theme(axis_text=element_text(size=18), axis_title=element_text(size=20))
                + ylab("Cell density")
                + xlab("Perturbation score")
                + theme(
                    legend_key_size=1,
                    legend_text=element_text(colour="black", size=14),
                    legend_title=element_blank(),
                    plot_title=element_text(size=16, face="bold"),
                )
            )
    return p2
