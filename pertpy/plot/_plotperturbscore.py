import copy
from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from plotnine import *


def plotperturbscore(
    adata: AnnData,
    labels: str,
    target_gene: str,
    mixscape_class="mixscape_class",
    color="orange",
    split_by: Optional[str] = None,
    before_mixscape=False,
    prtb_type: Optional[str] = "KO",
):
    """Density plots to visualize perturbation scores calculated from RunMixscape function.
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
        prtb_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.
        **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

    """
    prtb_score = None
    for key in adata.uns["mixscape"][target_gene].keys():
        prtb_score_temp = adata.uns["mixscape"][target_gene][key]
        prtb_score_temp["name"] = key
        if prtb_score is None:
            prtb_score = copy.deepcopy(prtb_score_temp)
        else:
            prtb_score = pd.concat([prtb_score, prtb_score_temp])
    prtb_score["mix"] = adata.obs[mixscape_class][prtb_score.index]
    gd = list(set(prtb_score[labels]).difference({target_gene}))[0]
    if before_mixscape is True:
        cols = {gd: "#7d7d7d", target_gene: color}
        p = ggplot(prtb_score, aes(x="pvec", color="gene_target")) + geom_density() + theme_classic()
        p_copy = copy.deepcopy(p)
        p_copy._build()
        top_r = max(p_copy.layers[0].data["density"])
        prtb_score["y_jitter"] = prtb_score["pvec"]
        prtb_score.loc[prtb_score["gene_target"] == gd, "y_jitter"] = np.random.uniform(
            low=0.001, high=top_r / 10, size=sum(prtb_score["gene_target"] == gd)
        )  # prtb_score['y_jitter'][prtb_score['gene_target'] == target_gene] = np.random.uniform(low=-top_r/10, high=0, size=sum(prtb_score['gene_target'] == target_gene))
        prtb_score.loc[prtb_score["gene_target"] == target_gene, "y_jitter"] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(prtb_score["gene_target"] == target_gene)
        )
        if split_by is not None:
            prtb_score["split"] = adata.obs[split_by][prtb_score.index]
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
    else:
        cols = {gd: "#7d7d7d", f"{target_gene} NP": "#c9c9c9", f"{target_gene} {prtb_type}": color}
        p = ggplot(prtb_score, aes(x="pvec", color="mix")) + geom_density() + theme_classic()
        p_copy = copy.deepcopy(p)
        p_copy._build()
        top_r = max(p_copy.layers[0].data["density"])
        prtb_score["y_jitter"] = prtb_score["pvec"]
        gd2 = list(set(prtb_score["mix"]).difference([f"{target_gene} NP", f"{target_gene} {prtb_type}"]))[0]
        prtb_score.loc[prtb_score["mix"] == gd2, "y_jitter"] = np.random.uniform(
            low=0.001, high=top_r / 10, size=sum(prtb_score["mix"] == gd2)
        )
        prtb_score.loc[prtb_score["mix"] == f"{target_gene} {prtb_type}", "y_jitter"] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(prtb_score["mix"] == f"{target_gene} {prtb_type}")
        )
        prtb_score.loc[prtb_score["mix"] == f"{target_gene} NP", "y_jitter"] = np.random.uniform(
            low=-top_r / 10, high=0, size=sum(prtb_score["mix"] == f"{target_gene} NP")
        )
        if split_by is not None:
            prtb_score["split"] = adata.obs[split_by][prtb_score.index]
            p2 = (
                ggplot(prtb_score, aes(x="pvec", color="mix"))
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
