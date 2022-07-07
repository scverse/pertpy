from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from sklearn.mixture import GaussianMixture

warnings.simplefilter("ignore")


def _define_normal_mixscape(X: np.ndarray | sparse.spmatrix | pd.DataFrame | None) -> list[float]:  # noqa: N803
    """Calculates the mean and standard deviation of a matrix.

    Args:
        X: The matrix to calculate the properties for.

    Returns:
        Mean and standard deviation of the matrix.
    """
    mu = X.mean()
    sd = X.std()

    return [mu, sd]


def mixscape(
    adata: AnnData,
    labels: str,
    control: str,
    new_class_name: str | None = "mixscape_class",
    min_de_genes: int | None = 5,
    layer: str | None = None,
    logfc_threshold: float | None = 0.25,
    iter_num: int | None = 10,
    split_by: str | None = None,
    pval_cutoff: float | None = 5e-2,
    perturbation_type: str | None = "KO",
    copy: bool | None = False,
):
    """Identify perturbed and non-perturbed gRNA expressing cells that accounts for multiple treatments/conditions/chemical perturbations.

    The implementation resembles https://satijalab.org/seurat/reference/runmixscape

    Args:
        adata: The annotated data object.
        pert_key: The column of `.obs` with perturbation categories, should also contain `control`.
        labels: The column of `.obs` with target gene labels.
        control: Control category from the `pert_key` column.
        new_class_name: Name of mixscape classification to be stored in `.obs`.
        min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
        layer: Key from adata.layers whose value will be used to perform tests on.
        logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells (default: 0.25).
        iter_num: Number of normalmixEM iterations to run if convergence does not occur.
        split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                  the perturbation signature for every replicate separately.
        pval_cutoff: P-value cut-off for selection of significantly DE genes.
        perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.
        copy: Determines whether a copy of the `adata` is returned.

    Returns:
        If `copy=True`, returns the copy of `adata` with the classification result in `.obs`.
        Otherwise writes the results directly to `.obs` of the provided `adata`.

        mixscape_class: pandas.Series (`adata.obs['mixscape_class']`).
        Classification result with cells being either classified as perturbed (KO, by default) or non-perturbed (NP) based on their target gene class.

        mixscape_class_global: pandas.Series (`adata.obs['mixscape_class_global']`).
        Global classification result (perturbed, NP or NT)

        mixscape_class_p_ko: pandas.Series (`adata.obs['mixscape_class_p_ko']`).
        Posterior probabilities used to determine if a cell is KO (default). Name of this item will change to match perturbation_type parameter setting. (>0.5) or NP
    """
    if copy:
        adata = adata.copy()

    if split_by is None:
        split_masks = [np.full(adata.n_obs, True, dtype=bool)]
        categories = ["all"]
    else:
        split_obs = adata.obs[split_by]
        categories = split_obs.unique()
        split_masks = [split_obs == category for category in categories]

    # determine gene sets across all splits/groups through differential gene expression
    perturbation_markers: dict[tuple, np.ndarray] = {}
    gv_list: dict[str, dict] = {}
    for split, split_mask in enumerate(split_masks):
        category = categories[split]
        # get gene sets for each split
        genes = list(set(adata[split_mask].obs[labels]).difference([control]))
        adata_split = adata[split_mask].copy()
        # find top DE genes between cells with targeting and non-targeting gRNAs
        sc.tl.rank_genes_groups(
            adata_split, layer=layer, groupby=labels, groups=genes, reference=control, method="t-test"
        )
        # get DE genes for each gene
        for gene in genes:
            logfc_threshold_mask = adata_split.uns["rank_genes_groups"]["logfoldchanges"][gene] >= logfc_threshold
            de_genes = adata_split.uns["rank_genes_groups"]["names"][gene][logfc_threshold_mask]
            pvals_adj = adata_split.uns["rank_genes_groups"]["pvals_adj"][gene][logfc_threshold_mask]
            de_genes = de_genes[pvals_adj < pval_cutoff]
            if len(de_genes) < min_de_genes:
                de_genes = np.array([])
            perturbation_markers[(category, gene)] = de_genes

    adata_comp = adata
    if layer is not None:
        X = adata_comp.layers[layer]
    else:
        X = adata_comp.X
    # initialize return variables
    adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"] = 0
    adata.obs[new_class_name] = adata.obs[labels].astype(str)
    adata.obs[f"{new_class_name}_global"] = np.empty(
        [
            adata.n_obs,
        ],
        dtype=np.object_,
    )

    for split, split_mask in enumerate(split_masks):
        category = categories[split]
        genes = list(set(adata[split_mask].obs[labels]).difference([control]))
        for gene in genes:
            post_prob = 0
            orig_guide_cells = (adata.obs[labels] == gene) & split_mask
            orig_guide_cells_index = list(orig_guide_cells.index[orig_guide_cells])
            nt_cells = (adata.obs[labels] == control) & split_mask
            all_cells = orig_guide_cells | nt_cells

            if len(perturbation_markers[(category, gene)]) == 0:
                adata.obs.loc[orig_guide_cells, new_class_name] = f"{gene} NP"
            else:
                de_genes = perturbation_markers[(category, gene)]
                de_genes_indices = _get_column_indices(adata, list(de_genes))
                dat = X[all_cells][:, de_genes_indices]
                converged = False
                n_iter = 0
                old_classes = adata.obs[labels][all_cells]
                while not converged and n_iter < iter_num:
                    # Get all cells in current split&Gene
                    guide_cells = (adata.obs[labels] == gene) & split_mask
                    # get average value for each gene over all selected cells
                    # all cells in current split&Gene minus all NT cells in current split
                    # Each row is for each cell, each column is for each gene, get mean for each column
                    vec = np.mean(X[guide_cells][:, de_genes_indices], axis=0) - np.mean(
                        X[nt_cells][:, de_genes_indices], axis=0
                    )
                    # project cells onto the perturbation vector
                    pvec = np.sum(np.multiply(dat.toarray(), vec), axis=1) / np.sum(np.multiply(vec, vec))
                    pvec = pd.Series(np.asarray(pvec).flatten(), index=list(all_cells.index[all_cells]))
                    if n_iter == 0:
                        gv = pd.DataFrame(columns=["pvec", labels])
                        gv["pvec"] = pvec
                        gv[labels] = control
                        gv.loc[guide_cells, labels] = gene
                        if gene not in gv_list.keys():
                            gv_list[gene] = {}
                        gv_list[gene][category] = gv

                    guide_norm = _define_normal_mixscape(pvec[guide_cells])
                    nt_norm = _define_normal_mixscape(pvec[nt_cells])
                    means_init = np.array([[nt_norm[0]], [guide_norm[0]]])
                    precisions_init = np.array([nt_norm[1], guide_norm[1]])
                    mm = GaussianMixture(
                        n_components=2,
                        covariance_type="spherical",
                        means_init=means_init,
                        precisions_init=precisions_init,
                    ).fit(np.asarray(pvec).reshape(-1, 1))
                    probabilities = mm.predict_proba(np.array(pvec[orig_guide_cells_index]).reshape(-1, 1))
                    lik_ratio = probabilities[:, 0] / probabilities[:, 1]
                    post_prob = 1 / (1 + lik_ratio)
                    # based on the posterior probability, assign cells to the two classes
                    adata.obs.loc[
                        [orig_guide_cells_index[cell] for cell in np.where(post_prob > 0.5)[0]], new_class_name
                    ] = gene
                    adata.obs.loc[
                        [orig_guide_cells_index[cell] for cell in np.where(post_prob <= 0.5)[0]], new_class_name
                    ] = f"{gene} NP"
                    if sum(adata.obs[new_class_name][split_mask] == gene) < min_de_genes:
                        adata.obs.loc[guide_cells, new_class_name] = "NP"
                        converged = True
                    if adata.obs[new_class_name][all_cells].equals(old_classes):
                        converged = True
                    old_classes = adata.obs[new_class_name][all_cells]
                    n_iter += 1

                adata.obs.loc[
                    (adata.obs[new_class_name] == gene) & split_mask, new_class_name
                ] = f"{gene} {perturbation_type}"

            adata.obs[f"{new_class_name}_global"] = [a.split(" ")[-1] for a in adata.obs[new_class_name]]
            adata.obs.loc[orig_guide_cells_index, f"{new_class_name}_p_{perturbation_type.lower()}"] = post_prob
    adata.uns["mixscape"] = gv_list

    if copy:
        return adata


def _get_column_indices(adata, col_names):
    """Fetches the column indices in X for a given list of column names

    Args:
        adata: :class:`~anndata.AnnData` object
        col_names: Column names to extract the indices for

    Returns:
        Set of column indices
    """
    if isinstance(col_names, str):  # pragma: no cover
        col_names = [col_names]

    indices = list()
    for idx, col in enumerate(adata.var_names):
        if col in col_names:
            indices.append(idx)

    return indices
