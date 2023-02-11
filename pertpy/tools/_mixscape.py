from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from pynndescent import NNDescent
from rich import print
from scanpy.tools._utils import _choose_representation
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.mixture import GaussianMixture

import pertpy as pt

warnings.simplefilter("ignore")


class Mixscape:
    """Python implementation of Mixscape."""

    def __init__(self):
        pass

    def pert_sign(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        split_by: str | None = None,
        n_neighbors: int = 20,
        use_rep: str | None = None,
        n_pcs: int | None = None,
        batch_size: int | None = None,
        copy: bool = False,
        **kwargs,
    ):
        """Calculate perturbation signature.

        For each cell, we identify `n_neighbors` cells from the control pool with the most similar mRNA expression profiles.
        The perturbation signature is calculated by subtracting the averaged mRNA expression profile of the control
        neighbors from the mRNA expression profile of each cell.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Control category from the `pert_key` column.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                the perturbation signature for every replicate separately.
            n_neighbors: Number of neighbors from the control to use for the perturbation signature.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                If `None`, the representation is chosen automatically:
                For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                If 'X_pca' is not present, it’s computed with default parameters.
            n_pcs: Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
            batch_size: Size of batch to calculate the perturbation signature.
                If 'None', the perturbation signature is calcuated in the full mode, requiring more memory.
                The batched mode is very inefficient for sparse data.
            copy: Determines whether a copy of the `adata` is returned.
            **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

        Returns:
            If `copy=True`, returns the copy of `adata` with the perturbation signature in `.layers["X_pert"]`.
            Otherwise writes the perturbation signature directly to `.layers["X_pert"]` of the provided `adata`.
        """
        if copy:
            adata = adata.copy()

        adata.layers["X_pert"] = adata.X.copy()

        control_mask = adata.obs[pert_key] == control

        if split_by is None:
            split_masks = [np.full(adata.n_obs, True, dtype=bool)]
        else:
            split_obs = adata.obs[split_by]
            cats = split_obs.unique()
            split_masks = [split_obs == cat for cat in cats]

        R = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

        for split_mask in split_masks:
            control_mask_split = control_mask & split_mask

            R_split = R[split_mask]
            R_control = R[control_mask_split]

            eps = kwargs.pop("epsilon", 0.1)
            nn_index = NNDescent(R_control, **kwargs)
            indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)

            X_control = np.expm1(adata.X[control_mask_split])

            n_split = split_mask.sum()
            n_control = X_control.shape[0]

            if batch_size is None:
                col_indices = np.ravel(indices)
                row_indices = np.repeat(np.arange(n_split), n_neighbors)

                neigh_matrix = csr_matrix(
                    (np.ones_like(col_indices, dtype=np.float64), (row_indices, col_indices)),
                    shape=(n_split, n_control),
                )
                neigh_matrix /= n_neighbors
                adata.layers["X_pert"][split_mask] -= np.log1p(neigh_matrix @ X_control)
            else:
                is_sparse = issparse(X_control)
                split_indices = np.where(split_mask)[0]
                for i in range(0, n_split, batch_size):
                    size = min(i + batch_size, n_split)
                    select = slice(i, size)

                    batch = np.ravel(indices[select])
                    split_batch = split_indices[select]

                    size = size - i

                    # sparse is very slow
                    means_batch = X_control[batch]
                    means_batch = means_batch.toarray() if is_sparse else means_batch
                    means_batch = means_batch.reshape(size, n_neighbors, -1).mean(1)

                    adata.layers["X_pert"][split_batch] -= np.log1p(means_batch)

        if copy:
            return adata

    def mixscape(
        self,
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
            layer: Key from adata.layers whose value will be used to perform tests on. Default is using `.layers["X_pert"]`.
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

        perturbation_markers = self._get_perturbation_markers(
            adata, split_masks, categories, labels, control, layer, pval_cutoff, min_de_genes, logfc_threshold
        )

        adata_comp = adata
        if layer is not None:
            X = adata_comp.layers[layer]
        else:
            try:
                X = adata_comp.layers["X_pert"]
            except KeyError:
                print(
                    '[bold yellow]No "X_pert" found in .layers! -- Please run pert_sign first to calculate perturbation signature!'
                )
                raise
        # initialize return variables
        adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"] = 0
        adata.obs[new_class_name] = adata.obs[labels].astype(str)
        adata.obs[f"{new_class_name}_global"] = np.empty(
            [
                adata.n_obs,
            ],
            dtype=np.object_,
        )
        gv_list: dict[str, dict] = {}
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
                    de_genes_indices = self._get_column_indices(adata, list(de_genes))
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

                        guide_norm = self._define_normal_mixscape(pvec[guide_cells])
                        nt_norm = self._define_normal_mixscape(pvec[nt_cells])
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

    def lda(
        self,
        adata: AnnData,
        labels: str,
        mixscape_class_global="mixscape_class_global",
        layer: str | None = None,
        control: str | None = "NT",
        n_comps: int | None = 10,
        min_de_genes: int | None = 5,
        logfc_threshold: float | None = 0.25,
        split_by: str | None = None,
        pval_cutoff: float | None = 5e-2,
        perturbation_type: str | None = "KO",
        copy: bool | None = False,
    ):
        """Linear Discriminant Analysis on pooled CRISPR screen data. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            layer: Key from `adata.layers` whose value will be used to perform tests on.
            control: Control category from the `pert_key` column. Default is 'NT'.
            n_comps: Number of principal components to use. Defaults to 10.
            min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
            logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells (default: 0.25).
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
            pval_cutoff: P-value cut-off for selection of significantly DE genes.
            perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications. Default is KO.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the LDA result in `.uns`.
            Otherwise writes the results directly to `.uns` of the provided `adata`.

            mixscape_lda: numpy.ndarray (`adata.uns['mixscape_lda']`).
            LDA result.
        """
        if copy:
            adata = adata.copy()
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        if split_by is None:
            split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = split_obs.unique()
            split_masks = [split_obs == category for category in categories]

        mixscape_identifier = pt.tl.Mixscape()
        # determine gene sets across all splits/groups through differential gene expression
        perturbation_markers = mixscape_identifier._get_perturbation_markers(
            adata=adata,
            split_masks=split_masks,
            categories=categories,
            labels=labels,
            control=control,
            layer=layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
        )
        adata_subset = adata[
            (adata.obs[mixscape_class_global] == perturbation_type) | (adata.obs[mixscape_class_global] == control)
        ].copy()
        projected_pcs: dict[str, np.ndarray] = {}
        # performs PCA on each mixscape class separately and projects each subspace onto all cells in the data.
        for _, (key, value) in enumerate(perturbation_markers.items()):
            if len(value) == 0:
                continue
            else:
                gene_subset = adata_subset[
                    (adata_subset.obs[labels] == key[1]) | (adata_subset.obs[labels] == control)
                ].copy()
                sc.pp.scale(gene_subset)
                sc.tl.pca(gene_subset, n_comps=n_comps)
                sc.pp.neighbors(gene_subset)
                # projects each subspace onto all cells in the data.
                sc.tl.ingest(adata=adata_subset, adata_ref=gene_subset, embedding_method="pca")
                projected_pcs[key[1]] = adata_subset.obsm["X_pca"]
        # concatenate all pcs into a single matrix.
        for index, (_, value) in enumerate(projected_pcs.items()):
            if index == 0:
                projected_pcs_array = value
            else:
                projected_pcs_array = np.concatenate((projected_pcs_array, value), axis=1)

        clf = LinearDiscriminantAnalysis(n_components=len(np.unique(adata_subset.obs["gene_target"])) - 1)
        clf.fit(projected_pcs_array, adata_subset.obs["gene_target"])
        cell_embeddings = clf.transform(projected_pcs_array)
        adata.uns["mixscape_lda"] = cell_embeddings

        if copy:
            return adata

    def _get_perturbation_markers(
        self,
        adata: AnnData,
        split_masks: list[np.ndarray],
        categories: list[str],
        labels: str,
        control: str,
        layer: str,
        pval_cutoff: float,
        min_de_genes: float,
        logfc_threshold: float,
    ) -> dict[tuple, np.ndarray]:
        """determine gene sets across all splits/groups through differential gene expression

        Args:
            adata: :class:`~anndata.AnnData` object
            col_names: Column names to extract the indices for

        Returns:
            Set of column indices
        """
        perturbation_markers: dict[tuple, np.ndarray] = {}
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

        return perturbation_markers

    def _get_column_indices(self, adata, col_names):
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

    def _define_normal_mixscape(
        self, X: np.ndarray | sparse.spmatrix | pd.DataFrame | None
    ) -> list[float]:  # noqa: N803
        """Calculates the mean and standard deviation of a matrix.

        Args:
            X: The matrix to calculate the properties for.

        Returns:
            Mean and standard deviation of the matrix.
        """
        mu = X.mean()
        sd = X.std()

        return [mu, sd]
