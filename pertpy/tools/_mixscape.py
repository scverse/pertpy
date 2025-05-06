from __future__ import annotations

import copy
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from fast_array_utils.stats import mean, mean_var
from scanpy import get
from scanpy._utils import _check_use_raw, sanitize_anndata
from scanpy.plotting import _utils
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_matrix, spmatrix
from sklearn.mixture import GaussianMixture

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.pyplot import Figure
    from scipy import sparse


class Mixscape:
    """Python implementation of Mixscape."""

    def __init__(self):
        pass

    def perturbation_signature(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        ref_selection_mode: Literal["nn", "split_by"] = "nn",
        split_by: str | None = None,
        n_neighbors: int = 20,
        use_rep: str | None = None,
        n_dims: int | None = 15,
        n_pcs: int | None = None,
        batch_size: int | None = None,
        copy: bool = False,
        **kwargs,
    ):
        """Calculate perturbation signature.

        The perturbation signature is calculated by subtracting the mRNA expression profile of each cell from the averaged
        mRNA expression profile of the control cells (selected according to `ref_selection_mode`).
        The implementation resembles https://satijalab.org/seurat/reference/runmixscape. Note that in the original implementation, the
        perturbation signature is calculated on unscaled data by default, and we therefore recommend to do the same.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Name of the control category from the `pert_key` column.
            ref_selection_mode: Method to select reference cells for the perturbation signature calculation. If `nn`,
                the `n_neighbors` cells from the control pool with the most similar mRNA expression profiles are selected. If `split_by`,
                the control cells from the same split in `split_by` (e.g. indicating biological replicates) are used to calculate the perturbation signature.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                the perturbation signature for every replicate separately.
            n_neighbors: Number of neighbors from the control to use for the perturbation signature.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                If `None`, the representation is chosen automatically:
                For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                If 'X_pca' is not present, it’s computed with default parameters.
            n_dims: Number of dimensions to use from the representation to calculate the perturbation signature.
                If `None`, use all dimensions.
            n_pcs: If PCA representation is used, the number of principal components to compute.
                If `n_pcs==0` use `.X` if `use_rep is None`.
            batch_size: Size of batch to calculate the perturbation signature.
                If 'None', the perturbation signature is calcuated in the full mode, requiring more memory.
                The batched mode is very inefficient for sparse data.
            copy: Determines whether a copy of the `adata` is returned.
            **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

        Returns:
            If `copy=True`, returns the copy of `adata` with the perturbation signature in `.layers["X_pert"]`.
            Otherwise, writes the perturbation signature directly to `.layers["X_pert"]` of the provided `adata`.

        Examples:
            Calcutate perturbation signature for each cell in the dataset:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
        """
        if ref_selection_mode not in ["nn", "split_by"]:
            raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
        if ref_selection_mode == "split_by" and split_by is None:
            raise ValueError("split_by must be provided if ref_selection_mode is 'split_by'.")

        if copy:
            adata = adata.copy()

        adata.layers["X_pert"] = adata.X.copy()

        control_mask = adata.obs[pert_key] == control

        if ref_selection_mode == "split_by":
            for split in adata.obs[split_by].unique():
                split_mask = adata.obs[split_by] == split
                control_mask_group = control_mask & split_mask
                control_mean_expr = mean(adata.X[control_mask_group], axis=0)
                adata.layers["X_pert"][split_mask] = (
                    np.repeat(control_mean_expr.reshape(1, -1), split_mask.sum(), axis=0)
                    - adata.layers["X_pert"][split_mask]
                )
        else:
            if split_by is None:
                split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            else:
                split_obs = adata.obs[split_by]
                split_masks = [split_obs == cat for cat in split_obs.unique()]

            representation = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
            if n_dims is not None and n_dims < representation.shape[1]:
                representation = representation[:, :n_dims]

            from pynndescent import NNDescent

            for split_mask in split_masks:
                control_mask_split = control_mask & split_mask

                R_split = representation[split_mask]
                R_control = representation[np.asarray(control_mask_split)]

                eps = kwargs.pop("epsilon", 0.1)
                nn_index = NNDescent(R_control, **kwargs)
                indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)

                X_control = np.expm1(adata.X[np.asarray(control_mask_split)])

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
                    adata.layers["X_pert"][np.asarray(split_mask)] = (
                        sc.pp.log1p(neigh_matrix @ X_control) - adata.layers["X_pert"][np.asarray(split_mask)]
                    )
                else:
                    split_indices = np.where(split_mask)[0]
                    for i in range(0, n_split, batch_size):
                        size = min(i + batch_size, n_split)
                        select = slice(i, size)

                        batch = np.ravel(indices[select])
                        split_batch = split_indices[select]

                        size = size - i

                        means_batch = X_control[batch]
                        batch_reshaped = means_batch.reshape(size, n_neighbors, -1)
                        means_batch, _ = mean_var(batch_reshaped, axis=1)

                        adata.layers["X_pert"][split_batch] = (
                            np.log1p(means_batch) - adata.layers["X_pert"][split_batch]
                        )

        if copy:
            return adata

    def mixscape(
        self,
        adata: AnnData,
        labels: str,
        control: str,
        *,
        new_class_name: str | None = "mixscape_class",
        layer: str | None = None,
        min_de_genes: int | None = 5,
        logfc_threshold: float | None = 0.25,
        de_layer: str | None = None,
        test_method: str | None = "wilcoxon",
        iter_num: int | None = 10,
        scale: bool | None = True,
        split_by: str | None = None,
        pval_cutoff: float | None = 5e-2,
        perturbation_type: str | None = "KO",
        random_state: int | None = 0,
        copy: bool | None = False,
        **gmmkwargs,
    ):
        """Identify perturbed and non-perturbed gRNA expressing cells that accounts for multiple treatments/conditions/chemical perturbations.

        The implementation resembles https://satijalab.org/seurat/reference/runmixscape.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            control: Control category from the `labels` column.
            new_class_name: Name of mixscape classification to be stored in `.obs`.
            layer: Key from adata.layers whose value will be used to perform tests on. Default is using `.layers["X_pert"]`.
            min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
            logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells (default: 0.25).
            de_layer: Layer to use for identifying differentially expressed genes. If `None`, adata.X is used.
            test_method: Method to use for differential expression testing.
            iter_num: Number of normalmixEM iterations to run if convergence does not occur.
            scale: Scale the data specified in `layer` before running the GaussianMixture model on it.
            split_by: Provide `.obs` column with experimental condition/cell type annotation, if perturbations are condition/cell type-specific.
            pval_cutoff: P-value cut-off for selection of significantly DE genes.
            perturbation_type: specify type of CRISPR perturbation expected for labeling mixscape classifications.
            random_state: Random seed for the GaussianMixture model.
            copy: Determines whether a copy of the `adata` is returned.
            **gmmkwargs: Passed to custom implementation of scikit-learn Gaussian Mixture Model.

        Returns:
            If `copy=True`, returns the copy of `adata` with the classification result in `.obs`.
            Otherwise, writes the results directly to `.obs` of the provided `adata`.

            - mixscape_class: pandas.Series (`adata.obs['mixscape_class']`).
              Classification result with cells being either classified as perturbed (KO, by default) or non-perturbed (NP) based on their target gene class.

            - mixscape_class_global: pandas.Series (`adata.obs['mixscape_class_global']`).
              Global classification result (perturbed, NP or NT).

            - mixscape_class_p_ko: pandas.Series (`adata.obs['mixscape_class_p_ko']`).
              Posterior probabilities used to determine if a cell is KO (default).
              Name of this item will change to match perturbation_type parameter setting. (>0.5) or NP.

        Examples:
            Calcutate perturbation signature for each cell in the dataset:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
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
            adata=adata,
            split_masks=split_masks,
            categories=categories,
            labels=labels,
            control=control,
            layer=de_layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )

        adata_comp = adata
        if layer is not None:
            X = adata_comp.layers[layer]
        else:
            try:
                X = adata_comp.layers["X_pert"]
            except KeyError:
                raise KeyError(
                    "No 'X_pert' found in .layers! Please run perturbation_signature first to calculate perturbation signature!"
                ) from None

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

        adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"] = 0.0
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            gene_targets = list(set(adata[split_mask].obs[labels]).difference([control]))
            for gene in gene_targets:
                post_prob = 0
                orig_guide_cells = (adata.obs[labels] == gene) & split_mask
                orig_guide_cells_index = list(orig_guide_cells.index[orig_guide_cells])
                nt_cells = (adata.obs[labels] == control) & split_mask
                all_cells = orig_guide_cells | nt_cells

                if len(perturbation_markers[(category, gene)]) == 0:
                    adata.obs.loc[orig_guide_cells, new_class_name] = f"{gene} NP"

                else:
                    de_genes = perturbation_markers[(category, gene)]
                    de_genes_indices = np.where(np.isin(adata.var_names, list(de_genes)))[0]

                    dat = X[np.asarray(all_cells)][:, de_genes_indices]
                    if scale:
                        dat = sc.pp.scale(dat)

                    converged = False
                    n_iter = 0
                    old_classes = adata.obs[new_class_name][all_cells]

                    nt_cells_dat_idx = all_cells[all_cells].index.get_indexer(nt_cells[nt_cells].index)
                    nt_cells_mean = np.mean(dat[nt_cells_dat_idx], axis=0)

                    while not converged and n_iter < iter_num:
                        # Get all cells in current split&Gene
                        guide_cells = (adata.obs[new_class_name] == gene) & split_mask

                        # get average value for each gene over all selected cells
                        # all cells in current split&Gene minus all NT cells in current split
                        # Each row is for each cell, each column is for each gene, get mean for each column
                        guide_cells_dat_idx = all_cells[all_cells].index.get_indexer(guide_cells[guide_cells].index)
                        guide_cells_mean = np.mean(dat[guide_cells_dat_idx], axis=0)
                        vec = guide_cells_mean - nt_cells_mean

                        # project cells onto the perturbation vector
                        if isinstance(dat, spmatrix):
                            pvec = dat.dot(vec) / np.dot(vec, vec)
                        else:
                            pvec = np.dot(dat, vec) / np.dot(vec, vec)
                        pvec = pd.Series(np.asarray(pvec).flatten(), index=list(all_cells.index[all_cells]))

                        if n_iter == 0:
                            gv = pd.DataFrame(columns=["pvec", labels])
                            gv["pvec"] = pvec
                            gv[labels] = control
                            gv.loc[guide_cells, labels] = gene
                            if gene not in gv_list:
                                gv_list[gene] = {}
                            gv_list[gene][category] = gv

                        means_init = np.array([[pvec[nt_cells].mean()], [pvec[guide_cells].mean()]])
                        std_init = np.array([pvec[nt_cells].std(), pvec[guide_cells].std()])
                        mm = MixscapeGaussianMixture(
                            n_components=2,
                            covariance_type="spherical",
                            means_init=means_init,
                            precisions_init=1 / (std_init**2),
                            random_state=random_state,
                            max_iter=100,
                            fixed_means=[pvec[nt_cells].mean(), None],
                            fixed_covariances=[pvec[nt_cells].std() ** 2, None],
                            **gmmkwargs,
                        ).fit(np.asarray(pvec).reshape(-1, 1))
                        probabilities = mm.predict_proba(np.array(pvec[orig_guide_cells_index]).reshape(-1, 1))
                        lik_ratio = probabilities[:, 0] / probabilities[:, 1]
                        post_prob = 1 / (1 + lik_ratio)

                        # based on the posterior probability, assign cells to the two classes
                        ko_mask = post_prob > 0.5
                        adata.obs.loc[np.array(orig_guide_cells_index)[ko_mask], new_class_name] = gene
                        adata.obs.loc[np.array(orig_guide_cells_index)[~ko_mask], new_class_name] = f"{gene} NP"

                        if sum(adata.obs[new_class_name][split_mask] == gene) < min_de_genes:
                            adata.obs.loc[guide_cells, new_class_name] = "NP"
                            converged = True
                        current_classes = adata.obs[new_class_name][all_cells]
                        if (current_classes == old_classes).all():
                            converged = True
                        old_classes = current_classes

                        n_iter += 1

                    adata.obs.loc[(adata.obs[new_class_name] == gene) & split_mask, new_class_name] = (
                        f"{gene} {perturbation_type}"
                    )

                adata.obs[f"{new_class_name}_global"] = [a.split(" ")[-1] for a in adata.obs[new_class_name]]
                adata.obs.loc[orig_guide_cells_index, f"{new_class_name}_p_{perturbation_type.lower()}"] = post_prob
        adata.uns["mixscape"] = gv_list

        if copy:
            return adata

    def lda(
        self,
        adata: AnnData,
        labels: str,
        control: str,
        *,
        mixscape_class_global: str | None = "mixscape_class_global",
        layer: str | None = None,
        n_comps: int | None = 10,
        min_de_genes: int | None = 5,
        logfc_threshold: float | None = 0.25,
        test_method: str | None = "wilcoxon",
        split_by: str | None = None,
        pval_cutoff: float | None = 5e-2,
        perturbation_type: str | None = "KO",
        copy: bool | None = False,
    ):
        """Linear Discriminant Analysis on pooled CRISPR screen data. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            control: Control category from the `pert_key` column.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            layer: Layer to use for identifying differentially expressed genes. If `None`, adata.X is used.
            n_comps: Number of principal components to use.
            min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
            logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells.
            test_method: Method to use for differential expression testing.
            split_by: Provide `.obs` column with experimental condition/cell type annotation, if perturbations are condition/cell type-specific.
            pval_cutoff: P-value cut-off for selection of significantly DE genes.
            perturbation_type: Specify type of CRISPR perturbation expected for labeling mixscape classifications.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            If `copy=True`, returns the copy of `adata` with the LDA result in `.uns`.
            Otherwise, writes the results directly to `.uns` of the provided `adata`.

            mixscape_lda: numpy.ndarray (`adata.uns['mixscape_lda']`).
            LDA result.

        Examples:
            Use LDA dimensionality reduction to visualize the perturbation effects:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.lda(mdata["rna"], "gene_target", "NT")
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

        # determine gene sets across all splits/groups through differential gene expression
        perturbation_markers = self._get_perturbation_markers(
            adata=adata,
            split_masks=split_masks,
            categories=categories,
            labels=labels,
            control=control,
            layer=layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )
        adata_subset = adata[
            (adata.obs[mixscape_class_global] == perturbation_type) | (adata.obs[mixscape_class_global] == control)
        ]
        X = adata_subset.X - adata_subset.X.mean(0)
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
                # project cells into PCA space of gene_subset
                projected_pcs[key[1]] = np.asarray(np.dot(X, gene_subset.varm["PCs"]))
        # concatenate all pcs into a single matrix.
        projected_pcs_array = np.concatenate(list(projected_pcs.values()), axis=1)

        clf = LinearDiscriminantAnalysis(n_components=len(np.unique(adata_subset.obs[labels])) - 1)
        clf.fit(projected_pcs_array, adata_subset.obs[labels])
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
        test_method: str,
    ) -> dict[tuple, np.ndarray]:
        """Determine gene sets across all splits/groups through differential gene expression.

        Args:
            adata: :class:`~anndata.AnnData` object
            split_masks: List of boolean masks for each split/group.
            categories: List of split/group names.
            labels: The column of `.obs` with target gene labels.
            control: Control category from the `labels` column.
            layer: Key from adata.layers whose value will be used to compare gene expression.
            pval_cutoff: P-value cut-off for selection of significantly DE genes.
            min_de_genes: Required number of genes that are differentially expressed for method to separate perturbed and non-perturbed cells.
            logfc_threshold: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells.
            test_method: Method to use for differential expression testing.

        Returns:
            Set of column indices.
        """
        perturbation_markers: dict[tuple, np.ndarray] = {}  # type: ignore
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            # get gene sets for each split
            gene_targets = list(set(adata[split_mask].obs[labels]).difference([control]))
            adata_split = adata[split_mask].copy()
            # find top DE genes between cells with targeting and non-targeting gRNAs
            sc.tl.rank_genes_groups(
                adata_split,
                layer=layer,
                groupby=labels,
                groups=gene_targets,
                reference=control,
                method=test_method,
                use_raw=False,
            )
            # get DE genes for each target gene
            for gene in gene_targets:
                logfc_threshold_mask = (
                    np.abs(adata_split.uns["rank_genes_groups"]["logfoldchanges"][gene]) >= logfc_threshold
                )
                de_genes = adata_split.uns["rank_genes_groups"]["names"][gene][logfc_threshold_mask]
                pvals_adj = adata_split.uns["rank_genes_groups"]["pvals_adj"][gene][logfc_threshold_mask]
                de_genes = de_genes[pvals_adj < pval_cutoff]
                if len(de_genes) < min_de_genes:
                    de_genes = np.array([])
                perturbation_markers[(category, gene)] = de_genes

        return perturbation_markers

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_barplot(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        guide_rna_column: str,
        *,
        mixscape_class_global: str = "mixscape_class_global",
        axis_text_x_size: int = 8,
        axis_text_y_size: int = 6,
        axis_title_size: int = 8,
        legend_title_size: int = 8,
        legend_text_size: int = 8,
        legend_bbox_to_anchor: tuple[float, float] = None,
        figsize: tuple[float, float] = (25, 25),
        return_fig: bool = False,
    ) -> Figure | None:
        """Barplot to visualize perturbation scores calculated by the `mixscape` function.

        Args:
            adata: The annotated data object.
            guide_rna_column: The column of `.obs` with guide RNA labels. The target gene labels.
                              The format must be <gene_target>g<#>. Examples are 'STAT2g1' and 'ATF2g1'.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            axis_text_x_size: Size of the x-axis text.
            axis_text_y_size: Size of the y-axis text.
            axis_title_size: Size of the axis title.
            legend_title_size: Size of the legend title.
            legend_text_size: Size of the legend text.
            legend_bbox_to_anchor: The bbox that the legend will be anchored.
            figsize: The size of the figure.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.plot_barplot(mdata["rna"], guide_rna_column="NT")

        Preview:
            .. image:: /_static/docstring_previews/mixscape_barplot.png
        """
        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run the `mixscape` function first.")
        count = pd.crosstab(index=adata.obs[mixscape_class_global], columns=adata.obs[guide_rna_column])
        all_cells_percentage = pd.melt(count / count.sum(), ignore_index=False).reset_index()
        KO_cells_percentage = all_cells_percentage[all_cells_percentage[mixscape_class_global] == "KO"]
        KO_cells_percentage = KO_cells_percentage.sort_values("value", ascending=False)

        new_levels = KO_cells_percentage[guide_rna_column]
        all_cells_percentage[guide_rna_column] = pd.Categorical(
            all_cells_percentage[guide_rna_column], categories=new_levels, ordered=False
        )
        all_cells_percentage[mixscape_class_global] = pd.Categorical(
            all_cells_percentage[mixscape_class_global], categories=["NT", "NP", "KO"], ordered=False
        )
        all_cells_percentage["gene"] = all_cells_percentage[guide_rna_column].str.rsplit("g", expand=True)[0]
        all_cells_percentage["guide_number"] = all_cells_percentage[guide_rna_column].str.rsplit("g", expand=True)[1]
        all_cells_percentage["guide_number"] = "g" + all_cells_percentage["guide_number"]
        NP_KO_cells = all_cells_percentage[all_cells_percentage["gene"] != "NT"]

        color_mapping = {"KO": "salmon", "NP": "lightgray", "NT": "grey"}
        unique_genes = NP_KO_cells["gene"].unique()
        fig, axs = plt.subplots(int(len(unique_genes) / 5), 5, figsize=figsize, sharey=True)
        for i, gene in enumerate(unique_genes):
            ax = axs[int(i / 5), i % 5]
            grouped_df = (
                NP_KO_cells[NP_KO_cells["gene"] == gene]
                .groupby(["guide_number", "mixscape_class_global"], observed=False)["value"]
                .sum()
                .unstack()
            )
            grouped_df.plot(
                kind="bar",
                stacked=True,
                color=[color_mapping[col] for col in grouped_df.columns],
                ax=ax,
                width=0.8,
                legend=False,
            )
            ax.set_title(gene, bbox={"facecolor": "white", "edgecolor": "black", "pad": 1}, fontsize=axis_title_size)
            ax.set(xlabel="sgRNA", ylabel="% of cells")
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
            ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=0, ha="right", fontsize=axis_text_x_size)
            ax.set_yticks(ax.get_yticks(), ax.get_yticklabels(), rotation=0, fontsize=axis_text_y_size)
            ax.legend(
                title="Mixscape Class",
                loc="center right",
                bbox_to_anchor=legend_bbox_to_anchor,
                frameon=True,
                fontsize=legend_text_size,
                title_fontsize=legend_title_size,
            )

        fig.subplots_adjust(right=0.8)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_heatmap(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        labels: str,
        target_gene: str,
        control: str,
        *,
        layer: str | None = None,
        method: str | None = "wilcoxon",
        subsample_number: int | None = 900,
        vmin: float | None = -2,
        vmax: float | None = 2,
        return_fig: bool = False,
        **kwds,
    ) -> Figure | None:
        """Heatmap plot using mixscape results. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            target_gene: Target gene name to visualize heatmap for.
            control: Control category from the `pert_key` column.
            layer: Key from `adata.layers` whose value will be used to perform tests on.
            method: The default method is 'wilcoxon', see `method` parameter in `scanpy.tl.rank_genes_groups` for more options.
            subsample_number: Subsample to this number of observations.
            vmin: The value representing the lower limit of the color scale. Values smaller than vmin are plotted with the same color as vmin.
            vmax: The value representing the upper limit of the color scale. Values larger than vmax are plotted with the same color as vmax.
            {common_plot_args}
            **kwds: Additional arguments to `scanpy.pl.rank_genes_groups_heatmap`.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.plot_heatmap(
            ...     adata=mdata["rna"], labels="gene_target", target_gene="IFNGR2", layer="X_pert", control="NT"
            ... )

        Preview:
            .. image:: /_static/docstring_previews/mixscape_heatmap.png
        """
        if "mixscape_class" not in adata.obs:
            raise ValueError("Please run `pt.tl.mixscape` first.")
        adata_subset = adata[(adata.obs[labels] == target_gene) | (adata.obs[labels] == control)].copy()
        sc.tl.rank_genes_groups(adata_subset, layer=layer, groupby=labels, method=method)
        sc.pp.scale(adata_subset, max_value=vmax)
        sc.pp.subsample(adata_subset, n_obs=subsample_number)

        fig = sc.pl.rank_genes_groups_heatmap(
            adata_subset,
            groupby="mixscape_class",
            vmin=vmin,
            vmax=vmax,
            n_genes=20,
            groups=["NT"],
            show=False,
            **kwds,
        )

        if return_fig:
            return fig
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_perturbscore(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        labels: str,
        target_gene: str,
        *,
        mixscape_class: str = "mixscape_class",
        color: str = "orange",
        palette: dict[str, str] = None,
        split_by: str = None,
        before_mixscape: bool = False,
        perturbation_type: str = "KO",
        return_fig: bool = False,
    ) -> Figure | None:
        """Density plots to visualize perturbation scores calculated by the `pt.tl.mixscape` function.

        Requires `pt.tl.mixscape` to be run first.

        https://satijalab.org/seurat/reference/plotperturbscore

        Args:
            adata: The annotated data object.
            labels: The column of `.obs` with target gene labels.
            target_gene: Target gene name to visualize perturbation scores for.
            mixscape_class: The column of `.obs` with mixscape classifications.
            color: Specify color of target gene class or knockout cell class. For control non-targeting and non-perturbed cells, colors are set to different shades of grey.
            palette: Optional full color palette to overwrite all colors.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate
                      the perturbation signature for every replicate separately.
            before_mixscape: Option to split densities based on mixscape classification (default) or original target gene classification.
                             Default is set to NULL and plots cells by original class ID.
            perturbation_type: Specify type of CRISPR perturbation expected for labeling mixscape classifications.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            Visualizing the perturbation scores for the cells in a dataset:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.plot_perturbscore(adata=mdata["rna"], labels="gene_target", target_gene="IFNGR2", color="orange")

        Preview:
            .. image:: /_static/docstring_previews/mixscape_perturbscore.png
        """
        if "mixscape" not in adata.uns:
            raise ValueError("Please run the `mixscape` function first.")
        perturbation_score = None
        for key in adata.uns["mixscape"][target_gene]:
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
            palette = {gd: "#7d7d7d", target_gene: color}
            plot_dens = sns.kdeplot(data=perturbation_score, x="pvec", hue=labels, fill=False, common_norm=False)
            top_r = max(plot_dens.get_lines()[cond].get_data()[1].max() for cond in range(len(plot_dens.get_lines())))
            plt.close()
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            rng = np.random.default_rng()
            perturbation_score.loc[perturbation_score[labels] == gd, "y_jitter"] = rng.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score[labels] == gd)
            )
            perturbation_score.loc[perturbation_score[labels] == target_gene, "y_jitter"] = rng.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score[labels] == target_gene)
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                sns.set_theme(style="whitegrid")
                g = sns.FacetGrid(
                    data=perturbation_score, col=split_by, hue=split_by, palette=palette, height=5, sharey=False
                )
                g.map(sns.kdeplot, "pvec", fill=True, common_norm=False, palette=palette)
                g.map(sns.scatterplot, "pvec", "y_jitter", s=10, alpha=0.5, palette=palette)
                g.set_axis_labels("Perturbation score", "Cell density")
                g.add_legend(title=split_by, fontsize=14, title_fontsize=16)
                g.despine(left=True)

            # If split_by is not provided, create a single plot
            else:
                sns.set_theme(style="whitegrid")
                sns.kdeplot(
                    data=perturbation_score, x="pvec", hue="gene_target", fill=True, common_norm=False, palette=palette
                )
                sns.scatterplot(
                    data=perturbation_score, x="pvec", y="y_jitter", hue="gene_target", palette=palette, s=10, alpha=0.5
                )
                plt.xlabel("Perturbation score", fontsize=16)
                plt.ylabel("Cell density", fontsize=16)
                plt.title("Density Plot", fontsize=18)
                plt.legend(title="gene_target", title_fontsize=14, fontsize=12)
                sns.despine()

        # If before_mixscape is False, split densities based on mixscape classifications
        else:
            if palette is None:
                palette = {gd: "#7d7d7d", f"{target_gene} NP": "#c9c9c9", f"{target_gene} {perturbation_type}": color}
            plot_dens = sns.kdeplot(data=perturbation_score, x="pvec", hue=labels, fill=False, common_norm=False)
            top_r = max(plot_dens.get_lines()[i].get_data()[1].max() for i in range(len(plot_dens.get_lines())))
            plt.close()
            perturbation_score["y_jitter"] = perturbation_score["pvec"]
            rng = np.random.default_rng()
            gd2 = list(
                set(perturbation_score["mix"]).difference([f"{target_gene} NP", f"{target_gene} {perturbation_type}"])
            )[0]
            perturbation_score.loc[perturbation_score["mix"] == gd2, "y_jitter"] = rng.uniform(
                low=0.001, high=top_r / 10, size=sum(perturbation_score["mix"] == gd2)
            ).astype(np.float32)
            perturbation_score.loc[perturbation_score["mix"] == f"{target_gene} {perturbation_type}", "y_jitter"] = (
                rng.uniform(
                    low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} {perturbation_type}")
                )
            )
            perturbation_score.loc[perturbation_score["mix"] == f"{target_gene} NP", "y_jitter"] = rng.uniform(
                low=-top_r / 10, high=0, size=sum(perturbation_score["mix"] == f"{target_gene} NP")
            )
            # If split_by is provided, split densities based on the split_by
            if split_by is not None:
                sns.set_theme(style="whitegrid")
                g = sns.FacetGrid(
                    data=perturbation_score, col=split_by, hue="mix", palette=palette, height=5, sharey=False
                )
                g.map(sns.kdeplot, "pvec", fill=True, common_norm=False, alpha=0.7)
                g.map(sns.scatterplot, "pvec", "y_jitter", s=10, alpha=0.5)
                g.set_axis_labels("Perturbation score", "Cell density")
                g.add_legend(title="mix", fontsize=14, title_fontsize=16)
                g.despine(left=True)

            # If split_by is not provided, create a single plot
            else:
                sns.set_theme(style="whitegrid")
                sns.kdeplot(
                    data=perturbation_score,
                    x="pvec",
                    hue="mix",
                    fill=True,
                    common_norm=False,
                    palette=palette,
                    alpha=0.7,
                )
                sns.scatterplot(
                    data=perturbation_score, x="pvec", y="y_jitter", hue="mix", palette=palette, s=10, alpha=0.5
                )
                plt.xlabel("Perturbation score", fontsize=16)
                plt.ylabel("Cell density", fontsize=16)
                plt.title("Density", fontsize=18)
                plt.legend(title="mixscape class", title_fontsize=14, fontsize=12)
                sns.despine()

        if return_fig:
            return plt.gcf()
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_violin(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        target_gene_idents: str | list[str],
        *,
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
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Axes | Figure | None:
        """Violin plot using mixscape results.

        Requires `pt.tl.mixscape` to be run first.

        Args:
            adata: The annotated data object.
            target_gene_idents: Target gene name to plot.
            keys: Keys for accessing variables of `.var_names` or fields of `.obs`. Default is 'mixscape_class_p_ko'.
            groupby: The key of the observation grouping to consider. Default is 'mixscape_class'.
            log: Plot on logarithmic axis.
            use_raw: Whether to use `raw` attribute of `adata`.
            stripplot: Add a stripplot on top of the violin plot.
            order: Order in which to show the categories.
            xlabel: Label of the x-axis. Defaults to `groupby` if `rotation` is `None`, otherwise, no label is shown.
            ylabel: Label of the y-axis. If `None` and `groupby` is `None`, defaults to `'value'`.
                    If `None` and `groubpy` is not `None`, defaults to `keys`.
            ax: A matplotlib axes object. Only works if plotting a single component.
            {common_plot_args}
            **kwargs: Additional arguments to `seaborn.violinplot`.

        Returns:
            If `return_fig` is `True`, returns the figure (as Axes list if it's a multi-panel plot), otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.plot_violin(
            ...     adata=mdata["rna"], target_gene_idents=["NT", "IFNGR2 NP", "IFNGR2 KO"], groupby="mixscape_class"
            ... )

        Preview:
            .. image:: /_static/docstring_previews/mixscape_violin.png
        """
        if isinstance(target_gene_idents, str):
            mixscape_class_mask = adata.obs[groupby] == target_gene_idents
        elif isinstance(target_gene_idents, list):
            mixscape_class_mask = np.full_like(adata.obs[groupby], False, dtype=bool)
            for ident in target_gene_idents:
                mixscape_class_mask |= adata.obs[groupby] == ident
        adata = adata[mixscape_class_mask]

        sanitize_anndata(adata)
        use_raw = _check_use_raw(adata, use_raw)
        if isinstance(keys, str):
            keys = [keys]
        keys = list(OrderedDict.fromkeys(keys))  # remove duplicates, preserving the order

        if isinstance(ylabel, str | type(None)):
            ylabel = [ylabel] * (1 if groupby is None else len(keys))
        if groupby is None:
            if len(ylabel) != 1:
                raise ValueError(f"Expected number of y-labels to be `1`, found `{len(ylabel)}`.")
        elif len(ylabel) != len(keys):
            raise ValueError(f"Expected number of y-labels to be `{len(keys)}`, found `{len(ylabel)}`.")

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
                **kwargs,
            )

            if stripplot:
                grouped_df = obs_tidy.groupby(x)
                for ax_id, key in zip(range(g.axes.shape[1]), keys, strict=False):
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
                for ax in g.axes[0]:  # noqa: PLR1704
                    ax.tick_params(axis="x", labelrotation=rotation)
        else:
            # set by default the violin plot cut=0 to limit the extend
            # of the violin plot (see stacked_violin code) for more info.
            kwargs.setdefault("cut", 0)
            kwargs.setdefault("inner")

            if ax is None:
                axs, _, _, _ = _utils.setup_axes(
                    ax=ax,
                    panels=["x"] if groupby is None else keys,
                    show_ticks=True,
                    right_margin=0.3,
                )
            else:
                axs = [ax]
            for ax, y, ylab in zip(axs, ys, ylabel, strict=False):
                ax = sns.violinplot(  # noqa: PLW2901
                    x=x,
                    y=y,
                    data=obs_tidy,
                    order=order,
                    orient="vertical",
                    scale=scale,
                    ax=ax,
                    hue=hue,
                    **kwargs,
                )
                # Get the handles and labels.
                handles, labels = ax.get_legend_handles_labels()
                if stripplot:
                    ax = sns.stripplot(  # noqa: PLW2901
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

        if hue is not None and stripplot is True:
            plt.legend(handles, labels)

        if return_fig:
            if multi_panel and groupby is None and len(ys) == 1:
                return g
            elif len(axs) == 1:
                return axs[0]
            else:
                return axs
        plt.show()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_lda(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        control: str,
        *,
        mixscape_class: str = "mixscape_class",
        mixscape_class_global: str = "mixscape_class_global",
        perturbation_type: str | None = "KO",
        lda_key: str | None = "mixscape_lda",
        n_components: int | None = None,
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes | None = None,
        return_fig: bool = False,
        **kwds,
    ) -> Figure | None:
        """Visualizing perturbation responses with Linear Discriminant Analysis. Requires `pt.tl.mixscape()` to be run first.

        Args:
            adata: The annotated data objectplot_heatmap.
            control: Control category from the `pert_key` column.
            mixscape_class: The column of `.obs` with the mixscape classification result.
            mixscape_class_global: The column of `.obs` with mixscape global classification result (perturbed, NP or NT).
            perturbation_type: Specify type of CRISPR perturbation expected for labeling mixscape classifications.
            n_components: The number of dimensions of the embedding.
            lda_key: If not specified, lda looks .uns["mixscape_lda"] for the LDA results.
            color_map: Matplotlib color map.
            palette: Matplotlib palette.
            ax: Matplotlib axes.
            {common_plot_args}
            **kwds: Additional arguments to `scanpy.pl.umap`.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
            >>> ms_pt.mixscape(mdata["rna"], "gene_target", "NT", layer="X_pert")
            >>> ms_pt.lda(mdata["rna"], "gene_target", "NT", split_by="replicate")
            >>> ms_pt.plot_lda(adata=mdata["rna"], control="NT")

        Preview:
            .. image:: /_static/docstring_previews/mixscape_lda.png
        """
        if mixscape_class not in adata.obs:
            raise ValueError(f'Did not find `.obs["{mixscape_class!r}"]`. Please run the `mixscape` function first.')
        if lda_key not in adata.uns:
            raise ValueError(f'Did not find `.uns["{lda_key!r}"]`. Please run the `lda` function first.')

        adata_subset = adata[
            (adata.obs[mixscape_class_global] == perturbation_type) | (adata.obs[mixscape_class_global] == control)
        ].copy()
        adata_subset.obsm[lda_key] = adata_subset.uns[lda_key]
        if n_components is None:
            n_components = adata_subset.uns[lda_key].shape[1]
        sc.pp.neighbors(adata_subset, use_rep=lda_key)
        sc.tl.umap(adata_subset, n_components=n_components)
        fig = sc.pl.umap(
            adata_subset,
            color=mixscape_class,
            palette=palette,
            color_map=color_map,
            return_fig=return_fig,
            show=False,
            ax=ax,
            **kwds,
        )

        if return_fig:
            return fig
        plt.show()
        return None


class MixscapeGaussianMixture(GaussianMixture):
    def __init__(
        self,
        n_components: int,
        fixed_means: Sequence[float] | None = None,
        fixed_covariances: Sequence[float] | None = None,
        **kwargs,
    ):
        """Custom Gaussian Mixture Model where means and covariances can be fixed for specific components.

        Args:
            n_components: Number of Gaussian components
            fixed_means: Means to fix (use None for those that should be estimated)
            fixed_covariances: Covariances to fix (use None for those that should be estimated)
            kwargs: Additional arguments passed to scikit-learn's GaussianMixture
        """
        super().__init__(n_components=n_components, **kwargs)
        self.fixed_means = fixed_means
        self.fixed_covariances = fixed_covariances

        self.fixed_mean_indices = []
        self.fixed_mean_values = []
        if fixed_means is not None:
            self.fixed_mean_indices = [i for i, m in enumerate(fixed_means) if m is not None]
            if self.fixed_mean_indices:
                self.fixed_mean_values = np.array([fixed_means[i] for i in self.fixed_mean_indices])

        self.fixed_cov_indices = []
        self.fixed_cov_values = []
        if fixed_covariances is not None:
            self.fixed_cov_indices = [i for i, c in enumerate(fixed_covariances) if c is not None]
            if self.fixed_cov_indices:
                self.fixed_cov_values = np.array([fixed_covariances[i] for i in self.fixed_cov_indices])

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray):
        """Modified M-step to respect fixed means and covariances."""
        super()._m_step(X, log_resp)

        if self.fixed_mean_indices:
            self.means_[self.fixed_mean_indices] = self.fixed_mean_values

        if self.fixed_cov_indices:
            self.covariances_[self.fixed_cov_indices] = self.fixed_cov_values

        return self
