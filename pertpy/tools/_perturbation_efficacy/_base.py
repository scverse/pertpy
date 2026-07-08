from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scanpy as sc
from fast_array_utils.stats import mean, mean_var
from pandas.errors import PerformanceWarning
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_array, csr_matrix, issparse, sparray

if TYPE_CHECKING:
    from anndata import AnnData


class PerturbationEfficacyAnalyzer:
    """Shared substrate for the perturbation efficacy analysis tools.

    It holds the steps that both the binary :class:`~pertpy.tools.Mixscape` classification and the continuous :class:`~pertpy.tools.Mixscale` scoring build on.
    These are computing the perturbation signature and detecting the differentially expressed marker genes per perturbation.
    """

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

        The perturbation signature is calculated by subtracting the mRNA expression profile of each cell from the averaged mRNA expression profile of the control cells (selected according to `ref_selection_mode`).
        The implementation resembles https://satijalab.org/seurat/reference/runmixscape.
        Note that in the original implementation, the perturbation signature is calculated on unscaled data by default, and we therefore recommend to do the same.

        Args:
            adata: The annotated data object.
            pert_key: The column  of `.obs` with perturbation categories, should also contain `control`.
            control: Name of the control category from the `pert_key` column.
            ref_selection_mode: Method to select reference cells for the perturbation signature calculation.
                If `nn`, the `n_neighbors` cells from the control pool with the most similar mRNA expression profiles are selected.
                If `split_by`, the control cells from the same split in `split_by` (e.g. indicating biological replicates) are used to calculate the perturbation signature.
            split_by: Provide the column `.obs` if multiple biological replicates exist to calculate the perturbation signature for every replicate separately.
            n_neighbors: Number of neighbors from the control to use for the perturbation signature.
            use_rep: Use the indicated representation. `'X'` or any key for `.obsm` is valid.
                If `None`, the representation is chosen automatically:
                For `.n_vars` < 50, `.X` is used, otherwise 'X_pca' is used.
                If 'X_pca' is not present, it's computed with default parameters.
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

        # pynndescent and the LIL workflow below only support legacy scipy sparse matrices, so a sparse array
        # input is computed on as a csr_matrix and converted back to a sparse array at the end.
        input_is_sparray = isinstance(adata.X, sparray)
        X = csr_matrix(adata.X) if input_is_sparray else adata.X
        adata.layers["X_pert"] = X.copy()

        # Work with LIL for efficient indexing but don't store it in AnnData as LIL is not supported anymore
        X_pert_lil = adata.layers["X_pert"].tolil() if issparse(adata.layers["X_pert"]) else adata.layers["X_pert"]

        control_mask = adata.obs[pert_key] == control

        if ref_selection_mode == "split_by":
            for split in adata.obs[split_by].unique():
                split_mask = adata.obs[split_by] == split
                control_mask_group = control_mask & split_mask
                control_mean_expr = mean(X[control_mask_group], axis=0)
                X_pert_lil[split_mask] = (
                    np.repeat(control_mean_expr.reshape(1, -1), split_mask.sum(), axis=0) - X_pert_lil[split_mask]
                )
        else:
            if split_by is None:
                split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            else:
                split_obs = adata.obs[split_by]
                split_masks = [split_obs == cat for cat in split_obs.unique()]

            representation = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
            if isinstance(representation, sparray):
                representation = csr_matrix(representation)
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
                X_control = np.expm1(X[np.asarray(control_mask_split)])
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
                    X_pert_lil[np.asarray(split_mask)] = (
                        sc.pp.log1p(neigh_matrix @ X_control) - X_pert_lil[np.asarray(split_mask)]
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
                        X_pert_lil[split_batch] = np.log1p(means_batch) - X_pert_lil[split_batch]

        if issparse(X_pert_lil):
            x_pert = X_pert_lil.tocsr()
            adata.layers["X_pert"] = csr_array(x_pert) if input_is_sparray else x_pert
        else:
            adata.layers["X_pert"] = X_pert_lil

        if copy:
            return adata

    def _get_perturbation_markers(
        self,
        adata: AnnData,
        *,
        split_masks: list[np.ndarray],
        categories: list[str],
        pert_key: str,
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
            pert_key: The column of `.obs` with target gene labels.
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
            gene_targets = list(set(adata[split_mask].obs[pert_key]).difference([control]))
            adata_split = adata[split_mask].copy()
            # find top DE genes between cells with targeting and non-targeting gRNAs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", PerformanceWarning)
                sc.tl.rank_genes_groups(
                    adata_split,
                    layer=layer,
                    groupby=pert_key,
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
