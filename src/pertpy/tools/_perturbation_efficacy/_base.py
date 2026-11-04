from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import scanpy as sc
from fast_array_utils.conv import to_dense
from fast_array_utils.stats import mean, mean_var
from numba import njit, prange
from pandas.errors import PerformanceWarning
from scanpy.tools._utils import _choose_representation
from scipy.sparse import csr_array, csr_matrix, sparray, vstack

from pertpy._types import CSBase, RankGenesMethod, cast_matrix

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


@njit(parallel=True)
def _csr_subset_kernel(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, rows: np.ndarray, col_map: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(rows.size + 1, dtype=np.int64)
    for i in prange(rows.size):
        for k in range(indptr[rows[i]], indptr[rows[i] + 1]):
            if col_map[indices[k]] >= 0:
                counts[i + 1] += 1
    out_indptr = np.cumsum(counts)
    out_indices = np.empty(out_indptr[-1], dtype=indices.dtype)
    out_data = np.empty(out_indptr[-1], dtype=data.dtype)
    for i in prange(rows.size):
        pos = out_indptr[i]
        for k in range(indptr[rows[i]], indptr[rows[i] + 1]):
            new_col = col_map[indices[k]]
            if new_col >= 0:
                out_indices[pos] = new_col
                out_data[pos] = data[k]
                pos += 1
    return out_indptr, out_indices, out_data


def _subset_rows_cols(X: np.ndarray | CSBase, rows: np.ndarray, cols: np.ndarray) -> np.ndarray | CSBase:
    """`X[rows][:, cols]` for unique `cols`, without materializing the full selected rows of a CSR matrix."""
    if not (isinstance(X, CSBase) and X.format == "csr"):
        return X[rows][:, cols]
    col_map = np.full(X.shape[1], -1, dtype=np.int64)
    col_map[cols] = np.arange(cols.size)
    indptr, indices, data = _csr_subset_kernel(X.indptr, X.indices, X.data, rows, col_map)
    return type(X)((data, indices, indptr), shape=(rows.size, cols.size))


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
            copy: Determines whether a copy of the `adata` is returned.
            **kwargs: Additional arguments for the `NNDescent` class from `pynndescent`.

        Returns:
            If `copy=True`, returns the copy of `adata` with the perturbation signature in `.layers["X_pert"]`.
            Otherwise, writes the perturbation signature directly to `.layers["X_pert"]` of the provided `adata`.

        Examples:
            Calcutate perturbation signature for each cell in the dataset:

            >>> import pertpy as pt
            >>> mdata = pt.ds.papalexi_2021()
            >>> ms_pt = pt.tl.Mixscape()
            >>> ms_pt.perturbation_signature(mdata["rna"], "perturbation", "NT", split_by="replicate")
        """
        if ref_selection_mode not in ["nn", "split_by"]:
            raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
        if ref_selection_mode == "split_by" and split_by is None:
            raise ValueError("split_by must be provided if ref_selection_mode is 'split_by'.")

        if copy:
            adata = adata.copy()

        input_is_sparray = isinstance(adata.X, sparray)
        X = csr_matrix(cast_matrix(adata.X)) if input_is_sparray else cast_matrix(adata.X)
        X_pert = None if isinstance(X, CSBase) else X.copy()
        sparse_blocks: list[tuple[np.ndarray, CSBase]] = []

        def _set_rows(rows: np.ndarray, values) -> None:
            if X_pert is None:
                sparse_blocks.append((rows, csr_matrix(values, dtype=X.dtype)))
            else:
                X_pert[rows] = values

        control_mask = np.asarray(adata.obs[pert_key] == control)

        if ref_selection_mode == "split_by":
            for split in adata.obs[split_by].unique():
                split_mask = np.asarray(adata.obs[split_by] == split)
                split_rows = np.flatnonzero(split_mask)
                control_mean_expr = mean(X[control_mask & split_mask], axis=0)
                _set_rows(
                    split_rows,
                    np.repeat(control_mean_expr.reshape(1, -1), split_rows.size, axis=0) - X[split_rows],
                )
        else:
            if split_by is None:
                split_masks = [np.full(adata.n_obs, True, dtype=bool)]
            else:
                split_obs = adata.obs[split_by]
                split_masks = [np.asarray(split_obs == cat) for cat in split_obs.unique()]

            representation = cast_matrix(_choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs))
            if isinstance(representation, sparray):
                representation = csr_matrix(representation)
            if n_dims is not None and n_dims < representation.shape[1]:
                representation = representation[:, :n_dims]

            from pynndescent import NNDescent

            eps = kwargs.pop("epsilon", 0.1)
            for split_mask in split_masks:
                control_mask_split = control_mask & split_mask
                split_rows = np.flatnonzero(split_mask)
                R_split = representation[split_mask]
                R_control = representation[np.asarray(control_mask_split)]
                nn_index = NNDescent(R_control, **kwargs)
                indices, _ = nn_index.query(R_split, k=n_neighbors, epsilon=eps)
                X_split_control = X[control_mask_split]
                X_control = (
                    X_split_control.expm1() if isinstance(X_split_control, CSBase) else np.expm1(X_split_control)
                )
                n_split = split_rows.size
                n_control = X_control.shape[0]

                if batch_size is None:
                    col_indices = np.ravel(indices)
                    row_indices = np.repeat(np.arange(n_split), n_neighbors)
                    neigh_matrix = csr_matrix(
                        (np.ones_like(col_indices, dtype=np.float64), (row_indices, col_indices)),
                        shape=(n_split, n_control),
                    )
                    neigh_matrix /= n_neighbors
                    _set_rows(split_rows, sc.pp.log1p(neigh_matrix @ X_control) - X[split_rows])
                else:
                    for i in range(0, n_split, batch_size):
                        size = min(i + batch_size, n_split)
                        select = slice(i, size)
                        batch = np.ravel(indices[select])
                        split_batch = split_rows[select]
                        size = size - i
                        means_batch = to_dense(X_control[batch])
                        batch_reshaped = means_batch.reshape(size, n_neighbors, -1)
                        means_batch, _ = mean_var(batch_reshaped, axis=1)
                        _set_rows(split_batch, np.log1p(means_batch) - X[split_batch])

        if X_pert is not None:
            adata.layers["X_pert"] = X_pert
        else:
            rows = np.concatenate([r for r, _ in sparse_blocks]) if sparse_blocks else np.array([], dtype=int)
            unset_rows = np.setdiff1d(np.arange(adata.n_obs), rows)
            if unset_rows.size:
                sparse_blocks.append((unset_rows, csr_matrix(X[unset_rows])))
                rows = np.concatenate([rows, unset_rows])
            x_pert = vstack([block for _, block in sparse_blocks], format="csr")[np.argsort(rows)]
            x_pert.eliminate_zeros()
            x_pert.sort_indices()
            adata.layers["X_pert"] = csr_array(x_pert) if input_is_sparray else x_pert

        if copy:
            return adata

    @staticmethod
    def _seurat_expression_stats(X: np.ndarray | CSBase) -> tuple[np.ndarray, np.ndarray]:
        """Detection fractions and Seurat-style log2 mean expression for log1p data.

        Seurat's default `data`-slot fold change uses
        log2((sum(expm1(expression)) + 1) / n_cells).
        """
        n_cells = X.shape[0]
        if isinstance(X, CSBase):
            detected = np.asarray((X > 0).sum(axis=0)).ravel() / n_cells
            linear_sum = np.asarray(X.expm1().sum(axis=0)).ravel()
        else:
            X = np.asarray(X)
            detected = np.count_nonzero(X > 0, axis=0) / n_cells
            linear_sum = np.expm1(X).sum(axis=0)
        return detected, np.log2((linear_sum + 1) / n_cells)

    def _get_perturbation_markers(
        self,
        adata: AnnData,
        *,
        split_masks: list[np.ndarray | pd.Series],
        categories: list[str],
        pert_key: str,
        control: str,
        layer: str | None,
        pval_cutoff: float,
        min_de_genes: float,
        logfc_threshold: float,
        test_method: RankGenesMethod,
    ) -> dict[tuple, np.ndarray]:
        """Select Mixscape marker genes with Seurat's `TopDEGenesMixscape` defaults.

        The Wilcoxon (or other requested) test is performed by Scanpy; selection
        uses Seurat's full-feature Bonferroni correction, 10% detection filter,
        3-cell minimum per group, and fold change for log1p-normalized data.
        The statistical test backends are not guaranteed numerically identical.
        """
        perturbation_markers: dict[tuple, np.ndarray] = {}
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            adata_split = adata[split_mask].copy()
            labels = adata_split.obs[pert_key]
            gene_targets = [gene for gene in labels.unique() if gene != control]
            group_sizes = labels.value_counts()
            valid_targets = [gene for gene in gene_targets if group_sizes[gene] >= 3]
            for gene in gene_targets:
                perturbation_markers[(category, gene)] = np.array([], dtype=str)
            if group_sizes.get(control, 0) < 3 or not valid_targets:
                continue

            expression = cast_matrix(adata_split.layers[layer] if layer is not None else adata_split.X)
            control_expression = expression[np.asarray(labels == control)]
            control_pct, control_mean = self._seurat_expression_stats(control_expression)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", PerformanceWarning)
                sc.tl.rank_genes_groups(
                    adata_split,
                    layer=layer,
                    groupby=pert_key,
                    groups=valid_targets,
                    reference=control,
                    method=test_method,
                    use_raw=False,
                    corr_method="bonferroni",
                )
                de_results = adata_split.uns["rank_genes_groups"]
                for gene in valid_targets:
                    target_expression = expression[np.asarray(labels == gene)]
                    target_pct, target_mean = self._seurat_expression_stats(target_expression)
                    gene_names = de_results["names"][gene]
                    gene_idx = adata_split.var_names.get_indexer(gene_names)
                    detection_mask = np.maximum(target_pct[gene_idx], control_pct[gene_idx]) >= 0.1
                    fold_change_mask = np.abs(target_mean[gene_idx] - control_mean[gene_idx]) >= logfc_threshold
                    pval_mask = de_results["pvals_adj"][gene] < pval_cutoff
                    markers = gene_names[detection_mask & fold_change_mask & pval_mask]
                    if len(markers) >= min_de_genes:
                        perturbation_markers[(category, gene)] = markers

        return perturbation_markers
