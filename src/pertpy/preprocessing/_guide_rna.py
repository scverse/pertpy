from __future__ import annotations

import uuid
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from numba import njit, prange
from scipy.sparse import csr_matrix, issparse, sparray, spmatrix
from scverse_misc import Deprecation, deprecated_arg

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy._jax import jax_import
from pertpy._types import CSBase, CSRBase, cast_frame, cast_matrix

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.pyplot import Figure

_LEARNING_RATE_DEPRECATION = Deprecation(
    "1.5.0", "The mixture model is fit by EM, which has no learning rate. The argument is ignored."
)


def _nonzero_columns(X: np.ndarray | CSBase) -> list[tuple[np.ndarray, np.ndarray]]:
    """Row indices and values of the non-zero entries of every column, dropping explicitly stored zeros."""
    columns: list[tuple[np.ndarray, np.ndarray]] = []
    if isinstance(X, CSBase):
        X_csc = X.tocsc()
        for g in range(X_csc.shape[1]):
            start, end = X_csc.indptr[g], X_csc.indptr[g + 1]
            rows, data = X_csc.indices[start:end], X_csc.data[start:end]
            keep = data != 0
            columns.append((rows, data) if keep.all() else (rows[keep], data[keep]))
    else:
        for g in range(X.shape[1]):
            nonzero = np.flatnonzero(X[:, g])
            columns.append((nonzero, X[nonzero, g]))
    return columns


def _format_guides(names: Sequence[str], limit: int = 10) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


class GuideAssignment:
    """Assign cells to guide RNAs."""

    @singledispatchmethod
    def assign_by_threshold(
        self,
        data: AnnData | np.ndarray | CSBase,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ):
        """Simple threshold based gRNA assignment function.

        Each cell is assigned to gRNA with at least `assignment_threshold` counts.
        This function expects unnormalized data as input.

        Args:
            data: The (annotated) data matrix of shape `n_obs` × `n_vars`.
                  Rows correspond to cells and columns to genes.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            output_layer: Assigned guide will be saved on adata.layers[output_key].

        Examples:
            Each cell is assigned to gRNA that occurs at least 5 times in the respective cell.

            >>> import pertpy as pt
            >>> mdata = pt.data.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_by_threshold(gdo, assignment_threshold=5)
        """
        raise NotImplementedError(
            f"No implementation found for {type(data)}. Must be numpy array, sparse matrix, or AnnData object."
        )

    @assign_by_threshold.register(AnnData)
    def _assign_by_threshold_anndata(
        self,
        adata: AnnData,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ) -> None:
        X = adata.X if layer is None else adata.layers[layer]
        guide_assignments = self.assign_by_threshold(X, assignment_threshold=assignment_threshold)
        adata.layers[output_layer] = guide_assignments

    @assign_by_threshold.register(np.ndarray)
    def _assign_by_threshold_numpy(self, X: np.ndarray, /, *, assignment_threshold: float) -> np.ndarray:
        return (assignment_threshold <= X).astype(np.int8)

    @assign_by_threshold.register(spmatrix)
    @assign_by_threshold.register(sparray)
    def _assign_by_threshold_sparse(self, X: CSBase, /, *, assignment_threshold: float) -> CSRBase:
        X_csr = csr_matrix(X)
        assigned = csr_matrix(
            ((X_csr.data >= assignment_threshold).astype(np.int8), X_csr.indices.copy(), X_csr.indptr.copy()),
            shape=X_csr.shape,
        )
        assigned.eliminate_zeros()
        return assigned

    @singledispatchmethod
    def assign_to_max_guide(
        self,
        data: AnnData | np.ndarray | CSBase,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> np.ndarray | None:
        """Simple threshold based max gRNA assignment function.

        Each cell is assigned to the most expressed gRNA if it has at least `assignment_threshold` counts.
        Ties are resolved in favour of the first of the tied gRNAs.
        This function expects unnormalized data as input.

        Args:
            data: The (annotated) data matrix of shape `n_obs` × `n_vars`.
                  Rows correspond to cells and columns to genes.
            assignment_threshold: The count threshold that is required for an assignment to be viable.
            layer: Key to the layer containing raw count values of the gRNAs.
                   adata.X is used if layer is None. Expects count data.
            obs_key: Assigned guide will be saved on adata.obs[output_key].
            no_grna_assigned_key: The key to return if no gRNA is expressed enough.

        Examples:
            Each cell is assigned to the most expressed gRNA if it has at least 5 counts.

            >>> import pertpy as pt
            >>> mdata = pt.ds.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_to_max_guide(gdo, assignment_threshold=5)
        """
        raise NotImplementedError(
            f"No implementation found for {type(data)}. Must be numpy array, sparse matrix, or AnnData object."
        )

    @assign_to_max_guide.register(AnnData)
    def assign_to_max_guide_anndata(
        self,
        adata: AnnData,
        /,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> None:
        X = adata.X if layer is None else adata.layers[layer]
        guide_assignments = self.assign_to_max_guide(
            X, var=adata.var, assignment_threshold=assignment_threshold, no_grna_assigned_key=no_grna_assigned_key
        )
        adata.obs[obs_key] = guide_assignments

    @assign_to_max_guide.register(np.ndarray)
    def assign_to_max_guide_numpy(
        self,
        X: np.ndarray,
        /,
        *,
        var: pd.DataFrame,
        assignment_threshold: float,
        no_grna_assigned_key: str = "Negative",
    ) -> np.ndarray:
        return self._max_guide(
            X.max(axis=1), X.argmax(axis=1), var, assignment_threshold=assignment_threshold, no_key=no_grna_assigned_key
        )

    @staticmethod
    @njit(parallel=True)
    def _assign_max_guide_sparse(indptr, data, indices, assignment_threshold, assigned_grna):
        n_rows = len(indptr) - 1
        for i in prange(n_rows):
            row_start = indptr[i]
            row_end = indptr[i + 1]

            if row_end > row_start:
                data_row = data[row_start:row_end]
                indices_row = indices[row_start:row_end]
                max_pos = np.argmax(data_row)
                if data_row[max_pos] >= assignment_threshold:
                    assigned_grna[i] = indices_row[max_pos]
        return assigned_grna

    @assign_to_max_guide.register(spmatrix)
    @assign_to_max_guide.register(sparray)
    def assign_to_max_guide_sparse(
        self, X: CSBase, /, *, var: pd.DataFrame, assignment_threshold: float, no_grna_assigned_key: str = "Negative"
    ) -> np.ndarray:
        X_csr = csr_matrix(X)
        assigned_positions = np.full(X_csr.shape[0], -1, dtype=np.int32)
        assigned_positions = self._assign_max_guide_sparse(
            X_csr.indptr, X_csr.data, X_csr.indices, assignment_threshold, assigned_positions
        )

        assigned_grna = np.full(X_csr.shape[0], no_grna_assigned_key, dtype=object)
        mask = assigned_positions >= 0
        assigned_grna[mask] = var.index.to_numpy()[assigned_positions[mask]]
        return assigned_grna

    @staticmethod
    def _max_guide(
        max_values: np.ndarray,
        max_positions: np.ndarray,
        var: pd.DataFrame,
        *,
        assignment_threshold: float,
        no_key: str,
    ) -> np.ndarray:
        max_values = np.asarray(max_values).ravel()
        max_positions = np.asarray(max_positions).ravel()
        return np.where(max_values >= assignment_threshold, var.index.to_numpy()[max_positions], no_key)

    @singledispatchmethod
    def assign_mixture_model(
        self,
        data: AnnData | np.ndarray | CSBase,
        /,
        *,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        layer: str | None = None,
        assigned_guides_key: str = "assigned_guide",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        show_progress: bool = False,
        n_iter: int = 1000,
        learning_rate: float = 0.01,
        n_init_seeds: int = 10,
        seed: int = 2024,
    ) -> np.ndarray | None:
        """Assigns gRNAs to cells using a Poisson-Gaussian mixture model.

        The model, priors, and per-guide thresholding rule are those of ``crispat.ga_poisson_gauss`` (Velten group, https://github.com/velten-group/crispat).
        For each guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)``, ``lam ~ LogNormal(0, 1)``.
        A cell is assigned to a guide if its UMI count is at least the smallest integer ``t`` for which ``P(Normal | log2(t)) > 0.5``.

        The MAP estimate is found by multi-start EM on the distinct count values of every guide, for all guides in parallel.
        The lowest-loss solution is kept after discarding label-switched optima and degenerate optima in which the Gaussian weight vanishes or the Gaussian collapses onto a single count value.

        Args:
            data: AnnData with gRNA counts, or a dense or sparse cell-by-guide count matrix.
            model: The mixture model to use; currently only ``"poisson_gauss_mixture"`` is supported.
            layer: Layer name to use when ``data`` is an AnnData (defaults to ``X``).
            assigned_guides_key: Per-cell assignment is saved on ``adata.obs[assigned_guides_key]``.
            no_grna_assigned_key: Key to use when a cell is negative for all gRNAs.
            max_assignments_per_cell: Maximum number of gRNAs that can be assigned to a cell.
            multiple_grna_assigned_key: Key to use when more than ``max_assignments_per_cell`` gRNAs are assigned.
            multiple_grna_assignment_string: Separator used to join multiple gRNAs assigned to one cell.
            only_return_results: If ``True``, do not modify ``adata`` and return the assignment array.
            show_progress: Whether to print a progress line.
            n_iter: Maximum number of EM iterations per start.
            learning_rate: Deprecated and ignored.
            n_init_seeds: Number of prior-median initialisations per guide.
            seed: Random seed used for the prior-median initialisations.

        Examples:
            >>> import pertpy as pt
            >>> mdata = pt.ds.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_mixture_model(gdo)
        """
        raise NotImplementedError(
            f"No implementation found for {type(data)}. Must be numpy array, sparse matrix, or AnnData object."
        )

    @assign_mixture_model.register(AnnData)
    @deprecated_arg("learning_rate", _LEARNING_RATE_DEPRECATION)
    def _assign_mixture_model_anndata(
        self,
        adata: AnnData,
        /,
        *,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        layer: str | None = None,
        assigned_guides_key: str = "assigned_guide",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        show_progress: bool = False,
        n_iter: int = 1000,
        learning_rate: float = 0.01,
        n_init_seeds: int = 10,
        seed: int = 2024,
    ) -> np.ndarray | None:
        if model != "poisson_gauss_mixture":
            raise ValueError("Model not implemented. Please use 'poisson_gauss_mixture'.")
        X = cast_matrix(adata.X if layer is None else adata.layers[layer])
        binary, estimates = self._fit_mixture_pg(
            X,
            guide_names=list(adata.var_names),
            n_iter=n_iter,
            n_init_seeds=n_init_seeds,
            seed=seed,
            show_progress=show_progress,
        )
        assignments = self._binary_to_per_cell_strings(
            binary,
            np.asarray(adata.var_names, dtype=object),
            no_grna_assigned_key=no_grna_assigned_key,
            max_assignments_per_cell=max_assignments_per_cell,
            multiple_grna_assigned_key=multiple_grna_assigned_key,
            multiple_grna_assignment_string=multiple_grna_assignment_string,
        )
        if only_return_results:
            return assignments

        for column, estimate in estimates.items():
            adata.var[column] = estimate
        adata.obs[assigned_guides_key] = pd.Categorical(assignments)
        return None

    @assign_mixture_model.register(np.ndarray)
    @assign_mixture_model.register(spmatrix)
    @assign_mixture_model.register(sparray)
    @deprecated_arg("learning_rate", _LEARNING_RATE_DEPRECATION)
    def _assign_mixture_model_matrix(
        self,
        X: np.ndarray | CSBase,
        /,
        *,
        var: pd.DataFrame,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        show_progress: bool = False,
        n_iter: int = 1000,
        learning_rate: float = 0.01,
        n_init_seeds: int = 10,
        seed: int = 2024,
    ) -> np.ndarray:
        if model != "poisson_gauss_mixture":
            raise ValueError("Model not implemented. Please use 'poisson_gauss_mixture'.")
        binary, _ = self._fit_mixture_pg(
            X,
            guide_names=list(var.index),
            n_iter=n_iter,
            n_init_seeds=n_init_seeds,
            seed=seed,
            show_progress=show_progress,
        )
        return self._binary_to_per_cell_strings(
            binary,
            np.asarray(var.index, dtype=object),
            no_grna_assigned_key=no_grna_assigned_key,
            max_assignments_per_cell=max_assignments_per_cell,
            multiple_grna_assigned_key=multiple_grna_assigned_key,
            multiple_grna_assignment_string=multiple_grna_assignment_string,
        )

    def _fit_mixture_pg(
        self,
        X: np.ndarray | CSBase,
        *,
        guide_names: list[str],
        n_iter: int,
        n_init_seeds: int,
        seed: int,
        show_progress: bool,
    ) -> tuple[CSRBase, dict[str, np.ndarray]]:
        """Fit the Poisson-Gaussian mixture for every guide.

        Returns a sparse binary cell-by-guide assignment matrix and the per-guide estimates and thresholds.
        """
        with jax_import("assign_mixture_model"):
            from pertpy.preprocessing._guide_rna_mixture import compute_count_thresholds, fit_poisson_gauss_mixture

        n_cells, n_guides = X.shape
        if issparse(X):
            X = X.tocsc()
        if (X.data.size > 0 and X.data.min() < 0) if isinstance(X, CSBase) else np.any(np.asarray(X) < 0):
            raise ValueError(
                "Data contains negative values. Please use non-negative data for guide assignment with the Mixture Model."
            )
        columns = _nonzero_columns(X)
        nonzero_counts = np.array([values.size for _, values in columns])
        max_counts = np.array([values.max() if values.size else 0 for _, values in columns], dtype=np.float64)

        # crispat fits only on cells with non-zero counts and requires max count >= 2.
        fittable = (nonzero_counts >= 2) & (max_counts >= 2)
        skipped = np.flatnonzero(~fittable)
        if skipped.size:
            warn(
                f"Skipping {skipped.size} guides with fewer than 2 expressing cells or a maximum count below 2: "
                f"{_format_guides([guide_names[g] for g in skipped])}.",
                stacklevel=5,
            )

        fittable_idx = np.flatnonzero(fittable)
        estimates = {
            name: np.full(n_guides, np.nan, dtype=np.float64)
            for name in (
                "poisson_rate",
                "gaussian_mean",
                "gaussian_std",
                "mix_probs_0",
                "mix_probs_1",
                "threshold",
                "final_loss",
            )
        }
        thresholds = estimates["threshold"]
        if fittable_idx.size > 0:
            uniques = [np.unique(columns[g][1], return_counts=True) for g in fittable_idx]
            values = np.zeros((fittable_idx.size, max(u.size for u, _ in uniques)), dtype=np.float32)
            weights = np.zeros_like(values)
            for i, (unique_counts, multiplicities) in enumerate(uniques):
                values[i, : unique_counts.size] = np.log2(unique_counts)
                weights[i, : unique_counts.size] = multiplicities

            if show_progress:
                print(f"Fitting Poisson-Gaussian mixture for {fittable_idx.size} guides in parallel.")

            fit = fit_poisson_gauss_mixture(values, weights, n_iter=n_iter, n_init_seeds=n_init_seeds, seed=seed)
            thresholds[fittable_idx] = compute_count_thresholds(fit, max_counts[fittable_idx])
            estimates["poisson_rate"][fittable_idx] = fit.poisson_rate
            estimates["gaussian_mean"][fittable_idx] = fit.gaussian_mean
            estimates["gaussian_std"][fittable_idx] = fit.gaussian_std
            estimates["mix_probs_0"][fittable_idx] = fit.mix_probs[:, 0]
            estimates["mix_probs_1"][fittable_idx] = fit.mix_probs[:, 1]
            estimates["final_loss"][fittable_idx] = fit.final_loss

            unassignable = fittable_idx[np.isnan(thresholds[fittable_idx])]
            if unassignable.size:
                warn(
                    f"No positive population found for {unassignable.size} guides, which are not assigned to any cell: "
                    f"{_format_guides([guide_names[g] for g in unassignable])}.",
                    stacklevel=5,
                )

        assigned = [
            (rows[values >= thresholds[g]], g)
            for g, (rows, values) in enumerate(columns)
            if not np.isnan(thresholds[g])
        ]
        row_idx = np.concatenate([rows for rows, _ in assigned]) if assigned else np.empty(0, dtype=np.intp)
        col_idx = (
            np.concatenate([np.full(rows.size, g) for rows, g in assigned]) if assigned else np.empty(0, dtype=np.intp)
        )
        binary = csr_matrix((np.ones(row_idx.size, dtype=np.int8), (row_idx, col_idx)), shape=(n_cells, n_guides))
        return binary, estimates

    @staticmethod
    def _binary_to_per_cell_strings(
        binary: CSRBase,
        guide_names: np.ndarray,
        *,
        no_grna_assigned_key: str,
        max_assignments_per_cell: int,
        multiple_grna_assigned_key: str,
        multiple_grna_assignment_string: str,
    ) -> np.ndarray:
        n_cells = binary.shape[0]
        n_assigned = np.diff(binary.indptr)
        if any(multiple_grna_assignment_string in name for name in guide_names):
            warn(
                f"Some guide names contain the separator {multiple_grna_assignment_string!r}, so assignments of several guides to one cell cannot be split unambiguously. "
                "Pass a different `multiple_grna_assignment_string`.",
                stacklevel=5,
            )
        assignments = np.full(n_cells, no_grna_assigned_key, dtype=object)
        single = n_assigned == 1
        assignments[single] = guide_names[binary.indices[binary.indptr[:-1][single]]]
        multi = (n_assigned > 1) & (n_assigned <= max_assignments_per_cell)
        if multi.any():
            cell_of_entry = np.repeat(np.arange(n_cells), n_assigned)
            entries = multi[cell_of_entry]
            joined = (
                pd.Series(guide_names[binary.indices[entries]])
                .groupby(cell_of_entry[entries])
                .agg(multiple_grna_assignment_string.join)
            )
            assignments[joined.index.to_numpy()] = joined.to_numpy()
        assignments[n_assigned > max_assignments_per_cell] = multiple_grna_assigned_key
        return assignments

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_heatmap(  # pragma: no cover # noqa: D417
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        order_by: np.ndarray | str | None = None,
        key_to_save_order: str | None = None,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Heatmap plotting of guide RNA expression matrix.

        Assuming guides have sparse expression, this function reorders cells
        and plots guide RNA expression so that a nice sparse representation is achieved.
        The cell ordering can be stored and reused in future plots to obtain consistent
        plots before and after analysis of the guide RNA expression.
        Note: This function expects a log-normalized or binary data.

        Args:
            adata: Annotated data matrix containing gRNA values
            layer: Key to the layer containing log normalized count values of the gRNAs.
                   adata.X is used if layer is None.
            order_by: The order of cells in y axis.
                      If None, cells will be reordered to have a nice sparse representation.
                      If a string is provided, adata.obs[order_by] will be used as the order.
                      If a numpy array is provided, the array will be used for ordering.
            key_to_save_order: The obs key to save cell orders in the current plot. Only saves if not None.
            {common_plot_args}
            kwargs: Are passed to sc.pl.heatmap.

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.
            The position of every cell in the y-axis will be saved on `adata.obs[key_to_save_order]` if provided,
            so that passing it as `order_by` reproduces the plot.

        Examples:
            Each cell is assigned to gRNA that occurs at least 5 times in the respective cell, which is then
            visualized using a heatmap.

            >>> import pertpy as pt
            >>> mdata = pt.ds.papalexi_2021()
            >>> gdo = mdata.mod["gdo"]
            >>> ga = pt.pp.GuideAssignment()
            >>> ga.assign_by_threshold(gdo, assignment_threshold=5)
            >>> ga.plot_heatmap(gdo)
        """
        data = cast_matrix(adata.X if layer is None else adata.layers[layer])

        if order_by is None:
            if isinstance(data, CSBase):
                max_values = data.max(axis=1).toarray().squeeze()
                data_argmax = np.asarray(data.argmax(axis=1)).squeeze()
                max_guide_index = np.where(max_values != data.min(axis=1).toarray().squeeze(), data_argmax, -1)
            else:
                max_guide_index = np.where(
                    data.max(axis=1).squeeze() != data.min(axis=1).squeeze(), data.argmax(axis=1).squeeze(), -1
                )
            order = np.argsort(max_guide_index)
        elif isinstance(order_by, str):
            order = np.argsort(adata.obs[order_by])
        else:
            order = order_by

        temp_col_name = f"_tmp_pertpy_grna_plot_{uuid.uuid4()}"
        adata.obs[temp_col_name] = pd.Categorical(["" for _ in range(adata.shape[0])])

        if key_to_save_order is not None:
            adata.obs[key_to_save_order] = np.argsort(order)

        try:
            fig = sc.pl.heatmap(
                adata[order, :],
                var_names=adata.var.index.tolist(),
                groupby=temp_col_name,
                use_raw=False,
                dendrogram=False,
                layer=layer,
                show=False,
                **kwargs,
            )
        finally:
            del cast_frame(adata.obs)[temp_col_name]

        if return_fig:
            return fig  # type: ignore[return-value]
        plt.show()
        return None
