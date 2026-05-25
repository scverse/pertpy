from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import entropy

from pertpy._logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _resolve_matrix(adata: AnnData, *, layer_key: str | None, embedding_key: str | None) -> np.ndarray:
    """Pick the cell-by-feature matrix from a layer, an obsm embedding, or ``.X``.

    Layer wins over embedding; both default back to ``.X``; passing both raises.
    """
    if layer_key is not None and embedding_key is not None:
        raise ValueError("Please, select just either layer or embedding for computation.")
    if layer_key is not None:
        if layer_key not in adata.layers:
            raise ValueError(f"Layer {layer_key!r} does not exist in the .layers attribute.")
        return np.asarray(adata.layers[layer_key])
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"Embedding {embedding_key!r} does not exist in the .obsm attribute.")
        return np.asarray(adata.obsm[embedding_key])
    return np.asarray(adata.X)


def _subtract_control_mean(
    matrix: np.ndarray,
    control_mask: np.ndarray,
    group_masks: list[np.ndarray],
    *,
    name: str,
) -> np.ndarray:
    """Return ``matrix`` with the within-group control mean subtracted from every row.

    Groups with no control cells are left untouched and a warning is emitted.
    """
    out = np.zeros_like(matrix, dtype=float)
    for mask in group_masks:
        in_group = mask & control_mask
        n_control = int(in_group.sum())
        if n_control == 0:
            logger.warning(
                f"No control cells found for one group when computing {name!r}; "
                "leaving those rows unchanged (no control subtraction applied)."
            )
            out[mask, :] = matrix[mask, :]
            continue
        control_mean = matrix[in_group, :] if n_control == 1 else np.mean(matrix[in_group, :], axis=0)
        out[mask, :] = matrix[mask, :] - control_mean
    return out


class PerturbationSpace:
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single data points in an embeddings summarize a cell, whereas in a perturbation space, data points summarize whole perturbations.
    """

    def __init__(self):
        self.control_diff_computed = False

    def compute_control_diff(  # type: ignore
        self,
        adata: AnnData,
        *,
        target_col: str = "perturbation",
        group_col: str = None,
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = "control_diff",
        embedding_key: str = None,
        new_embedding_key: str = "control_diff",
        all_data: bool = False,
        copy: bool = True,
    ) -> AnnData:
        """Subtract mean of the control from the perturbation.

        Args:
            adata: Anndata object of size cells x genes.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.
            group_col: `.obs` column name that stores the label of the group of each cell. If None, ignore groups.
            reference_key: The key of the control values.
            layer_key: Key of the AnnData layer to use for computation.
            new_layer_key: the results are stored in the given layer.
            embedding_key: `obsm` key of the AnnData embedding to use for computation.
            new_embedding_key: Results are stored in a new embedding in `obsm` with this key.
            all_data: if True, do the computation in all data representations (X, all layers and all embeddings)
            copy: If True returns a new AnnData; otherwise updates the input AnnData in place.

        Returns:
            Updated AnnData object.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> diff_adata = ps.compute_control_diff(mdata["rna"], target_col="gene_target", reference_key="NT")
        """
        if reference_key not in adata.obs[target_col].unique():
            raise ValueError(
                f"Reference key {reference_key} not found in {target_col}. {reference_key} must be in obs column {target_col}."
            )

        if embedding_key is not None and embedding_key not in adata.obsm:
            raise ValueError(f"Embedding key {embedding_key} not found in obsm keys of the anndata.")

        if layer_key is not None and layer_key not in adata.layers:
            raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")

        if copy:
            adata = adata.copy()

        control_mask = (adata.obs[target_col] == reference_key).to_numpy()
        if group_col is None:
            group_masks: list[np.ndarray] = [np.ones(adata.n_obs, dtype=bool)]
        else:
            group_masks = [(adata.obs[group_col] == sample).to_numpy() for sample in adata.obs[group_col].unique()]

        if layer_key:
            adata.layers[new_layer_key] = _subtract_control_mean(
                adata.layers[layer_key], control_mask, group_masks, name=new_layer_key
            )

        if embedding_key:
            adata.obsm[new_embedding_key] = _subtract_control_mean(
                adata.obsm[embedding_key], control_mask, group_masks, name=new_embedding_key
            )

        if (not layer_key and not embedding_key) or all_data:
            adata.X = _subtract_control_mean(np.asarray(adata.X), control_mask, group_masks, name="X")

        if all_data:
            for local_layer_key in list(adata.layers.keys()):
                if local_layer_key in {layer_key, new_layer_key}:
                    continue
                new_key = local_layer_key + "_control_diff"
                adata.layers[new_key] = _subtract_control_mean(
                    adata.layers[local_layer_key], control_mask, group_masks, name=new_key
                )

            for local_embedding_key in list(adata.obsm.keys()):
                if local_embedding_key in {embedding_key, new_embedding_key}:
                    continue
                new_key = local_embedding_key + "_control_diff"
                adata.obsm[new_key] = _subtract_control_mean(
                    adata.obsm[local_embedding_key], control_mask, group_masks, name=new_key
                )

        self.control_diff_computed = True

        return adata

    def _combine(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        op: Callable[[np.ndarray, np.ndarray], np.ndarray],
        reference_key: str,
        new_pert_name: str,
        ensure_consistency: bool,
        target_col: str,
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Backend for both ``add`` and ``subtract``.

        ``op`` is applied as ``op(running_total, adata[perturbation].X)`` so callers pick addition or subtraction (or anything else operating in-place on a numpy array).
        """
        perturbations = list(perturbations)
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )

        if ensure_consistency:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)
            rename_back = {
                key: key.removesuffix("_control_diff")
                for key in [*adata.layers.keys(), *adata.obsm.keys()]
                if key.endswith("_control_diff")
            }
        else:
            warnings.warn(
                "Combining perturbations without `ensure_consistency=True` is only well-defined "
                "if the input was already differenced against control "
                "(otherwise perturbation - perturbation != control).",
                stacklevel=3,
            )
            rename_back = {}

        def _running(values: np.ndarray) -> np.ndarray:
            result = values[adata.obs_names.get_loc(reference_key)].astype(float, copy=True)
            for perturbation in perturbations:
                op(result, values[adata.obs_names.get_loc(perturbation)])
            return result

        new_layers: dict[str, np.ndarray] = {}
        for layer_key, mat in adata.layers.items():
            new_layers[rename_back.get(layer_key, layer_key)] = np.concatenate(
                (np.asarray(mat), _running(np.asarray(mat))[None, :]), axis=0
            )

        new_obsm: dict[str, np.ndarray] = {}
        for embedding_key, mat in adata.obsm.items():
            new_obsm[rename_back.get(embedding_key, embedding_key)] = np.concatenate(
                (np.asarray(mat), _running(np.asarray(mat))[None, :]), axis=0
            )

        X = np.asarray(adata.X)
        new_X = np.concatenate((X, _running(X)[None, :]), axis=0)

        new_perturbation = AnnData(X=new_X)
        new_index = adata.obs_names.append(pd.Index([new_pert_name]))
        new_perturbation.obs_names = new_index
        new_perturbation.obs = adata.obs.reindex(new_index)

        for key, value in new_layers.items():
            new_perturbation.layers[key] = value
        for key, value in new_obsm.items():
            new_perturbation.obsm[key] = value

        new_perturbation.obs[target_col] = new_perturbation.obs_names.astype("category")

        if ensure_consistency:
            return new_perturbation, adata
        return new_perturbation

    def add(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = True,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Add perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to add.
            reference_key: perturbation source from which the perturbation summation starts.
            ensure_consistency: If True, differentiate against control via `compute_control_diff` before combining so that "perturbation - perturbation == control" holds in the resulting space. Set False only if the input has already been differenced.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the addition of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.add(ps_adata, perturbations=["ATF2", "CD86"], reference_key="NT")
        """
        perturbations = list(perturbations)
        new_pert_name = "+".join(perturbations)

        def _iadd(a: np.ndarray, b: np.ndarray) -> None:
            a += b

        return self._combine(
            adata,
            perturbations=perturbations,
            op=_iadd,
            reference_key=reference_key,
            new_pert_name=new_pert_name,
            ensure_consistency=ensure_consistency,
            target_col=target_col,
        )

    def subtract(
        self,
        adata: AnnData,
        *,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = True,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Subtract perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to subtract.
            reference_key: Perturbation source from which the perturbation subtraction starts.
            ensure_consistency: If True, differentiate against control via `compute_control_diff` before combining so that "perturbation - perturbation == control" holds in the resulting space. Set False only if the input has already been differenced.
            target_col: `.obs` column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the subtraction of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.subtract(ps_adata, reference_key="ATF2", perturbations=["BRD4", "CUL3"])
        """
        perturbations = list(perturbations)
        new_pert_name = reference_key + "-" + "-".join(perturbations)

        def _isub(a: np.ndarray, b: np.ndarray) -> None:
            a -= b

        return self._combine(
            adata,
            perturbations=perturbations,
            op=_isub,
            reference_key=reference_key,
            new_pert_name=new_pert_name,
            ensure_consistency=ensure_consistency,
            target_col=target_col,
        )

    def label_transfer(
        self,
        adata: AnnData,
        *,
        target_column: str = "perturbation",
        column_uncertainty_score_key: str = "perturbation_transfer_uncertainty",
        target_val: str = "unknown",
        neighbors_key: str = "neighbors",
    ) -> None:
        """Impute missing values in the specified column using KNN imputation in the space defined by `use_rep`.

        Uncertainty is calculated as the entropy of the label distribution in the neighborhood of the target cell.
        In other words, a cell where all neighbors have the same set of labels will have an uncertainty of 0, whereas a cell where all neighbors have many different labels will have high uncertainty.

        Args:
            adata: The AnnData object containing single-cell data.
            target_column: The column name in adata.obs to perform imputation on.
            column_uncertainty_score_key: The column name in adata.obs to store the uncertainty score of the label transfer.
            target_val: The target value to impute.
            neighbors_key: The key in adata.uns where the neighbors are stored.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> import numpy as np
            >>> adata = sc.datasets.pbmc68k_reduced()
            >>> # randomly dropout 10% of the data annotations
            >>> adata.obs["perturbation"] = adata.obs["louvain"].astype(str).copy()
            >>> random_cells = np.random.choice(adata.obs.index, int(adata.obs.shape[0] * 0.1), replace=False)
            >>> adata.obs.loc[random_cells, "perturbation"] = "unknown"
            >>> sc.pp.neighbors(adata)
            >>> sc.tl.umap(adata)
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps.label_transfer(adata)
        """
        if neighbors_key not in adata.uns:
            raise ValueError(f"Key {neighbors_key} not found in adata.uns. Please run `sc.pp.neighbors` first.")

        labels = adata.obs[target_column].astype(str)
        target_cells = labels == target_val

        connectivities = adata.obsp[adata.uns[neighbors_key]["connectivities_key"]]
        # convert labels to an incidence matrix
        one_hot_encoded_labels = adata.obs[target_column].astype(str).str.get_dummies()
        # convert to distance-weighted neighborhood incidence matrix
        weighted_label_occurence = pd.DataFrame(
            (one_hot_encoded_labels.values.T * connectivities).T,
            index=adata.obs_names,
            columns=one_hot_encoded_labels.columns,
        )
        # choose best label for each target cell
        best_labels = weighted_label_occurence.drop(target_val, axis=1)[target_cells].idxmax(axis=1)
        adata.obs[target_column] = labels
        adata.obs.loc[target_cells, target_column] = best_labels

        # calculate uncertainty
        uncertainty = np.zeros(adata.n_obs)
        uncertainty[target_cells] = entropy(weighted_label_occurence.drop(target_val, axis=1)[target_cells], axis=1)
        adata.obs[column_uncertainty_score_key] = uncertainty
