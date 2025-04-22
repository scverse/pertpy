from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from lamin_utils import logger
from rich import print
from scipy.stats import entropy

if TYPE_CHECKING:
    from collections.abc import Iterable


class PerturbationSpace:
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single data points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.
    """

    def __init__(self):
        self.control_diff_computed = False

    def compute_control_diff(  # type: ignore
        self,
        adata: AnnData,
        target_col: str = "perturbation",
        group_col: str = None,
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = "control_diff",
        embedding_key: str = None,
        new_embedding_key: str = "control_diff",
        all_data: bool = False,
        copy: bool = False,
    ) -> AnnData:
        """Subtract mean of the control from the perturbation.

        Args:
            adata: Anndata object of size cells x genes.
            target_col: .obs column name that stores the label of the perturbation applied to each cell.
            group_col: .obs column name that stores the label of the group of each cell. If None, ignore groups.
            reference_key: The key of the control values.
            layer_key: Key of the AnnData layer to use for computation.
            new_layer_key: the results are stored in the given layer.
            embedding_key: `obsm` key of the AnnData embedding to use for computation.
            new_embedding_key: Results are stored in a new embedding in `obsm` with this key.
            all_data: if True, do the computation in all data representations (X, all layers and all embeddings)
            copy: If True returns a new Anndata of same size with the new column; otherwise it updates the initial AnnData object.

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

        if embedding_key is not None and embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding key {embedding_key} not found in obsm keys of the anndata.")

        if layer_key is not None and layer_key not in adata.layers:
            raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")

        if copy:
            adata = adata.copy()

        control_mask = adata.obs[target_col] == reference_key
        group_masks = (
            [(adata.obs[group_col] == sample) for sample in adata.obs[group_col].unique()]
            if group_col
            else [[True] * adata.n_obs]
        )

        if layer_key:
            adata.layers[new_layer_key] = np.zeros((adata.n_obs, adata.n_vars))
            for mask in group_masks:
                num_control = (control_mask & mask).sum()
                if num_control == 1:
                    control_expression = adata.layers[layer_key][(control_mask & mask), :]
                elif num_control > 1:
                    control_expression = np.mean(adata.layers[layer_key][(control_mask & mask), :], axis=0)
                else:
                    control_expression = np.zeros((1, adata.n_vars))
                adata.layers[new_layer_key][mask, :] = adata.layers[layer_key][mask, :] - control_expression

        if embedding_key:
            adata.obsm[new_embedding_key] = np.zeros(adata.obsm[embedding_key].shape)
            for mask in group_masks:
                num_control = (control_mask & mask).sum()
                if num_control == 1:
                    control_expression = adata.obsm[embedding_key][(control_mask & mask), :]
                elif num_control > 1:
                    control_expression = np.mean(adata.obsm[embedding_key][(control_mask & mask), :], axis=0)
                else:
                    control_expression = np.zeros((1, adata.n_vars))
                adata.obsm[new_embedding_key][mask, :] = adata.obsm[embedding_key][mask, :] - control_expression

        if (not layer_key and not embedding_key) or all_data:
            adata_x = np.zeros((adata.n_obs, adata.n_vars))
            for mask in group_masks:
                num_control = (control_mask & mask).sum()
                if num_control == 1:
                    control_expression = adata.X[(control_mask & mask), :]
                elif num_control > 1:
                    control_expression = np.mean(adata.X[(control_mask & mask), :], axis=0)
                else:
                    control_expression = np.zeros((1, adata.n_vars))
                adata_x[mask, :] = adata.X[mask, :] - control_expression
            adata.X = adata_x

        if all_data:
            layers_keys = list(adata.layers.keys())
            for local_layer_key in layers_keys:
                if local_layer_key not in (layer_key, new_layer_key):
                    adata.layers[local_layer_key + "_control_diff"] = np.zeros((adata.n_obs, adata.n_vars))
                    for mask in group_masks:
                        adata.layers[local_layer_key + "_control_diff"][mask, :] = adata.layers[local_layer_key][
                            mask, :
                        ] - np.mean(adata.layers[local_layer_key][(control_mask & mask), :], axis=0)

            embedding_keys = list(adata.obsm_keys())
            for local_embedding_key in embedding_keys:
                if local_embedding_key not in (embedding_key, new_embedding_key):
                    adata.obsm[local_embedding_key + "_control_diff"] = np.zeros(adata.obsm[local_embedding_key].shape)
                    for mask in group_masks:
                        adata.obsm[local_embedding_key + "_control_diff"][mask, :] = adata.obsm[local_embedding_key][
                            mask, :
                        ] - np.mean(adata.obsm[local_embedding_key][(control_mask & mask), :], axis=0)

        self.control_diff_computed = True

        return adata

    def add(
        self,
        adata: AnnData,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = False,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Add perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to add.
            reference_key: perturbation source from which the perturbation summation starts.
            ensure_consistency: If True, runs differential expression on all data matrices to ensure consistency of linear space.
            target_col: .obs column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the addition of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object
            provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.add(ps_adata, perturbations=["ATF2", "CD86"], reference_key="NT")
        """
        new_pert_name = ""
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )
            new_pert_name += perturbation + "+"

        if not ensure_consistency:
            logger.warning(
                "Operation might be done in non-consistent space (perturbation - perturbation != control). \n"
                "Subtract control perturbation needed for consistency of space in all data representations. \n"
                "Run with ensure_consistency=True"
            )
        else:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)

        data: dict[str, np.array] = {}

        for local_layer_key in adata.layers:
            data["layers"] = {}
            control_local = adata[reference_key].layers[local_layer_key].copy()
            for perturbation in perturbations:
                control_local += adata[perturbation].layers[local_layer_key]
            original_data = adata.layers[local_layer_key].copy()
            new_data = np.concatenate((original_data, control_local))
            data["layers"][local_layer_key] = new_data

        for local_embedding_key in adata.obsm_keys():
            data["embeddings"] = {}
            control_local = adata[reference_key].obsm[local_embedding_key].copy()
            for perturbation in perturbations:
                control_local += adata[perturbation].obsm[local_embedding_key]
            original_data = adata.obsm[local_embedding_key].copy()
            new_data = np.concatenate((original_data, control_local))
            data["embeddings"][local_embedding_key] = new_data

        # Operate in X
        control = adata[reference_key].X.copy()
        for perturbation in perturbations:
            control += adata[perturbation].X

        # Fill all obs fields with NaNs
        new_pert_obs = [np.nan for _ in adata.obs]

        original_data = adata.X.copy()
        new_data = np.concatenate((original_data, control))
        new_perturbation = AnnData(X=new_data)

        original_obs_names = adata.obs_names
        new_obs_names = original_obs_names.append(pd.Index([new_pert_name[:-1]]))
        new_perturbation.obs_names = new_obs_names

        new_obs = adata.obs.copy()
        new_obs.loc[new_pert_name[:-1]] = new_pert_obs
        new_perturbation.obs = new_obs

        if "layers" in data:
            for key in data["layers"]:
                key_name = key
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.layers[key_name] = data["layers"][key]

        if "embeddings" in data:
            key_name = key
            for key in data["embeddings"]:
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.obsm[key_name] = data["embeddings"][key]

        new_perturbation.obs[target_col] = new_perturbation.obs_names.astype("category")

        if ensure_consistency:
            return new_perturbation, adata

        return new_perturbation

    def subtract(
        self,
        adata: AnnData,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = False,
        target_col: str = "perturbation",
    ) -> tuple[AnnData, AnnData] | AnnData:
        """Subtract perturbations linearly. Assumes input of size n_perts x dimensionality.

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to subtract.
            reference_key: Perturbation source from which the perturbation subtraction starts.
            ensure_consistency: If True, runs differential expression on all data matrices to ensure consistency of linear space.
            target_col: .obs column name that stores the label of the perturbation applied to each cell.

        Returns:
            Anndata object of size (n_perts+1) x dim, where the last row is the subtraction of the specified perturbations.
            If ensure_consistency is True, returns a tuple of (new_perturbation, adata) where adata is the AnnData object
            provided as input but updated using compute_control_diff.

        Examples:
            Example usage with PseudobulkSpace:

            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.subtract(ps_adata, reference_key="ATF2", perturbations=["BRD4", "CUL3"])
        """
        new_pert_name = reference_key + "-"
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )
            new_pert_name += perturbation + "-"

        if not ensure_consistency:
            logger.warning(
                "Operation might be done in non-consistent space (perturbation - perturbation != control).\n"
                "Subtract control perturbation needed for consistency of space in all data representations.\n"
                "Run with ensure_consistency=True"
            )
        else:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)

        data: dict[str, np.array] = {}

        for local_layer_key in adata.layers:
            data["layers"] = {}
            control_local = adata[reference_key].layers[local_layer_key].copy()
            for perturbation in perturbations:
                control_local -= adata[perturbation].layers[local_layer_key]
            original_data = adata.layers[local_layer_key].copy()
            new_data = np.concatenate((original_data, control_local))
            data["layers"][local_layer_key] = new_data

        for local_embedding_key in adata.obsm_keys():
            data["embeddings"] = {}
            control_local = adata[reference_key].obsm[local_embedding_key].copy()
            for perturbation in perturbations:
                control_local -= adata[perturbation].obsm[local_embedding_key]
            original_data = adata.obsm[local_embedding_key].copy()
            new_data = np.concatenate((original_data, control_local))
            data["embeddings"][local_embedding_key] = new_data

        # Operate in X
        control = adata[reference_key].X.copy()
        for perturbation in perturbations:
            control -= adata[perturbation].X

        # Fill all obs fields with NaNs
        new_pert_obs = [np.nan for _ in adata.obs]

        original_data = adata.X.copy()
        new_data = np.concatenate((original_data, control))
        new_perturbation = AnnData(X=new_data)

        original_obs_names = adata.obs_names
        new_obs_names = original_obs_names.append(pd.Index([new_pert_name[:-1]]))
        new_perturbation.obs_names = new_obs_names

        new_obs = adata.obs.copy()
        new_obs.loc[new_pert_name[:-1]] = new_pert_obs
        new_perturbation.obs = new_obs

        if "layers" in data:
            for key in data["layers"]:
                key_name = key
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.layers[key_name] = data["layers"][key]

        if "embeddings" in data:
            key_name = key
            for key in data["embeddings"]:
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.obsm[key_name] = data["embeddings"][key]

        new_perturbation.obs[target_col] = new_perturbation.obs_names.astype("category")

        if ensure_consistency:
            return new_perturbation, adata

        return new_perturbation

    def label_transfer(
        self,
        adata: AnnData,
        column: str = "perturbation",
        column_uncertainty_score_key: str = "perturbation_transfer_uncertainty",
        target_val: str = "unknown",
        neighbors_key: str = "neighbors",
    ) -> None:
        """Impute missing values in the specified column using KNN imputation in the space defined by `use_rep`.

        Uncertainty is calculated as the entropy of the label distribution in the neighborhood of the target cell.
        In other words, a cell where all neighbors have the same set of labels will have an uncertainty of 0, whereas a cell
        where all neighbors have many different labels will have high uncertainty.

        Args:
            adata: The AnnData object containing single-cell data.
            column: The column name in adata.obs to perform imputation on.
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

        labels = adata.obs[column].astype(str)
        target_cells = labels == target_val

        connectivities = adata.obsp[adata.uns[neighbors_key]["connectivities_key"]]
        # convert labels to an incidence matrix
        one_hot_encoded_labels = adata.obs[column].astype(str).str.get_dummies()
        # convert to distance-weighted neighborhood incidence matrix
        weighted_label_occurence = pd.DataFrame(
            (one_hot_encoded_labels.values.T * connectivities).T,
            index=adata.obs_names,
            columns=one_hot_encoded_labels.columns,
        )
        # choose best label for each target cell
        best_labels = weighted_label_occurence.drop(target_val, axis=1)[target_cells].idxmax(axis=1)
        adata.obs[column] = labels
        adata.obs.loc[target_cells, column] = best_labels

        # calculate uncertainty
        uncertainty = np.zeros(adata.n_obs)
        uncertainty[target_cells] = entropy(weighted_label_occurence.drop(target_val, axis=1)[target_cells], axis=1)
        adata.obs[column_uncertainty_score_key] = uncertainty
