from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from anndata import AnnData
from rich import print

if TYPE_CHECKING:
    from collections.abc import Iterable


class PerturbationSpace:
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single dota points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.
    """

    def __init__(self):
        self.control_diff_computed = False

    def compute_control_diff(  # type: ignore
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = "control_diff",
        embedding_key: str = None,
        new_embedding_key: str = "control_diff",
        all_data: bool = False,
        copy: bool = False,
    ):
        """Subtract mean of the control from the perturbation.

        Args:
            adata: Anndata object of size cells x genes.
            target_col: .obs column name that stores the label of the perturbation applied to each cell. Defaults to 'perturbations'.
            reference_key: The key of the control values. Defaults to 'control'.
            layer_key: Key of the AnnData layer to use for computation. Defaults to the `X` matrix otherwise.
            new_layer_key: the results are stored in the given layer. Defaults to 'differential diff'.
            embedding_key: `obsm` key of the AnnData embedding to use for computation. Defaults to the 'X' matrix otherwise.
            new_embedding_key: Results are stored in a new embedding in `obsm` with this key. Defaults to 'control diff'.
            all_data: if True, do the computation in all data representations (X, all layers and all embeddings)
            copy: If True returns a new Anndata of same size with the new column; otherwise it updates the initial AnnData object.

        Examples:
            Example usage with PseudobulkSpace:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> diff_adata = ps.compute_control_diff(mdata["rna"], target_col="gene_target", reference_key='NT')
        """
        if reference_key not in adata.obs[target_col].unique():
            raise ValueError(
                f"Reference key {reference_key} not found in {target_col}. {reference_key} must be in obs column {target_col}."
            )

        if embedding_key is not None and embedding_key not in adata.obsm_keys():
            raise ValueError(f"Embedding key {embedding_key} not found in obsm keys of the anndata.")

        if layer_key is not None and layer_key not in adata.layers.keys():
            raise ValueError(f"Layer {layer_key!r} does not exist in the anndata.")

        if copy:
            adata = adata.copy()

        control_mask = adata.obs[target_col] == reference_key
        num_control = control_mask.sum()

        if layer_key:
            if num_control == 1:
                control_expression = adata.layers[layer_key][control_mask, :]
            else:
                control_expression = np.mean(adata.layers[layer_key][control_mask, :], axis=0)
            diff_matrix = adata.layers[layer_key] - control_expression
            adata.layers[new_layer_key] = diff_matrix

        if embedding_key:
            if num_control == 1:
                control_expression = adata.obsm[embedding_key][control_mask, :]
            else:
                control_expression = np.mean(adata.obsm[embedding_key][control_mask, :], axis=0)
            diff_matrix = adata.obsm[embedding_key] - control_expression
            adata.obsm[new_embedding_key] = diff_matrix

        if (not layer_key and not embedding_key) or all_data:
            if num_control == 1:
                control_expression = adata.X[control_mask, :]
            else:
                control_expression = np.mean(adata.X[control_mask, :], axis=0)
            diff_matrix = adata.X - control_expression
            adata.X = diff_matrix

        if all_data:
            layers_keys = list(adata.layers.keys())
            for local_layer_key in layers_keys:
                if local_layer_key != layer_key and local_layer_key != new_layer_key:
                    diff_matrix = adata.layers[local_layer_key] - np.mean(
                        adata.layers[local_layer_key][control_mask, :], axis=0
                    )
                    adata.layers[local_layer_key + "_control_diff"] = diff_matrix

            embedding_keys = list(adata.obsm_keys())
            for local_embedding_key in embedding_keys:
                if local_embedding_key != embedding_key and local_embedding_key != new_embedding_key:
                    diff_matrix = adata.obsm[local_embedding_key] - np.mean(
                        adata.obsm[local_embedding_key][control_mask, :], axis=0
                    )
                    adata.obsm[local_embedding_key + "_control_diff"] = diff_matrix

        self.control_diff_computed = True

        return adata

    def add(
        self,
        adata: AnnData,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = False,
        target_col: str = "perturbations",
    ):
        """Add perturbations linearly. Assumes input of size n_perts x dimensionality

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to add.
            reference_key: perturbation source from which the perturbation summation starts.
            ensure_consistency: If True, runs differential expression on all data matrices to ensure consistency of linear space.
            target_col: .obs column name that stores the label of the perturbation applied to each cell. Defaults to 'perturbations'.

        Examples:
            Example usage with PseudobulkSpace:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.add(ps_adata, perturbations=["ATF2", "CD86"], reference_key='NT')
        """
        new_pert_name = ""
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )
            new_pert_name += perturbation + "+"

        if not ensure_consistency:
            print(
                "[bold yellow]Operation might be done in non-consistent space (perturbation - perturbation != control). \n"
                "Subtract control perturbation needed for consistency of space in all data representations. \n"
                "Run with ensure_consistency=True"
            )
        else:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)

        data: dict[str, np.array] = {}

        for local_layer_key in adata.layers.keys():
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

        if "layers" in data.keys():
            for key in data["layers"]:
                key_name = key
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.layers[key_name] = data["layers"][key]

        if "embeddings" in data.keys():
            key_name = key
            for key in data["embeddings"]:
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.obsm[key_name] = data["embeddings"][key]

        if ensure_consistency:
            return new_perturbation, adata

        return new_perturbation

    def subtract(
        self,
        adata: AnnData,
        perturbations: Iterable[str],
        reference_key: str = "control",
        ensure_consistency: bool = False,
        target_col: str = "perturbations",
    ):
        """Subtract perturbations linearly. Assumes input of size n_perts x dimensionality

        Args:
            adata: Anndata object of size n_perts x dim.
            perturbations: Perturbations to subtract,
            reference_key: Perturbation source from which the perturbation subtraction starts
            ensure_consistency: If True, runs differential expression on all data matrices to ensure consistency of linear space.
            target_col: .obs column name that stores the label of the perturbation applied to each cell. Defaults to 'perturbations'.

        Examples:
            Example usage with PseudobulkSpace:
            >>> import pertpy as pt
            >>> mdata = pt.dt.papalexi_2021()
            >>> ps = pt.tl.PseudobulkSpace()
            >>> ps_adata = ps.compute(mdata["rna"], target_col="gene_target", groups_col="gene_target")
            >>> new_perturbation = ps.add(ps_adata, reference_key="ATF2", perturbations=["BRD4", "CUL3"])
        """
        new_pert_name = reference_key + "-"
        for perturbation in perturbations:
            if perturbation not in adata.obs_names:
                raise ValueError(
                    f"Perturbation {perturbation} not found in adata.obs_names. {perturbation} must be in adata.obs_names."
                )
            new_pert_name += perturbation + "-"

        if not ensure_consistency:
            print(
                "[bold yellow]Operation might be done in non-consistent space (perturbation - perturbation != control).\n"
                "Subtract control perturbation needed for consistency of space in all data representations.\n"
                "Run with ensure_consistency=True"
            )
        else:
            adata = self.compute_control_diff(adata, copy=True, all_data=True, target_col=target_col)

        data: dict[str, np.array] = {}

        for local_layer_key in adata.layers.keys():
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

        if "layers" in data.keys():
            for key in data["layers"]:
                key_name = key
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.layers[key_name] = data["layers"][key]

        if "embeddings" in data.keys():
            key_name = key
            for key in data["embeddings"]:
                if key.endswith("_control_diff"):
                    key_name = key.removesuffix("_control_diff")
                new_perturbation.obsm[key_name] = data["embeddings"][key]

        if ensure_consistency:
            return new_perturbation, adata

        return new_perturbation
