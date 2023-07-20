from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class PerturbationSpace:
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single dota points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.
    """

    def compute_control_diff(  # type: ignore
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = "control_diff",
        embedding_key: str = None,
        new_embedding_key: str = "control_diff",
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
            copy: If True returns a new Anndata of same size with the new column; otherwise it updates the initial AnnData object.
        """
        if reference_key not in adata.obs[target_col].unique():
            raise ValueError(
                f"Reference key {reference_key} not found in {target_col}. {reference_key} must be in obs column {target_col}."
            )

        if copy:
            adata = adata.copy()

        control_mask = adata.obs[target_col] == reference_key

        if layer_key:
            diff_matrix = adata.layers[layer_key] - np.mean(adata.layers[layer_key][~control_mask, :], axis=0)
            adata[new_layer_key] = diff_matrix

        elif embedding_key:
            diff_matrix = adata.obsm[embedding_key] - np.mean(adata.obsm[embedding_key][~control_mask, :], axis=0)
            adata.obsm[new_embedding_key] = diff_matrix
        else:
            diff_matrix = adata.X - np.mean(adata.X[~control_mask, :], axis=0)
            adata.X = diff_matrix

        return adata

    def add(self):
        raise NotImplementedError

    def subtract(self):
        raise NotImplementedError
