from abc import ABC, abstractmethod
from anndata import AnnData
import numpy as np


class PerturbationSpace(ABC):
    """Implements various ways of interacting with PerturbationSpaces.

    We differentiate between a cell space and a perturbation space.
    Visually speaking, in cell spaces single dota points in an embeddings summarize a cell,
    whereas in a perturbation space, data points summarize whole perturbations.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    
    def differential_expression(  # type: ignore
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        reference_key: str = "control",
        layer_key: str = None,
        new_layer_key: str = "differential_response",
        embedding_key: str = None,
        new_embedding_key: str = "differential_response",
        copy: bool = False,
    ):
        """
            Takes as input an Anndata object of size cells x genes. 
            The .obs column 'target_col' stores the label of the perturbation applied to each cell.
            The label 'reference_key' indicates the control perturbation
            If 'layer_key' is specified and exists in the adata, the pseudobulk computation is done by using it. Otherwise, computation is done with .X
            If 'embedding_key' is specified and exists in the adata, the clustering is done with that embedding. Otherwise, computation is done with .X
            If 'copy' is True, create a new Anndata, otherwise update the existent
            Return a Anndata of size cells x genes with the differential expression in the selected .X, layer or embedding.
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
