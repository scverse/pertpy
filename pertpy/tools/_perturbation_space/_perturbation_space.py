from typing import Literal

import numpy as np
import scipy as sp
from anndata import AnnData


class PerturbationSpace:
    """Implements various ways of summarizing perturbations into a condensed space."""

    def calculate_mean_gene_expression(self, adata: AnnData) -> AnnData:
        pass

    def simple_differential_response(
        self,
        adata: AnnData,
        target_col: str = "perturbations",
        reference_key: str = "control",
        *,
        layer_key: str = None,
        new_layer_key: str = "differential_response",
        embedding_key: str = None,
        new_embedding_key: str = "differential_response",
        copy: bool = False,
    ) -> AnnData:
        """Obtains a simple differential response by subtracting the mean of the reference (e.g. control) from the targets.

        Args:
            adata: The AnnData object containing
            target_col: The column containing the sample states (perturbations). Must contain the 'reference_key'. Defaults to 'perturbations'.
            reference_key: The reference sample. Defaults to 'control'.
            layer_key: The layer of which to use the transcription values for to determine the differential response.
            new_layer_key: The name of the new layer to add to the AnnData object. Defaults to 'differential response'.
            embedding_key: The obsm matrix of which to use the embedding values for to determine the differential response.
            new_embedding_key: The name of the new obsm key to add to the AnnData object. Defaults to 'differential_response'.

        Returns:
            An AnnData object with either a new layer ('differential_response') or a new obsm matrix (key: 'differential_response').
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

    def clustering(
        self,
        adata: AnnData,
        method: Literal["dendrogram", "k-means", "dbscan", "spectral clustering", "gaussian mixture"],
    ) -> AnnData:
        pass

    def discriminator_classifier(self, adata: AnnData) -> AnnData:
        """

         Leveraging discriminator classifier: The idea here is that we fit either a regressor model for gene expression (see Supplemental Materials.
         here https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7289078/ (Dose-response analysis) and Sup 17-19)
         and we use either coefficient of the model for each perturbation as a feature or train a classifier example
         (simple MLP or logistic regression and take the penultimate layer as feature space and apply pseudo bulking approach above)

        Args:
            adata:

        Returns:

        """
        pass

    def calculate_meta_cells(self, adata: AnnData) -> AnnData:
        """

        See https://metacells.readthedocs.io/en/latest/Metacells_Vignette.html

        Args:
            adata:

        Returns:

        """
        pass

    def calculate_sea_cells(self, adata: AnnData) -> AnnData:
        """

        See https://github.com/dpeerlab/SEACells

        Args:
            adata:

        Returns:

        """
        pass
