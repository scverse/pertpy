from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from collections.abc import Iterable

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from anndata import AnnData

class MetaDataPlot:
    """Plotting functions for Metadata."""
    @staticmethod
    def plot_correlation (
        adata: AnnData,
        corr: pd.DataFrame,
        pval: pd.DataFrame,
        identifier: str = "DepMap_ID",
        metadata_key: str = "bulk_rna_broad",
        category: str = "cell line",
        subset_identifier: Union[str, int, Iterable[str], Iterable[int], None] = None,
    ) -> None:
        """Visualise the correlation of cell lines with annotated metadata.

        Args:
            adata: Input data object.
            corr: Pearson correlation scores. If not available, please call the function `pt.md.CellLine.correlate()` first.
            pval: P-values for pearson correlation. If not available, please call the function `pt.md.CellLine.correlate()` first.
            identifier: Column in `.obs` containing the identifiers. Defaults to "DepMap_ID".
            metadata_key: Key of the AnnData obsm for comparison with the X matrix. Defaults to "bulk_rna_broad".
            category: The category for correlation comparison. Defaults to "cell line".
            subset_identifier: Selected identifiers for scatter plot visualization between the X matrix and `metadata_key`.
                              If None, all cell lines will be plotted.
                              If not None, only the chosen cell line will be plotted, either speficied as a value in `identifier` (string) or as an index number.
                              Defaults to None.
        Returns:
            Pearson correlation coefficients and their corresponding p-values for matched and unmatched cell lines separately.
        """
        if category == "cell line":
            if subset_identifier is None:
                annotation = "\n".join(
                    (
                        f"Mean pearson correlation: {np.mean(np.diag(corr)):.4f}",
                        f"Mean p-value: {np.mean(np.diag(pval)):.4f}",
                    )
                )
                plt.scatter(x=adata.obsm[metadata_key], y=adata.X)
                plt.xlabel(metadata_key)
                plt.ylabel("Baseline")
            else:
                subset_identifier_list = [subset_identifier] if isinstance(subset_identifier, (str, int)) else list(subset_identifier)

                if all(isinstance(id, int) and 0 <= id < adata.n_obs for id in subset_identifier_list):
                    # Visualize the chosen cell line at the given index
                    subset_identifier_list = adata.obs[identifier].values[subset_identifier_list]
                elif not all(isinstance(id, str) for id in subset_identifier_list) or not set(subset_identifier_list).issubset(adata.obs[identifier].unique()):
                    # The chosen cell line must be found in `identifier`
                    raise ValueError("`Subset_identifier` must contain either all strings or all integers within the index.")

                plt.scatter(x=adata.obsm[metadata_key].loc[subset_identifier_list], 
                            y=adata[adata.obs[identifier].isin(subset_identifier_list)].X)
                plt.xlabel(f"{metadata_key}: {subset_identifier_list[0]}" if len(subset_identifier_list) == 1 else f"{metadata_key}")
                plt.ylabel(f"Baseline: {subset_identifier_list[0]}" if len(subset_identifier_list) == 1 else "Baseline")

                # Annotate with the correlation coefficient and p-value of the chosen cell lines
                subset_cor = np.mean(np.diag(corr.loc[subset_identifier_list, subset_identifier_list]))
                subset_pval = np.mean(np.diag(pval.loc[subset_identifier_list, subset_identifier_list]))
                annotation = "\n".join(
                    (
                        f"Pearson correlation: {subset_cor:.4f}",
                        f"P-value: {subset_pval:.4f}",
                    )
                )

            plt.text(
                0.05,
                0.95,
                annotation,
                fontsize=10,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "alpha": 0.5, "facecolor": "white", "edgecolor": "black"},
            )
            plt.show()
        else:
            raise NotImplementedError

