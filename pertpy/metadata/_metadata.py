from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lamin_utils import logger

if TYPE_CHECKING:
    from collections.abc import Sequence


class MetaData:
    """Superclass for pertpy's MetaData components."""

    def _warn_unmatch(
        self,
        total_identifiers: int,
        unmatched_identifiers: Sequence[str],
        query_id: str,
        reference_id: str,
        metadata_type: Literal[
            "cell line",
            "protein expression",
            "bulk RNA",
            "drug response",
            "moa",
            "compound",
        ] = "cell line",
        verbosity: int | str = 5,
    ) -> None:
        """Helper function to print out the unmatched identifiers.

        Args:
            total_identifiers: The total number of identifiers in the `adata` object.
            unmatched_identifiers: Unmatched identifiers in the `adata` object.
            query_id: The column of `.obs` with cell line information.
            reference_id: The type of cell line identifier in the metadata.
            metadata_type: The type of metadata where some identifiers are not matched during annotation such as
                           cell line, protein expression, bulk RNA expression, drug response, moa or compound.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or 'all'.
        """
        if isinstance(verbosity, str):
            if verbosity != "all":
                raise ValueError("Only a non-negative value or 'all' is accepted.")
            else:
                verbosity = len(unmatched_identifiers)

        if len(unmatched_identifiers) == total_identifiers:
            hint = ""
            if metadata_type in ["protein expression", "bulk RNA", "drug response"]:
                hint = "Additionally, call the `CellLineMetaData.annotate()` function to acquire more possible query IDs that can be used for cell line annotation purposes."
            raise ValueError(
                f"Attempting to match the query id {query_id} in 'adata.obs' to the reference id {reference_id} in the metadata.\n"
                "However, none of the query IDs could be found in the {metadata_type} annotation data.\n"
                "To resolve this issue, call the `lookup()` function to create a LookUp object.\n"
                "This enables obtaining the count of matched identifiers in the AnnData object for different types of reference and query IDs.\n"
                f"{hint}"
            )
        if len(unmatched_identifiers) == 0:
            return
        if isinstance(verbosity, int) and verbosity >= 0:
            verbosity = min(verbosity, len(unmatched_identifiers))
            if verbosity > 0:
                logger.info(
                    f"There are {total_identifiers} identifiers in `adata.obs`."
                    f"However, {len(unmatched_identifiers)} identifiers can't be found in the {metadata_type} annotation, "
                    "leading to the presence of NA values for their respective metadata.\n"
                    f"Please check again: *unmatched_identifiers[:verbosity]..."
                )
        else:
            raise ValueError("Only 'all' or a non-negative value is accepted.")
