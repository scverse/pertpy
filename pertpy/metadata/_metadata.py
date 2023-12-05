from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence


class MetaData:
    def __init__(self):
        pass

    def _print_unmatched_ids(
        self,
        total_identifiers: int = 0,
        unmatched_identifiers: Sequence[str] = None,
        metadata_type: Literal[
            "cell line", "protein expression", "bulk RNA expression", "drug response", "moa", "compound"
        ] = "cell line",
        verbosity: int | str = 5,
    ) -> None:
        """Helper function to print out the unmatched identifiers.

        Args:
            total_identifiers: The total number of identifiers in the `adata` object.
            unmatched_identifiers: Unmatched identifiers in the `adata` object.
            metadata_type: The type of metadata where some identifiers are not matched during annotation, cell line, protein expression, bulk RNA expression, drug response, moa or compound. Defaults to "cell line".
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
        """
        if isinstance(verbosity, str):
            if verbosity != "all":
                raise ValueError("Only a non-negative value or 'all' is accepted.")
            else:
                verbosity = len(unmatched_identifiers)
        if isinstance(verbosity, int) and verbosity >= 0:
            verbosity = min(verbosity, len(unmatched_identifiers))
            if verbosity > 0:
                print(
                    f"[bold blue]There are {total_identifiers} identifiers in `adata.obs`."
                    f"However, {len(unmatched_identifiers)} identifiers can't be found in the {metadata_type} annotation,"
                    "leading to the presence of NA values for their respective metadata.\n",
                    "Please check again: ",
                    *unmatched_identifiers[:verbosity],
                    sep="\n- ",
                )
        else:
            raise ValueError("Only 'all' or a non-negative value is accepted.")
