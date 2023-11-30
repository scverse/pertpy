from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pandas as pd
import pubchempy as pcp
from rich import print
from scanpy import settings

from pertpy.data._dataloader import _download

from ._look_up import LookUp

if TYPE_CHECKING:
    from anndata import AnnData


class CompoundMetaData:
    """Utilities to fetch metadata for compounds."""

    def __init__(self):
        settings.cachedir = ".pertpy_cache"

    def __print_unmatched_ids(
        self,
        total_identifiers: int = 0,
        unmatched_identifiers: list[str] | None = None,
        metadata_type: Literal["pubchem"] | None = None,
        show_warning: int | str = 5,
    ) -> None:
        """Helper function to print out the unmatched identifiers.

        Args:
            total_identifiers: The total number of identifiers in the `adata` object.
            unmatched_identifiers: A list of unmatched identifiers to print.
            metadata_type: The type of metadata. Defaults to None.
            show_warning: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.

        """
        if isinstance(show_warning, str):
            if show_warning != "all":
                raise ValueError("Only a non-negative value or 'all' is accepted.")
            else:
                show_warning = len(unmatched_identifiers)
        if isinstance(show_warning, int) and show_warning >= 0:
            show_warning = min(show_warning, len(unmatched_identifiers))
            print(
                f"[bold blue]There are {total_identifiers} identifiers in `adata.obs`."
                f"However, {len(unmatched_identifiers)} identifiers can't be found in the {metadata_type} annotation,"
                "leading to the presence of NA values for their respective metadata.\n",
                "Please check again: ",
                *unmatched_identifiers[:show_warning],
                sep="\n- ",
            )
        else:
            raise ValueError("Only 'all' or a non-negative value is accepted.")

    def annotate_compounds(
        self,
        adata: AnnData,
        query_id: str = "perturbation",
        query_id_type: Literal["name", "cid"] = "name",
        show_warning: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch compound annotation.

        For each cell, we fetch compound annotation via pubchempy.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with compound identifiers. Defaults to "perturbation".
            query_id_type: The type of compound identifiers, 'name' or 'cid'. Defaults to "name".
            show_warning: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with compound annotation.
        """
        if copy:
            adata = adata.copy()

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`. \n" "Please check again. ")

        query_dict = {}
        not_matched_identifiers = []
        for compound in adata.obs[query_id].dropna().astype(str).unique():
            if query_id_type == "name":
                cids = pcp.get_compounds(compound, "name")
                if len(cids) == 0:  # search did not work
                    not_matched_identifiers.append(compound)
                if len(cids) >= 1:
                    # If the name matches the first synonym offered by PubChem (outside of capitalization),
                    # it is not changed (outside of capitalization). Otherwise, it is replaced with the first synonym.
                    query_dict[compound] = [cids[0].synonyms[0], cids[0].cid, cids[0].canonical_smiles]
            else:
                try:
                    cid = pcp.Compound.from_cid(compound)
                    query_dict[compound] = [cid.synonyms[0], compound, cid.canonical_smiles]
                except pcp.BadRequestError:
                    # pubchempy throws badrequest if a cid is not found
                    not_matched_identifiers.append(compound)

        identifier_num_all = len(adata.obs[query_id].unique())
        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"Attempting to find the query id {query_id} in the adata.obs in pubchem database.\n"
                "However, none of them could be found.\n"
                "The annotation process has been halted.\n"
                "To resolve this issue, please call the `CompoundMetaData.lookup()` function to create a LookUp object.\n"
                "By using the `LookUp.compound()` method. "
            )

        if len(not_matched_identifiers) > 0:
            self.__print_unmatched_ids(
                total_identifiers=identifier_num_all,
                unmatched_identifiers=not_matched_identifiers,
                show_warning=show_warning,
                metadata_type="pubchem",
            )

        query_df = pd.DataFrame.from_dict(query_dict, orient="index", columns=["pubchem_name", "pubchem_ID", "smiles"])
        # Merge and remove duplicate columns
        # Column is converted to float after merging due to unmatches
        # Convert back to integers
        if query_id_type == "cid":
            query_df.pubchem_ID = query_df.pubchem_ID.astype("Int64")
            adata.obs = (
                adata.obs.merge(
                    query_df, left_on=query_id, right_on="pubchem_ID", how="left", suffixes=("", "_fromMeta")
                )
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
        else:
            adata.obs = (
                adata.obs.merge(query_df, left_on=query_id, right_index=True, how="left", suffixes=("", "_fromMeta"))
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
            adata.obs.pubchem_ID = adata.obs.pubchem_ID.astype("Int64")
        return adata
