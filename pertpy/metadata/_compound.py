from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pubchempy as pcp

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


class Compound(MetaData):
    """Utilities to fetch metadata for compounds."""

    def __init__(self):
        super().__init__()

    def annotate_compounds(
        self,
        adata: AnnData,
        query_id: str = "perturbation",
        query_id_type: Literal["name", "cid"] = "name",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch compound annotation from pubchempy.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with compound identifiers.
            query_id_type: The type of compound identifiers, 'name' or 'cid'.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all".
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with compound annotation.
        """
        if copy:
            adata = adata.copy()

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`.\n Please check again.")

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
                    query_dict[compound] = [
                        cids[0].synonyms[0],
                        cids[0].cid,
                        cids[0].canonical_smiles,
                    ]
            else:
                try:
                    cid = pcp.Compound.from_cid(compound)
                    query_dict[compound] = [
                        cid.synonyms[0],
                        compound,
                        cid.canonical_smiles,
                    ]
                except pcp.BadRequestError:
                    # pubchempy throws badrequest if a cid is not found
                    not_matched_identifiers.append(compound)

        identifier_num_all = len(adata.obs[query_id].unique())
        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id=query_id_type,
            metadata_type="compound",
            verbosity=verbosity,
        )

        query_df = pd.DataFrame.from_dict(query_dict, orient="index", columns=["pubchem_name", "pubchem_ID", "smiles"])
        # Merge and remove duplicate columns
        # Column is converted to float after merging due to unmatches
        # Convert back to integers afterwards
        if query_id_type == "cid":
            query_df.pubchem_ID = query_df.pubchem_ID.astype("Int64")
            adata.obs = (
                adata.obs.merge(
                    query_df,
                    left_on=query_id,
                    right_on="pubchem_ID",
                    how="left",
                    suffixes=("", "_fromMeta"),
                )
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
        else:
            adata.obs = (
                adata.obs.merge(
                    query_df,
                    left_on=query_id,
                    right_index=True,
                    how="left",
                    suffixes=("", "_fromMeta"),
                )
                .filter(regex="^(?!.*_fromMeta)")
                .set_index(adata.obs.index)
            )
            adata.obs.pubchem_ID = adata.obs.pubchem_ID.astype("Int64")

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for CompoundMetaData.

        The LookUp object provides an overview of the metadata to annotate.
        Each annotate_{metadata} function has a corresponding lookup function in the LookUp object,
        where users can search the reference_id in the metadata and compare with the query_id in their own data.

        Returns:
            Returns a LookUp object specific for compound annotation.
        """
        return LookUp(type="compound")
