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
    """Utilities to fetch metadata for compound."""

    def __init__(self):
        settings.cachedir = ".pertpy_cache"

    def annotate_compound(
        self,
        adata: AnnData,
        query_id: str = "pert_iname",
        copy: bool = False,
    ) -> AnnData:
        """Fetch compound annotation.

        For each cell, we fetch compound annotation via pubchempy.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with compound identifier. Defaults to "pert_iname".
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
        for p in adata.obs[query_id].cat.categories:  # list(adata.obs.perturbation_name.unique())
            cids = pcp.get_cids(p, "name", list_return="flat")
            if len(cids) == 0:  # search did not work
                not_matched_identifiers.append(p)
            if len(cids) >= 1:
                ## If the name matches the first synonym offered by PubChem (outside of capitalization), it is not
                ## changed (outside of capitalization). Otherwise, it is replaced with the first synonym.
                cmpd = pcp.Compound.from_cid(cids[0])
                query_dict[p] = [cmpd.synonyms[0], cids[0], cmpd.to_dict()["canonical_smiles"]]

        identifier_num_all = len(adata.obs[query_id].unique())
        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"Attempting to match the query id {query_id} in the adata.obs to the pert_iname in the metadata.\n"
                "However, none of the query IDs could be found in the compound annotation data.\n"
                "The annotation process has been halted.\n"
                "To resolve this issue, please call the `CompoundMetaData.lookup()` function to create a LookUp object.\n"
                "By using the `LookUp.compound()` method. "
            )

        if len(not_matched_identifiers) > 0:
            print(
                f"[bold blue]There are {identifier_num_all} types of perturbation in `adata.obs`."
                f"However, {len(not_matched_identifiers)} can't be found in the compound annotation,"
                "leading to the presence of NA values for their respective metadata.\n",
                "Please check again: ",
                *not_matched_identifiers,
                sep="\n- ",
            )
        query_df = pd.DataFrame.from_dict(query_dict, orient="index", columns=["pubchem_name", "pubchem_ID", "smiles"])
        adata.obs = adata.obs.merge(query_df, left_on=query_id, right_index=True, how="left")
