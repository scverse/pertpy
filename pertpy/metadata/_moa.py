from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pertpy._types import as_frame

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


class Moa(MetaData):
    """Utilities to fetch metadata for mechanism of action studies."""

    def __init__(self):
        self.clue = None

    def _download_clue(self) -> None:
        clue_path = self._download_metadata("repurposing_drugs_20200324.txt")
        self.clue = pd.read_csv(clue_path, sep="	", skiprows=9)[["pert_iname", "moa", "target"]]

    def annotate(
        self,
        adata: AnnData,
        query_id: str = "perturbation",
        target: str | None = None,
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Annotate cells affected by perturbations by mechanism of action.

        For each cell, we fetch the mechanism of action and molecular targets of the compounds sourced from clue.io.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with the name of a perturbagen.
            target: The column of `.obs` with target information. If set to None, all MoAs are retrieved without comparing molecular targets.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or 'all'.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with MoA annotation.
        """
        if copy:
            adata = adata.copy()

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`.\nPlease check again.")

        if self.clue is None:
            self._download_clue()

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id].str.lower()) - set(self.clue["pert_iname"].str.lower()))
        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id="pert_iname",
            metadata_type="moa",
            verbosity=verbosity,
        )

        obs = as_frame(adata.obs)
        adata.obs = (
            obs.merge(
                self.clue,
                left_on=obs[query_id].str.lower().to_numpy(),
                right_on=self.clue["pert_iname"].str.lower().to_numpy(),
                how="left",
                suffixes=("", "_fromMeta"),
            )
            .set_index(obs.index)
            .drop("key_0", axis=1)
        )

        # If target column is given, check whether it is one of the targets listed in the metadata
        # If inconsistent, treat this perturbagen as unmatched and overwrite the annotated metadata with NaN
        if target is not None:
            annotated = as_frame(adata.obs)
            target_meta = "target" if target != "target" else "target_fromMeta"
            annotated[target_meta] = annotated[target_meta].mask(
                ~annotated.apply(lambda row: str(row[target]) in str(row[target_meta]), axis=1)
            )
            pertname_meta = "pert_iname" if query_id != "pert_iname" else "pert_iname_fromMeta"
            annotated.loc[annotated[target_meta].isna(), [pertname_meta, "moa"]] = np.nan

        # If query_id and reference_id have different names, there will be a column for each of them after merging
        # which is redundant as they refer to the same information.
        if query_id != "pert_iname":
            del as_frame(adata.obs)["pert_iname"]

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for Moa metadata.

        The LookUp object provides an overview of the metadata to annotate.

        Returns:
            Returns a LookUp object specific for MoA annotation.
        """
        if self.clue is None:
            self._download_clue()

        return LookUp(
            type="moa",
            transfer_metadata=[self.clue],
        )
