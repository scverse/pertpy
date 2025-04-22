from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scanpy import settings

from pertpy.data._dataloader import _download

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


class Moa(MetaData):
    """Utilities to fetch metadata for mechanism of action studies."""

    def __init__(self):
        self.clue = None

    def _download_clue(self) -> None:
        clue_path = Path(settings.cachedir) / "repurposing_drugs_20200324.txt"
        if not Path(clue_path).exists():
            _download(
                url="https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt",
                output_file_name="repurposing_drugs_20200324.txt",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.clue = pd.read_csv(clue_path, sep="	", skiprows=9)
        self.clue = self.clue[["pert_iname", "moa", "target"]]

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

        adata.obs = (
            adata.obs.merge(
                self.clue,
                left_on=adata.obs[query_id].str.lower(),
                right_on=self.clue["pert_iname"].str.lower(),
                how="left",
                suffixes=("", "_fromMeta"),
            )
            .set_index(adata.obs.index)
            .drop("key_0", axis=1)
        )

        # If target column is given, check whether it is one of the targets listed in the metadata
        # If inconsistent, treat this perturbagen as unmatched and overwrite the annotated metadata with NaN
        if target is not None:
            target_meta = "target" if target != "target" else "target_fromMeta"
            adata.obs[target_meta] = adata.obs[target_meta].mask(
                ~adata.obs.apply(lambda row: str(row[target]) in str(row[target_meta]), axis=1)
            )
            pertname_meta = "pert_iname" if query_id != "pert_iname" else "pert_iname_fromMeta"
            adata.obs.loc[adata.obs[target_meta].isna(), [pertname_meta, "moa"]] = np.nan

        # If query_id and reference_id have different names, there will be a column for each of them after merging
        # which is redundant as they refer to the same information.
        if query_id != "pert_iname":
            del adata.obs["pert_iname"]

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for Moa metadata.

        The LookUp object provides an overview of the metadata to annotate.
        annotate_moa function has a corresponding lookup function in the LookUp object,
        where users can search the query_ids and targets in the metadata.

        Returns:
            Returns a LookUp object specific for MoA annotation.
        """
        if self.clue is None:
            self._download_clue()

        return LookUp(
            type="moa",
            transfer_metadata=[self.clue],
        )
