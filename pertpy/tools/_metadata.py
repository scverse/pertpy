from __future__ import annotations

from pathlib import Path

import anndata
import pandas as pd
from scanpy import settings

from pertpy.data._dataloader import _download


class MetaData:
    """Utilities to fetch cell or perturbation metadata."""

    def __init__(self):
        """Download cell line metadata from DepMap"""
        cell_line_file_name = "sample_info.csv"
        cell_line_file_path = settings.cachedir.__str__() + "/" + cell_line_file_name
        if not Path(cell_line_file_path).exists():
            _download(
                url="https://ndownloader.figshare.com/files/35020903",
                output_file_name=cell_line_file_name,
                output_path=settings.cachedir,
                is_zip=False,
            )
            self.cell_line_meta = pd.read_csv(settings.cachedir.__str__() + "/" + cell_line_file_name)
        else:
            self.cell_line_meta = pd.read_csv(cell_line_file_path)

    def annotate_cell_lines(
        self,
        adata: anndata,
        cell_line_column: str = "DepMap_ID",
        cell_line_type: str = "DepMap_ID",
        cell_line_information: list[str] = None,
        copy: bool = False,
    ) -> anndata:
        """Fetch cell line annotation.

        For each cell, we fetch cell line annotation from Dependency Map (DepMap).

        Args:
            adata: The data object to annotate.
            cell_line_column: The column of `.obs` with cell line information. (default: "DepMap_ID")
            cell_line_type: The type of cell line information, e.g. DepMap_ID, cell_line_name or stripped_cell_line_name. (default: "DepMap_ID")
            cell_line_information: The metadata to fetch. All metadata will be fetched by default. (default: all)
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with cell line annotation.
        """
        if copy:
            adata = adata.copy()
        if cell_line_type in self.cell_line_meta.columns:
            """If the specified cell line type can be found in the DepMap database,
            we can compare these keys and fetch the corresponding metadata."""
            if cell_line_information is None:
                """If no cell_line_information is specified, all metadata is fetched by default.
                Sometimes there is already different cell line information in the `adata`.
                In order to avoid redundant information,
                we will remove the duplicate information from metadata after merging."""
                adata.obs = (
                    adata.obs.reset_index()
                    .merge(
                        self.cell_line_meta,
                        left_on=cell_line_column,
                        right_on=cell_line_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index.names)
                )
                """ If cell_line_column and cell_line_type have different names,
                there will be a column for each of them after merging,
                which is redundant as they refer to the same information.
                We will move the cell_line_type column."""
                if cell_line_column != cell_line_type:
                    del adata.obs[cell_line_type]

            elif set(cell_line_information).issubset(set(self.cell_line_meta.columns)):
                """If cell_line_information is specified and can be found in the DepMap database,
                We will subset the original metadata dataframe correspondingly and add them to the `adata`.
                Again, redundant information will be removed."""
                if cell_line_type not in cell_line_information:
                    cell_line_information.append(cell_line_type)
                cell_line_meta_part = self.cell_line_meta[cell_line_information]
                adata.obs = (
                    adata.obs.reset_index()
                    .merge(
                        cell_line_meta_part,
                        left_on=cell_line_column,
                        right_on=cell_line_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index.names)
                )
                """ Again, redundant information will be removed."""
                if cell_line_column != cell_line_type:
                    del adata.obs[cell_line_type]
            else:
                raise ValueError(
                    "The specified cell line metadata can't be found in the DepMap database."
                    "Please give the cell line metadata available in the DepMap,"
                    "or fetch all the metadata by default."
                )
        else:
            raise ValueError(
                "The specified cell line type is not available in the DepMap database."
                "Please give the type of cell line information available in the DepMap."
                "e.g. DepMap_ID, cell_line_name or stripped_cell_line_name."
                "DepMap_ID is compared by default."
            )

        return adata
