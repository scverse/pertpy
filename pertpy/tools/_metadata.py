import anndata
import pandas as pd


class MetaData:
    """Python implementation of MetaData."""

    def __init__(self):
        self.cell_line_meta = pd.read_csv("https://ndownloader.figshare.com/files/35020903")

    def annotate_cell_lines(
        self,
        adata: anndata,
        cell_line_column: str = "DepMap_ID",
        cell_line_type: str = "DepMap_ID",
        cell_line_information: list = None,
    ):
        """Fetch cell line annotation.

        For each cell, we fetch cell line annotation from Dependency Map (DepMap).

        Args:
            adata: The data object to annotate.
            cell_line_column: The column of `.obs` with cell line information.
            cell_line_type: The type of cell line information, e.g. DepMap_ID, cell_line_name or stripped_cell_line_name.
            cell_line_information: The metadata to fetch. All metada will be fetched by default.

        Returns:
            Returns an `adata` with cell line annotation.
        """
        adata_copy = adata.copy()
        if cell_line_type in self.cell_line_meta.columns:
            if cell_line_information is None:
                adata_copy.obs = (
                    adata_copy.obs.reset_index()
                    .merge(
                        self.cell_line_meta,
                        left_on=cell_line_column,
                        right_on=cell_line_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata_copy.obs.index.names)
                )
                if cell_line_column != cell_line_type:
                    del adata_copy.obs[cell_line_type]
            elif set(cell_line_information).issubset(set(self.cell_line_meta.columns)):
                if cell_line_type not in cell_line_information:
                    cell_line_information.append(cell_line_type)
                cell_line_meta_part = self.cell_line_meta[cell_line_information]
                adata_copy.obs = (
                    adata_copy.obs.reset_index()
                    .merge(
                        cell_line_meta_part,
                        left_on=cell_line_column,
                        right_on=cell_line_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata_copy.obs.index.names)
                )
                if cell_line_column != cell_line_type:
                    del adata_copy.obs[cell_line_type]
            else:
                raise ValueError("Cell line metadata not found.")

        else:
            raise ValueError("Please choose correct cell line type.")
        return adata_copy
