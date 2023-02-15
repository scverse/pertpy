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
            print("No DepMap metadata file found. Start downloading now.")
            _download(
                url="https://ndownloader.figshare.com/files/35020903",
                output_file_name=cell_line_file_name,
                output_path=settings.cachedir,
                is_zip=False,
            )
        self.cell_line_meta = pd.read_csv(cell_line_file_path)

        """Download meta data for driver genes of the intOGen analysis from DepMap_Sanger"""
        driver_gene_intOGen_file_name = "IntOGen-Drivers.zip"
        driver_gene_intOGen_file_path = (
            settings.cachedir.__str__() + "/2020-02-02_IntOGen-Drivers-20200213/Compendium_Cancer_Genes.tsv"
        )
        if not Path(driver_gene_intOGen_file_path).exists():
            print("No metadata file was found for driver genes of the intOGen analysis. Start downloading now.")
            _download(
                url="https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip",
                output_file_name=driver_gene_intOGen_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )

        self.driver_gene_intOGen = pd.read_table(driver_gene_intOGen_file_path)

        """Download meta data for driver genes of the COSMIC Tier 1 gene from DepMap_Sanger """
        self.driver_gene_cosmic = pd.read_csv("Census_allSun Feb 12 10_56_28 2023.csv")

        """Download bulk RNA-seq data collated from the Wellcome Sanger Institute and the Broad Institute """
        bulk_rna_sanger_file_name = "rnaseq_sanger.zip"
        bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316.csv"
        if not Path(bulk_rna_sanger_file_path).exists():
            print("No metadata file was found for bulk RNA-seq data of Sanger cell line. Start downloading now.")
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/rnaseq_sanger_20210316.zip",
                output_file_name=bulk_rna_sanger_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )

        self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_file_path)

        bulk_rna_broad_file_name = "rnaseq_broad.zip"
        bulk_rna_broad_file_path = settings.cachedir.__str__() + "/rnaseq_broad_20210317.csv"
        if not Path(bulk_rna_broad_file_path).exists():
            print("No metadata file was found for bulk RNA-seq data of broad cell line. Start downloading now.")
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/rnaseq_broad_20210317.zip",
                output_file_name=bulk_rna_broad_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )

        self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path)

    def getinfo_annotate_driver_genes(self, driver_gene_set: str = "intOGen") -> None:
        """Fetch driver gene annotation.

        The user can use this function to know which kind of information to query when annotating driver gene information from DepMap_Sanger.
        The list of driver genes is the union of two complementary gene sets: intOGen & COSMIC Tier 1.
        """

        """Print the columns of the DepMap_Sanger driver gene annotation and the number of driver genes."""
        if driver_gene_set == "intOGen":
            print(
                "Current available information in the DepMap_Sanger driver gene annotation for intOGen genes: ",
                *list(self.driver_gene_intOGen.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.driver_gene_intOGen.index)} driver genes are saved in this file")
        if driver_gene_set == "COSMIC":
            print(
                "Current available information in the DepMap_Sanger driver gene annotation for COSMIC Tier 1 genes: ",
                *list(self.driver_gene_cosmic.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.driver_gene_cosmic.index)} driver genes are saved in this file")

    def getinfo_annotate_bulk_rna_expression(self, cell_line_source: str = "broad") -> None:
        """Fetch bulk rna expression.

        The user can use this function to know which kind of information to query when annotating bulk rna expression.
        There are two sources for the cell line bulk rna expression data: broad or Sanger.
        """

        """Print the columns of the cell line bulk rna expression data."""
        if cell_line_source == "broad":
            print(
                "Current available information in the RNA-Seq Data for broad cell line only: ",
                *list(self.bulk_rna_broad.columns.values),
                sep="\n- ",
            )
            print(
                f"{len(self.bulk_rna_broad.model_name.unique())} unique cell lines are saved in this file under the "
                f"column model_name"
            )
            print(f"{len(self.bulk_rna_broad.gene_id.unique())} unique genes are saved in this file")

        if cell_line_source == "sanger":
            print(
                "Current available information in the RNA-Seq Data for Sanger cell line only: ",
                *list(self.bulk_rna_sanger.columns.values),
                sep="\n- ",
            )
            print(
                f"{len(self.bulk_rna_sanger.model_name.unique())} unique cell lines are saved in this file under the "
                f"column model_name"
            )
            print(f"{len(self.bulk_rna_sanger.gene_id.unique())} unique genes are saved in this file")

    def getinfo_annotate_cell_lines(self) -> None:
        """Fetch cell line annotation.

        The user can use this function to know which kind of information to query when annotating cell line information from Dependency Map (DepMap).
        """

        """Print the columns of the DepMap cell line annotation and the number of cell lines."""
        print(
            "Current available information in the DepMap cell line annotation: ",
            *list(self.cell_line_meta.columns.values),
            sep="\n- ",
        )
        print(f"{len(self.cell_line_meta.index)} cell lines are saved in this file")
        # print('Please also have a brief overview about what the DepMap cell annotation file looks like ',
        # self.cell_line_meta.head())

    def annotate_cell_lines(
        self,
        adata: anndata,
        cell_line_identifier: str = "DepMap_ID",
        identifier_type: str = "DepMap_ID",
        cell_line_information: list[str] = None,
        copy: bool = False,
    ) -> anndata:
        """Fetch cell line annotation.

        For each cell, we fetch cell line annotation from Dependency Map (DepMap).

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: "DepMap_ID")
            identifier_type: The type of cell line information, e.g. DepMap_ID, cell_line_name or stripped_cell_line_name. (default: "DepMap_ID")
            cell_line_information: The metadata to fetch. All metadata will be fetched by default. (default: all)
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with cell line annotation.
        """
        if copy:
            adata = adata.copy()
        if identifier_type in self.cell_line_meta.columns:
            """If the specified cell line type can be found in the DepMap database,
            we can compare these keys and fetch the corresponding metadata."""

            not_matched_identifiers = list(
                set(adata.obs[cell_line_identifier]) - set(self.cell_line_meta[identifier_type])
            )
            if len(not_matched_identifiers) > 0:
                print(
                    "Following identifiers can not be found in DepMap cell line annotation file,"
                    " so their corresponding meta data are NA values. Please check it again:",
                    *not_matched_identifiers,
                    sep="\n- ",
                )

            if cell_line_information is None:
                """If no cell_line_information is specified, all metadata is fetched by default.
                Sometimes there is already different cell line information in the `adata`.
                In order to avoid redundant information,
                we will remove the duplicate information from metadata after merging."""
                adata.obs = (
                    adata.obs.reset_index()
                    .merge(
                        self.cell_line_meta,
                        left_on=cell_line_identifier,
                        right_on=identifier_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index.names)
                )
                """ If cell_line_identifier and identifier_type have different names,
                there will be a column for each of them after merging,
                which is redundant as they refer to the same information.
                We will move the identifier_type column."""
                if cell_line_identifier != identifier_type:
                    del adata.obs[identifier_type]

            elif set(cell_line_information).issubset(set(self.cell_line_meta.columns)):
                """If cell_line_information is specified and can be found in the DepMap database,
                We will subset the original metadata dataframe correspondingly and add them to the `adata`.
                Again, redundant information will be removed."""
                if identifier_type not in cell_line_information:
                    cell_line_information.append(identifier_type)
                cell_line_meta_part = self.cell_line_meta[cell_line_information]
                adata.obs = (
                    adata.obs.reset_index()
                    .merge(
                        cell_line_meta_part,
                        left_on=cell_line_identifier,
                        right_on=identifier_type,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index.names)
                )
                """ Again, redundant information will be removed."""
                if cell_line_identifier != identifier_type:
                    del adata.obs[identifier_type]
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

    def annotate_bulk_rna_expression(
        self,
        adata: anndata,
        cell_line_identifier: str = "cell_line_name",
        cell_line_source: str = "broad",
        cell_line_information: str = "read_count",
        copy: bool = False,
    ) -> anndata:
        """Fetch bulk rna expression.

        For each cell, we fetch bulk rna expression from either Broad or Sanger cell line.

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: "cell_line_name")
            cell_line_source: The bulk rna expression data from either Broad or Sanger cell line. (default: "broad")
            cell_line_information: The metadata to fetch. (default: read_count)
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with bulk rna expression annotation.
        """
        if copy:
            adata = adata.copy()
        if cell_line_source == "broad":
            bulk_rna = self.bulk_rna_broad
        else:
            bulk_rna = self.bulk_rna_sanger

        """If the specified cell line type can be found in the bulk rna expression data,
        we can compare these keys and fetch the corresponding metadata."""

        not_matched_identifiers = list(set(adata.obs[cell_line_identifier]) - set(bulk_rna["model_name"]))
        if len(not_matched_identifiers) > 0:
            print(
                "Following identifiers can not be found in bulk RNA expression data,"
                " so their corresponding meta data are NA values. Please check it again:",
                *not_matched_identifiers,
                sep="\n- ",
            )

        bulk_rna_gb_cellline = bulk_rna.groupby("model_name")[["gene_symbol", cell_line_information]].agg(
            lambda x: list(x)
        )
        bulk_rna_expression = pd.DataFrame(
            bulk_rna_gb_cellline[cell_line_information].tolist(), index=bulk_rna_gb_cellline.index
        )
        bulk_rna_expression.columns = bulk_rna_gb_cellline.gene_symbol[1]
        bulk_rna_expression.index.names = [None]
        bulk_rna_expression_mapped = pd.merge(
            adata.obs[cell_line_identifier],
            bulk_rna_expression,
            left_on=cell_line_identifier,
            right_index=True,
            how="left",
        )
        adata.obsm["bulk_rna_expression_" + cell_line_source] = bulk_rna_expression_mapped

        return adata
