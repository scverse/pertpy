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
            print("[bold yellow]No DepMap metadata file found. Starting download now.")
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
            print("[bold yellow]No metadata file was found for driver genes of the intOGen analysis. Starting download now.")
            _download(
                url="https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip",
                output_file_name=driver_gene_intOGen_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.driver_gene_intOGen = pd.read_table(driver_gene_intOGen_file_path)

        """Download meta data for driver genes of the COSMIC Tier 1 gene from DepMap_Sanger """
        self.driver_gene_cosmic = pd.read_csv("https://www.dropbox.com/s/8azkmt7vqz56e2m/COSMIC_tier1.csv?dl=1")

        """Download bulk RNA-seq data collated from the Wellcome Sanger Institute and the Broad Institute """
        bulk_rna_sanger_file_name = "rnaseq_sanger.zip"
        bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316.csv"
        if not Path(bulk_rna_sanger_file_path).exists():
            print("[bold yellow]No metadata file was found for bulk RNA-seq data of Sanger cell line. Starting download now.")
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

        """Download proteomics data (ProCan-DepMapSanger) which saves protein intensity values
         acquired using data-independent acquisition mass spectrometry (DIA-MS). """

        proteomics_file_name = "Proteomics_20221214.zip"
        proteomics_file_path = settings.cachedir.__str__() + "/proteomics_all_20221214.csv"
        if not Path(proteomics_file_path).exists():
            print("No metadata file was found for proteomics data (ProCan-DepMapSanger). Start downloading now.")
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/Proteomics_20221214.zip",
                output_file_name=proteomics_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.proteomics_data = pd.read_csv(proteomics_file_path)

        """Download CCLE expression data, which contains Gene expression TPM values
        of the protein coding genes for DepMap cell lines.
       """

        ccle_expr_file_name = "CCLE_expression.csv"
        ccle_expr_file_path = settings.cachedir.__str__() + "/CCLE_expression.csv"
        if not Path(ccle_expr_file_path).exists():
            print("No metadata file was found for CCLE expression data. Start downloading now.")
            _download(
                url="https://figshare.com/ndownloader/files/34989919",
                output_file_name=ccle_expr_file_name,
                output_path=settings.cachedir,
                is_zip=False,
            )
        self.ccle_expr = pd.read_csv(ccle_expr_file_path, index_col=0)

    def getinfo_annotate_driver_genes(self, driver_gene_set: str = "intOGen") -> None:
        """
        The user can use this function to know which kind of information to query when annotating
        driver gene information from DepMap_Sanger.
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
        """
        The user can use this function to know which kind of information to query when
        annotating bulk rna expression.
        There are two sources for the cell Line bulk rna expression data: broad or Sanger.
        """

        """Print the columns of the DepMap_Sanger driver gene annotation and the number of driver genes."""
        if cell_line_source == "broad":
            print(
                "Current available information in the RNA-Seq Data for broad cell line only: ",
                *list(self.bulk_rna_broad.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.bulk_rna_broad.model_name.unique())} unique cell lines are saved in this file.")
            print(f"{len(self.bulk_rna_broad.gene_id.unique())} unique genes are saved in this file")

        elif cell_line_source == "sanger":
            print(
                "Current available information in the RNA-Seq Data for Sanger cell line only: ",
                *list(self.bulk_rna_sanger.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.bulk_rna_sanger.model_name.unique())} unique cell lines are saved in this file.")
            print(f"{len(self.bulk_rna_sanger.gene_id.unique())} unique genes are saved in this file.")
        else:
            raise ValueError(
                "The specified source of bulk rna expression data is not available. "
                "Please choose either broad or sanger."
            )

    def getinfo_annotate_cell_lines(self) -> None:
        """
        The user can use this function to know which kind of information to query when
        annotating cell line information from Dependency Map (DepMap).
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

    def getinfo_annotate_protein_expression(self) -> None:
        """
        The user can use this function to know which kind of information to query when annotating protein expression.
        Protein intensity values acquired using data-independent acquisition mass spectrometry (DIA-MS).

        Data was then log2-transformed. MS runs across replicates of each cell line were combined by
        calculating the geometric mean.
        The final dataset, termed ProCan-DepMapSanger, was derived from 6,864 mass spectrometry runs covering
        949 cell lines and quantifying a total of 8,498 proteins.
        """

        """Print the columns of the protein expression data and the number of proteins."""
        print(
            "Current available information in the proteomics data: ",
            *list(self.proteomics_data.columns.values),
            sep="\n- ",
        )
        print(f"{len(self.proteomics_data.model_name.unique())} unique cell lines are saved in this file.")
        print(f"{len(self.proteomics_data.uniprot_id.unique())} unique proteins are saved in this file.")

    def getinfo_annotate_ccle_expression(self) -> None:
        """
        The user can use this function to know which kind of information to query when annotating
        CCLE expression data.
        """

        """Print the number of genes and cell lines of the CCLE expression datas."""

        print(f" Expression of {len(self.ccle_expr.columns.unique())} genes is saved in this file.")
        print(f"{len(self.ccle_expr.index.unique())} unique cell lines are available in this file.")

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
        identifier_type: str = "model_name",
        bulk_rna_source: str = "broad",
        bulk_rna_information: str = "read_count",
        copy: bool = False,
    ) -> anndata:
        """Fetch bulk rna expression.

        For each cell, we fetch bulk rna expression from either Broad or Sanger cell line.

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: "cell_line_name")
            identifier_type: The type of cell line information, e.g. model_name or model_id. (default: 'model_name')
            bulk_rna_source: The bulk rna expression data from either Broad or Sanger cell line. (default: "broad")
            bulk_rna_information: The metadata to fetch. (default: read_count)
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with bulk rna expression annotation.
        """
        if copy:
            adata = adata.copy()
        if bulk_rna_source == "broad":
            bulk_rna = self.bulk_rna_broad
        elif bulk_rna_source == "sanger":
            bulk_rna = self.bulk_rna_sanger
        else:
            raise ValueError(
                "The specified source of bulk rna expression data is not available. "
                "Please choose either broad or sanger."
            )

        """If the specified cell line type can be found in the bulk rna expression data,
        we can compare these keys and fetch the corresponding metadata."""

        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified cell line identifier can't be found in the adata.obs. "
                "Please fetch the cell line meta data first using the functiion "
                "annotate_cell_lines()."
            )

        if identifier_type not in bulk_rna.columns:
            raise ValueError(
                "The specified identifier type can't be found in the meta data. "
                "Please check the available identifier type in the meta data using "
                "the function getinfo_annotate_bulk_rna_expression()."
            )

        not_matched_identifiers = list(set(adata.obs[cell_line_identifier]) - set(bulk_rna[identifier_type]))
        if len(not_matched_identifiers) > 0:
            print(
                "Following identifiers can not be found in bulk RNA expression data,"
                " so their corresponding meta data are NA values. Please check it again:",
                *not_matched_identifiers,
                sep="\n- ",
            )

        rna_exp = pd.pivot(bulk_rna, index=identifier_type, columns="gene_id", values=bulk_rna_information)
        # order according to adata.obs
        rna_exp = rna_exp.reindex(adata.obs[cell_line_identifier])
        # have same index as adata.obs
        rna_exp = rna_exp.set_index(adata.obs.index)
        adata.obsm["bulk_rna_expression_" + bulk_rna_source] = rna_exp

        return adata

    def annotate_protein_expression(
        self,
        adata: anndata,
        cell_line_identifier: str = "cell_line_name",
        identifier_type: str = "model_name",
        copy: bool = False,
    ) -> anndata:
        """Fetch protein expression.

        For each cell, we fetch protein intensity values acquired using data-independent acquisition mass spectrometry (DIA-MS).

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: 'cell_line_name")
            identifier_type: The type of cell line information, e.g. model_name or model_id. (default: 'model_name')
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with protein expression annotation.
        """
        if copy:
            adata = adata.copy()

        """If the specified cell line type can be found in the protein expression data,
        we can compare these keys and fetch the corresponding metadata."""

        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified cell line identifier can't be found in the adata.obs. "
                "Please fetch the cell line meta data first using the functiion "
                "annotate_cell_lines()."
            )

        if identifier_type not in self.proteomics_data.columns:
            raise ValueError(
                "The specified identifier type can't be found in the meta data. "
                "Please check the available identifier type in the meta data using "
                "the function getinfo_annotate_protein_expression()."
            )

        not_matched_identifiers = list(
            set(adata.obs[cell_line_identifier]) - set(self.proteomics_data[identifier_type])
        )
        if len(not_matched_identifiers) > 0:
            print(
                "[bold yellow]Following identifiers can not be found in the protein expression data,"
                " so their corresponding meta data are NA values. Please check it again:",
                *not_matched_identifiers,
                sep="\n- ",
            )
        # convert the original protein intensities table from long format to wide format, group by the cell lines
        prot_exp = pd.pivot(
            self.proteomics_data, index=identifier_type, columns="uniprot_id", values="protein_intensity"
        )
        # order according to adata.obs
        prot_exp = prot_exp.reindex(adata.obs[cell_line_identifier])
        # have same index with adata.obs
        prot_exp = prot_exp.set_index(adata.obs.index)
        # save in the adata.obsm
        adata.obsm["proteomics_protein_intensity"] = prot_exp

        return adata

    def annotate_ccle_expression(
        self,
        adata: anndata,
        cell_line_identifier: str = "DepMap_ID",
        copy: bool = False,
    ) -> anndata:
        """Fetch CCLE expression.

        For each cell, we fetch .

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: 'DepMap_ID")
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with CCLE expression annotation.
        """
        if copy:
            adata = adata.copy()

        """If the specified cell line type can be found in the CCLE expression data,
        we can compare these keys and fetch the corresponding metadata."""

        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified cell line identifier can't be found in the adata.obs. "
                "Please fetch the cell line meta data first using the functiion "
                "annotate_cell_lines()."
            )

        not_matched_identifiers = list(set(adata.obs[cell_line_identifier]) - set(self.ccle_expr.index))
        if len(not_matched_identifiers) > 0:
            print(
                "[bold yellow]Following identifiers can not be found in the CCLE expression data,"
                " their corresponding meta data are NA values. Please check it again:",
                *not_matched_identifiers,
                sep="\n- ",
            )

        # order the cell line according to adata.obs
        ccle_expression = self.ccle_expr.reindex(adata.obs[cell_line_identifier])
        # set the index same as in adata.obs
        ccle_expression = ccle_expression.set_index(adata.obs.index)
        # save in the adata.obsm
        adata.obsm["CCLE_expression"] = ccle_expression

        return adata
