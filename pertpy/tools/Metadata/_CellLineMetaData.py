from __future__ import annotations

from pathlib import Path

import anndata
import pandas as pd
from rich import print
from scanpy import settings

from pertpy.data._dataloader import _download


class CellLineMetaData:
    """Utilities to fetch cell or perturbation metadata."""

    def __init__(self):
        # Download cell line metadata from DepMap
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

        # Download cell line metadata from The Genomics of Drug Sensitivity in Cancer Project
        cell_line_cancer_project_file_name = "cell_line_cancer_project.csv"
        cell_line_cancer_project_file_path = settings.cachedir.__str__() + "/" + cell_line_cancer_project_file_name
        if not Path(cell_line_cancer_project_file_path).exists():
            print("[bold yellow]No cell line metadata file from The Genomics of Drug Sensitivity "
                  "in Cancer Project found. Starting download now.")
            _download(
                url="https://www.cancerrxgene.org/api/celllines?list=all&sEcho=1&iColumns=7&sColumns=&"
                    "iDisplayStart=0&iDisplayLength=25&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&"
                    "mDataProp_4=4&mDataProp_5=5&mDataProp_6=6&sSearch=&bRegex=false&sSearch_0=&bRegex_0=false&"
                    "bSearchable_0=true&sSearch_1=&bRegex_1=false&bSearchable_1=true&sSearch_2=&bRegex_2=false&"
                    "bSearchable_2=true&sSearch_3=&bRegex_3=false&bSearchable_3=true&sSearch_4=&bRegex_4=false&"
                    "bSearchable_4=true&sSearch_5=&bRegex_5=false&bSearchable_5=true&sSearch_6=&bRegex_6=false&"
                    "bSearchable_6=true&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&"
                    "bSortable_2=true&bSortable_3=true&bSortable_4=true&bSortable_5=true&bSortable_6=true&export=csv",
                output_file_name=cell_line_cancer_project_file_name,
                output_path=settings.cachedir,
                is_zip=False,
            )

        self.cell_line_cancer_project_meta = pd.read_csv(cell_line_cancer_project_file_path)
        # remove white space in column names
        self.cell_line_cancer_project_meta.columns = self.cell_line_cancer_project_meta.columns.str.strip()
        # have a column of stripped names
        self.cell_line_cancer_project_meta['stripped_cell_line_name'] = \
            self.cell_line_cancer_project_meta['Cell line Name'].str.replace(r'\-|\.', '', regex=True)
        self.cell_line_cancer_project_meta['stripped_cell_line_name'] = pd.Categorical(
            self.cell_line_cancer_project_meta['stripped_cell_line_name'].str.upper())
        # pivot the data frame so that each cell line has only one row of metadata
        index_col = list(set(self.cell_line_cancer_project_meta.columns) - {'Datasets', 'number of drugs'})
        self.cell_line_cancer_project_meta = self.cell_line_cancer_project_meta.pivot(index=index_col,
                                                                                      columns='Datasets',
                                                                                      values='number of drugs')
        self.cell_line_cancer_project_meta.columns.name = None
        self.cell_line_cancer_project_meta = self.cell_line_cancer_project_meta.reset_index()

        # Download metadata for driver genes of the intOGen analysis from DepMap_Sanger
        driver_gene_intOGen_file_name = "IntOGen-Drivers.zip"
        driver_gene_intOGen_file_path = (
            settings.cachedir.__str__() + "/2020-02-02_IntOGen-Drivers-20200213/Compendium_Cancer_Genes.tsv"
        )
        if not Path(driver_gene_intOGen_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for driver genes of the intOGen analysis. Starting download now."
            )
            _download(
                url="https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip",
                output_file_name=driver_gene_intOGen_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.driver_gene_intOGen = pd.read_table(driver_gene_intOGen_file_path)

        # Download metadata for driver genes of the COSMIC Tier 1 gene
        self.driver_gene_cosmic = pd.read_csv("https://www.dropbox.com/s/8azkmt7vqz56e2m/COSMIC_tier1.csv?dl=1")

        # Download bulk RNA-seq data collated from the Wellcome Sanger Institute and the Broad Institute
        bulk_rna_sanger_file_name = "rnaseq_sanger.zip"
        bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316.csv"
        if not Path(bulk_rna_sanger_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for bulk RNA-seq data of Sanger cell line. Starting download now."
            )
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
            print(
                "[bold yellow]No metadata file was found for bulk RNA-seq data of broad cell line. Starting download now."
            )
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/rnaseq_broad_20210317.zip",
                output_file_name=bulk_rna_broad_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path)

        # Download proteomics data from ProCan-DepMapSanger
        proteomics_file_name = "Proteomics_20221214.zip"
        proteomics_file_path = settings.cachedir.__str__() + "/proteomics_all_20221214.csv"
        if not Path(proteomics_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for proteomics data (ProCan-DepMapSanger). Starting download now."
            )
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/Proteomics_20221214.zip",
                output_file_name=proteomics_file_name,
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.proteomics_data = pd.read_csv(proteomics_file_path)

        # Download CCLE expression data from DepMap
        ccle_expr_file_name = "CCLE_expression.csv"
        ccle_expr_file_path = settings.cachedir.__str__() + "/CCLE_expression.csv"
        if not Path(ccle_expr_file_path).exists():
            print("[bold yellow]No metadata file was found for CCLE expression data. Starting download now.")
            _download(
                url="https://figshare.com/ndownloader/files/34989919",
                output_file_name=ccle_expr_file_name,
                output_path=settings.cachedir,
                is_zip=False,
            )
        self.ccle_expr = pd.read_csv(ccle_expr_file_path, index_col=0)

    def getinfo_annotate_driver_genes(self, driver_gene_set: str = "intOGen") -> None:
        """A brief summary of genes in cancer driver annotation data.

        Args:
            driver_gene_set: gene set for cancer driver annotation: intOGen or COSMIC. (default: "intOGen")

        Returns:
            None
        """

        # Print the columns of the driver gene annotation and the number of driver genes from DepMap_Sanger.
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
        """A brief summary of bulk RNA expression data.

        Args:
            cell_line_source: the source of RNA-seq data: broad or sanger. (default: "broad")

        Returns:
            None
        """

        # Print the columns and the number of the genes in the bulk rna expression data
        # from the Wellcome Sanger Institute or the Broad Institute.
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

    def getinfo_annotate_cell_lines(self, cell_line_source: str = "DepMap") -> None:
        """A brief summary of cell line metadata.

        Args:
            cell_line_source: the source of cell line annotation: DepMap or Cancerrxgene. (default: "DepMap")

        Returns:
            None
        """

        # Print the columns and the number of cell lines in the cell line annotation.

        if cell_line_source == "DepMap":
            print(
                "Current available information in the DepMap cell line annotation: ",
                *list(self.cell_line_meta.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.cell_line_meta.index)} cell lines are saved in this file")

        elif cell_line_source == "Cancerrxgene":
            print(
                "Current available information in the cell line annotation from the project "
                "Genomics of Drug Sensitivity in Cancer: ",
                *list(self.cell_line_cancer_project_meta.columns.values),
                sep="\n- ",
            )
            print(f"{len(self.cell_line_cancer_project_meta.index)} cell lines are saved in this file")

        else:
            raise ValueError(
                "The specified source of cell line annotation is not available. "
                "Please choose either DepMap or Cancerrxgene."
            )

    def getinfo_annotate_protein_expression(self) -> None:
        """A brief summary of protein expression data.

        Returns:
            None
        """

        # Print the columns and the number of proteins in the protein expression data.
        print(
            "Current available information in the proteomics data: ",
            *list(self.proteomics_data.columns.values),
            sep="\n- ",
        )
        print(f"{len(self.proteomics_data.model_name.unique())} unique cell lines are saved in this file.")
        print(f"{len(self.proteomics_data.uniprot_id.unique())} unique proteins are saved in this file.")

    def getinfo_annotate_ccle_expression(self) -> None:
        """A brief summary of CCLE expression data.

        Returns:
            None
        """

        # Print the number of genes and cell lines in the CCLE expression datas from Depmap.
        print(f" Expression of {len(self.ccle_expr.columns.unique())} genes is saved in this file.")
        print(f"{len(self.ccle_expr.index.unique())} unique cell lines are available in this file.")

    def annotate_cell_lines(
        self,
        adata: anndata,
        cell_line_identifier: str = "DepMap_ID",
        identifier_type: str = "DepMap_ID",
        cell_line_information: list[str] = None,
        source: str = "DepMap",
        copy: bool = False,
    ) -> anndata:
        """Fetch cell line annotation.

        For each cell, we fetch cell line annotation from Dependency Map (DepMap).

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: "DepMap_ID")
            identifier_type: The type of cell line information, e.g. DepMap_ID, cell_line_name or
                stripped_cell_line_name. To fetch cell line metadata from Cancerrxgene, it is recommended to choose
                "stripped_cell_line_name". (default: "DepMap_ID")
            cell_line_information: The metadata to fetch. All metadata will be fetched by default. (default: all)
            source: The source of cell line metadata, DepMap or Cancerrxgene. (default: "DepMap")
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with cell line annotation.
        """

        if copy:
            adata = adata.copy()
        if source == "DepMap":
            cell_line_meta = self.cell_line_meta
        elif source == "Cancerrxgene":
            cell_line_meta = self.cell_line_cancer_project_meta
        else:
            raise ValueError(
                "The specified source of cell line metadata is not available."
                "Please choose either DepMap or Cancerrxgene."
                "Default is DepMap."
            )

        if identifier_type in cell_line_meta.columns:
            """If the specified cell line type can be found in the database,
            we can compare these keys and fetch the corresponding metadata."""

            not_matched_identifiers = list(
                set(adata.obs[cell_line_identifier]) - set(cell_line_meta[identifier_type]))
            if len(not_matched_identifiers) > 0:
                print('Following identifiers can not be found in cell line annotation file,' \
                      ' so their corresponding meta data are NA values. Please check it again:',
                      *not_matched_identifiers, sep='\n- ')

            if cell_line_information is None:
                """If no cell_line_information is specified, all metadata is fetched by default.
                Sometimes there is already different cell line information in the `adata`.
                In order to avoid redundant information,
                we will remove the duplicate information from metadata after merging."""
                adata.obs = (
                    adata.obs.reset_index()
                    .merge(
                        cell_line_meta,
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

            elif set(cell_line_information).issubset(set(cell_line_meta.columns)):
                """If cell_line_information is specified and can be found in the DepMap database,
                We will subset the original metadata dataframe correspondingly and add them to the `adata`.
                Again, redundant information will be removed."""
                if identifier_type not in cell_line_information:
                    cell_line_information.append(identifier_type)
                cell_line_meta_part = cell_line_meta[cell_line_information]
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

        # Make sure that the specified `cell_line_type` can be found in the bulk rna expression data,
        # then we can compare these keys and fetch the corresponding metadata.

        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified `cell_line_identifier` can't be found in the `adata.obs`. "
                "Please fetch the cell line metadata first using the function "
                "`annotate_cell_lines()`."
            )

        if identifier_type not in bulk_rna.columns:
            raise ValueError(
                "The specified `identifier_type` can't be found in the metadata. "
                "Please check the available `identifier_type` in the bulk expression data using "
                "the function getinfo_annotate_bulk_rna_expression()."
            )

        not_matched_identifiers = list(set(adata.obs[cell_line_identifier]) - set(bulk_rna[identifier_type]))
        if len(not_matched_identifiers) > 0:
            print(
                "[bold yellow]Following identifiers can not be found in bulk RNA expression data,"
                " their corresponding meta data are NA values. Please check it again:",
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

        # Make sure that the specified `cell_line_type` can be found in the protein expression data,
        # then we can compare these keys and fetch the corresponding metadata.

        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified `cell_line_identifier` can't be found in the `adata.obs`. "
                "Please fetch the cell line meta data first using the function "
                "`annotate_cell_lines()`."
            )

        if identifier_type not in self.proteomics_data.columns:
            raise ValueError(
                "The specified `identifier_type` can't be found in the meta data. "
                "Please check the available identifier types in the protein expression data calling "
                "the function `getinfo_annotate_protein_expression()`."
            )

        not_matched_identifiers = list(
            set(adata.obs[cell_line_identifier]) - set(self.proteomics_data[identifier_type])
        )
        if len(not_matched_identifiers) > 0:
            print(
                "[bold yellow]Following identifiers can not be found in the protein expression data,"
                " their corresponding meta data are NA values. Please check it again:",
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
        """Fetch CCLE expression data.

         For each cell, we fetch gene expression TPM values of the protein coding genes for its corresponding DepMap
         cell line.

        Args:
            adata: The data object to annotate.
            cell_line_identifier: The column of `.obs` with cell line information. (default: 'DepMap_ID")
            copy: Determines whether a copy of the `adata` is returned. (default: False)

        Returns:
            Returns an AnnData object with CCLE expression annotation.
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified cell line type can be found in the CCLE expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if cell_line_identifier not in adata.obs.columns:
            raise ValueError(
                "The specified `cell_line_identifier` can't be found in the `adata.obs`. "
                "Please fetch the cell line meta data first using the function "
                "`annotate_cell_lines()`."
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
