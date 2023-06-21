from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import pandas as pd
from anndata import AnnData
from rich import print
from scanpy import settings

from pertpy.data._dataloader import _download

from ._look_up import LookUp


class CellLineMetaData:
    """Utilities to fetch cell line metadata."""

    def __init__(self):
        # Download cell line metadata from DepMap
        cell_line_file_path = settings.cachedir.__str__() + "/sample_info.csv"
        if not Path(cell_line_file_path).exists():
            print("[bold yellow]No DepMap metadata file found. Starting download now.")
            _download(
                url="https://ndownloader.figshare.com/files/35020903",
                output_file_name="sample_info.csv",
                output_path=settings.cachedir,
                is_zip=False,
            )
        self.cell_line_meta = pd.read_csv(cell_line_file_path)

        # Download cell line metadata from The Genomics of Drug Sensitivity in Cancer Project
        cell_line_cancer_project_file_path = settings.cachedir.__str__() + "/cell_line_cancer_project.csv"
        cell_line_cancer_project_transformed_path = (
            settings.cachedir.__str__() + "/cell_line_cancer_project_transformed.csv"
        )
        if not Path(cell_line_cancer_project_transformed_path).exists():
            if not Path(cell_line_cancer_project_file_path).exists():
                print(
                    "[bold yellow]No cell line metadata file from The Genomics of Drug Sensitivity "
                    "in Cancer Project found. Starting download now."
                )
                _download(
                    url="https://www.cancerrxgene.org/api/celllines?list=all&sEcho=1&iColumns=7&sColumns=&"
                    "iDisplayStart=0&iDisplayLength=25&mDataProp_0=0&mDataProp_1=1&mDataProp_2=2&mDataProp_3=3&"
                    "mDataProp_4=4&mDataProp_5=5&mDataProp_6=6&sSearch=&bRegex=false&sSearch_0=&bRegex_0=false&"
                    "bSearchable_0=true&sSearch_1=&bRegex_1=false&bSearchable_1=true&sSearch_2=&bRegex_2=false&"
                    "bSearchable_2=true&sSearch_3=&bRegex_3=false&bSearchable_3=true&sSearch_4=&bRegex_4=false&"
                    "bSearchable_4=true&sSearch_5=&bRegex_5=false&bSearchable_5=true&sSearch_6=&bRegex_6=false&"
                    "bSearchable_6=true&iSortCol_0=0&sSortDir_0=asc&iSortingCols=1&bSortable_0=true&bSortable_1=true&"
                    "bSortable_2=true&bSortable_3=true&bSortable_4=true&bSortable_5=true&bSortable_6=true&export=csv",
                    output_file_name="cell_line_cancer_project.csv",
                    output_path=settings.cachedir,
                    is_zip=False,
                )

            self.cl_cancer_project_meta = pd.read_csv(cell_line_cancer_project_file_path)
            self.cl_cancer_project_meta.columns = self.cl_cancer_project_meta.columns.str.strip()
            self.cl_cancer_project_meta["stripped_cell_line_name"] = self.cl_cancer_project_meta[
                "Cell line Name"
            ].str.replace(r"\-|\.", "", regex=True)
            self.cl_cancer_project_meta["stripped_cell_line_name"] = pd.Categorical(
                self.cl_cancer_project_meta["stripped_cell_line_name"].str.upper()
            )
            # pivot the data frame so that each cell line has only one row of metadata
            index_col = list(set(self.cl_cancer_project_meta.columns) - {"Datasets", "number of drugs"})
            self.cl_cancer_project_meta = self.cl_cancer_project_meta.pivot(
                index=index_col, columns="Datasets", values="number of drugs"
            )
            self.cl_cancer_project_meta.columns.name = None
            self.cl_cancer_project_meta = self.cl_cancer_project_meta.reset_index()
            self.cl_cancer_project_meta = self.cl_cancer_project_meta.rename(
                columns={"Cell line Name": "cell_line_name"}
            )
            self.cl_cancer_project_meta.to_csv(cell_line_cancer_project_transformed_path)

        else:
            self.cl_cancer_project_meta = pd.read_csv(cell_line_cancer_project_transformed_path, index_col=0)

        # Download metadata for driver genes of the intogen analysis from DepMap_Sanger
        driver_gene_intogen_file_path = (
            settings.cachedir.__str__() + "/2020-02-02_IntOGen-Drivers-20200213/Compendium_Cancer_Genes.tsv"
        )
        if not Path(driver_gene_intogen_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for driver genes of the intogen analysis."
                " Starting download now."
            )
            _download(
                url="https://www.intogen.org/download?file=IntOGen-Drivers-20200201.zip",
                output_file_name="IntOGen-Drivers.zip",
                output_path=settings.cachedir,
                is_zip=True,
            )
        self.driver_gene_intogen = pd.read_table(driver_gene_intogen_file_path)
        self.driver_gene_intogen.rename(columns=lambda col: col.lower(), inplace=True)

        # Download metadata for driver genes of the COSMIC Tier 1 gene
        self.driver_gene_cosmic = pd.read_csv("https://www.dropbox.com/s/8azkmt7vqz56e2m/COSMIC_tier1.csv?dl=1")

        # Download bulk RNA-seq data collated from the Wellcome Sanger Institute and the Broad Institute
        bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316.csv"
        bulk_rna_sanger_trimm_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316_trimm.csv"

        if not Path(bulk_rna_sanger_trimm_path).exists():
            if not Path(bulk_rna_sanger_file_path).exists():
                print(
                    "[bold yellow]No metadata file was found for bulk RNA-seq data of Sanger cell line."
                    " Starting download now..."
                )
                _download(
                    url="https://cog.sanger.ac.uk/cmp/download/rnaseq_sanger_20210316.zip",
                    output_file_name="rnaseq_sanger.zip",
                    output_path=settings.cachedir,
                    is_zip=True,
                )
            self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_file_path)
            self.bulk_rna_sanger.drop(["data_source", "gene_symbol"], axis=1, inplace=True)
            self.bulk_rna_sanger[["model_id", "model_name", "gene_id"]] = self.bulk_rna_sanger[
                ["model_id", "model_name", "gene_id"]
            ].astype("category")
            self.bulk_rna_sanger.to_csv(bulk_rna_sanger_trimm_path)
        else:
            self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_trimm_path, index_col=0)

        bulk_rna_broad_file_path = settings.cachedir.__str__() + "/rnaseq_broad_20210317.csv"
        bulk_rna_broad_trimm_path = settings.cachedir.__str__() + "/rnaseq_broad_20210317_trimm.csv"

        if not Path(bulk_rna_broad_trimm_path).exists():
            if not Path(bulk_rna_broad_file_path).exists():
                print(
                    "[bold yellow]No metadata file was found for bulk RNA-seq data of broad cell line."
                    " Starting download now."
                )
                _download(
                    url="https://cog.sanger.ac.uk/cmp/download/rnaseq_broad_20210317.zip",
                    output_file_name="rnaseq_broad.zip",
                    output_path=settings.cachedir,
                    is_zip=True,
                )
            self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path)

            # gene symbol can not be the column name of fetched bulk rna expression data
            # there are 37263 unique gene ids
            # but 37262 unique gene symbols (SEPTIN4)
            self.bulk_rna_broad.drop(["data_source", "gene_symbol"], axis=1, inplace=True)
            self.bulk_rna_broad[["model_id", "model_name", "gene_id"]] = self.bulk_rna_broad[
                ["model_id", "model_name", "gene_id"]
            ].astype("category")
            self.bulk_rna_broad.to_csv(bulk_rna_broad_trimm_path)
        else:
            self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_trimm_path, index_col=0)

        # Download proteomics data from ProCan-DepMapSanger
        proteomics_file_path = settings.cachedir.__str__() + "/proteomics_all_20221214.csv"
        proteomics_trimm_path = settings.cachedir.__str__() + "/proteomics_all_20221214_trimm.csv"
        if not Path(proteomics_trimm_path).exists():
            if not Path(proteomics_file_path).exists():
                print(
                    "[bold yellow]No metadata file was found for proteomics data (ProCan-DepMapSanger)."
                    " Starting download now."
                )
                _download(
                    url="https://cog.sanger.ac.uk/cmp/download/Proteomics_20221214.zip",
                    output_file_name="Proteomics_20221214.zip",
                    output_path=settings.cachedir,
                    is_zip=True,
                )
            self.proteomics_data = pd.read_csv(proteomics_file_path)
            self.proteomics_data[["uniprot_id", "model_id", "model_name", "symbol"]] = self.proteomics_data[
                ["uniprot_id", "model_id", "model_name", "symbol"]
            ].astype("category")
            self.proteomics_data.to_csv(proteomics_trimm_path)
        else:
            self.proteomics_data = pd.read_csv(proteomics_trimm_path, index_col=0)

        # Download CCLE expression data from DepMap
        ccle_expr_file_path = settings.cachedir.__str__() + "/CCLE_expression.csv"
        if not Path(ccle_expr_file_path).exists():
            print("[bold yellow]No metadata file was found for CCLE expression data." " Starting download now.")
            _download(
                url="https://figshare.com/ndownloader/files/34989919",
                output_file_name="CCLE_expression.csv",
                output_path=settings.cachedir,
                is_zip=False,
            )
        self.ccle_expr = pd.read_csv(ccle_expr_file_path, index_col=0)

    def annotate_cell_lines(
        self,
        adata: AnnData,
        query_id: str = "DepMap_ID",
        reference_id: str = "DepMap_ID",
        cell_line_information: list[str] | None = None,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        copy: bool = False,
    ) -> AnnData:
        """Fetch cell line annotation.

        For each cell, we fetch cell line annotation from Dependency Map (DepMap).

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "DepMap_ID".
            reference_id: The type of cell line identifier in the meta data, e.g. DepMap_ID, cell_line_name or
                stripped_cell_line_name. If fetch cell line metadata from Cancerrxgene, it is recommended to choose
                "stripped_cell_line_name". Defaults to "DepMap_ID".
            cell_line_information: The metadata to fetch. All metadata will be fetched by default. Defaults to None (=all).
            cell_line_source: The source of cell line metadata, DepMap or Cancerrxgene. Defaults to "DepMap".
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with cell line annotation.
        """
        if copy:
            adata = adata.copy()

        if cell_line_source == "DepMap":
            cell_line_meta = self.cell_line_meta
        else:
            reference_id = "stripped_cell_line_name"
            if query_id == "DepMap_ID":
                query_id = "stripped_cell_line_name"
                print(
                    "To annotate cell line metadata from Cancerrxgene, ",
                    "we use `stripped_cell_line_name` as reference and query indentifier. ",
                    "Please make sure that `stripped_cell_line_name` is available in the adata.obs. ",
                    "or use the DepMap as cell_line_source to annotate the cell line first ",
                )
            cell_line_meta = self.cl_cancer_project_meta

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`. \n" "Please check again. ")

        if reference_id in cell_line_meta.columns:
            # If the specified cell line type can be found in the database,
            # we can compare these keys and fetch the corresponding metadata.
            identifier_num_all = len(adata.obs[query_id].unique())
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(cell_line_meta[reference_id]))
            if len(not_matched_identifiers) == identifier_num_all:
                raise ValueError(
                    f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata. \n"
                    "However, none of the query IDs could be found in the cell line annotation data. \n"
                    "It is recommended to stop the annotation process. \n"
                    "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                    "By using the `LookUp.cell_line()` method, \n"
                    "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs."
                )

            if len(not_matched_identifiers) > 0:
                print(f"There are {identifier_num_all} identifiers in `adata.obs`.")
                print(
                    f"But following {len(not_matched_identifiers)} identifiers can't be found in the cell line annotation file. "
                )
                print(
                    "Resulting in NA values for their corresponding metadata. ",
                    "Please check again: ",
                    *not_matched_identifiers,
                    sep="\n- ",
                )

            if cell_line_information is None:
                # If no cell_line_information is specified, all metadata is fetched by default.
                # Sometimes there is already different cell line information in the `adata`.
                # To avoid redundant information we will remove the duplicate information from metadata after merging.
                adata.obs = (
                    adata.obs.merge(
                        cell_line_meta,
                        left_on=query_id,
                        right_on=reference_id,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index)
                )
                # If query_id and reference_id have different names,
                # there will be a column for each of them after merging,
                # which is redundant as they refer to the same information.
                # We will move the reference_id column.
                if query_id != reference_id:
                    del adata.obs[reference_id]

            elif set(cell_line_information).issubset(set(cell_line_meta.columns)):
                # If cell_line_information is specified and can be found in the DepMap database,
                # We will subset the original metadata dataframe correspondingly and add them to the `adata`.
                # Again, redundant information will be removed.
                if reference_id not in cell_line_information:
                    cell_line_information.append(reference_id)
                cell_line_meta_part = cell_line_meta[cell_line_information]
                adata.obs = (
                    adata.obs.merge(
                        cell_line_meta_part,
                        left_on=query_id,
                        right_on=reference_id,
                        how="left",
                        suffixes=("", "_fromMeta"),
                    )
                    .filter(regex="^(?!.*_fromMeta)")
                    .set_index(adata.obs.index)
                )
                # Again, redundant information will be removed.
                if query_id != reference_id:
                    del adata.obs[reference_id]

        else:
            raise ValueError(
                f"The requested cell line type {reference_id} is currently unavailable in the database. \n"
                "Please refer to the available reference identifier in the chosen database. \n"
                "DepMap_ID is compared by default. \n"
                "To solve the issue, you can also call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.cell_line()` method, you can obtain the available reference identifiers in the metadata. \n"
            )

        return adata

    def annotate_bulk_rna_expression(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        reference_id: Literal["model_name", "model_id"] = "model_name",
        cell_line_source: Literal["broad", "sanger"] = "broad",
        bulk_rna_information: Literal["read_count", "fpkm"] = "read_count",
        copy: bool = False,
    ) -> AnnData:
        """Fetch bulk rna expression.

        For each cell, we fetch bulk rna expression from either Broad or Sanger cell line.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "cell_line_name".
            reference_id: The type of cell line identifier in the meta data, model_name or model_id. Defaults to "model_name".
            cell_line_source: The bulk rna expression data from either broad or sanger cell line. Defaults to "broad".
            bulk_rna_information: The metadata to fetch, read_count or fpkm. Defaults to "read_count".
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with bulk rna expression annotation.
        """
        if copy:
            adata = adata.copy()
        bulk_rna = self.bulk_rna_broad if cell_line_source == "broad" else self.bulk_rna_sanger

        # Make sure that the specified `cell_line_type` can be found in the bulk rna expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Please ensure that you are using one of the available query IDs present in the adata.obs for the annotation. \n"
                "If the desired query ID is not available, you can fetch the cell line metadata \n"
                "using the `annotate_cell_lines()` function before calling annotate_ccle_expression(). \n"
                "This will help ensure that the required query ID is included in your data,  e.g. stripped_cell_line_name, DepMap ID."
            )

        if reference_id not in bulk_rna.columns:
            raise ValueError(
                "The specified `reference_id` {reference_id} is not available in the bulk RNA expression data. \n"
                "Please use the reference identifier available in the metadata. \n"
                "To solve the issue, you can call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.bulk_rna_expression()` method, you can obtain the available reference identifiers in the metadata. \n"
            )
        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(bulk_rna[reference_id]))

        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata. \n"
                "However, none of the query IDs could be found in the bulk RNA expression data. \n"
                "It is recommended to stop the annotation process. \n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.bulk_rna_expression()` method, \n"
                "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs. \n"
                "Additionally, you can call the `CellLineMetaData.annotate_cell_lines()` function \n"
                "to acquire more possible query IDs that can be used for annotation purposes."
            )

        if len(not_matched_identifiers) > 0:
            print(f"There are {identifier_num_all} identifiers in `adata.obs`.")
            print(
                f"[bold yellow]Following {len(not_matched_identifiers)} identifiers can't be found in bulk RNA expression data. "
            )
            print(
                "Resulting in NA values for their corresponding metadata. Please check again: ",
                *not_matched_identifiers,
                sep="\n- ",
            )
        bulk_rna = bulk_rna[[reference_id, "gene_id", bulk_rna_information]]
        rna_exp = pd.pivot(bulk_rna, index=reference_id, columns="gene_id", values=bulk_rna_information)
        rna_exp = rna_exp.reindex(adata.obs[query_id])
        rna_exp.index = adata.obs.index
        adata.obsm["bulk_rna_expression_" + cell_line_source] = rna_exp

        return adata

    def annotate_protein_expression(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        reference_id: Literal["model_name", "model_id"] = "model_name",
        protein_information: Literal["protein_intensity", "zscore"] = "protein_intensity",
        protein_id: Literal["uniprot_id", "symbol"] = "uniprot_id",
        copy: bool = False,
    ) -> AnnData:
        """Fetch protein expression.

        For each cell, we fetch protein intensity values acquired using data-independent acquisition mass spectrometry (DIA-MS).

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "cell_line_name".
            reference_id: The type of cell line identifier in the meta data, model_name or model_id. Defaults to "model_name".
            protein_information: The type of protein expression data to fetch, protein_intensity or zscore. Defaults to "protein_intensity".
            protein_id: The protein identifier saved in the fetched meta data, uniprot_id or symbol. Defaults to "uniprot_id".
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with protein expression annotation.
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified `cell_line_type` can be found in the protein expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Please ensure that you are using one of the available query IDs present in the adata.obs for the annotation. \n"
                "If the desired query ID is not available, you can fetch the cell line metadata \n"
                "using the `annotate_cell_lines()` function before calling annotate_protein_expression(). \n"
                "This will help ensure that the required query ID is included in your data"
            )

        if reference_id not in self.proteomics_data.columns:
            raise ValueError(
                f"The specified `reference_id`{reference_id} can't be found in the protein expression data. \n"
                "Please use the reference identifier available in the metadata.  \n"
                "To solve the issue, you can call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.protein_expression()` method, you can obtain the available reference identifiers in the metadata. "
            )

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.proteomics_data[reference_id]))

        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata. \n"
                "However, none of the query IDs could be found in the proteomics data. \n"
                "It is recommended to stop the annotation process. \n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.protein_expression()` method, \n"
                "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs. \n"
                "Additionally, you can call the `CellLineMetaData.annotate_cell_lines` function \n"
                "to acquire more possible query IDs that can be used for annotation purposes."
            )

        if len(not_matched_identifiers) > 0:
            print(f"There are {identifier_num_all} identifiers in `adata.obs`.")
            print(
                f"[bold yellow]But following {len(not_matched_identifiers)} identifiers can't be found in the protein expression data. "
            )
            print(
                "Resulting in NA values for their corresponding metadata. Please check again: ",
                *not_matched_identifiers,
                sep="\n- ",
            )
        # convert the original protein intensities table from long format to wide format, group by the cell lines
        prot_exp = self.proteomics_data[[reference_id, protein_id, protein_information]]
        prot_exp = pd.pivot(prot_exp, index=reference_id, columns=protein_id, values=protein_information)
        prot_exp = prot_exp.reindex(adata.obs[query_id])
        prot_exp.index = adata.obs.index
        adata.obsm["proteomics_" + protein_information] = prot_exp

        return adata

    def annotate_ccle_expression(
        self,
        adata: AnnData,
        query_id: str = "DepMap_ID",
        copy: bool = False,
    ) -> AnnData:
        """Fetch CCLE expression data.

         For each cell, we fetch gene expression TPM values of the protein coding genes for its corresponding DepMap
         cell line.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "DepMap_ID".
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with CCLE expression annotation.
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified cell line type can be found in the CCLE expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Please ensure that you are using one of the available query IDs present in the adata.obs for the annotation. \n"
                "If the desired query ID is not available, you can fetch the cell line metadata \n"
                "using the `annotate_cell_lines()` function before calling annotate_ccle_expression(). \n"
                "This will help ensure that the required query ID is included in your data."
            )

        not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.ccle_expr.index))
        if len(not_matched_identifiers) > 0:
            print(
                "[bold yellow]Following identifiers can not be found in the CCLE expression data,"
                " their corresponding meta data are NA values. Please check it again:",
                *not_matched_identifiers,
                sep="\n- ",
            )
        identifier_num_all = len(adata.obs[query_id].unique())
        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the DepMap_id in the metadata. \n"
                "However, none of the query IDs could be found in the proteomics data. \n"
                "It is recommended to stop the annotation process. \n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.ccle_expression()` method, \n"
                "you can obtain the count of matched identifiers in the adata for different query IDs. \n"
                "Additionally, you can call the CellLineMetaData.annotate_cell_lines function to \n"
                "acquire DepMap_ID information that can be used for annotation purposes."
            )

        ccle_expression = self.ccle_expr.reindex(adata.obs[query_id])
        ccle_expression.index = adata.obs.index
        adata.obsm["CCLE_expression"] = ccle_expression

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for CellLineMetaData.

        The LookUp object provides an overview of the metadata to annotate.
        Each annotate_{metadata} function has a corresponding lookup function in the LookUp object,
        where users can search the reference_id in the metadata and
        compare with the query_id in their own data.

        Returns:
            Returns a LookUp object specific for cell line annotation.
        """
        return LookUp(
            type="cell_line",
            transfer_metadata=[
                self.cell_line_meta,
                self.cl_cancer_project_meta,
                self.driver_gene_intogen,
                self.driver_gene_cosmic,
                self.bulk_rna_sanger,
                self.bulk_rna_broad,
                self.proteomics_data,
                self.ccle_expr,
            ],
        )
