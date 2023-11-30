from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from rich import print
from scanpy import settings

from pertpy.data._dataloader import _download

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


class CellLineMetaData(MetaData):
    """Utilities to fetch cell line metadata."""

    def __init__(self):
        settings.cachedir = ".pertpy_cache"
        # Download cell line metadata from DepMap
        # Source: https://depmap.org/portal/download/all/ (DepMap Public 22Q2)
        cell_line_file_path = settings.cachedir.__str__() + "/sample_info.csv"
        if not Path(cell_line_file_path).exists():
            print("[bold yellow]No DepMap metadata file found. Starting download now.")
            _download(
                url="https://ndownloader.figshare.com/files/35020903",
                output_file_name="sample_info.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.cell_line_meta = pd.read_csv(cell_line_file_path)

        # Download cell line metadata from The Genomics of Drug Sensitivity in Cancer Project
        # Source: https://www.cancerrxgene.org/celllines
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
                    block_size=4096,
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

        # Download metadata for driver genes from DepMap.Sanger
        # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Gene annotation)
        gene_annotation_file_path = settings.cachedir.__str__() + "/gene_identifiers_20191101.csv"
        if not Path(gene_annotation_file_path).exists():
            print("[bold yellow]No metadata file was found for gene annotation." " Starting download now.")
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/gene_identifiers_20191101.csv",
                output_file_name="gene_identifiers_20191101.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.gene_annotation = pd.read_table(gene_annotation_file_path, delimiter=",")

        # Download bulk RNA-seq data collated by the Wellcome Sanger Institute and the Broad Institute from DepMap.Sanger
        # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Expression data)
        # issue: read count values contain random whitespace, not sure what it supposes to mean
        # solution: remove the white space and convert to int before depmap updates the metadata
        bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_read_count_20220624_processed.csv"
        if not Path(bulk_rna_sanger_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for bulk RNA-seq data of Sanger cell line."
                " Starting download now..."
            )
            _download(
                url="https://figshare.com/ndownloader/files/42467103",
                output_file_name="rnaseq_read_count_20220624_processed.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_file_path, index_col=0)

        # Download CCLE expression data from DepMap
        # Source: https://depmap.org/portal/download/all/ (DepMap Public 22Q2)
        bulk_rna_broad_file_path = settings.cachedir.__str__() + "/CCLE_expression_full.csv"
        if not Path(bulk_rna_broad_file_path).exists():
            print("[bold yellow]No metadata file was found for CCLE expression data. Starting download now.")
            _download(
                url="https://figshare.com/ndownloader/files/34989922",
                output_file_name="CCLE_expression_full.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path, index_col=0)

        # Download proteomics data processed by DepMap.Sanger
        # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Proteomics)
        proteomics_file_path = settings.cachedir.__str__() + "/proteomics_all_20221214_processed.csv"
        if not Path(proteomics_file_path).exists():
            print("[bold yellow]No metadata file was found for proteomics data (DepMap.Sanger). Starting download now.")
            _download(
                url="https://figshare.com/ndownloader/files/42468393",
                output_file_name="proteomics_all_20221214_processed.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.proteomics_data = pd.read_csv(proteomics_file_path, index_col=0)

        # Download GDSC drug response data
        # Source: https://www.cancerrxgene.org/downloads/bulk_download (Drug Screening - IC50s)
        drug_response_gdsc1_file_path = settings.cachedir.__str__() + "/ic50_gdsc1.xlsx"
        if not Path(drug_response_gdsc1_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for drug response data of GDSC1 dataset."
                " Starting download now."
            )
            _download(
                url="https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_fitted_dose_response_24Jul22.xlsx",
                output_file_name="ic50_gdsc1.xlsx",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.drug_response_gdsc1 = pd.read_excel(drug_response_gdsc1_file_path)
        self.drug_response_gdsc1 = self.drug_response_gdsc1.iloc[:, [3, 4, 5, 7, 8, 15, 16]]
        self.drug_response_gdsc1.rename(columns=lambda col: col.lower(), inplace=True)
        self.drug_response_gdsc1 = self.drug_response_gdsc1.loc[
            self.drug_response_gdsc1.groupby(["cell_line_name", "drug_name"])["auc"].idxmax()
        ]
        self.drug_response_gdsc1 = self.drug_response_gdsc1.reset_index(drop=True)

        drug_response_gdsc2_file_path = settings.cachedir.__str__() + "/ic50_gdsc2.xlsx"
        if not Path(drug_response_gdsc2_file_path).exists():
            print(
                "[bold yellow]No metadata file was found for drug response data of GDSC2 dataset."
                " Starting download now."
            )
            _download(
                url="https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC2_fitted_dose_response_24Jul22.xlsx",
                output_file_name="ic50_gdsc2.xlsx",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.drug_response_gdsc2 = pd.read_excel(drug_response_gdsc2_file_path)
        self.drug_response_gdsc2 = self.drug_response_gdsc2.iloc[:, [3, 4, 5, 7, 8, 15, 16]]
        self.drug_response_gdsc2.rename(columns=lambda col: col.lower(), inplace=True)
        self.drug_response_gdsc2 = self.drug_response_gdsc2.loc[
            self.drug_response_gdsc2.groupby(["cell_line_name", "drug_name"])["auc"].idxmax()
        ]
        self.drug_response_gdsc2 = self.drug_response_gdsc2.reset_index(drop=True)

    def annotate_cell_lines(
        self,
        adata: AnnData,
        query_id: str = "DepMap_ID",
        reference_id: str = "DepMap_ID",
        cell_line_information: list[str] | None = None,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        verbosity: int | str = 5,
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
            print_warning: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with cell line annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs['cell_line_name'] = 'MCF7'
            >>> pt_metadata = pt.tl.CellLineMetaData()
            >>> adata_annotated = pt_metadata.annotate_cell_lines(adata=adata, reference_id='cell_line_name', query_id='cell_line_name', copy=True)
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
                    "[bold blue]`stripped_cell_line_name` is used as reference and query indentifier ",
                    " to annotate cell line metadata from Cancerrxgene. "
                    "So please make sure that `stripped_cell_line_name` is available in the adata.obs. ",
                    "or use the DepMap as `cell_line_source` to annotate the cell line first ",
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
                    f"Attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata.\n"
                    "However, none of the query IDs could be found in the cell line annotation data.\n"
                    "The annotation process has been halted.\n"
                    "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object.\n"
                    "By using the `LookUp.cell_line()` method, "
                    "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs."
                )

            if len(not_matched_identifiers) > 0:
                self._print_unmatched_ids(
                    total_identifiers=identifier_num_all,
                    unmatched_identifiers=not_matched_identifiers,
                    verbosity=verbosity,
                    metadata_type="cell line",
                )

            if cell_line_information is None:
                # If no cell_line_information is specified, all metadata is fetched by default.
                # Sometimes there is already different cell line information in the AnnData object.
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
                # We will subset the original metadata dataframe correspondingly and add them to the AnnData object.
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
                f"The requested cell line type {reference_id} is currently unavailable in the database.\n"
                "To solve ths issue, please refer to the available reference identifier in the chosen database.\n"
                "DepMap_ID is compared by default.\n"
                "Alternatively, you can call the `CellLineMetaData.lookup()` function to create a LookUp object.\n"
                "By using the `LookUp.cell_line()` method, you can obtain the available reference identifiers in the metadata."
            )

        return adata

    def annotate_bulk_rna_expression(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        cell_line_source: Literal["broad", "sanger"] = "sanger",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch bulk rna expression.

        For each cell, we fetch bulk rna expression from either Broad or Sanger cell line.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "cell_line_name" if `cell_line_source` is sanger, otherwise "DepMap_ID".
            cell_line_source: The bulk rna expression data from either broad or sanger cell line. Defaults to "sanger".
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with bulk rna expression annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs['cell_line_name'] = 'MCF7'
            >>> pt_metadata = pt.tl.CellLineMetaData()
            >>> adata_annotated = pt_metadata.annotate_cell_lines(adata=adata, reference_id='cell_line_name', query_id='cell_line_name', copy=True)
            >>> pt_metadata.annotate_bulk_rna_expression(adata_annotated)
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified `cell_line_type` can be found in the bulk rna expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`.\n"
                "Please ensure that you are using one of the available query IDs present in the adata.obs for the annotation.\n"
                "If the desired query ID is not available, you can fetch the cell line metadata "
                "using the `annotate_cell_lines()` function before calling annotate_ccle_expression(). "
                "This will help ensure that the required query ID is included in your data, e.g. stripped_cell_line_name, DepMap ID."
            )

        identifier_num_all = len(adata.obs[query_id].unique())
        if cell_line_source == "sanger":
            reference_id = "model_name"
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.bulk_rna_sanger.index))
        else:
            reference_id = "DepMap_ID"
            print(
                "To annotate bulk RNA expression data from Broad Institue, ",
                "`DepMap_ID` is used as default reference and query indentifier if no `reference_id` is given. ",
                "Please make sure that `DepMap_ID` is available in the adata.obs. ",
                "Alternatively, use the `annotate_cell_lines()` function to annotate the cell line first ",
            )

            if query_id == "cell_line_name":
                query_id = "DepMap_ID"
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.bulk_rna_broad.index))

        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata."
                "However, none of the query IDs could be found in the bulk RNA expression data.\n"
                "The annotation process has been halted.\n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object.\n"
                "By using the `LookUp.bulk_rna_expression()` method, "
                "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs.\n"
                "Additionally, you can call the `CellLineMetaData.annotate_cell_lines()` function "
                "to acquire more possible query IDs that can be used for annotation purposes."
            )

        if len(not_matched_identifiers) > 0:
            self._print_unmatched_ids(
                total_identifiers=identifier_num_all,
                unmatched_identifiers=not_matched_identifiers,
                verbosity=verbosity,
                metadata_type="bulk RNA expression",
            )

        if cell_line_source == "sanger":
            sanger_rna_exp = self.bulk_rna_sanger[self.bulk_rna_sanger.index.isin(adata.obs[query_id])]
            sanger_rna_exp = sanger_rna_exp.reindex(adata.obs[query_id])
            sanger_rna_exp.index = adata.obs.index
            adata.obsm["bulk_rna_expression_sanger"] = sanger_rna_exp
        else:
            broad_rna_exp = self.bulk_rna_broad[self.bulk_rna_broad.index.isin(adata.obs[query_id])]
            ccle_expression = broad_rna_exp.reindex(adata.obs[query_id])
            ccle_expression.index = adata.obs.index
            adata.obsm["bulk_rna_expression_broad"] = ccle_expression

        return adata

    def annotate_protein_expression(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        reference_id: Literal["model_name", "model_id"] = "model_name",
        protein_information: Literal["protein_intensity", "zscore"] = "protein_intensity",
        protein_id: Literal["uniprot_id", "symbol"] = "uniprot_id",
        verbosity: int | str = 5,
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
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with protein expression annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs['cell_line_name'] = 'MCF7'
            >>> pt_metadata = pt.tl.CellLineMetaData()
            >>> adata_annotated = pt_metadata.annotate_cell_lines(adata=adata, reference_id='cell_line_name', query_id='cell_line_name', copy=True)
            >>> pt_metadata.annotate_protein_expression(adata_annotated)
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
                "To solve the issue, please use the reference identifier available in the metadata.  \n"
                "Alternatively, you can call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.protein_expression()` method, you can obtain the available reference identifiers in the metadata. "
            )

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.proteomics_data[reference_id]))

        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata."
                "However, none of the query IDs could be found in the proteomics data. \n"
                "The annotation process has been halted. \n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.protein_expression()` method, "
                "you can obtain the count of matched identifiers in the adata for different types of reference IDs and query IDs. \n"
                "Additionally, you can call the `CellLineMetaData.annotate_cell_lines` function "
                "to acquire more possible query IDs that can be used for annotation purposes."
            )

        if len(not_matched_identifiers) > 0:
            self._print_unmatched_ids(
                total_identifiers=identifier_num_all,
                unmatched_identifiers=not_matched_identifiers,
                verbosity=verbosity,
                metadata_type="protein expression",
            )

        # convert the original protein intensities table from long format to wide format, group by the cell lines
        prot_exp = self.proteomics_data[[reference_id, protein_id, protein_information]]
        prot_exp = pd.pivot(prot_exp, index=reference_id, columns=protein_id, values=protein_information)
        prot_exp = prot_exp.reindex(adata.obs[query_id])
        prot_exp.index = adata.obs.index
        adata.obsm["proteomics_" + protein_information] = prot_exp

        return adata

    def annotate_from_gdsc(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        reference_id: Literal["cell_line_name", "sanger_model_id", "cosmic_id"] = "cell_line_name",
        query_perturbation: str = "perturbation",
        reference_perturbation: Literal["drug_name", "drug_id"] = "drug_name",
        gdsc_dataset: Literal[1, 2] = 1,
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch drug response data.

        For each cell, we fetch drug response data as natural log of the fitted IC50 for its corresponding cell line and perturbation from GDSC fitted data results file.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information. Defaults to "cell_line_name".
            reference_id: The type of cell line identifier in the meta data, cell_line_name, sanger_model_id or cosmic_id. Defaults to "cell_line_name".
            query_perturbation: The column of `.obs` with perturbation information. Defaults to "perturbation".
            reference_perturbation: The type of perturbation in the meta data, drug_name or drug_id. Defaults to "drug_name".
            gdsc_dataset: The GDSC dataset, 1 or 2. Defaults to 1. The GDSC1 dataset updates previous releases with additional drug screening data from the Wellcome Sanger Institute and Massachusetts General Hospital. It covers 970 Cell lines and 403 Compounds with 333292 IC50s. GDSC2 is new and has 243,466 IC50 results from the latest screening at the Wellcome Sanger Institute using improved experimental procedures.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all". Defaults to 5.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            Returns an AnnData object with drug response annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.mcfarland_2020()
            >>> pt_metadata = pt.tl.CellLineMetaData()
            >>> pt_metadata.annotate_from_gdsc(adata, query_id='cell_line')
        """
        if copy:
            adata = adata.copy()
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Please ensure that you are using one of the available query IDs present in the adata.obs for the annotation. \n"
                "If the desired query ID is not available, you can fetch the cell line metadata "
                "using the `annotate_cell_lines()` function before calling `annotate_from_gdsc()`. "
                "This will help ensure that the required query ID is included in your data."
            )
        if gdsc_dataset == 1:
            gdsc_data = self.drug_response_gdsc1
        else:
            gdsc_data = self.drug_response_gdsc2

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(gdsc_data[reference_id]))
        if len(not_matched_identifiers) > 0:
            self._print_unmatched_ids(
                total_identifiers=identifier_num_all,
                unmatched_identifiers=not_matched_identifiers,
                verbosity=verbosity,
                metadata_type="drug response",
            )

        if len(not_matched_identifiers) == identifier_num_all:
            raise ValueError(
                f"You are attempting to match the query id {query_id} in the adata.obs to the reference id {reference_id} in the metadata. \n"
                "However, none of the query IDs could be found in the drug response data. \n"
                "The annotation process has been halted. \n"
                "To resolve this issue, please call the `CellLineMetaData.lookup()` function to create a LookUp object. \n"
                "By using the `LookUp.drug_response_gdsc()` method, \n"
                "you can obtain the count of matched identifiers in the adata for different query IDs. \n"
                "Additionally, you can call the `CellLineMetaData.annotate_cell_lines()` function to \n"
                "acquire more cell line information that can be used for annotation purposes."
            )
        old_index_name = "index" if adata.obs.index.name is None else adata.obs.index.name
        adata.obs = (
            adata.obs.reset_index()
            .set_index([query_id, query_perturbation])
            .assign(ln_ic50=self.drug_response_gdsc1.set_index([reference_id, reference_perturbation]).ln_ic50)
            .reset_index()
            .set_index(old_index_name)
        )

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for CellLineMetaData.

        The LookUp object provides an overview of the metadata to annotate.
        Each annotate_{metadata} function has a corresponding lookup function in the LookUp object,
        where users can search the reference_id in the metadata and
        compare with the query_id in their own data.

        Returns:
            Returns a LookUp object specific for cell line annotation.

        Examples:
            >>> import pertpy as pt
            >>> pt_metadata = pt.tl.CellLineMetaData()
            >>> lookup = pt_metadata.lookup()
        """
        return LookUp(
            type="cell_line",
            transfer_metadata=[
                self.cell_line_meta,
                self.cl_cancer_project_meta,
                self.gene_annotation,
                self.bulk_rna_sanger,
                self.bulk_rna_broad,
                self.proteomics_data,
                self.drug_response_gdsc1,
                self.drug_response_gdsc2,
            ],
        )
