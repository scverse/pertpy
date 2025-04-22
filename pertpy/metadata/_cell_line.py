from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from lamin_utils import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.pyplot import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scanpy import settings
from scipy import stats

from pertpy._doc import _doc_params, doc_common_plot_args
from pertpy.data._dataloader import _download

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


class CellLine(MetaData):
    """Utilities to fetch cell line metadata."""

    def __init__(self):
        super().__init__()
        self.depmap = None
        self.cancerxgene = None
        self.gene_annotation = None
        self.bulk_rna_sanger = None
        self.bulk_rna_broad = None
        self.proteomics = None
        self.drug_response_gdsc1 = None
        self.drug_response_gdsc2 = None
        self.drug_response_prism = None

    def _download_cell_line(self, cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap") -> None:
        if cell_line_source == "DepMap":
            # Download cell line metadata from DepMap
            # Source: https://depmap.org/portal/download/all/ (DepMap Public 23Q4)
            depmap_cell_line_path = Path(settings.cachedir) / "depmap_23Q4_info.csv"
            if not Path(depmap_cell_line_path).exists():
                _download(
                    url="https://ndownloader.figshare.com/files/43746708",
                    output_file_name="depmap_23Q4_info.csv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.depmap = pd.read_csv(depmap_cell_line_path)
            self.depmap = self.depmap.reset_index().rename(columns={"CellLineName": "cell_line_name"})
        else:
            # Download cell line metadata from The Genomics of Drug Sensitivity in Cancer Project
            # Source: https://www.cancerrxgene.org/celllines
            cancerxgene_cell_line_path = Path(settings.cachedir) / "cell_line_cancer_project.csv"
            transformed_cancerxgene_cell_line_path = Path(settings.cachedir) / "cancerrxgene_info.csv"

            if not Path(transformed_cancerxgene_cell_line_path).exists():
                if not Path(cancerxgene_cell_line_path).exists():
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
                self.cancerxgene = pd.read_csv(cancerxgene_cell_line_path)
                self.cancerxgene.columns = self.cancerxgene.columns.str.strip()
                self.cancerxgene["stripped_cell_line_name"] = (
                    self.cancerxgene["Cell line Name"]
                    .str.replace(r"\-|\.", "", regex=True)
                    .str.upper()
                    .astype("category")
                )
                # pivot the data frame so that each cell line has only one row of metadata
                index_col = set(self.cancerxgene.columns) - {
                    "Datasets",
                    "number of drugs",
                }
                self.cancerxgene = self.cancerxgene.pivot(index=index_col, columns="Datasets", values="number of drugs")
                self.cancerxgene.columns.name = None
                self.cancerxgene = self.cancerxgene.reset_index().rename(columns={"Cell line Name": "cell_line_name"})
                self.cancerxgene.to_csv(transformed_cancerxgene_cell_line_path)
            else:
                self.cancerxgene = pd.read_csv(transformed_cancerxgene_cell_line_path, index_col=0)

    def _download_gene_annotation(self) -> None:
        # Download metadata for driver genes from DepMap.Sanger
        # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Gene annotation)
        gene_annotation_file_path = Path(settings.cachedir) / "genes_info.csv"
        if not Path(gene_annotation_file_path).exists():
            _download(
                url="https://cog.sanger.ac.uk/cmp/download/gene_identifiers_20191101.csv",
                output_file_name="genes_info.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.gene_annotation = pd.read_table(gene_annotation_file_path, delimiter=",")

    def _download_bulk_rna(self, cell_line_source: Literal["broad", "sanger"] = "broad") -> None:
        if cell_line_source == "sanger":
            # Download bulk RNA-seq data collated by the Wellcome Sanger Institute and the Broad Institute from DepMap.Sanger
            # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Expression data)
            # issue: read count values contain random whitespace
            # solution: remove the white space and convert to int before depmap updates the metadata
            bulk_rna_sanger_file_path = Path(settings.cachedir) / "rnaseq_sanger_info.csv"
            if not Path(bulk_rna_sanger_file_path).exists():
                _download(
                    url="https://figshare.com/ndownloader/files/42467103",
                    output_file_name="rnaseq_sanger_info.csv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_file_path, index_col=0, dtype="unicode")
        else:
            # Download CCLE expression data from DepMap
            # Source: https://depmap.org/portal/download/all/ (DepMap Public 22Q2)
            bulk_rna_broad_file_path = Path(settings.cachedir) / "rnaseq_depmap_info.csv"
            if not Path(bulk_rna_broad_file_path).exists():
                _download(
                    url="https://figshare.com/ndownloader/files/34989922",
                    output_file_name="rnaseq_depmap_info.csv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path, index_col=0)

    def _download_proteomics(self) -> None:
        # Download proteomics data processed by DepMap.Sanger
        # Source: https://cellmodelpassports.sanger.ac.uk/downloads (Proteomics)
        proteomics_file_path = Path(settings.cachedir) / "proteomics_info.csv"
        if not Path(proteomics_file_path).exists():
            _download(
                url="https://figshare.com/ndownloader/files/42468393",
                output_file_name="proteomics_info.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.proteomics = pd.read_csv(proteomics_file_path, index_col=0)

    def _download_gdsc(self, gdsc_dataset: Literal[1, 2] = 1) -> None:
        if gdsc_dataset == 1:
            # Download GDSC drug response data
            # Source: https://www.cancerrxgene.org/downloads/bulk_download (Drug Screening - IC50s and AUC)
            # URL: https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.4/GDSC1_fitted_dose_response_24Jul22.xlsx
            drug_response_gdsc1_file_path = Path(settings.cachedir) / "gdsc1_info.csv"
            if not Path(drug_response_gdsc1_file_path).exists():
                _download(
                    url="https://figshare.com/ndownloader/files/43757235",
                    output_file_name="gdsc1_info.csv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.drug_response_gdsc1 = pd.read_csv(drug_response_gdsc1_file_path, index_col=0)
        if gdsc_dataset == 2:
            drug_response_gdsc2_file_path = Path(settings.cachedir) / "gdsc2_info.csv"
            if not Path(drug_response_gdsc2_file_path).exists():
                _download(
                    url="https://figshare.com/ndownloader/files/43757232",
                    output_file_name="gdsc2_info.csv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.drug_response_gdsc2 = pd.read_csv(drug_response_gdsc2_file_path, index_col=0)

    def _download_prism(self) -> None:
        # Download PRISM drug response data
        # Source: DepMap PRISM Repurposing 19Q4 secondary screen dose response curve parameters
        drug_response_prism_file_path = Path(settings.cachedir) / "prism_info.csv"
        if not Path(drug_response_prism_file_path).exists():
            _download(
                url="https://figshare.com/ndownloader/files/20237739",
                output_file_name="prism_info.csv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        df = pd.read_csv(drug_response_prism_file_path, index_col=0)[["depmap_id", "name", "ic50", "ec50", "auc"]]
        df = df.dropna(subset=["depmap_id", "name"])
        df = df.groupby(["depmap_id", "name"]).mean().reset_index()
        self.drug_response_prism = df

    def annotate(
        self,
        adata: AnnData,
        query_id: str = "DepMap_ID",
        reference_id: str = "ModelID",
        fetch: list[str] | None = None,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Annotate cell lines.

        For each cell, we fetch cell line annotation from either the Dependency Map (DepMap) or The Genomics of Drug Sensitivity in Cancer Project (Cancerxgene).

        Args:
            adata: The data object to annotate.
            query_id: The column of ``.obs`` with cell line information.
            reference_id: The type of cell line identifier in the metadata, e.g. ModelID, CellLineName	or StrippedCellLineName.
                          If fetching cell line metadata from Cancerrxgene, it is recommended to choose "stripped_cell_line_name".
            fetch: The metadata to fetch.
            cell_line_source: The source of cell line metadata, DepMap or Cancerrxgene.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all".
            copy: Determines whether a copy of ``adata`` is returned.

        Returns:
            Returns an AnnData object with cell line annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs["cell_line_name"] = "MCF7"
            >>> pt_metadata = pt.md.CellLine()
            >>> adata_annotated = pt_metadata.annotate(adata=adata,
            >>>                                        reference_id='cell_line_name',
            >>>                                        query_id='cell_line_name',
            >>>                                        fetch=["cell_line_name", "Age", "OncotreePrimaryDisease"],
            >>>                                        copy=True)
        """
        if copy:
            adata = adata.copy()

        if cell_line_source == "DepMap":
            if self.depmap is None:
                self._download_cell_line(cell_line_source="DepMap")
            cell_line_meta = self.depmap
        else:
            reference_id = "stripped_cell_line_name"
            if query_id == "DepMap_ID":
                query_id = "stripped_cell_line_name"
                logger.error(
                    "`stripped_cell_line_name` is used as reference and query identifier to annotate cell line metadata from Cancerrxgene. "
                    "Ensure that stripped cell line names are available in 'adata.obs.' or use the DepMap as `cell_line_source` to annotate the cell line first."
                )
            if self.cancerxgene is None:
                self._download_cell_line(cell_line_source="Cancerrxgene")
            cell_line_meta = self.cancerxgene

        if query_id not in adata.obs.columns:
            raise ValueError(f"The requested query_id {query_id} is not in `adata.obs`.")

        if reference_id in cell_line_meta.columns:
            # If the specified cell line type can be found in the database,
            # we can compare these keys and fetch the corresponding metadata.
            identifier_num_all = len(adata.obs[query_id].unique())
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(cell_line_meta[reference_id]))

            self._warn_unmatch(
                total_identifiers=identifier_num_all,
                unmatched_identifiers=not_matched_identifiers,
                query_id=query_id,
                reference_id=reference_id,
                metadata_type="cell line",
                verbosity=verbosity,
            )

            if fetch is not None:
                # If fetch is specified and can be found in the DepMap database,
                # We will subset the original metadata dataframe correspondingly and add them to the AnnData object.
                # Redundant information will be removed.
                if set(fetch).issubset(set(cell_line_meta.columns)):
                    if reference_id not in fetch:
                        fetch.append(reference_id)
                else:
                    raise ValueError(
                        "Selected cell line information is not present in the metadata.\n"
                        "Please create a `CellLineMetaData.lookup()` object to obtain the available cell line information in the metadata."
                    )

            # If no fetch is specified, all metadata is fetched by default.
            # Sometimes there is already different cell line information in the AnnData object.
            # To avoid redundant information we will remove duplicate information from metadata after merging.
            adata.obs = (
                adata.obs.merge(
                    cell_line_meta if fetch is None else cell_line_meta[fetch],
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

        else:
            raise ValueError(
                f"The requested cell line type {reference_id} is currently unavailable in the database.\n"
                "Refer to the available reference identifier in the chosen database.\n"
                "DepMap_ID is compared by default.\n"
                "Alternatively, create a `CellLineMetaData.lookup()` object to "
                "obtain the available reference identifiers in the metadata."
            )

        return adata

    def annotate_bulk_rna(
        self,
        adata: AnnData,
        query_id: str = None,
        cell_line_source: Literal["broad", "sanger"] = "sanger",
        verbosity: int | str = 5,
        gene_identifier: Literal["gene_name", "gene_ID", "both"] = "gene_ID",
        copy: bool = False,
    ) -> AnnData:
        """Fetch bulk rna expression from the Broad or Sanger.

        For each cell, we fetch bulk rna expression from either Broad or Sanger cell line.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information.
                Defaults to "cell_line_name" if `cell_line_source` is sanger, otherwise "DepMap_ID".
            cell_line_source: The bulk rna expression data from either broad or sanger cell line.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all".
            gene_identifier: The type of gene identifier saved in the fetched meta data, 'gene_name', 'gene_ID' or 'both'.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with bulk rna expression annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs["cell_line_name"] = "MCF7"
            >>> pt_metadata = pt.md.CellLine()
            >>> adata_annotated = pt_metadata.annotate(
            ...     adata=adata, reference_id="cell_line_name", query_id="cell_line_name", copy=True
            ... )
            >>> pt_metadata.annotate_bulk_rna(adata_annotated)
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified `cell_line_type` can be found in the bulk rna expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns and query_id is not None:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Ensure that you are using one of the available query IDs present in the adata.obs for the annotation."
                "If the desired query ID is not available, you can fetch the cell line metadata "
                "using the `annotate()` function before calling 'annotate_bulk_rna()'. "
                "This ensures that the required query ID is included in your data, e.g. stripped_cell_line_name, DepMap ID."
            )
        if query_id is None:
            query_id = "cell_line_name" if cell_line_source == "sanger" else "DepMap_ID"
        identifier_num_all = len(adata.obs[query_id].unique())

        # Lazily download the bulk rna expression data
        if cell_line_source == "sanger":
            if query_id not in adata.obs.columns:
                raise ValueError(
                    "To annotate bulk RNA data from Wellcome Sanger Institute, `cell_line_name` is used as default reference and query identifier if no `query_id` is given."
                    "Ensure that you have column `cell_line_name` in `adata.obs` or specify column name in which cell line name is stored."
                    "If cell line name isn't available in 'adata.obs', use `annotate()` to annotate the cell line first."
                )
            if self.bulk_rna_sanger is None:
                self._download_bulk_rna(cell_line_source="sanger")
            reference_id = "model_name"
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.bulk_rna_sanger.index))
        else:
            if query_id not in adata.obs.columns:
                raise ValueError(
                    "To annotate bulk RNA data from Broad Institue, `DepMap_ID` is used as default reference and query identifier if no `query_id` is given."
                    "Ensure that you have column `DepMap_ID` in `adata.obs` or specify column name in which DepMap ID is stored."
                    "If DepMap ID isn't available in 'adata.obs', use `annotate()` to annotate the cell line first."
                )
            reference_id = "DepMap_ID"

            if self.bulk_rna_broad is None:
                self._download_bulk_rna(cell_line_source="broad")
            not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.bulk_rna_broad.index))

        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id=reference_id,
            metadata_type="bulk RNA",
            verbosity=verbosity,
        )

        if cell_line_source == "sanger":
            sanger_rna_exp = self.bulk_rna_sanger[self.bulk_rna_sanger.index.isin(adata.obs[query_id])]
            sanger_rna_exp = sanger_rna_exp.reindex(adata.obs[query_id])
            sanger_rna_exp.index = adata.obs.index
            adata.obsm["bulk_rna_sanger"] = sanger_rna_exp
        else:
            if gene_identifier == "gene_ID":
                self.bulk_rna_broad.columns = [
                    (gene_name.split(" (")[1].split(")")[0] if "(" in gene_name else gene_name)
                    for gene_name in self.bulk_rna_broad.columns
                ]
            elif gene_identifier == "gene_name":
                self.bulk_rna_broad.columns = [
                    gene_name.split(" (")[0] if "(" in gene_name else gene_name
                    for gene_name in self.bulk_rna_broad.columns
                ]
            broad_rna_exp = self.bulk_rna_broad[self.bulk_rna_broad.index.isin(adata.obs[query_id])]
            ccle_expression = broad_rna_exp.reindex(adata.obs[query_id])
            ccle_expression.index = adata.obs.index
            adata.obsm["bulk_rna_broad"] = ccle_expression

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
            query_id: The column of `.obs` with cell line information.
            reference_id: The type of cell line identifier in the meta data, model_name or model_id.
            protein_information: The type of protein expression data to fetch, protein_intensity or zscore.
            protein_id: The protein identifier saved in the fetched meta data, uniprot_id or symbol.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or "all".
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with protein expression annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.dialogue_example()
            >>> adata.obs["cell_line_name"] = "MCF7"
            >>> pt_metadata = pt.md.CellLine()
            >>> adata_annotated = pt_metadata.annotate(
            ...     adata=adata, reference_id="cell_line_name", query_id="cell_line_name", copy=True
            ... )
            >>> pt_metadata.annotate_protein_expression(adata_annotated)
        """
        if copy:
            adata = adata.copy()

        # Make sure that the specified `cell_line_type` can be found in the protein expression data,
        # then we can compare these keys and fetch the corresponding metadata.
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in `adata.obs`. \n"
                "If the desired query ID is not available, you can fetch the cell line metadata \n"
                "using the `annotate()` function before calling annotate_protein_expression(). \n"
                "This ensures that the required query ID is included in your data."
            )
        # Lazily download the proteomics data
        if self.proteomics is None:
            self._download_proteomics()
        if reference_id not in self.proteomics.columns:
            raise ValueError(
                f"The specified `reference_id`{reference_id} can't be found in the protein expression data. \n"
                "To solve the issue, please use the reference identifier available in the metadata.  \n"
                "Alternatively, create a `CellLineMetaData.lookup()` object to obtain the available reference identifiers in the metadata."
            )

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(self.proteomics[reference_id]))

        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id=reference_id,
            metadata_type="protein expression",
            verbosity=verbosity,
        )

        # convert the original protein intensities table from long format to wide format, group by the cell lines
        adata.obsm["proteomics_" + protein_information] = (
            self.proteomics[[reference_id, protein_id, protein_information]]
            .pivot(index=reference_id, columns=protein_id, values=protein_information)
            .reindex(adata.obs[query_id])
            .set_index(adata.obs.index)
        )
        return adata

    def annotate_from_gdsc(
        self,
        adata: AnnData,
        query_id: str = "cell_line_name",
        reference_id: Literal["cell_line_name", "sanger_model_id", "cosmic_id"] = "cell_line_name",
        query_perturbation: str = "perturbation",
        reference_perturbation: Literal["drug_name", "drug_id"] = "drug_name",
        gdsc_dataset: Literal["gdsc_1", "gdsc_2"] = "gdsc_1",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch drug response data from GDSC.

        For each cell, we fetch drug response data as natural log of the fitted IC50 and AUC for its
        corresponding cell line and perturbation from GDSC fitted data results file.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information.
            reference_id: The type of cell line identifier in the metadata, cell_line_name, sanger_model_id or cosmic_id.
            query_perturbation: The column of `.obs` with perturbation information.
            reference_perturbation: The type of perturbation in the metadata, drug_name or drug_id.
            gdsc_dataset: The GDSC dataset, 1 or 2, specified as 'gdsc_1' or 'gdsc_2'.
                          The GDSC1 dataset updates previous releases with additional drug screening data from the
                          Sanger Institute and Massachusetts General Hospital.
                          It covers 970 Cell lines and 403 Compounds with 333292 IC50s.
                          GDSC2 is new and has 243,466 IC50 results from the latest screening at the Sanger Institute.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or 'all'.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with drug response annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.mcfarland_2020()
            >>> pt_metadata = pt.md.CellLine()
            >>> pt_metadata.annotate_from_gdsc(adata, query_id="cell_line")
        """
        if copy:
            adata = adata.copy()
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Ensure that you are using one of the available query IDs present in 'adata.obs' for the annotation.\n"
                "If the desired query ID is not available, you can fetch the cell line metadata "
                "using the `annotate()` function before calling `annotate_from_gdsc()`. "
                "This ensures that the required query ID is included in your data."
            )
        # Lazily download the GDSC data
        if gdsc_dataset == "gdsc_1":
            if self.drug_response_gdsc1 is None:
                self._download_gdsc(gdsc_dataset=1)
            gdsc_data = self.drug_response_gdsc1
        elif gdsc_dataset == "gdsc_2":
            if self.drug_response_gdsc2 is None:
                self._download_gdsc(gdsc_dataset=2)
            gdsc_data = self.drug_response_gdsc2
        else:
            raise ValueError("The GDSC dataset specified in `gdsc_dataset` must be either 'gdsc_1' or 'gdsc_2'.")

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(gdsc_data[reference_id]))
        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id=reference_id,
            metadata_type="drug response",
            verbosity=verbosity,
        )

        old_index_name = "index" if adata.obs.index.name is None else adata.obs.index.name
        adata.obs = (
            adata.obs.reset_index()
            .set_index([query_id, query_perturbation])
            .assign(ln_ic50_gdsc=gdsc_data.set_index([reference_id, reference_perturbation]).ln_ic50)
            .assign(auc_gdsc=gdsc_data.set_index([reference_id, reference_perturbation]).auc)
            .reset_index()
            .set_index(old_index_name)
        )

        return adata

    def annotate_from_prism(
        self,
        adata: AnnData,
        query_id: str = "DepMap_ID",
        query_perturbation: str = "perturbation",
        verbosity: int | str = 5,
        copy: bool = False,
    ) -> AnnData:
        """Fetch drug response data from PRISM.

        For each cell, we fetch drug response data as IC50, EC50 and AUC for its
        corresponding cell line and perturbation from PRISM fitted data results file.
        Note that all rows where either `depmap_id` or `name` is missing will be dropped.

        Args:
            adata: The data object to annotate.
            query_id: The column of `.obs` with cell line information.
            query_perturbation: The column of `.obs` with perturbation information.
            verbosity: The number of unmatched identifiers to print, can be either non-negative values or 'all'.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            Returns an AnnData object with drug response annotation.

        Examples:
            >>> import pertpy as pt
            >>> adata = pt.dt.mcfarland_2020()
            >>> pt_metadata = pt.md.CellLine()
            >>> pt_metadata.annotate_from_prism(adata, query_id="DepMap_ID")
        """
        if copy:
            adata = adata.copy()
        if query_id not in adata.obs.columns:
            raise ValueError(
                f"The specified `query_id` {query_id} can't be found in the `adata.obs`. \n"
                "Ensure that you are using one of the available query IDs present in 'adata.obs' for the annotation.\n"
                "If the desired query ID is not available, you can fetch the cell line metadata "
                "using the `annotate()` function before calling `annotate_from_prism()`. "
                "This ensures that the required query ID is included in your data."
            )
        if self.drug_response_prism is None:
            self._download_prism()
        prism_data = self.drug_response_prism
        # PRISM starts most drug names with a lowercase letter, so we want to make it case-insensitive
        prism_data["name_lower"] = prism_data["name"].str.lower()
        adata.obs["perturbation_lower"] = adata.obs[query_perturbation].str.lower()

        identifier_num_all = len(adata.obs[query_id].unique())
        not_matched_identifiers = list(set(adata.obs[query_id]) - set(prism_data["depmap_id"]))
        self._warn_unmatch(
            total_identifiers=identifier_num_all,
            unmatched_identifiers=not_matched_identifiers,
            query_id=query_id,
            reference_id="depmap_id",
            metadata_type="drug response",
            verbosity=verbosity,
        )

        old_index_name = "index" if adata.obs.index.name is None else adata.obs.index.name
        adata.obs = (
            adata.obs.reset_index()
            .set_index([query_id, "perturbation_lower"])
            .assign(ic50_prism=prism_data.set_index(["depmap_id", "name"]).ic50)
            .assign(ec50_prism=prism_data.set_index(["depmap_id", "name"]).ec50)
            .assign(auc_prism=prism_data.set_index(["depmap_id", "name"]).auc)
            .reset_index()
            .set_index(old_index_name)
            .drop(columns="perturbation_lower")
        )

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for CellLineMetaData.

        The LookUp object provides an overview of the metadata to annotate.
        Each annotate_{metadata} function has a corresponding lookup function in the LookUp object,
        where users can search the reference_id in the metadata and
        compare with the query_id in their own data.

        Returns:
            A LookUp object specific for cell line annotation.

        Examples:
            >>> import pertpy as pt
            >>> pt_metadata = pt.md.CellLine()
            >>> lookup = pt_metadata.lookup()
        """
        # Fetch the metadata if it hasn't been downloaded yet
        if self.depmap is None:
            self._download_cell_line(cell_line_source="DepMap")
        if self.cancerxgene is None:
            self._download_cell_line(cell_line_source="Cancerrxgene")
        if self.gene_annotation is None:
            self._download_gene_annotation()
        if self.bulk_rna_broad is None:
            self._download_bulk_rna(cell_line_source="broad")
        if self.bulk_rna_sanger is None:
            self._download_bulk_rna(cell_line_source="sanger")
        if self.proteomics is None:
            self._download_proteomics()
        if self.drug_response_gdsc1 is None:
            self._download_gdsc(gdsc_dataset=1)
        if self.drug_response_gdsc2 is None:
            self._download_gdsc(gdsc_dataset=2)
        if self.drug_response_prism is None:
            self._download_prism()

        # Transfer the data
        return LookUp(
            type="cell_line",
            transfer_metadata=[
                self.depmap,
                self.cancerxgene,
                self.gene_annotation,
                self.bulk_rna_sanger,
                self.bulk_rna_broad,
                self.proteomics,
                self.drug_response_gdsc1,
                self.drug_response_gdsc2,
                self.drug_response_prism,
            ],
        )

    def _pairwise_correlation(
        self, mat1: np.array, mat2: np.array, row_name: Iterable, col_name: Iterable
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate the row-wise pearson correlation between two matrices.

        Args:
            mat1: Input array
            mat2: Input array
            row_name: Row name of the output dataframes
            col_name: Row name of the output dataframes

        Returns:
            Returns DataFrames for both the Pearson correlation coefficients and their associated p-values.
        """
        corr = np.empty((mat1.shape[0], mat2.shape[0]))
        pvals = np.empty((mat1.shape[0], mat2.shape[0]))

        for i in range(mat1.shape[0]):
            for j in range(mat2.shape[0]):
                if i > j:
                    corr[i, j] = corr[j, i]
                    pvals[i, j] = pvals[j, i]
                else:
                    corr[i, j], pvals[i, j] = stats.pearsonr(mat1[i], mat2[j])
        corr = pd.DataFrame(corr, index=row_name, columns=col_name)
        pvals = pd.DataFrame(pvals, index=row_name, columns=col_name)

        return corr, pvals

    def correlate(
        self,
        adata: AnnData,
        identifier: str = "DepMap_ID",
        metadata_key: str = "bulk_rna_broad",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        """Correlate cell lines with annotated metadata.

        Args:
            adata: Input data object.
            identifier: Column in `.obs` containing cell line identifiers.
            metadata_key: Key of the AnnData obsm for comparison with the X matrix.

        Returns:
            Returns pearson correlation coefficients and their corresponding p-values for matched and unmatched cell lines separately.
        """
        if metadata_key not in adata.obsm:
            raise ValueError("The metadata can not be found in adata.obsm")
        if identifier not in adata.obs:
            raise ValueError("The identifier can not be found in adata.obs")
        if adata.X.shape[1] != adata.obsm[metadata_key].shape[1]:
            raise ValueError(
                "Dimensions of adata.X do not match those of metadata. Ensure that they have the same gene list."
            )
        # Raise error if the genes are not the same
        if (
            isinstance(adata.obsm[metadata_key], pd.DataFrame)
            and sum(adata.obsm[metadata_key].columns != adata.var.index.values) > 0
        ):
            raise ValueError(
                "Column name of metadata is not the same as the index of adata.var. Ensure that the genes are in the same order."
            )

        # Divide cell lines into those are present and not present in the metadata
        overlapped_cl = adata[~adata.obsm[metadata_key].isna().all(axis=1), :]
        missing_cl = adata[adata.obsm[metadata_key].isna().all(axis=1), :]

        corr, pvals = self._pairwise_correlation(
            overlapped_cl.X,
            overlapped_cl.obsm[metadata_key].values,
            row_name=overlapped_cl.obs[identifier],
            col_name=overlapped_cl.obs[identifier],
        )
        if missing_cl is not None:
            new_corr, new_pvals = self._pairwise_correlation(
                missing_cl.X,
                overlapped_cl.obsm[metadata_key].values,
                row_name=missing_cl.obs[identifier],
                col_name=overlapped_cl.obs[identifier],
            )
        else:
            new_corr = new_pvals = None

        return corr, pvals, new_corr, new_pvals

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_correlation(  # noqa: D417
        self,
        adata: AnnData,
        corr: pd.DataFrame,
        pval: pd.DataFrame,
        *,
        identifier: str = "DepMap_ID",
        metadata_key: str = "bulk_rna_broad",
        category: str = "cell line",
        subset_identifier: str | int | Iterable[str] | Iterable[int] | None = None,
        return_fig: bool = False,
    ) -> Figure | None:
        """Visualise the correlation of cell lines with annotated metadata.

        Args:
            adata: Input data object.
            corr: Pearson correlation scores.
            pval: P-values for pearson correlation.
            identifier: Column in `.obs` containing the identifiers.
            metadata_key: Key of the AnnData obsm for comparison with the X matrix.
            category: The category for correlation comparison.
            subset_identifier: Selected identifiers for scatter plot visualization between the X matrix and `metadata_key`.
                              If not None, only the chosen cell line will be plotted, either specified as a value in `identifier` (string) or as an index number.
                              If None, all cell lines will be plotted.
            {common_plot_args}

        Returns:
            Pearson correlation coefficients and their corresponding p-values for matched and unmatched cell lines separately.
        """
        if corr is None or pval is None:
            raise ValueError(
                "Missing required input parameter: 'corr' or 'pval'. Please call the function `pt.md.CellLine.correlate()` to generate these outputs before proceeding."
            )

        if category == "cell line":
            if subset_identifier is None:
                annotation = "\n".join(
                    (
                        f"Mean pearson correlation: {np.mean(np.diag(corr)):.4f}",
                        f"Mean p-value: {np.mean(np.diag(pval)):.4f}",
                    )
                )
                plt.scatter(x=adata.obsm[metadata_key], y=adata.X)
                plt.xlabel(metadata_key)
                plt.ylabel("Baseline")
            else:
                subset_identifier_list = (
                    [subset_identifier] if isinstance(subset_identifier, str | int) else list(subset_identifier)
                )
                # Convert the valid identifiers to the index list
                if all(isinstance(id, str) for id in subset_identifier_list):
                    if set(subset_identifier_list).issubset(adata.obs[identifier].unique()):
                        subset_identifier_list = np.where(
                            np.isin(adata.obs[identifier].values, subset_identifier_list)
                        )[0]
                    else:
                        raise ValueError("`Subset_identifier` must be found in adata.obs.`identifier`.")
                elif all(isinstance(id, int) and 0 <= id < adata.n_obs for id in subset_identifier_list):
                    pass
                elif all(isinstance(id, int) and (id < 0 or id >= adata.n_obs) for id in subset_identifier_list):
                    raise ValueError("`Subset_identifier` out of index.")
                else:
                    raise ValueError("`Subset_identifier` must contain either all strings or all integers.")

                plt.scatter(
                    x=adata.obsm[metadata_key].iloc[subset_identifier_list],
                    y=adata[subset_identifier_list].X,
                )
                plt.xlabel(
                    f"{metadata_key}: {adata.obs[identifier].values[subset_identifier_list[0]]}"
                    if len(subset_identifier_list) == 1
                    else f"{metadata_key}"
                )
                plt.ylabel(
                    f"Baseline: {adata.obs[identifier].values[subset_identifier_list[0]]}"
                    if len(subset_identifier_list) == 1
                    else "Baseline"
                )

                # Annotate with the correlation coefficient and p-value of the chosen cell lines
                subset_cor = np.mean(np.diag(corr.iloc[subset_identifier_list, subset_identifier_list]))
                subset_pval = np.mean(np.diag(pval.iloc[subset_identifier_list, subset_identifier_list]))
                annotation = "\n".join(
                    (
                        f"Pearson correlation: {subset_cor:.4f}",
                        f"P-value: {subset_pval:.4f}",
                    )
                )

            plt.text(
                0.05,
                0.95,
                annotation,
                fontsize=10,
                transform=plt.gca().transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.5,
                    "facecolor": "white",
                    "edgecolor": "black",
                },
            )

            if return_fig:
                return plt.gcf()
            plt.show()
            return None
        else:
            raise NotImplementedError("Only 'cell line' category is supported for correlation comparison.")
