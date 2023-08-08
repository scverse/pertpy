from __future__ import annotations

from typing import Literal

import pandas as pd
from rich import print


class LookUp:
    """Generate LookUp object for different type of metadata."""

    class LookUpDict(dict):
        """Create a simple dictionary for LookUp object"""

        __getattr__ = dict.get

    def __init__(self, type: str = "cell_line", transfer_metadata: list[pd.DataFrame] | None = None):
        """
        Args:
            type: metadata type. Default: cell_line. Currrently, LookUp object is only implemented for CellLineMetaData.
            transfer_metadata: a list of dataframes used to generate Lookup object. To ensure efficient transfer of
                metadata during initialization, LookUp object should always be generated by the corresponding MetaData
                class. Also, different MetaData classes have different required metadata to transfer.
        """
        if type == "cell_line":
            self.type = type
            self.cell_line_meta = transfer_metadata[0]
            self.cl_cancer_project_meta = transfer_metadata[1]
            self.gene_annotation = transfer_metadata[2]
            self.bulk_rna_sanger = transfer_metadata[3]
            self.bulk_rna_broad = transfer_metadata[4]
            self.proteomics_data = transfer_metadata[5]
            self.drug_response_gdsc1 = transfer_metadata[6]
            self.drug_response_gdsc2 = transfer_metadata[7]

            depmap_data = {
                "n_cell_line": len(self.cell_line_meta.index),
                "n_metadata": len(self.cell_line_meta.columns),
                "cell_line": self.cell_line_meta.DepMap_ID.values,
                "metadata": self.cell_line_meta.columns.values,
                "reference_id": ["DepMap_ID", "cell_line_name", "stripped_cell_line_name", "CCLE_Name"],
                "reference_id_example": "DepMap_ID: ACH-000016 | cell_line_name: SLR 21 | stripped_cell_line_name: SLR21 | CCLE_Name: SLR21_KIDNEY",
            }
            depmap_dict = self.LookUpDict(depmap_data)

            cancerrxgene_data = {
                "n_cell_line": len(self.cl_cancer_project_meta.index),
                "n_metadata": len(self.cl_cancer_project_meta.columns),
                "cell_line": self.cl_cancer_project_meta.stripped_cell_line_name.values,
                "metadata": self.cl_cancer_project_meta.columns.values,
                "reference_id": ["cell_line_name", "stripped_cell_line_name", "Model ID", "COSMIC ID"],
                "reference_id_example": "cell_line_name: SNU-283 | stripped_cell_line_name: SNU283 | Model ID: SIDM00215 | COSMIC ID: 1659929",
            }
            cancerrxgene_dict = self.LookUpDict(cancerrxgene_data)
            self.cell_lines_dict = self.LookUpDict({"depmap": depmap_dict, "cancerrxgene": cancerrxgene_dict})

            broad_data = {
                "n_cell_line": len(self.bulk_rna_broad.index),
                "n_gene": len(self.bulk_rna_broad.columns),
                "cell_line": self.bulk_rna_broad.index.values,
                "gene": self.bulk_rna_broad.columns.values,
                "reference_id": "DepMap_ID",
                "reference_id_example": "DepMap_ID: ACH-001113",
            }
            broad_dict = self.LookUpDict(broad_data)

            sanger_data = {
                "n_cell_line": len(self.bulk_rna_sanger.index),
                "n_gene": len(self.bulk_rna_sanger.columns),
                "cell_line": self.bulk_rna_sanger.index.values,
                "gene": self.bulk_rna_sanger.columns.values,
                "reference_id": "model_name",
                "reference_id_example": "model_name: MEC-1",
            }
            sanger_dict = self.LookUpDict(sanger_data)
            self.bulk_rna_dict = self.LookUpDict({"broad": broad_dict, "sanger": sanger_dict})

            proteomics_data = {
                "n_cell_line": len(self.proteomics_data["model_name"].unique()),
                "n_protein": len(self.proteomics_data.uniprot_id.unique()),
                "cell_line": self.proteomics_data["model_name"].unique(),
                "protein": self.proteomics_data.uniprot_id.unique(),
                "metadata": self.proteomics_data.columns.values,
                "reference_id": ["model_id", "model_name"],
                "reference_id_example": "model_id: SIDM00483 | model_name: SK-GT-4",
            }
            self.proteomics_dict = self.LookUpDict(proteomics_data)

            gdsc1_data = {
                "n_cell_line": len(self.drug_response_gdsc1["cell_line_name"].unique()),
                "n_drug": len(self.drug_response_gdsc1.drug_name.unique()),
                "cell_line": self.drug_response_gdsc1.cell_line_name.unique(),
                "drug_name": self.drug_response_gdsc1.drug_name.unique(),
                "metadata": self.drug_response_gdsc1.columns.values,
                "reference_id": ["cell_line_name", "sanger_model_id", "cosmic_id"],
                "reference_id_example": "cell_line_name: ES5 | sanger_model_id: SIDM00263 | cosmic_id: 684057",
            }
            gdsc1_dict = self.LookUpDict(gdsc1_data)

            gdsc2_data = {
                "n_cell_line": len(self.drug_response_gdsc2["cell_line_name"].unique()),
                "n_drug_": len(self.drug_response_gdsc2.drug_name.unique()),
                "cell_line": self.drug_response_gdsc2.cell_line_name.unique(),
                "drug_name": self.drug_response_gdsc2.drug_name.unique(),
                "metadata": self.drug_response_gdsc2.columns.values,
                "reference_id": ["cell_line_name", "sanger_model_id", "cosmic_id"],
                "reference_id_example": "cell_line_name: PFSK-1 | sanger_model_id: SIDM01132 | cosmic_id: 683667",
            }
            gdsc2_dict = self.LookUpDict(gdsc2_data)

            self.drug_response_dict = self.LookUpDict({"gdsc1": gdsc1_dict, "gdsc2": gdsc2_dict})
        else:
            raise NotImplementedError

    def cell_lines(
        self,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        reference_id: str = "DepMap_ID",
        query_id_list: list[str] | None = None,
    ) -> None:
        """A brief summary of cell line metadata.

        Args:
            cell_line_source: the source of cell line annotation, DepMap or Cancerrxgene. Defaults to "DepMap".
            reference_id: The type of cell line identifier in the meta data, e.g. DepMap_ID, cell_line_name or
                stripped_cell_line_name. If fetch cell line metadata from Cancerrxgene, it is recommended to choose
                "stripped_cell_line_name". Defaults to "DepMap_ID".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the
                metadata. Defaults to None.

        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specifically for CellLineMetaData!")

        if cell_line_source == "DepMap":
            print("[bold blue]Default parameters to annotate cell line metadata:")
            default_param = {
                "cell_line_source": "DepMap",
                "query_id": "DepMap_ID",
                "reference_id": "DepMap_ID",
                "cell_line_information": "None",
            }
            print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))
        else:
            print("[bold blue]Default parameters for Genomics of Drug sensitivity in Cancer are:")
            default_param = {
                "query_id": "stripped_cell_line_name",
                "reference_id": "stripped_cell_line_name",
                "cell_line_information": "None",
            }
            print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            if cell_line_source == "DepMap":
                if reference_id not in self.cell_line_meta.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available in the DepMap cell line annotation data. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.cell_line_meta[reference_id]))
            else:
                if reference_id == "DepMap_ID":
                    reference_id = "stripped_cell_line_name"
                if reference_id not in self.cl_cancer_project_meta.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available "
                        f"in the cell line annotation from the project Genomics of Drug Sensitivity in Cancer. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.cl_cancer_project_meta[reference_id]))

            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def bulk_rna_expression(
        self,
        cell_line_source: Literal["broad", "sanger"] = "sanger",
        query_id_list: list[str] | None = None,
    ) -> None:
        """A brief summary of bulk RNA expression data.

        Args:
            cell_line_source: the source of RNA-seq data, broad or sanger. Defaults to "sanger".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the
                metadata. Defaults to None.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        if cell_line_source == "broad":
            bulk_rna = self.bulk_rna_broad
        else:
            bulk_rna = self.bulk_rna_sanger

        print("[bold blue]Default parameters to annotate bulk RNA expression: ")
        default_param = {
            "query_id": "cell_line_name",
            "cell_line_source": "sanger",
        }
        print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(bulk_rna.index))

            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def protein_expression(
        self, reference_id: Literal["model_name", "model_id"] = "model_name", query_id_list: list[str] | None = None
    ) -> None:
        """A brief summary of protein expression data.

        Args:
            reference_id: The type of cell line identifier in the meta data, model_name or model_id.
                Defaults to "model_name".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the
                metadata. Defaults to None.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        print("[bold blue]Default parameters to annotate protein expression: ")
        default_param = {
            "query_id": "cell_line_name",
            "reference_id": "model_name",
            "bulk_rna_information": "read_count",
            "protein_information": "protein_intensity",
            "protein_id": "uniprot_id",
        }
        print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)

            if reference_id not in self.proteomics_data.columns:
                raise ValueError(
                    f"The specified `reference_id` {reference_id} is not available in the proteomics data. "
                )
            not_matched_identifiers = list(set(query_id_list) - set(self.proteomics_data[reference_id]))
            print(f"[bold blue]{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"[bold yellow]{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def drug_response(
        self,
        gdsc_dataset: Literal[1, 2] = 1,
        reference_id: Literal["cell_line_name", "sanger_model_id", "cosmic_id"] = "cell_line_name",
        query_id_list: list[str] | None = None,
        reference_perturbation: Literal["drug_name", "drug_id"] = "drug_name",
        query_perturbation_list: list[str] | None = None,
    ) -> None:
        """A brief summary of drug response data.

        Args:
            gdsc_dataset: The GDSC dataset, 1 or 2. Defaults to 1. The GDSC1 dataset updates previous releases with additional drug screening data from the Wellcome Sanger Institute and Massachusetts General Hospital. It covers 970 Cell lines and 403 Compounds with 333292 IC50s. GDSC2 is new and has 243,466 IC50 results from the latest screening at the Wellcome Sanger Institute using improved experimental procedures.
            reference_id: The type of cell line identifier in the meta data, cell_line_name, sanger_model_id or cosmic_id. Defaults to "cell_line_name".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the metadata. Defaults to None.
            reference_perturbation: The perturbation information in the meta data, drug_name or drug_id. Defaults to "drug_name".
            query_perturbation_list: A list of unique perturbation types to test the number of matched ones present in the metadata. Defaults to None.

        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")
        if gdsc_dataset == 1:
            gdsc_data = self.drug_response_gdsc1
        else:
            gdsc_data = self.drug_response_gdsc2

        print("[bold blue]Default parameters to annotate cell line metadata: ")
        default_param = {
            "gdsc_dataset": "1",
            "query_id": "cell_line_name",
            "reference_id": "cell_line_name",
            "query_perturbation": "perturbation",
            "reference_perturbation": "drug_name",
        }
        print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))

        if query_id_list is not None:
            if reference_id not in gdsc_data.columns:
                raise ValueError(
                    f"The specified `reference_id` {reference_id} is not available in the GDSC drug response data. "
                )
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(gdsc_data[reference_id]))
            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

        if query_perturbation_list is not None:
            if reference_perturbation not in gdsc_data.columns:
                raise ValueError(
                    f"The specified `reference_perturbation` {reference_perturbation} is not available in the GDSC drug response data. "
                )
            identifier_num_all = len(query_perturbation_list)
            not_matched_identifiers = list(set(query_perturbation_list) - set(gdsc_data[reference_perturbation]))
            print(f"{len(not_matched_identifiers)} perturbation types are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} perturbation types are found! ")

    def genes_annotation(
        self,
        reference_id: Literal["gene_id", "ensembl_gene_id", "hgnc_id", "hgnc_symbol"] = "ensembl_gene_id",
        query_id_list: list[str] | None = None,
    ) -> None:
        """A brief summary of gene annotation metadata

        Args:
            reference_id: The type of gene identifier in the meta data, gene_id, ensembl_gene_id, hgnc_id, hgnc_symbol. Defaults to "ensembl_gene_id".
            query_id_list: A list of unique gene identifiers to test the number of matched ids present in the metadata. Defaults to None.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        print("To summarize: in the DepMap_Sanger gene annotation file, you can find: ")
        print(f"{len(self.gene_annotation.index)} driver genes")
        print(
            f"{len(self.gene_annotation.columns)} meta data including: ",
            *list(self.gene_annotation.columns.values),
            sep="\n- ",
        )
        print("Overview of gene annotation: ")
        print(self.gene_annotation.head().to_string())
        """
        #not implemented yet
        print("Default parameters to annotate gene annotation: ")
        default_param = {
            "query_id": "ensembl_gene_id",
        }
        print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))
        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(self.gene_annotation[reference_id]))
            print(f"{len(not_matched_identifiers)} genes are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} genes are found! ")
        """
