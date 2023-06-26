from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import pandas as pd
from scanpy import settings


class LookUp:
    def __init__(self, type: str = "cell_line", transfer_metadata: list[pd.DataFrame] | None = None):
        if type == "cell_line":
            self.type = type
            self.cell_line_meta = transfer_metadata[0]
            self.cl_cancer_project_meta = transfer_metadata[1]
            self.driver_gene_intogen = transfer_metadata[2]
            self.driver_gene_cosmic = transfer_metadata[3]
            self.bulk_rna_sanger = transfer_metadata[4]
            self.bulk_rna_broad = transfer_metadata[5]
            self.proteomics_data = transfer_metadata[6]
            self.ccle_expr = transfer_metadata[7]
            self.drug_response_gdsc1 = transfer_metadata[8]
            self.drug_response_gdsc2 = transfer_metadata[9]

    def cell_lines(
        self,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        reference_id: str = "DepMap_ID",
        query_id_list: list[str] | None = None,
    ) -> None:
        """A brief summary of cell line metadata.

        Args:
            cell_line_source: the source of cell line annotation, DepMap or Cancerrxgene. Defaults tp "DepMap".
            reference_id: The type of cell line identifier in the meta data, e.g. DepMap_ID, cell_line_name or
                stripped_cell_line_name. If fetch cell line metadata from Cancerrxgene, it is recommended to choose
                "stripped_cell_line_name". Defaults to "DepMap_ID".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the
                metadata. Defaults to None.
        """
        # only available for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specifically for CellLineMetaData!")

        if cell_line_source == "DepMap":
            print("To summarize: in the DepMap cell line annotation you can find: ")
            print(f"{len(self.cell_line_meta.index)} cell lines")
            print(
                f"{len(self.cell_line_meta.columns)} meta data, including ",
                *list(self.cell_line_meta.columns.values),
                sep="\n- ",
            )
            print("Overview of possible cell line reference identifiers: ")
            print(
                self.cell_line_meta[["DepMap_ID", "cell_line_name", "stripped_cell_line_name", "CCLE_Name"]]
                .head()
                .to_string()
            )
            print("Default parameters to annotate cell line metadata: ")
            default_param = {
                "cell_line_source": "DepMap",
                "query_id": "DepMap_ID",
                "reference_id": "DepMap_ID",
                "cell_line_information": "None",
            }
            print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))
        else:
            print(
                "To summarize: in the cell line annotation from the project Genomics of Drug Sensitivity in Cancer",
                "you can find: ",
            )
            print(f"{len(self.cl_cancer_project_meta.index)} cell lines")
            print(
                f"{len(self.cl_cancer_project_meta.columns)} meta data, including ",
                *list(self.cl_cancer_project_meta.columns.values),
                sep="\n- ",
            )
            print("Overview of possible cell line reference identifiers: ")
            print(
                self.cl_cancer_project_meta[["cell_line_name", "stripped_cell_line_name", "Model ID", "COSMIC ID"]]
                .head()
                .to_string()
            )

            print("Default is to annotate cell line meta data from DepMap database. ")
            print("However, to annotate cell line meta data from the project Genomics of Drug Sensitivity in Cancer, ")
            print("Default parameters are:")
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
                    print(
                        "To annotate cell line metadata from Cancerrxgene, ",
                        "we use `stripped_cell_line_name` as reference indentifier. ",
                    )
                    reference_id = "stripped_cell_line_name"
                if reference_id not in self.cl_cancer_project_meta.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available in the cell line annotation from the project Genomics of Drug Sensitivity in Cancer. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.cl_cancer_project_meta[reference_id]))

            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def bulk_rna_expression(
        self,
        cell_line_source: Literal["broad", "sanger"] = "broad",
        reference_id: Literal["model_name", "model_id"] = "model_name",
        query_id_list: list[str] | None = None,
    ) -> None:
        """A brief summary of bulk RNA expression data.

        Args:
            cell_line_source: the source of RNA-seq data, broad or sanger. Defaults to "broad".
            reference_id: The type of cell line identifier in the meta data, model_name, or model_id.
                Defaults to "model_name".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the
                metadata. Defaults to None.
        """
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")

        if cell_line_source == "broad":
            print("To summarize: in the RNA-Seq Data for broad cell line only, you can find: ")
            print(f"{len(self.bulk_rna_broad.model_name.unique())} cell lines")
            print(f"{len(self.bulk_rna_broad.gene_id.unique())} genes")
            print(
                f"{len(self.bulk_rna_broad.columns)} meta data, including ",
                *list(self.bulk_rna_broad.columns.values),
                sep="\n- ",
            )
            print("Overview of possible cell line reference identifiers: ")
            print(self.bulk_rna_broad[["model_id", "model_name"]].head().to_string())

        else:
            print("To summarize: in the RNA-Seq Data for Sanger cell line only, you can find: ")
            print(f"{len(self.bulk_rna_sanger.model_name.unique())} cell lines")
            print(f"{len(self.bulk_rna_sanger.gene_id.unique())} genes")
            print(
                f"{len(self.bulk_rna_sanger.columns)} meta data, including ",
                *list(self.bulk_rna_sanger.columns.values),
                sep="\n- ",
            )
            print("Overview of possible cell line reference identifiers: ")
            print(self.bulk_rna_sanger[["model_id", "model_name"]].head().to_string())

        print("Default parameters to annotate bulk RNA expression: ")
        default_param = {
            "query_id": "cell_line_name",
            "reference_id": "model_name",
            "cell_line_source": "broad",
            "bulk_rna_information": "read_count",
        }
        print("\n".join(f"- {k}: {v}" for k, v in default_param.items()))

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            if cell_line_source == "broad":
                if reference_id not in self.bulk_rna_broad.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available in the RNA-Seq Data for broad cell line. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.bulk_rna_broad[reference_id]))
            else:
                if reference_id not in self.bulk_rna_sanger.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available in the RNA-Seq Data for Sanger cell line. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.bulk_rna_sanger[reference_id]))

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
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")

        print("To summarize: in the proteomics data you can find: ")
        print(f"{len(self.proteomics_data.model_name.unique())} cell lines")
        print(f"{len(self.proteomics_data.uniprot_id.unique())} proteins")
        print(
            f"{len(self.proteomics_data.columns)} meta data, including ",
            *list(self.proteomics_data.columns.values),
            sep="\n- ",
        )
        print("Overview of possible cell line reference identifiers: ")
        print(self.proteomics_data[["model_id", "model_name"]].head().to_string())

        print("Default parameters to annotate protein expression: ")
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
            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def ccle_expression(self, query_id_list: list[str] | None = None) -> None:
        """A brief summary of CCLE expression data.

        Args:
            query_id_list: A list of unique cell line identifiers (here DepMap_ID) to test the number of
                matched ids present in the metadata. Defaults to None.

        """
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")

        print("To summarize: in the CCLE expression data you can find: ")
        print(f"{len(self.ccle_expr.index.unique())} cell lines")
        print(f"{len(self.ccle_expr.columns.unique())} genes")
        print("Only DepMap_ID is allowed to use as `reference_id`")

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(self.ccle_expr.index))
            print(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            print(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

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
            gdsc_dataset: The GDSC dataset, 1 or 2. Defaults to 1.
            reference_id: The type of cell line identifier in the meta data, cell_line_name, sanger_model_id or cosmic_id. Defaults to "cell_line_name".
            query_id_list: A list of unique cell line identifiers to test the number of matched ids present in the metadata. Defaults to None.
            reference_perturbation: The perturbation information in the meta data, drug_name or drug_id. Defaults to "drug_name".
            query_perturbation_list: A list of unique perturbation types to test the number of matched ones present in the metadata. Defaults to None.

        """
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")
        if gdsc_dataset == 1:
            gdsc_data = self.drug_response_gdsc1
        else:
            gdsc_data = self.drug_response_gdsc2
        print(f"To summarize: in the drug reponse GDSC{gdsc_dataset} data you can find: ")
        print(f"{len(gdsc_data.cell_line_name.unique())} cell lines")
        print(f"{len(gdsc_data.drug_id.unique())} different drugs ids")
        print(f"{len(gdsc_data.drug_name.unique())} different drugs names")

        print("Overview of possible cell line reference identifiers: ")
        print(gdsc_data[["cell_line_name", "sanger_model_id", "cosmic_id"]].head().to_string())
        print("Overview of possible perturbation reference identifiers: ")
        print(gdsc_data[["drug_name", "drug_id"]].head().to_string())
        print("Default parameters to annotate cell line metadata: ")
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

    def driver_genes(self, driver_gene_set: Literal["intogen", "cosmic"] = "intogen") -> None:
        """A brief summary of genes in cancer driver annotation data.

        Args:
            driver_gene_set: gene set for cancer driver annotation: intogen or cosmic. Defaults to "intogen".
        """
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")

        if driver_gene_set == "intogen":
            print("To summarize: in the DepMap_Sanger driver gene annotation for intogen genes, you can find: ")
            print(f"{len(self.driver_gene_intogen.index)} driver genes")
            print(
                f"{len(self.driver_gene_intogen.columns)} meta data including: ",
                *list(self.driver_gene_intogen.columns.values),
                sep="\n- ",
            )
        else:
            print("To summarize: in the DepMap_Sanger driver gene annotation for COSMIC Tier 1 genes, you can find: ")
            print(f"{len(self.driver_gene_cosmic.index)} driver genes")
            print(
                f"{len(self.driver_gene_cosmic.columns)} meta data including: ",
                *list(self.driver_gene_cosmic.columns.values),
                sep="\n- ",
            )
