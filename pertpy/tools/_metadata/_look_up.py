from __future__ import annotations
from typing import Literal
from pathlib import Path
from scanpy import settings
import pandas as pd

class LookUp:
    def __init__(self,
                 type: str = "cell_line"):
        self.type = "cell_line"
        if type == "cell_line":
            cell_line_file_path = settings.cachedir.__str__() + "/sample_info.csv"
            if not Path(cell_line_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.cell_line_meta = pd.read_csv(cell_line_file_path)

            cell_line_cancer_project_file_path = settings.cachedir.__str__() + "/cell_line_cancer_project_transformed.csv"
            if not Path(cell_line_cancer_project_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.cl_cancer_project_meta = pd.read_csv(cell_line_cancer_project_file_path)
            
            driver_gene_intOGen_file_path = (
                settings.cachedir.__str__() + "/2020-02-02_IntOGen-Drivers-20200213/Compendium_Cancer_Genes.tsv"
            )
            if not Path(driver_gene_intOGen_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.driver_gene_intOGen = pd.read_table(driver_gene_intOGen_file_path)
            self.driver_gene_intOGen.rename(columns=lambda x: x.lower(), inplace=True)

            self.driver_gene_cosmic = pd.read_csv("https://www.dropbox.com/s/8azkmt7vqz56e2m/COSMIC_tier1.csv?dl=1")
        
            bulk_rna_sanger_file_path = settings.cachedir.__str__() + "/rnaseq_sanger_20210316_trimm.csv"
            if not Path(bulk_rna_sanger_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.bulk_rna_sanger = pd.read_csv(bulk_rna_sanger_file_path, index_col=0)
        
            bulk_rna_broad_file_path = settings.cachedir.__str__() + "/rnaseq_broad_20210317_trimm.csv"
            if not Path(bulk_rna_broad_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.bulk_rna_broad = pd.read_csv(bulk_rna_broad_file_path, index_col=0)

            proteomics_file_path = settings.cachedir.__str__() + "/proteomics_all_20221214_trimm.csv"
            if not Path(proteomics_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.proteomics_data = pd.read_csv(proteomics_file_path, index_col=0)
        
            ccle_expr_file_path = settings.cachedir.__str__() + "/CCLE_expression.csv"
            if not Path(ccle_expr_file_path).exists():
                raise ValueError("CellLineMetaData was not sucessfully initialized!")
            self.ccle_expr = pd.read_csv(ccle_expr_file_path, index_col=0)
              
    def cell_lines(
        self,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        reference_id: str = "DepMap_ID",
        query_id_list: list[str]| None = None,
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
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")
        
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

        print("Default parameters to annotate cell line metadata: ")
        default_param = {
            "query_id": "DepMap_ID",
            "reference_id": "DepMap_ID",
            "cell_line_information": "None",
            "cell_line_source": "DepMap",
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

    def driver_genes(self, driver_gene_set: Literal["intOGen", "cosmic"] = "intOGen") -> None:
        """A brief summary of genes in cancer driver annotation data.

        Args:
            driver_gene_set: gene set for cancer driver annotation: intOGen or cosmic. Defaults to "intOGen".
        """
        # only availble for CellLineMetaData.lookup
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object spefic for CellLineMetaData!")
        
        if driver_gene_set == "intOGen":
            print("To summarize: in the DepMap_Sanger driver gene annotation for intOGen genes, you can find: ")
            print(f"{len(self.driver_gene_intOGen.index)} driver genes")
            print(
                f"{len(self.driver_gene_intOGen.columns)} meta data including: ",
                *list(self.driver_gene_intOGen.columns.values),
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

