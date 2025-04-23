from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Literal

from lamin_utils import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

if TYPE_CHECKING:
    import pandas as pd

import pubchempy as pcp


class LookUp:
    """Generate LookUp object for different type of metadata."""

    def __init__(
        self,
        type: Literal["cell_line", "moa", "compound", "drug"] = "cell_line",
        transfer_metadata: Sequence[pd.DataFrame] | None = None,
    ):
        """Lookup object for different type of metadata.

        Args:
            type: Metadata type for annotation. One of 'cell_line', 'compound', 'moa' or 'drug.
            transfer_metadata: DataFrames used to generate Lookup object.
                           This is currently set to None for CompoundMetaData which does not require any dataframes for transfer.
        """
        self.type = type
        if type == "cell_line":
            self.cell_line_meta = transfer_metadata[0]
            self.cl_cancer_project_meta = transfer_metadata[1]
            self.gene_annotation = transfer_metadata[2]
            self.bulk_rna_sanger = transfer_metadata[3]
            self.bulk_rna_broad = transfer_metadata[4]
            self.proteomics_data = transfer_metadata[5]
            self.drug_response_gdsc1 = transfer_metadata[6]
            self.drug_response_gdsc2 = transfer_metadata[7]

            cell_line_annotation = namedtuple(
                "cell_line_annotation",
                "n_cell_line cell_line n_metadata metadata reference_id reference_id_example default_parameter",
            )
            cell_lines = namedtuple("cell_lines", ["depmap", "cancerrxgene"])

            depmap_data = {
                "n_cell_line": len(self.cell_line_meta.index),
                "n_metadata": len(self.cell_line_meta.columns),
                "cell_line": self.cell_line_meta.ModelID.values,
                "metadata": self.cell_line_meta.columns.values,
                "reference_id": [
                    "ModelID",
                    "CellLineName",
                    "StrippedCellLineName",
                    "CCLE_Name",
                ],
                "reference_id_example": "ModelID: ACH-000001 | CellLineName: NIH:OVCAR-3 | StrippedCellLineName: NIHOVCAR3 | CCLEName: NIHOVCAR3_OVARY",
                "default_parameter": {
                    "cell_line_source": "DepMap",
                    "query_id": "DepMap_ID",
                    "reference_id": "ModelID",
                    "fetch": "None",
                },
            }
            depmap_record = cell_line_annotation(**depmap_data)

            cancerrxgene_data = {
                "n_cell_line": len(self.cl_cancer_project_meta.index),
                "n_metadata": len(self.cl_cancer_project_meta.columns),
                "cell_line": self.cl_cancer_project_meta.stripped_cell_line_name.values,
                "metadata": self.cl_cancer_project_meta.columns.values,
                "reference_id": [
                    "cell_line_name",
                    "stripped_cell_line_name",
                    "Model ID",
                    "COSMIC ID",
                ],
                "reference_id_example": "cell_line_name: SNU-283 | stripped_cell_line_name: SNU283 | Model ID: SIDM00215 | COSMIC ID: 1659929",
                "default_parameter": {
                    "query_id": "stripped_cell_line_name",
                    "reference_id": "stripped_cell_line_name",
                    "fetch": "None",
                },
            }
            cancerrxgene_record = cell_line_annotation(**cancerrxgene_data)
            self.cell_lines = cell_lines(depmap_record, cancerrxgene_record)

            bulk_rna_annotation = namedtuple(
                "bulk_rna_annotation",
                "n_cell_line cell_line n_gene gene reference_id reference_id_example default_parameter",
            )
            bulk_rna_expression = namedtuple("bulk_rna_expression", ["broad", "sanger"])

            broad_data = {
                "n_cell_line": len(self.bulk_rna_broad.index),
                "n_gene": len(self.bulk_rna_broad.columns),
                "cell_line": self.bulk_rna_broad.index.values,
                "gene": self.bulk_rna_broad.columns.values,
                "reference_id": "DepMap_ID",
                "reference_id_example": "DepMap_ID: ACH-001113",
                "default_parameter": {
                    "query_id": "DepMap_ID",
                    "cell_line_source": "broad",
                },
            }
            broad_record = bulk_rna_annotation(**broad_data)

            sanger_data = {
                "n_cell_line": len(self.bulk_rna_sanger.index),
                "n_gene": len(self.bulk_rna_sanger.columns),
                "cell_line": self.bulk_rna_sanger.index.values,
                "gene": self.bulk_rna_sanger.columns.values,
                "reference_id": "model_name",
                "reference_id_example": "model_name: MEC-1",
                "default_parameter": {
                    "query_id": "cell_line_name",
                    "cell_line_source": "sanger",
                },
            }
            sanger_record = bulk_rna_annotation(**sanger_data)
            self.bulk_rna = bulk_rna_expression(broad_record, sanger_record)

            proteomics = namedtuple(
                "proteomics",
                "n_cell_line cell_line n_protein protein metadata reference_id reference_id_example default_parameter",
            )
            proteomics_data = {
                "n_cell_line": len(self.proteomics_data["model_name"].unique()),
                "n_protein": len(self.proteomics_data.uniprot_id.unique()),
                "cell_line": self.proteomics_data["model_name"].unique(),
                "protein": self.proteomics_data.uniprot_id.unique(),
                "metadata": self.proteomics_data.columns.values,
                "reference_id": ["model_id", "model_name"],
                "reference_id_example": "model_id: SIDM00483 | model_name: SK-GT-4",
                "default_parameter": {
                    "query_id": "cell_line_name",
                    "reference_id": "model_name",
                    "bulk_rna_information": "read_count",
                    "protein_information": "protein_intensity",
                    "protein_id": "uniprot_id",
                },
            }
            self.proteomics = proteomics(**proteomics_data)

            drug_response_annotation = namedtuple(
                "drug_response_annotation",
                "n_cell_line cell_line n_drug drug_name metadata reference_id reference_id_example default_parameter",
            )
            drug_response = namedtuple("drug_response", ["gdsc1", "gdsc2"])

            gdsc1_data = {
                "n_cell_line": len(self.drug_response_gdsc1["cell_line_name"].unique()),
                "n_drug": len(self.drug_response_gdsc1.drug_name.unique()),
                "cell_line": self.drug_response_gdsc1.cell_line_name.unique(),
                "drug_name": self.drug_response_gdsc1.drug_name.unique(),
                "metadata": self.drug_response_gdsc1.columns.values,
                "reference_id": ["cell_line_name", "sanger_model_id", "cosmic_id"],
                "reference_id_example": "cell_line_name: ES5 | sanger_model_id: SIDM00263 | cosmic_id: 684057",
                "default_parameter": {
                    "gdsc_dataset": "1",
                    "query_id": "cell_line_name",
                    "reference_id": "cell_line_name",
                    "query_perturbation": "perturbation",
                    "reference_perturbation": "drug_name",
                },
            }
            gdsc1_dict = drug_response_annotation(**gdsc1_data)

            gdsc2_data = {
                "n_cell_line": len(self.drug_response_gdsc2["cell_line_name"].unique()),
                "n_drug": len(self.drug_response_gdsc2.drug_name.unique()),
                "cell_line": self.drug_response_gdsc2.cell_line_name.unique(),
                "drug_name": self.drug_response_gdsc2.drug_name.unique(),
                "metadata": self.drug_response_gdsc2.columns.values,
                "reference_id": ["cell_line_name", "sanger_model_id", "cosmic_id"],
                "reference_id_example": "cell_line_name: PFSK-1 | sanger_model_id: SIDM01132 | cosmic_id: 683667",
                "default_parameter": {
                    "gdsc_dataset": "1",
                    "query_id": "cell_line_name",
                    "reference_id": "cell_line_name",
                    "query_perturbation": "perturbation",
                    "reference_perturbation": "drug_name",
                },
            }
            gdsc2_dict = drug_response_annotation(**gdsc2_data)

            self.drug_response = drug_response(gdsc1_dict, gdsc2_dict)

        elif type == "moa":
            self.moa_meta = transfer_metadata[0]
            moa_annotation = namedtuple(
                "moa_annotation",
                "n_pert n_moa query_id query_id_example target_example default_parameter",
            )
            moa_data = {
                "n_pert": len(self.moa_meta.pert_iname.unique()),
                "n_moa": len(self.moa_meta.moa.unique()),
                "query_id": "pert_iname",
                "query_id_example": [
                    "(R)-(-)-apomorphine",
                    "9-aminocamptothecin",
                    "A-803467",
                ],
                "target_example": [
                    "ADRA2A|ADRA2B|ADRA2C|CALY|DRD1|DRD2|DRD3|DRD4|DRD5|HTR1A|HTR1B|HTR1D|HTR2A|HTR2B|HTR2C|HTR5A",
                    "SCN10A",
                    "TOP1",
                ],
                "default_parameter": {
                    "query_id": "pert_iname",
                    "target": None,
                },
            }
            self.moa = moa_annotation(**moa_data)

        elif type == "compound":
            compound_annotation = namedtuple("compound_annotation", "query_id query_id_example default_parameter")
            compound_data = {
                "query_id_type": ["name", "cid"],
                "query_id_example": "name: ACH-000016 | cid: SLR 21",
                "default_parameter": {
                    "query_id": "perturbation",
                    "query_id_type": "name",
                },
            }
            self.compound = compound_annotation(**compound_data)

        elif type == "drug":
            self.chembl = transfer_metadata[0]
            self.dgidb = transfer_metadata[1]
            self.pharmgkb = transfer_metadata[2]

            drug_annotation = namedtuple(
                "drug_annotation",
                "n_compound compound_example n_target target_example n_disease disease_example",
            )
            drugs = namedtuple("drugs", ["chembl", "dgidb", "pharmgkb"])

            dgidb_data = {
                "n_compound": len(self.dgidb.drug_claim_name.unique()),
                "n_target": len(self.dgidb.gene_claim_name.unique()),
                "compound_example": self.dgidb.drug_claim_name.values[0:5],
                "target_example": self.dgidb.gene_claim_name.unique()[0:5],
                "n_disease": 0,
                "disease_example": "",
            }
            dgidb_record = drug_annotation(**dgidb_data)

            chembl_targets = list(
                {t for target in self.chembl.targets.tolist() for t in target}
            )  # flatten the target column and remove duplicates
            chembl_data = {
                "n_compound": len(self.chembl.compounds),
                "n_target": len(chembl_targets),
                "compound_example": self.chembl.compounds.values[0:5],
                "target_example": chembl_targets[0:5],
                "n_disease": 0,
                "disease_example": "",
            }
            chembl_record = drug_annotation(**chembl_data)

            pharmgkb_data = {
                "n_compound": len(self.pharmgkb[self.pharmgkb.Type == "Chemical"]["Compound|Disease"].unique()),
                "n_target": len(self.pharmgkb.Gene.unique()),
                "compound_example": self.pharmgkb[self.pharmgkb.Type == "Chemical"]["Compound|Disease"].unique()[0:5],
                "target_example": self.pharmgkb.Gene.unique()[0:5],
                "n_disease": len(self.pharmgkb[self.pharmgkb.Type == "Disease"]["Compound|Disease"].unique()),
                "disease_example": self.pharmgkb[self.pharmgkb.Type == "Disease"]["Compound|Disease"].unique()[0:5],
            }
            pharmgkb_record = drug_annotation(**pharmgkb_data)
            self.drugs = drugs(chembl_record, dgidb_record, pharmgkb_record)

        else:
            raise NotImplementedError

    def available_cell_lines(
        self,
        cell_line_source: Literal["DepMap", "Cancerrxgene"] = "DepMap",
        reference_id: str = "ModelID",
        query_id_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of cell line metadata.

        Args:
            cell_line_source: the source of cell line annotation, DepMap or Cancerrxgene.
            reference_id: The type of cell line identifier in the meta data, e.g. ModelID, CellLineName	or StrippedCellLineName.
                If fetch cell line metadata from Cancerrxgene, it is recommended to choose "stripped_cell_line_name".
            query_id_list: Unique cell line identifiers to test the number of matched ids present in the
                metadata. If set to None, the query of metadata identifiers will be disabled.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specifically for CellLineMetaData!")

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            if cell_line_source == "DepMap":
                if reference_id not in self.cell_line_meta.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available in the DepMap cell line annotation data. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.cell_line_meta[reference_id]))
            else:
                if reference_id == "ModelID":
                    reference_id = "stripped_cell_line_name"
                if reference_id not in self.cl_cancer_project_meta.columns:
                    raise ValueError(
                        f"The specified `reference_id` {reference_id} is not available "
                        f"in the cell line annotation from the project Genomics of Drug Sensitivity in Cancer. "
                    )
                not_matched_identifiers = list(set(query_id_list) - set(self.cl_cancer_project_meta[reference_id]))

            logger.info(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def available_bulk_rna(
        self,
        cell_line_source: Literal["broad", "sanger"] = "sanger",
        query_id_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of bulk RNA expression data.

        Args:
            cell_line_source: the source of RNA-seq data, broad or sanger.
            query_id_list: Unique cell line identifiers to test the number of matched ids present in the
                metadata. If set to None, the query of metadata identifiers will be disabled.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        bulk_rna = self.bulk_rna_broad if cell_line_source == "broad" else self.bulk_rna_sanger

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(bulk_rna.index))

            logger.info(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def available_protein_expression(
        self,
        reference_id: Literal["model_name", "model_id"] = "model_name",
        query_id_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of protein expression data.

        Args:
            reference_id: The type of cell line identifier in the meta data, model_name or model_id.
            query_id_list: Unique cell line identifiers to test the number of matched ids present in the
                metadata. If set to None, the query of metadata identifiers will be disabled.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        if query_id_list is not None:
            identifier_num_all = len(query_id_list)

            if reference_id not in self.proteomics_data.columns:
                raise ValueError(
                    f"The specified `reference_id` {reference_id} is not available in the proteomics data. "
                )
            not_matched_identifiers = list(set(query_id_list) - set(self.proteomics_data[reference_id]))
            logger.info(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

    def available_drug_response(
        self,
        gdsc_dataset: Literal[1, 2] = 1,
        reference_id: Literal["cell_line_name", "sanger_model_id", "cosmic_id"] = "cell_line_name",
        query_id_list: Sequence[str] | None = None,
        reference_perturbation: Literal["drug_name", "drug_id"] = "drug_name",
        query_perturbation_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of drug response data.

        Args:
            gdsc_dataset: The GDSC dataset, 1 or 2.
                          The GDSC1 dataset updates previous releases with additional drug screening data from the Wellcome Sanger Institute and Massachusetts General Hospital.
                          It covers 970 Cell lines and 403 Compounds with 333292 IC50s.
                          GDSC2 is new and has 243,466 IC50 results from the latest screening at the Wellcome Sanger Institute using improved experimental procedures.
            reference_id: The type of cell line identifier in the meta data, cell_line_name, sanger_model_id or cosmic_id.
            query_id_list: Unique cell line identifiers to test the number of matched ids present in the metadata.
                           If set to None, the query of metadata identifiers will be disabled.
            reference_perturbation: The perturbation information in the meta data, drug_name or drug_id.
            query_perturbation_list: Unique perturbation types to test the number of matched ones present in the metadata.
                                     If set to None, the query of perturbation types will be disabled.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")
        gdsc_data = self.drug_response_gdsc1 if gdsc_dataset == 1 else self.drug_response_gdsc2

        if query_id_list is not None:
            if reference_id not in gdsc_data.columns:
                raise ValueError(
                    f"The specified `reference_id` {reference_id} is not available in the GDSC drug response data. "
                )
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(gdsc_data[reference_id]))
            logger.info(f"{len(not_matched_identifiers)} cell lines are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} cell lines are found! ")

        if query_perturbation_list is not None:
            if reference_perturbation not in gdsc_data.columns:
                raise ValueError(
                    f"The specified `reference_perturbation` {reference_perturbation} is not available in the GDSC drug response data. "
                )
            identifier_num_all = len(query_perturbation_list)
            not_matched_identifiers = list(set(query_perturbation_list) - set(gdsc_data[reference_perturbation]))
            logger.info(f"{len(not_matched_identifiers)} perturbation types are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} perturbation types are found! ")

    def available_genes_annotation(
        self,
        reference_id: Literal["gene_id", "ensembl_gene_id", "hgnc_id", "hgnc_symbol"] = "ensembl_gene_id",
        query_id_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of gene annotation metadata.

        Args:
            reference_id: The type of gene identifier in the meta data, gene_id, ensembl_gene_id, hgnc_id, hgnc_symbol.
            query_id_list: Unique gene identifiers to test the number of matched ids present in the metadata.
        """
        if self.type != "cell_line":
            raise ValueError("This is not a LookUp object specific for CellLineMetaData!")

        logger.info("To summarize: in the DepMap_Sanger gene annotation file, you can find: ")
        logger.info(f"{len(self.gene_annotation.index)} driver genes")
        logger.info(
            f"{len(self.gene_annotation.columns)} meta data including: ",
            *list(self.gene_annotation.columns.values),
            sep="\n- ",
        )
        logger.info("Overview of gene annotation: ")
        logger.info(self.gene_annotation.head().to_string())
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

    def available_moa(
        self,
        query_id_list: Sequence[str] | None = None,
        target_list: Sequence[str] | None = None,
    ) -> None:
        """A brief summary of MoA annotation.

        Args:
            query_id_list: Unique perturbagens to test the number of matched ones present in the metadata.
                           If set to None, the query of metadata perturbagens will be disabled.
            target_list: Unique molecular targets to test the number of matched ones present in the metadata.
                         If set to None, the comparison of molecular targets in the query of metadata perturbagens will be disabled.
        """
        if query_id_list is not None:
            if self.type != "moa":
                raise ValueError("This is not a LookUp object specific for MoaMetaData!")
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = list(set(query_id_list) - set(self.moa_meta.pert_iname))
            logger.info(f"{len(not_matched_identifiers)} perturbagens are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} perturbagens are found! ")

        if target_list is not None:
            targets = self.moa_meta.target.astype(str).apply(lambda x: x.split("|"))
            all_targets = [t for tl in targets for t in tl]
            identifier_num_all = len(target_list)
            not_matched_identifiers = list(set(target_list) - set(all_targets))
            logger.info(f"{len(not_matched_identifiers)} molecular targets are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} molecular targets are found! ")

    def available_compounds(
        self,
        query_id_list: Sequence[str] | None = None,
        query_id_type: Literal["name", "cid"] = "name",
    ) -> None:
        """A brief summary of compound annotation.

        Args:
            query_id_list: Unique compounds to test the number of matched ones present in the metadata.
                        If set to None, query of compound identifiers will be disabled.
            query_id_type: The type of compound identifiers, name or cid.
        """
        if self.type != "compound":
            raise ValueError("This is not a LookUp object specific for CompoundData!")
        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = []

            for compound in query_id_list:
                if query_id_type == "name":
                    cids = pcp.get_compounds(compound, "name")
                    if len(cids) == 0:  # search did not work
                        not_matched_identifiers.append(compound)
                else:
                    try:
                        pcp.Compound.from_cid(compound)
                    except pcp.BadRequestError:
                        not_matched_identifiers.append(compound)

            logger.info(f"{len(not_matched_identifiers)} compounds are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} compounds are found! ")

    def available_drug_annotation(
        self,
        drug_annotation_source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
        query_id_list: Sequence[str] | None = None,
        query_id_type: Literal["target", "compound", "disease"] = "target",
    ) -> None:
        """A brief summary of drug annotation.

        Args:
            drug_annotation_source: the source of drug annotation data, chembl, dgidb or pharmgkb.
            query_id_list: Unique target or compound names to test the number of matched ones present in the metadata.
                        If set to None, query of compound identifiers will be disabled.
            query_id_type: The type of identifiers, target, compound and disease(pharmgkb only).
        """
        if self.type != "drug":
            raise ValueError("This is not a LookUp object specific for DrugMetaData!")
        if query_id_list is not None:
            identifier_num_all = len(query_id_list)
            not_matched_identifiers = []

            if drug_annotation_source == "chembl":
                if query_id_type == "target":
                    chembl_targets = {t for target in self.chembl.targets.tolist() for t in target}
                    # flatten the target column and remove duplicates
                    not_matched_identifiers = list(set(query_id_list) - chembl_targets)
                elif query_id_type == "compound":
                    not_matched_identifiers = list(set(query_id_list) - self.chembl["compounds"])
                else:
                    raise ValueError(
                        "Gene-disease association is not available in chembl dataset, please try with pharmgkb."
                    )

            elif drug_annotation_source == "dgidb":
                if query_id_type == "target":
                    not_matched_identifiers = list(set(query_id_list) - set(self.dgidb["gene_claim_name"]))
                elif query_id_type == "compound":
                    not_matched_identifiers = list(set(query_id_list) - set(self.dgidb["drug_claim_name"]))
                else:
                    raise ValueError(
                        "Gene-disease association is not available in dgidb dataset, please try with pharmgkb."
                    )
            elif query_id_type == "target":
                not_matched_identifiers = list(set(query_id_list) - set(self.pharmgkb["Gene"]))
            elif query_id_type == "compound":
                compounds = self.pharmgkb[self.pharmgkb["Type"] == "Chemical"]
                not_matched_identifiers = list(set(query_id_list) - set(compounds["Compound|Disease"]))
            else:
                diseases = self.pharmgkb[self.pharmgkb["Type"] == "Disease"]
                not_matched_identifiers = list(set(query_id_list) - set(diseases["Compound|Disease"]))

            logger.info(f"{len(not_matched_identifiers)} {query_id_type}s are not found in the metadata.")
            logger.info(f"{identifier_num_all - len(not_matched_identifiers)} {query_id_type}s are found! ")
