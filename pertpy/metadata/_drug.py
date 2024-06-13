from __future__ import annotations

import json
from collections import ChainMap
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from scanpy import settings

from pertpy.data._dataloader import _download

from ._look_up import LookUp
from ._metadata import MetaData

if TYPE_CHECKING:
    from anndata import AnnData


def _download_drug_annotation(
    source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
) -> pd.DataFrame | dict[str, dict[str, list[str]]]:
    if source == "chembl":
        # Prepared in https://github.com/theislab/pertpy-datasets/blob/main/chembl_data.ipynb
        chembl_path = Path(settings.cachedir) / "chembl.json"
        if not Path(chembl_path).exists():
            _download(
                url="https://figshare.com/ndownloader/files/43871718",
                output_file_name="chembl.json",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        with chembl_path.open() as file:
            chembl_json = json.load(file)
        return chembl_json

    elif source == "dgidb":
        dgidb_path = Path(settings.cachedir) / "dgidb.tsv"
        if not Path(dgidb_path).exists():
            _download(
                url="https://www.dgidb.org/data/latest/interactions.tsv",
                output_file_name="dgidb.tsv",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        dgidb_df = pd.read_table(dgidb_path)
        return dgidb_df

    else:
        pharmgkb_path = Path(settings.cachedir) / "pharmgkb.tsv"
        if not Path(pharmgkb_path).exists():
            _download(
                url="https://api.pharmgkb.org/v1/download/file/data/relationships.zip",
                output_file_name="pharmgkb.zip",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=True,
            )
            Path.rename(Path(settings.cachedir) / "relationships.tsv", pharmgkb_path)

        pharmgkb_df = pd.read_table(pharmgkb_path)
        pharmgkb_df = pharmgkb_df[pharmgkb_df["Association"] != "not associated"]
        pharmgkb_df = pharmgkb_df[
            (pharmgkb_df["Entity1_type"] == "Gene")
            & ((pharmgkb_df["Entity2_type"] == "Chemical") | (pharmgkb_df["Entity2_type"] == "Disease"))
        ]
        pharmgkb_df.rename(
            columns={
                "Entity2_name": "Compound|Disease",
                "Entity1_name": "Gene",
                "Entity2_type": "Type",
            },
            inplace=True,
        )
        pharmgkb_df.drop(["Entity1_type", "Entity1_id", "Entity2_id"], axis=1, inplace=True)

        return pharmgkb_df


class Drug(MetaData):
    """Utilities to fetch metadata for drug studies."""

    def __init__(self):
        self.chembl = self.DrugDataBase(database="chembl")
        self.dgidb = self.DrugDataBase(database="dgidb")
        self.pharmgkb = self.DrugDataBase(database="pharmgkb")

    def annotate(
        self,
        adata: AnnData,
        source: Literal["chembl", "dgidb", "pharmgkb"] = "chembl",
        copy: bool = False,
    ) -> AnnData:
        """Annotates genes by their involvement in applied drugs.

        Genes need to be in HGNC format.

        Args:
            adata: AnnData object containing log-normalised data.
            source: Source of the metadata, chembl, dgidb or pharmgkb.
            copy: Determines whether a copy of the `adata` is returned.

        Returns:
            An AnnData object with a new column `drug` in the var slot.
        """
        if copy:
            adata = adata.copy()

        if source == "chembl":
            if not self.chembl.loaded:
                self.chembl.set()
            interaction = self.chembl.dataframe
        elif source == "dgidb":
            if not self.dgidb.loaded:
                self.dgidb.set()
            interaction = self.dgidb.dataframe
        else:
            if not self.pharmgkb.loaded:
                self.pharmgkb.set()
            interaction = self.pharmgkb.data

        if source != "pharmgkb":
            exploded_df = interaction.explode("targets")
            gene_compound_dict = (
                exploded_df.groupby("targets")["compounds"]
                .apply(lambda compounds: "|".join(sorted(set(compounds))))
                .to_dict()
            )

            adata.var["compounds"] = adata.var_names.map(lambda gene: gene_compound_dict.get(gene, ""))
        else:
            compounds = interaction[interaction["Type"] == "Chemical"]
            exploded_df = compounds.explode("Gene")
            gene_compound_dict = (
                exploded_df.groupby("Gene")["Compound|Disease"]
                .apply(lambda compounds: "|".join(sorted(set(compounds))))
                .to_dict()
            )

            adata.var["compounds"] = adata.var_names.map(lambda gene: gene_compound_dict.get(gene, ""))
            diseases = interaction[interaction["Type"] == "Disease"]
            exploded_df = diseases.explode("Gene")
            gene_disease_dict = (
                exploded_df.groupby("Gene")["Compound|Disease"]
                .apply(lambda diseases: "|".join(sorted(set(diseases))))
                .to_dict()
            )

            adata.var["diseases"] = adata.var_names.map(lambda gene: gene_disease_dict.get(gene, ""))
        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for Drug.

        The LookUp object provides an overview of the metadata to annotate.
        annotate function has a corresponding lookup function in the LookUp object,
        where users can search the compound and targets in the metadata.

        Returns:
            Returns a LookUp object specific for drug annotation.
        """
        if not self.chembl.loaded:
            self.chembl.set()
        if not self.dgidb.loaded:
            self.dgidb.set()
        if not self.pharmgkb.loaded:
            self.pharmgkb.set()

        return LookUp(
            type="drug",
            transfer_metadata=[
                self.chembl.dataframe,
                self.dgidb.data,
                self.pharmgkb.data,
            ],
        )

    class DrugDataBase:
        def __init__(self, database: Literal["chembl", "dgidb", "pharmgkb"] = "chembl"):
            self.database = database
            self.loaded = False

        def set(self) -> None:
            self.loaded = True
            data = _download_drug_annotation(source=self.database)
            self.data = data
            if self.database == "chembl":
                if not isinstance(data, dict):
                    raise ValueError(
                        "The chembl data is in a wrong format. Please clear the cache and reinitialize the object."
                    )
                self.dictionary = data
                targets = dict(ChainMap(*[data[cat] for cat in data]))
                self.dataframe = pd.DataFrame([{"Compound": k, "Targets": v} for k, v in targets.items()])
                self.dataframe.rename(
                    columns={"Targets": "targets", "Compound": "compounds"},
                    inplace=True,
                )
            elif self.database == "dgidb":
                if not isinstance(data, pd.DataFrame):
                    raise ValueError(
                        "The dgidb data is in a wrong format. Please clear the cache and reinitialize the object."
                    )
                self.dataframe = data.groupby("drug_claim_name")["gene_claim_name"].apply(list).reset_index()
                self.dataframe.rename(
                    columns={
                        "gene_claim_name": "targets",
                        "drug_claim_name": "compounds",
                    },
                    inplace=True,
                )
                self.dictionary = self.dataframe.set_index("compounds")["targets"].to_dict()
            else:
                if not isinstance(data, pd.DataFrame):
                    raise ValueError(
                        "The pharmGKB data is in a wrong format. Please clear the cache and reinitialize the object."
                    )
                self.dataframe = data.groupby("Compound|Disease")["Gene"].apply(list).reset_index()
                self.dataframe.rename(
                    columns={
                        "Gene": "targets",
                        "Compound|Disease": "compounds|diseases",
                    },
                    inplace=True,
                )
                self.dictionary = self.dataframe.set_index("compounds|diseases")["targets"].to_dict()

        def df(self) -> pd.DataFrame:
            if not self.loaded:
                self.set()
            return self.dataframe

        def dict(self) -> dict[str, list[str]] | dict[str, dict[str, list[str]]]:
            if not self.loaded:
                self.set()
            return self.dictionary
