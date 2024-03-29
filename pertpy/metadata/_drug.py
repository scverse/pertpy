from __future__ import annotations

import json
from collections import ChainMap
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


class Drug(MetaData):
    """Utilities to fetch metadata for drug studies."""

    def __init__(self):
        self.chembl = None
        self.dgidb = None

    def _download_drug_annotation(
        self,
        source: Literal["chembl", "dgidb"] = "chembl",
    ) -> None:
        if source == "chembl" and self.chembl is None:
            # Prepared in https://github.com/theislab/pertpy-datasets/blob/main/chembl_data.ipynb
            chembl_path = Path(settings.cachedir) / "chembl.json"
            if not Path(chembl_path).exists():
                print("[bold yellow]No metadata file was found for chembl. Starting download now.")
                _download(
                    url="https://figshare.com/ndownloader/files/43871718",
                    output_file_name="chembl.json",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            with chembl_path.open() as file:
                chembl_json = json.load(file)
            self._chembl_json = chembl_json
            targets = dict(ChainMap(*[chembl_json[cat] for cat in chembl_json]))
            self.chembl = pd.DataFrame([{"Compound": k, "Targets": v} for k, v in targets.items()])
            self.chembl.rename(columns={"Targets": "targets", "Compound": "compounds"}, inplace=True)
        if source == "dgidb" and self.dgidb is None:
            dgidb_path = Path(settings.cachedir) / "dgidb.tsv"
            if not Path(dgidb_path).exists():
                print("[bold yellow]No metadata file was found for dgidb. Starting download now.")
                _download(
                    url="https://www.dgidb.org/data/latest/interactions.tsv",
                    output_file_name="dgidb.tsv",
                    output_path=settings.cachedir,
                    block_size=4096,
                    is_zip=False,
                )
            self.dgidb_df = pd.read_table(dgidb_path)
            self.dgidb = self.dgidb_df.groupby("drug_claim_name")["gene_claim_name"].apply(list).reset_index()
            self.dgidb.rename(
                columns={"gene_claim_name": "targets", "drug_claim_name": "compounds"},
                inplace=True,
            )
            self.dgidb_dict = self.dgidb.set_index("compounds")["targets"].to_dict()

    def annotate(
        self,
        adata: AnnData,
        source: Literal["chembl", "dgidb"] = "chembl",
        copy: bool = False,
    ) -> AnnData:
        """Annotates genes by their involvement in applied drugs.

        Genes need to be in HGNC format.

        Args:
            adata: AnnData object containing log-normalised data.
            source: Source of the metadata, chembl or dgidb. Defaults to chembl.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            An AnnData object with a new column `drug` in the var slot.
        """
        if copy:
            adata = adata.copy()

        self._download_drug_annotation(source)

        interaction = None
        if source == "chembl":
            interaction = self.chembl
        else:
            interaction = self.dgidb

        exploded_df = interaction.explode("targets")

        gene_compound_dict = (
            exploded_df.groupby("targets")["compounds"]
            .apply(lambda compounds: "|".join(sorted(set(compounds))))
            .to_dict()
        )

        adata.var["compounds"] = adata.var_names.map(lambda gene: gene_compound_dict.get(gene, ""))

        return adata

    def lookup(self) -> LookUp:
        """Generate LookUp object for Drug.

        The LookUp object provides an overview of the metadata to annotate.
        annotate function has a corresponding lookup function in the LookUp object,
        where users can search the compound and targets in the metadata.

        Returns:
            Returns a LookUp object specific for drug annotation.
        """
        if self.chembl is None:
            self._download_drug_annotation()
        if self.dgidb is None:
            self._download_drug_annotation(source="dgidb")

        return LookUp(
            type="drug",
            transfer_metadata=[self.chembl, self.dgidb_df],
        )
