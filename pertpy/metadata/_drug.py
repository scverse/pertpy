from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
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
        # Prepared in https://github.com/theislab/pertpy-datasets/blob/main/chembl_data.ipynb
        chembl_path = Path(settings.cachedir) / "chembl.parquet"
        if not Path(chembl_path).exists():
            print("[bold yellow]No metadata file was found for chembl. Starting download now.")
            _download(
                url="https://figshare.com/ndownloader/files/43848687",
                output_file_name="chembl.parquet",
                output_path=settings.cachedir,
                block_size=4096,
                is_zip=False,
            )
        self.chembl = pd.read_parquet(chembl_path)
        self.chembl.rename(columns={"Targets": "targets", "Compound": "compounds"}, inplace=True)

    def annotate(self, adata: AnnData, copy: bool = False) -> AnnData:
        """Annotates genes by their involvement in applied drugs.

        Genes need to be in HGNC format.

        Args:
            adata: AnnData object containing log-normalised data.
            copy: Determines whether a copy of the `adata` is returned. Defaults to False.

        Returns:
            An AnnData object with a new column `drug` in the var slot.
        """
        if copy:
            adata = adata.copy()

        exploded_df = self.chembl.explode("targets")

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
        annotate_moa function has a corresponding lookup function in the LookUp object,
        where users can search the query_ids and targets in the metadata.

        Returns:
            Returns a LookUp object specific for MoA annotation.
        """
        return LookUp(
            type="moa",
            transfer_metadata=[self.chembl],
        )
