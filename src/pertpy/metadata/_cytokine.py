from __future__ import annotations

from functools import cached_property
from typing import Literal

import pandas as pd

from ._metadata import MetaData


class Cytokine(MetaData):
    """Cytokine response signatures of human PBMCs from the Human Cytokine Dictionary."""

    @cached_property
    def dictionary(self) -> pd.DataFrame:
        """Genes that 90 cytokines regulate in 24 PBMC cell types, with an adjusted p-value of at most 0.05.

        Columns are `celltype`, `cytokine`, `gene`, `log_fc` (log2 fold change against PBS), `adj_p_value` and `well_biased`.
        """
        return pd.read_parquet(self._download_metadata("human_cytokine_dictionary.parquet"))

    def gene_sets(
        self,
        celltype: str,
        *,
        direction: Literal["up", "down"] = "up",
        max_adj_p_value: float = 0.01,
        min_abs_log_fc: float = 1.0,
        exclude_well_biased: bool = True,
    ) -> dict[str, list[str]]:
        """Gene sets of the genes that each cytokine regulates in a cell type.

        The gene sets can be passed as `targets` to the tests of :class:`~pertpy.tools.Enrichment`.

        Args:
            celltype: Cell type of the dictionary, one of the categories of `dictionary["celltype"]`.
            direction: Whether to collect the genes that the cytokines up- or downregulate.
            max_adj_p_value: Largest adjusted p-value of a gene in a set.
                The dictionary only holds genes up to 0.05.
            min_abs_log_fc: Smallest absolute log2 fold change of a gene in a set.
            exclude_well_biased: Whether to leave out the genes that the dictionary flags as differing between culture wells regardless of the cytokine.

        Returns:
            Gene sets keyed by cytokine, leaving out cytokines without any gene that passes the thresholds.

        Examples:
            >>> import pertpy as pt
            >>> targets = pt.md.Cytokine().gene_sets("CD14_Mono")
        """
        dictionary = self.dictionary
        if celltype not in dictionary["celltype"].cat.categories:
            raise ValueError(
                f"{celltype!r} is not a cell type of the Human Cytokine Dictionary, "
                f"use one of {', '.join(sorted(dictionary['celltype'].cat.categories))}."
            )
        log_fc = dictionary["log_fc"] if direction == "up" else -dictionary["log_fc"]
        mask = (
            (dictionary["celltype"] == celltype)
            & (dictionary["adj_p_value"] <= max_adj_p_value)
            & (log_fc >= min_abs_log_fc)
        )
        if exclude_well_biased:
            mask &= ~dictionary["well_biased"]
        genes = dictionary.loc[mask].groupby("cytokine", observed=True)["gene"]
        return {str(cytokine): list(cytokine_genes) for cytokine, cytokine_genes in genes}
